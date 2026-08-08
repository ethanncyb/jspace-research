# Anthropic-style probe-swap (two-hop coordinate swap).
# Model + lens are selected in scripts/_common.py (MODEL_NAME toggle).
#
# Implements data/experiments/README.md "probe-swap" with J-space directions:
#   - baseline: greedy next-token == item.answer (the experiment's own filter);
#   - swap: at every band layer (lens default for the model's depth) and every
#     prompt token position, add
#       strength * ||h_clean|| * (unit J-dir(swap_to) - unit J-dir(intermediate))
#     i.e. remove the intermediate concept's J-space direction and write the
#     replacement's, leaving the rest of the activation unchanged;
#   - score: greedy next-token == item.swap_answer ("intended flip");
#   - control: same construction with matched-norm random unit vectors.
#
# Artifacts: results/probe_swap_<model-tag>.json, results/probe_swap_<model-tag>.txt
from __future__ import annotations

import datetime
import json
import statistics
from pathlib import Path

import torch
from _common import (
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    MODEL_TAG,
    load_model_and_lens,
)

from jlens.hooks import ActivationRecorder, ActivationSteerer

RESULTS = Path("results")
STRENGTHS = (0.1, 0.2, 0.4)


def single_token_id(tok, text: str) -> int | None:
    for variant in (text, " " + text):
        ids = tok(variant, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def matches(decoded: str, gold: str) -> bool:
    return decoded.strip().lower() == gold.lower()


def main() -> None:
    hf, model, lens, tok, device = load_model_and_lens()
    band = lens._default_steering_layers(model)
    print(f"band: layers {band[0]}..{band[-1]}")

    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    records: list[dict] = []
    skipped: list[dict] = []

    for index, item in enumerate(items):
        inter_id = single_token_id(tok, item["intermediate"])
        swap_id = single_token_id(tok, item["swap_to"])
        if inter_id is None or swap_id is None:
            skipped.append({"name": item["name"], "reason": "multi-token concept"})
            continue

        ids = model.encode(item["prompt"])
        positions = list(range(ids.shape[1]))
        final_layer = model.n_layers - 1

        # One clean pass: baseline answer + per-site norms for delta scaling.
        with ActivationRecorder(model.layers, at=set(band) | {final_layer}) as recorder:
            with torch.no_grad():
                model.forward(ids)
        clean_logits = model.unembed(recorder.activations[final_layer][0, -1].float())
        clean_argmax = tok.decode([int(clean_logits.argmax())])
        baseline_ok = matches(clean_argmax, item["answer"])

        record: dict = {
            "name": item["name"],
            "category": item["category"],
            "intermediate": item["intermediate"],
            "answer": item["answer"],
            "swap_to": item["swap_to"],
            "swap_answer": item["swap_answer"],
            "baseline_argmax": clean_argmax,
            "baseline_ok": baseline_ok,
            "trials": [],
        }
        if not baseline_ok:
            records.append(record)
            print(f"[{index + 1}/{len(items)}] {item['name']}: baseline miss "
                  f"(got {clean_argmax!r}, want {item['answer']!r}) — excluded from swap scoring")
            continue

        norms = {
            layer: recorder.activations[layer][0, :, :].float().norm(dim=-1)
            for layer in band
        }  # [T] per layer, reused across strengths (clean pass is shared)

        swap_answer_first_id = tok(item["swap_answer"], add_special_tokens=False).input_ids[0]

        for strength in STRENGTHS:
            for control in ("jspace", "random"):
                deltas: dict[int, torch.Tensor] = {}
                for layer in band:
                    if control == "jspace":
                        dir_a = lens.direction(model, layer, inter_id)
                        dir_b = lens.direction(model, layer, swap_id)
                    else:
                        gen = torch.Generator().manual_seed(
                            10_007 * index + int(strength * 1000) + layer
                        )
                        ref = lens.direction(model, layer, inter_id)
                        dir_a = torch.nn.functional.normalize(
                            torch.randn(model.d_model, generator=gen), dim=0
                        ).to(ref.device)
                        dir_b = torch.nn.functional.normalize(
                            torch.randn(model.d_model, generator=gen), dim=0
                        ).to(ref.device)
                    diff = dir_b - dir_a
                    deltas[layer] = strength * norms[layer][:, None] * diff[None, :]

                with ActivationSteerer(model.layers, deltas=deltas, positions=positions), \
                    ActivationRecorder(model.layers, at=[final_layer]) as steered_recorder:
                    with torch.no_grad():
                        model.forward(ids)
                steered_logits = model.unembed(
                    steered_recorder.activations[final_layer][0, -1].float()
                )
                argmax_text = tok.decode([int(steered_logits.argmax())])
                sa_rank = int(
                    (steered_logits.argsort(descending=True) == swap_answer_first_id)
                    .nonzero()[0]
                )
                record["trials"].append(
                    {
                        "strength": strength,
                        "control": control,
                        "argmax": argmax_text,
                        "flip_to_swap_answer": matches(argmax_text, item["swap_answer"]),
                        "still_answer": matches(argmax_text, item["answer"]),
                        "swap_answer_token_rank": sa_rank,
                    }
                )
        records.append(record)
        n_done = sum(1 for r in records if r["baseline_ok"])
        print(f"[{index + 1}/{len(items)}] {item['name']}: baseline ok ({n_done} usable so far)")

    usable = [r for r in records if r["baseline_ok"]]
    per_strength: list[dict] = []
    for strength in STRENGTHS:
        for control in ("jspace", "random"):
            trials = [
                t for r in usable for t in r["trials"]
                if t["strength"] == strength and t["control"] == control
            ]
            if not trials:
                continue
            per_strength.append(
                {
                    "strength": strength,
                    "control": control,
                    "n": len(trials),
                    "flip_rate": statistics.fmean(t["flip_to_swap_answer"] for t in trials),
                    "still_answer_rate": statistics.fmean(t["still_answer"] for t in trials),
                    "median_swap_answer_rank": float(
                        statistics.median(t["swap_answer_token_rank"] for t in trials)
                    ),
                }
            )

    # Per-category flip rate at the strongest jspace setting (categories with n >= 3).
    best = STRENGTHS[-1]
    per_category: dict[str, dict] = {}
    for r in usable:
        trial = next(
            t for t in r["trials"]
            if t["strength"] == best and t["control"] == "jspace"
        )
        bucket = per_category.setdefault(r["category"], {"n": 0, "flips": 0})
        bucket["n"] += 1
        bucket["flips"] += int(trial["flip_to_swap_answer"])
    per_category = {
        k: {**v, "flip_rate": v["flips"] / v["n"]}
        for k, v in sorted(per_category.items(), key=lambda kv: -kv[1]["n"])
        if v["n"] >= 3
    }

    out = {
        "experiment": "probe-swap (two-hop J-space coordinate swap)",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "lens": f"{LENS_REPO} {LENS_REVISION} {LENS_FILE}",
        "config": {
            "band_layers": [band[0], band[-1]],
            "positions": "all prompt token positions",
            "strengths": list(STRENGTHS),
            "direction": "strength * ||h_clean|| * (unit J-dir(swap_to) - unit J-dir(intermediate))",
            "control": "matched-norm random unit-vector pairs",
            "baseline_filter": "greedy next-token == answer",
        },
        "n_items": len(items),
        "n_baseline_ok": len(usable),
        "n_skipped_multitoken": len(skipped),
        "per_strength": per_strength,
        "per_category_at_max_strength": per_category,
        "skipped": skipped,
        "items": records,
    }
    RESULTS.mkdir(exist_ok=True)
    json_path = RESULTS / f"probe_swap_{MODEL_TAG}.json"
    json_path.write_text(json.dumps(out, indent=2))

    lines = [
        f"probe-swap summary — {MODEL_NAME} + J-space coordinate swap",
        f"items: {len(items)}, baseline-ok: {len(usable)}, skipped (multi-token): {len(skipped)}",
        f"band: layers {band[0]}..{band[-1]}, every prompt position",
        "",
        f"{'strength':>8} {'control':>8} {'n':>4} {'flip_rate':>10} {'still_answer':>12} {'med_sa_rank':>12}",
    ]
    for row in per_strength:
        lines.append(
            f"{row['strength']:>8} {row['control']:>8} {row['n']:>4} "
            f"{row['flip_rate']:>10.3f} {row['still_answer_rate']:>12.3f} "
            f"{row['median_swap_answer_rank']:>12.1f}"
        )
    txt_path = RESULTS / f"probe_swap_{MODEL_TAG}.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {json_path}\nwrote {txt_path}")


if __name__ == "__main__":
    main()
