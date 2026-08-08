# J-space steering demo.
# Model + lens are selected in scripts/_common.py (MODEL_NAME toggle).
#
# Loads the Neuronpedia n=1000 Jacobian lens, steers a single concept token's
# J-space direction into the residual stream (default band for the model's depth,
# prompt token), and records:
#   1. target-token rank / logit lift / KL across strengths, jspace vs random
#      matched-norm controls (via JacobianLens.steer);
#   2. greedy continuations with vs without the intervention active during
#      the prompt pass.
#
# Artifacts: results/steering_demo_<model-tag>.txt (+ .svg rank chart).
from __future__ import annotations

import argparse
import datetime
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

from jlens.examples import EXAMPLES, resolve_prompt
from jlens.hooks import ActivationRecorder

RESULTS = Path("results")
STRENGTHS = (0.0, 0.1, 0.2)

# Concept-token candidates per bundled example; the first single-token
# candidate whose clean rank leaves headroom (not already rank 0) is used.
TARGETS = {
    "overdose-flag": [" emergency", " hospital", " poison", " 911"],
    "greatest-fear": [" spiders", " death", " darkness", " heights"],
}


def single_token_id(tok, text: str) -> int | None:
    ids = tok(text, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def pick_target(model, tok, prompt: str, candidates: list[str]) -> tuple[str, int, int]:
    """Choose a target concept token: single-token, not already clean rank 0.

    Returns (text, token_id, clean_rank).
    """
    ids = model.encode(prompt)
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        with torch.no_grad():
            model.forward(ids)
    logits = model.unembed(recorder.activations[final_layer][0, -1].float())
    order = logits.argsort(descending=True)
    for text in candidates:
        token_id = single_token_id(tok, text)
        if token_id is None:
            continue
        rank = int((order == token_id).nonzero()[0])
        if rank >= 1:  # headroom: steering should move something not already top
            return text, token_id, rank
    # Fallback: best-ranked single-token candidate.
    for text in candidates:
        token_id = single_token_id(tok, text)
        if token_id is not None:
            return text, token_id, int((order == token_id).nonzero()[0])
    raise RuntimeError(f"no single-token candidate among {candidates}")


def steer_generate(hf, model, lens, tok, prompt: str, target_id: int, strength: float, max_new: int) -> str:
    """Greedy continuation with the J-space delta applied at the final prompt
    token during the prompt pass only (cached decoding steps are untouched)."""
    device = model.input_device
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]
    band = lens._default_steering_layers(model)

    # Clean pass to get the per-site activation norms that scale the delta
    # (same convention as JacobianLens.steer).
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=band) as recorder:
        with torch.no_grad():
            model.forward(ids)

    handles = []

    def make_hook(layer: int):
        direction = lens.direction(model, layer, target_id)
        clean_norm = recorder.activations[layer][0, -1, :].float().norm()
        delta = (strength * clean_norm * direction).to(torch.float16)

        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.shape[1] < prompt_len:
                return None  # incremental decoding step: nothing to steer
            changed = hidden.clone()
            changed[:, prompt_len - 1, :] += delta.to(hidden.device, hidden.dtype)
            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        return hook

    try:
        for layer in band:
            handles.append(model.layers[layer].register_forward_hook(make_hook(layer)))
        out = hf.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    return tok.decode(out[0, prompt_len:], skip_special_tokens=True)


def write_svg(path: Path, rows: list[dict]) -> None:
    """Minimal dependency-free line chart: steered target rank vs strength,
    jspace vs random control (log-scaled rank axis)."""
    import math

    width, height, pad = 640, 400, 60
    strengths = sorted({r["strength"] for r in rows})
    series = {
        control: [r for r in rows if r["control"] == control]
        for control in ("jspace", "random")
    }
    max_rank = max(max(r["steered_rank"] for r in rows), max(r["clean_rank"] for r in rows), 10)
    lo, hi = 0.0, math.log10(max_rank + 1)

    def x(strength: float) -> float:
        i = strengths.index(strength)
        return pad + i * (width - 2 * pad) / max(1, len(strengths) - 1)

    def y(rank: int) -> float:
        return height - pad - (math.log10(rank + 1) - lo) / (hi - lo) * (height - 2 * pad)

    colors = {"jspace": "#c23b22", "random": "#555555"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:white;font:12px sans-serif">',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="14">'
        f"J-space steering on {MODEL_NAME}: steered target rank vs strength</text>",
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle">strength</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle">'
        f"target rank (log)</text>",
    ]
    for s in strengths:
        parts.append(
            f'<text x="{x(s)}" y="{height-pad+16}" text-anchor="middle">{s}</text>'
        )
    for tick in (0, 9, 99, 999):
        if tick <= max_rank:
            parts.append(
                f'<text x="{pad-6}" y="{y(tick)+4}" text-anchor="end">{tick}</text>'
                f'<line x1="{pad-3}" y1="{y(tick)}" x2="{pad}" y2="{y(tick)}" stroke="black"/>'
            )
    clean_rank = rows[0]["clean_rank"]
    parts.append(
        f'<line x1="{pad}" y1="{y(clean_rank)}" x2="{width-pad}" y2="{y(clean_rank)}" '
        f'stroke="#2b6cb0" stroke-dasharray="6 4"/>'
        f'<text x="{width-pad-4}" y="{y(clean_rank)-6}" text-anchor="end" fill="#2b6cb0">'
        f"clean rank {clean_rank}</text>"
    )
    for control, points in series.items():
        points = sorted(points, key=lambda r: r["strength"])
        pts = " ".join(f"{x(r['strength'])},{y(r['steered_rank'])}" for r in points)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[control]}" stroke-width="2"/>')
        for r in points:
            parts.append(
                f'<circle cx="{x(r["strength"])}" cy="{y(r["steered_rank"])}" r="4" fill="{colors[control]}"/>'
            )
        parts.append(
            f'<text x="{x(points[-1]["strength"])+8}" y="{y(points[-1]["steered_rank"])+4}" '
            f'fill="{colors[control]}">{control}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default="overdose-flag", choices=[e.slug for e in EXAMPLES])
    parser.add_argument("--target", default=None, help="override the concept token, e.g. ' emergency'")
    parser.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    hf, model, lens, tok, device = load_model_and_lens()
    example = next(e for e in EXAMPLES if e.slug == args.slug)
    prompt = resolve_prompt(example, tok)

    if args.target is not None:
        target_text = args.target
        target_id = single_token_id(tok, target_text)
        if target_id is None:
            raise SystemExit(f"--target {target_text!r} is not a single token")
        clean_rank = -1
    else:
        target_text, target_id, clean_rank = pick_target(
            model, tok, prompt, TARGETS.get(args.slug, [" emergency"])
        )

    ids = model.encode(prompt)
    band = lens._default_steering_layers(model)

    rows: list[dict] = []
    for strength in STRENGTHS:
        controls = ("jspace",) if strength == 0 else ("jspace", "random")
        for control in controls:
            result = lens.steer(
                model,
                ids,
                target_token_id=target_id,
                positions=(-1,),
                strength=strength,
                direction_mode=control,
            )
            rows.append(
                {
                    "strength": strength,
                    "control": control,
                    "clean_rank": int(result.clean_target_ranks[0]),
                    "steered_rank": int(result.steered_target_ranks[0]),
                    "target_logit_lift": float(result.target_logit_lift[0]),
                    "kl_divergence": float(result.kl_divergence[0]),
                }
            )
            print(rows[-1])

    generations: dict[str, str] = {}
    for label, strength in (("baseline", 0.0), ("jspace_0.1", 0.1), ("jspace_0.2", 0.2)):
        print(f"generating {label} ...")
        generations[label] = steer_generate(
            hf, model, lens, tok, prompt, target_id, strength, args.max_new
        )

    RESULTS.mkdir(exist_ok=True)
    txt_path = RESULTS / f"steering_demo_{MODEL_TAG}.txt"
    svg_path = RESULTS / f"steering_demo_{MODEL_TAG}.svg"
    write_svg(svg_path, rows)

    lines = [
        f"J-space steering demo — {MODEL_NAME} + Neuronpedia Jacobian lens",
        f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"example slug : {example.slug} ({example.section})",
        f"prompt       : {prompt!r}",
        f"lens         : {LENS_REPO} rev {LENS_REVISION} file {LENS_FILE} "
        f"(n_prompts={lens.n_prompts}, source_layers {lens.source_layers[0]}..{lens.source_layers[-1]})",
        f"target token : {target_text!r} (id={target_id}, clean rank={rows[0]['clean_rank']})",
        f"band         : layers {band[0]}..{band[-1]}, position = final prompt token",
        "",
        "rank metrics (JacobianLens.steer; strength 0 = clean baseline):",
        f"{'strength':>8} {'control':>8} {'clean_rank':>10} {'steered_rank':>12} {'logit_lift':>10} {'KL':>8}",
    ]
    for r in rows:
        lines.append(
            f"{r['strength']:>8} {r['control']:>8} {r['clean_rank']:>10} "
            f"{r['steered_rank']:>12} {r['target_logit_lift']:>10.3f} {r['kl_divergence']:>8.4f}"
        )
    lines += ["", "greedy continuations (intervention at final prompt token, prompt pass only):"]
    for label, text in generations.items():
        lines += ["", f"--- {label} ---", text.strip()]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {txt_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
