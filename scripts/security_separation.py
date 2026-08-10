# Security-separation: does harmful intent separate from benign intent in the
# Jacobian-lens readout?  The make-or-break measurement for an internal
# (white-box) security guard.
#
# Dataset: data/experiments/security-separation.json — 120 one-line prompts
# across 8 categories: 40 original harmful (explicit threat vocabulary), 40
# original benign, 24 "stealth" harmful (harmful intent, no threat words), and
# 16 "hard" benign (benign intent, contain threat words); plus two concept
# sets: "safety" (threat words) and "neutral" (control words).  For each item
# we run ONE forward pass over the encoded prompt, record activations at every
# band layer (the lens default steering band), and at every (band layer,
# prompt token position) read out vocabulary ranks via the Jacobian lens
# (J_l-transport + unembed, as in JacobianLens.apply).  The headline score is
# the MINIMUM rank any single-token safety concept reaches anywhere in the
# band — lower means the model "sees" threat.  The neutral concept set is
# scored the same way as the control: harmful-vs-benign separation should NOT
# appear for neutral tokens at the same magnitude.
#
# Headline metrics (computed exactly as in the n80 run, for comparability):
# AUROC of harmful-vs-benign against the safety min-rank, per-label median
# min-rank, best single-threshold accuracy (all also for the neutral control),
# and per-category safety medians.
#
# Added diagnostics:
#   * per-item richer records: per-layer min safety/neutral rank, per-layer
#     min safety rank at the LAST prompt token, per-concept and per-position
#     min safety ranks, and per-position-bucket min safety ranks.
#   * per-layer AUROC (which layers separate), per-position-bucket AUROC
#     (all-positions vs last-token-only vs prompt halves), per-concept AUROC
#     (which safety words carry the signal), and alternative aggregates beyond
#     raw min-rank (median over layers of per-layer min, mean over positions
#     of per-position min, last-token min).  Note: monotone transforms of a
#     score (e.g. log-rank) cannot change AUROC, so the aggregates vary the
#     AGGREGATION, not the transform.
#   * cell-stratified AUROC: original-80 (original harmful vs original
#     benign), stealth-harmful vs hard-benign (the key vocabulary-confound
#     test), and all-120 pooled.
#
# Artifacts: results/security_separation_<model-tag>.json,
#            results/security_separation_<model-tag>.txt
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

from jlens.hooks import ActivationRecorder
from jlens.vis import _ranks_of

RESULTS = Path("results")
DATA = Path("data/experiments/security-separation.json")

# Position buckets for the min-rank score: which prompt tokens the minimum is
# taken over (always also over band layers x safety concepts).
POSITION_BUCKETS = ("all_positions", "last_token", "first_half", "second_half")

# Alternative item-level aggregates of the safety rank field.  "min_anywhere"
# is the headline score; the rest are diagnostics.
AGGREGATES = (
    "min_anywhere",
    "last_token_min",
    "layer_median_of_min",
    "position_mean_of_min",
)


def single_token_id(tok, text: str) -> tuple[int, str] | None:
    """Token id + surface form if ``text`` is exactly one token, else None."""
    for variant in (text, " " + text):
        ids = tok(variant, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0], variant
    return None


def filter_concepts(tok, words: list[str]) -> tuple[list[dict], list[str]]:
    """Keep only concepts that tokenize to exactly one token."""
    kept: list[dict] = []
    dropped: list[str] = []
    for word in words:
        hit = single_token_id(tok, word)
        if hit is None:
            dropped.append(word)
        else:
            token_id, variant = hit
            kept.append({"word": word, "variant": variant, "token_id": token_id})
    return kept, dropped


@torch.no_grad()
def band_ranks(model, lens, band: list[int], prompt: str, concept_ids: dict):
    """One forward pass; lens ranks of each concept set at every band layer.

    Returns (n_tokens, ranks_by_set) with ranks_by_set[name][layer] an int64
    tensor [seq_len, n_concepts] of full-vocab ranks (0 = top).
    """
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=band) as recorder:
        model.forward(ids)
    ranks_by_set: dict[str, dict[int, torch.Tensor]] = {
        name: {} for name in concept_ids
    }
    for layer in band:
        residual = recorder.activations[layer][0].float()  # [seq_len, d_model]
        logits = model.unembed(lens.transport(residual, layer)).float().cpu()
        for name, ids_tensor in concept_ids.items():
            ranks_by_set[name][layer] = _ranks_of(logits, ids_tensor)
        del logits
    return ids.shape[1], ranks_by_set


def headline_best(ranks_by_layer: dict[int, torch.Tensor], band: list[int]):
    """Min rank over band x positions x concepts, first-achieved argmin.

    Exactly the scoring of the n80 run: iterate layers in band order, keep the
    strictly smallest rank; argmin flattens [seq_len, n_concepts] row-major.
    """
    best: tuple[int, int, int, int] | None = None
    for layer in band:
        ranks = ranks_by_layer[layer]
        r_min = int(ranks.min())
        if best is None or r_min < best[0]:
            pos, cidx = divmod(int(ranks.argmin()), ranks.shape[1])
            best = (r_min, layer, pos, cidx)
    return best


def position_bucket_mins(stacked: torch.Tensor) -> dict[str, int]:
    """Min rank within each position bucket.

    ``stacked`` is [n_layers, seq_len, n_concepts].
    """
    seq_len = stacked.shape[1]
    half = max(1, seq_len // 2)
    second = stacked[:, half:, :] if half < seq_len else stacked[:, -1:, :]
    return {
        "all_positions": int(stacked.min()),
        "last_token": int(stacked[:, -1, :].min()),
        "first_half": int(stacked[:, :half, :].min()),
        "second_half": int(second.min()),
    }


def auroc_lower_is_positive(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUROC where label 1 is predicted by LOWER scores."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for s_pos in pos:
        for s_neg in neg:
            if s_pos < s_neg:
                wins += 1
            elif s_pos == s_neg:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_threshold(scores: list[float], labels: list[int]) -> dict:
    """Best accuracy of the rule 'harmful iff score <= threshold'."""
    candidates = sorted(set(scores))
    edges = [candidates[0] - 1] + [
        (a + b) / 2 for a, b in zip(candidates, candidates[1:], strict=False)
    ]
    best = {"threshold": edges[0], "accuracy": -1.0}
    for thr in edges:
        correct = sum(
            int((s <= thr) == (y == 1)) for s, y in zip(scores, labels, strict=True)
        )
        accuracy = correct / len(scores)
        if accuracy > best["accuracy"]:
            best = {"threshold": thr, "accuracy": accuracy}
    return best


def summarize(scores: list[float], labels: list[int]) -> dict:
    by_label = {
        name: [s for s, y in zip(scores, labels, strict=True) if y == value]
        for name, value in (("harmful", 1), ("benign", 0))
    }
    thr = best_threshold(scores, labels)
    return {
        "auroc": auroc_lower_is_positive(scores, labels),
        "median_min_rank": {
            name: statistics.median(vals) for name, vals in by_label.items()
        },
        "best_threshold": thr,
    }


def stratum_of(item_id: str) -> str:
    """Dataset cell: 'original' (<cat>-harmful-NN / <cat>-benign-NN) or the
    new confound cell ('stealth' harmful / 'hard' benign)."""
    if "-stealth-" in item_id or "-hard-" in item_id:
        return "stealth_hard"
    return "original"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found — the dataset contract is "
            "{'concepts': {'safety': [...], 'neutral': [...]}, 'items': [...]}"
        )
    dataset = json.loads(DATA.read_text())
    items = dataset["items"]
    labels = [1 if item["label"] == "harmful" else 0 for item in items]
    n_harmful = sum(labels)
    print(
        f"dataset: {len(items)} items "
        f"({n_harmful} harmful / {len(items) - n_harmful} benign), "
        f"categories: {sorted({i['category'] for i in items})}"
    )

    hf, model, lens, tok, device = load_model_and_lens()
    band = lens._default_steering_layers(model)
    print(f"band: layers {band[0]}..{band[-1]} ({len(band)} layers)")

    concept_sets: dict[str, dict] = {}
    concept_ids: dict[str, torch.Tensor] = {}
    for name in ("safety", "neutral"):
        kept, dropped = filter_concepts(tok, dataset["concepts"][name])
        concept_sets[name] = {"kept": kept, "dropped": dropped}
        concept_ids[name] = torch.tensor(
            [entry["token_id"] for entry in kept], dtype=torch.long
        )
        print(
            f"{name} concepts: kept {len(kept)}/{len(kept) + len(dropped)} "
            f"(dropped: {dropped or 'none'})"
        )
    safety_words = [e["word"] for e in concept_sets["safety"]["kept"]]

    records: list[dict] = []
    for index, item in enumerate(items):
        n_tokens, ranks_by_set = band_ranks(
            model, lens, band, item["prompt"], concept_ids
        )
        record = {
            "id": item["id"],
            "label": item["label"],
            "category": item["category"],
            "cell": stratum_of(item["id"]),
            "n_tokens": n_tokens,
        }
        # Headline score, identical to the n80 run.
        for name in ("safety", "neutral"):
            r_min, layer, pos, cidx = headline_best(ranks_by_set[name], band)
            record[f"{name}_min_rank"] = r_min
            record[f"{name}_argmin"] = {
                "layer": layer,
                "position": pos,
                "concept": concept_sets[name]["kept"][cidx]["word"],
            }
        # Richer per-item structure for diagnostics.
        stacked = {
            name: torch.stack([ranks_by_set[name][layer] for layer in band])
            for name in ("safety", "neutral")
        }  # [n_layers, seq_len, n_concepts]
        record["safety_min_rank_per_layer"] = {
            str(layer): int(ranks_by_set["safety"][layer].min()) for layer in band
        }
        record["neutral_min_rank_per_layer"] = {
            str(layer): int(ranks_by_set["neutral"][layer].min()) for layer in band
        }
        record["safety_min_rank_last_token_per_layer"] = {
            str(layer): int(ranks_by_set["safety"][layer][-1, :].min())
            for layer in band
        }
        per_concept = stacked["safety"].amin(dim=(0, 1))  # [n_concepts]
        record["safety_min_rank_per_concept"] = {
            word: int(per_concept[i]) for i, word in enumerate(safety_words)
        }
        per_position = stacked["safety"].amin(dim=(0, 2))  # [seq_len]
        record["safety_min_rank_per_position"] = [
            int(v) for v in per_position.tolist()
        ]
        record["safety_min_rank_by_position_bucket"] = position_bucket_mins(
            stacked["safety"]
        )
        per_layer_min = stacked["safety"].amin(dim=(1, 2))  # [n_layers]
        record["safety_scores"] = {
            "min_anywhere": float(record["safety_min_rank"]),
            "last_token_min": float(
                record["safety_min_rank_by_position_bucket"]["last_token"]
            ),
            "layer_median_of_min": float(
                statistics.median(per_layer_min.tolist())
            ),
            "position_mean_of_min": float(per_position.float().mean()),
        }
        records.append(record)
        print(
            f"[{index + 1}/{len(items)}] {item['id']} ({item['label']}): "
            f"safety_min={record['safety_min_rank']} "
            f"neutral_min={record['neutral_min_rank']} "
            f"safety_last_token={record['safety_scores']['last_token_min']:.0f}"
        )

    # Headline summary, identical to the n80 run.
    summary: dict[str, dict] = {}
    for name in ("safety", "neutral"):
        scores = [float(r[f"{name}_min_rank"]) for r in records]
        summary[name] = summarize(scores, labels)

    per_category: dict[str, dict] = {}
    for r in records:
        bucket = per_category.setdefault(r["category"], {})
        bucket.setdefault(r["label"], []).append(r["safety_min_rank"])
    per_category = {
        cat: {
            label: statistics.median(vals) for label, vals in sorted(by_label.items())
        }
        for cat, by_label in sorted(per_category.items())
    }

    # --- diagnostics -------------------------------------------------------
    diagnostics: dict = {}

    # Per-layer AUROC: which layers separate (score = per-layer min safety rank).
    per_layer_auroc = {}
    for layer in band:
        scores = [
            float(r["safety_min_rank_per_layer"][str(layer)]) for r in records
        ]
        per_layer_auroc[str(layer)] = auroc_lower_is_positive(scores, labels)
    best_layer = max(per_layer_auroc, key=per_layer_auroc.get)
    diagnostics["per_layer_auroc"] = {
        "score": "min safety rank over positions x concepts at that layer",
        "auroc": per_layer_auroc,
        "best_layer": int(best_layer),
        "best_auroc": per_layer_auroc[best_layer],
    }

    # Position-bucket AUROC: last-token-only vs all-positions vs halves.
    per_bucket = {}
    for bucket in POSITION_BUCKETS:
        scores = [
            float(r["safety_min_rank_by_position_bucket"][bucket]) for r in records
        ]
        per_bucket[bucket] = auroc_lower_is_positive(scores, labels)
    diagnostics["position_bucket_auroc"] = {
        "score": "min safety rank over band layers x concepts within the bucket",
        "auroc": per_bucket,
        "best_bucket": max(per_bucket, key=per_bucket.get),
    }

    # Per-concept AUROC: which safety words carry the signal.
    per_concept_auroc = {}
    for word in safety_words:
        scores = [float(r["safety_min_rank_per_concept"][word]) for r in records]
        per_concept_auroc[word] = auroc_lower_is_positive(scores, labels)
    diagnostics["per_concept_auroc"] = {
        "score": "min rank of that concept over band layers x positions",
        "auroc": dict(
            sorted(per_concept_auroc.items(), key=lambda kv: -kv[1])
        ),
    }

    # Alternative aggregates (vary the aggregation, not the transform —
    # monotone transforms cannot change AUROC).
    aggregate_auroc = {}
    for agg in AGGREGATES:
        scores = [r["safety_scores"][agg] for r in records]
        aggregate_auroc[agg] = auroc_lower_is_positive(scores, labels)
    diagnostics["aggregate_auroc"] = {
        "aggregates": {
            "min_anywhere": "headline: min over layers x positions x concepts",
            "last_token_min": "min over layers x concepts at the last prompt token",
            "layer_median_of_min": "median over layers of per-layer min",
            "position_mean_of_min": "mean over positions of per-position min",
        },
        "auroc": aggregate_auroc,
        "best_aggregate": max(aggregate_auroc, key=aggregate_auroc.get),
    }

    # Cell-stratified AUROC: the stealth-vs-hard cell is the key confound
    # test — harmful intent without threat vocabulary vs benign prompts that
    # DO contain threat vocabulary.
    strata: dict[str, dict] = {}
    for name, pred in (
        ("original_80", lambda r: r["cell"] == "original"),
        ("stealth_vs_hard", lambda r: r["cell"] == "stealth_hard"),
        ("pooled_120", lambda r: True),
    ):
        sub = [r for r in records if pred(r)]
        sub_labels = [1 if r["label"] == "harmful" else 0 for r in sub]
        entry: dict[str, object] = {
            "n_harmful": sum(sub_labels),
            "n_benign": len(sub_labels) - sum(sub_labels),
        }
        for concept in ("safety", "neutral"):
            scores = [float(r[f"{concept}_min_rank"]) for r in sub]
            entry[concept] = summarize(scores, sub_labels)
        strata[name] = entry
    diagnostics["cell_stratified"] = strata

    out = {
        "experiment": "security-separation (harmful-vs-benign lens min-rank)",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "lens": f"{LENS_REPO} {LENS_REVISION} {LENS_FILE}",
        "config": {
            "band_layers": [band[0], band[-1]],
            "positions": "all prompt token positions",
            "score": "min lens rank of any single-token concept over band x positions",
            "concepts": concept_sets,
        },
        "n_items": len(items),
        "n_harmful": n_harmful,
        "summary": summary,
        "per_category_safety_median_min_rank": per_category,
        "diagnostics": diagnostics,
        "items": records,
    }
    RESULTS.mkdir(exist_ok=True)
    json_path = RESULTS / f"security_separation_{MODEL_TAG}.json"
    json_path.write_text(json.dumps(out, indent=2))

    def row(name: str, s: dict) -> str:
        med = s["median_min_rank"]
        return (
            f"{name:>8} {s['auroc']:>7.3f} {med['harmful']:>10.1f} "
            f"{med['benign']:>8.1f} {s['best_threshold']['threshold']:>10.1f} "
            f"{s['best_threshold']['accuracy']:>9.3f}"
        )

    def stratum_row(name: str, e: dict) -> str:
        return (
            f"{name:>16} {e['n_harmful']:>7} {e['n_benign']:>6} "
            f"{e['safety']['auroc']:>8.3f} {e['neutral']['auroc']:>8.3f} "
            f"{e['safety']['median_min_rank']['harmful']:>9.1f} "
            f"{e['safety']['median_min_rank']['benign']:>9.1f}"
        )

    stealth = strata["stealth_vs_hard"]["safety"]["auroc"]
    layer_lines = [
        f"  layer {layer:>2}: {per_layer_auroc[str(layer)]:.3f}" for layer in band
    ]
    concept_ranking = sorted(per_concept_auroc.items(), key=lambda kv: -kv[1])
    concept_lines = [
        f"  {word:>10}: {value:.3f}" for word, value in concept_ranking
    ]

    lines = [
        f"security-separation — {MODEL_NAME} + Jacobian-lens min-rank readout",
        f"items: {len(items)} ({n_harmful} harmful / "
        f"{len(items) - n_harmful} benign), band: layers {band[0]}..{band[-1]}, "
        "every prompt position",
        "kept safety concepts: "
        + ", ".join(e["word"] for e in concept_sets["safety"]["kept"]),
        "kept neutral concepts: "
        + ", ".join(e["word"] for e in concept_sets["neutral"]["kept"]),
        "",
        f"{'concept':>8} {'auroc':>7} {'med_harm':>10} {'med_ben':>8} "
        f"{'best_thr':>10} {'thr_acc':>9}",
        row("safety", summary["safety"]),
        row("neutral", summary["neutral"]),
        "",
        "safety row is the signal; neutral row is the control (expect ~0.5).",
        "",
        "cell-stratified AUROC (headline safety min-rank score):",
        f"{'stratum':>16} {'n_harm':>7} {'n_ben':>6} {'auroc_sf':>8} "
        f"{'auroc_ne':>8} {'med_harm':>9} {'med_ben':>9}",
        stratum_row("original_80", strata["original_80"]),
        stratum_row("stealth_vs_hard", strata["stealth_vs_hard"]),
        stratum_row("pooled_120", strata["pooled_120"]),
        "",
        f">>> KEY CONFOUND TEST — stealth-harmful vs hard-benign AUROC: "
        f"{stealth:.3f}",
        "    (stealth harmful carry NO threat words; hard benign DO contain "
        "threat words.",
        "     ~0.5 => the headline separation is just topic vocabulary, not "
        "intent;",
        "     clearly >0.5 => intent separates beyond vocabulary.)",
        "",
        "diagnostics (safety score, pooled 120):",
        "position-bucket AUROC (min over band x concepts within bucket):",
        f"  all_positions: {per_bucket['all_positions']:.3f}   "
        f"last_token: {per_bucket['last_token']:.3f}   "
        f"first_half: {per_bucket['first_half']:.3f}   "
        f"second_half: {per_bucket['second_half']:.3f}",
        f"alternative aggregates AUROC: "
        + "   ".join(f"{a}: {aggregate_auroc[a]:.3f}" for a in AGGREGATES),
        f"best layer: {best_layer} (AUROC {per_layer_auroc[best_layer]:.3f}); "
        f"best aggregate: {max(aggregate_auroc, key=aggregate_auroc.get)}",
        "",
        "per-layer AUROC (min safety rank at that layer):",
        *layer_lines,
        "",
        "per-concept AUROC (min rank of that safety word, sorted):",
        *concept_lines,
    ]
    txt_path = RESULTS / f"security_separation_{MODEL_TAG}.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {json_path}\nwrote {txt_path}")


if __name__ == "__main__":
    main()
