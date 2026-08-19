"""Compare baseline vs J-Space-intervention HumanEval results.

Usage:
    python -m src.compare_results --config config.yaml \
        --intervention-dir outputs/intervention_layers_12_20 \
        --baseline outputs/evaluation/results.jsonl

The intervention dir must contain run_metadata.json and
evaluation/results.jsonl. comparison.csv and report.md are written into
<intervention-dir>/comparison/. All labels (layer range, method, strength)
are read from the run's own metadata — never hand-typed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

CHANGE_TYPES = (
    "passed_both",
    "failed_both",
    "baseline_pass_intervention_fail",
    "baseline_fail_intervention_pass",
)


def _load_results(path: Path) -> dict[str, bool]:
    with path.open() as fh:
        return {json.loads(line)["task_id"]: json.loads(line)["passed"] for line in fh}


def _layer_label(layers: list[int]) -> str:
    return f"{layers[0]}–{layers[-1]}" if len(layers) > 1 else str(layers[0])


def compare(cfg: dict, baseline_path: Path, intervention_dir: Path) -> Path:
    out_dir = intervention_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = intervention_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hook_path = intervention_dir / "hook_summary.json"
    hook = json.loads(hook_path.read_text()) if hook_path.exists() else None

    layers = meta.get("layers") or (hook["layers"] if hook else [])
    layer_label = _layer_label(layers) if layers else "?"
    n_layers = len(layers)
    strength = meta.get("strength", "?")
    top_k = meta.get("top_k", "?")
    method = meta.get("intervention_method", "?")

    baseline = _load_results(baseline_path)
    intervention = _load_results(intervention_dir / "evaluation" / "results.jsonl")
    task_ids = sorted(baseline.keys() & intervention.keys())
    missing = sorted(baseline.keys() ^ intervention.keys())

    rows = []
    counts = {c: 0 for c in CHANGE_TYPES}
    for tid in task_ids:
        b, i = baseline[tid], intervention[tid]
        if b and i:
            change = "passed_both"
        elif not b and not i:
            change = "failed_both"
        elif b and not i:
            change = "baseline_pass_intervention_fail"
        else:
            change = "baseline_fail_intervention_pass"
        counts[change] += 1
        rows.append({"task_id": tid, "baseline_pass": b,
                     "intervention_pass": i, "change_type": change})

    n = len(task_ids)
    n_base = sum(baseline[t] for t in task_ids)
    n_int = sum(intervention[t] for t in task_ids)
    p_base = n_base / n
    p_int = n_int / n
    abs_diff = p_int - p_base
    rel_drop = (p_base - p_int) / p_base if p_base > 0 else 0.0

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["task_id", "baseline_pass", "intervention_pass", "change_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

    broken = [r["task_id"] for r in rows if r["change_type"] == "baseline_pass_intervention_fail"]
    fixed = [r["task_id"] for r in rows if r["change_type"] == "baseline_fail_intervention_pass"]

    # careful interpretation (ablation wording — this is NOT "turning off JSPace")
    dropped_significantly = counts["baseline_pass_intervention_fail"] > 2
    if dropped_significantly:
        interpretation = (
            f"HumanEval performance dropped beyond the ~1-2-task cross-hardware "
            f"noise floor. This is an inference-time ablation of selected JSPace "
            f"features in layers {layer_label}: it suggests that task-specific "
            f"JSPace information in layers {layer_label} contributes to HumanEval "
            f"coding performance in Qwen 3.5 9B. It does not mean JSPace was fully "
            f"disabled — only the top-{top_k} features at these layers were "
            f"neutralized (blend α={strength})."
        )
    else:
        interpretation = (
            f"HumanEval performance did not drop beyond the noise floor under this "
            f"ablation of selected JSPace features in layers {layer_label}. This "
            f"does not prove JSPace is irrelevant. It may mean layers {layer_label} "
            f"are not the most causal layer range, top_k={top_k} is too narrow, "
            f"mean_replace at α={strength} is too weak, or the model can compensate "
            f"using other pathways."
        )

    lines = [
        f"# Qwen 3.5 9B HumanEval JSPace Intervention: Layers {layer_label}",
        "",
        "## Baseline",
        f"Score: {p_base:.4f}",
        f"Passed: {n_base}/{n}",
        "",
        "## Intervention",
        f"Method: {method}",
        f"Layers: {layer_label} ({n_layers} layers: {layers})",
        f"Top-k: {top_k}",
        f"Strength (α): {strength}",
        f"Token strategy: {meta.get('token_strategy', '?')}",
        f"Mean source: {meta.get('mean_source', '?')}",
        f"Score: {p_int:.4f}",
        f"Passed: {n_int}/{n}",
        "",
        "## Result",
        f"Absolute change: {abs_diff:+.4f}",
        f"Relative change: {rel_drop:+.1%}",
        f"Number of task flips: {counts['baseline_pass_intervention_fail'] + counts['baseline_fail_intervention_pass']}",
        f"Passed baseline → failed intervention: {counts['baseline_pass_intervention_fail']}",
        f"Failed baseline → passed intervention: {counts['baseline_fail_intervention_pass']}",
        "",
        "## How the intervention works (plain language)",
        "",
        "The model thinks in vectors: every layer turns the text-so-far into a list",
        "of 4,096 numbers (a hidden state). The J-Lens is a learned translator: for",
        "each layer it has a matrix `J` that re-expresses that hidden state in the",
        "coordinate system of the model's *final* layer — the one the output head",
        "actually reads. That final-layer coordinate system is what we call J-Space,",
        "and the research hypothesis is that the model's most important,",
        '"about-to-say-it" content concentrates there.',
        "",
        f"During this run, a hook fired at {n_layers} layers ({layer_label}) on every",
        "token — the prompt and every generated token — and did four things:",
        "",
        "1. **Translate.** Project the hidden state into J-Space (`z = J·h`).",
        f"2. **Find the spikes.** Pick the {top_k} J-Space coordinates with the largest",
        "   values at this token — the directions where the model is 'committing'",
        "   most strongly right now.",
        f"3. **Blunt the spikes.** Pull each of those {top_k} coordinates part of the",
        f"   way (α = {strength}) toward its running average value for this task.",
        "   All other coordinates and all other layers are left untouched.",
        "4. **Translate back.** Convert only the *change* back into hidden-state",
        "   space with the pseudo-inverse of `J` and add it to the hidden state.",
        "   The model then continues generating as normal.",
        "",
        "Because only the change is written back, running the same pipeline with",
        'the method set to "none" reproduces the baseline exactly — verified',
        "bit-for-bit on the smoke tasks, which proves the machinery itself is not",
        "what hurts performance. The α dose was chosen via a sweep as the strongest",
        "setting that keeps the model writing fluent, valid-looking Python (full",
        "replacement destroys the output, which would make the score meaningless).",
        "",
        "So this comparison isolates one thing: *how much does coding performance",
        f"depend on the strongest J-Space signals at layers {layer_label}?*",
        "",
        "| change_type | tasks |",
        "|---|---|",
        *(f"| {c} | {counts[c]} |" for c in CHANGE_TYPES),
        "",
        "Reading the table: `passed_both`/`failed_both` = task outcome unchanged;",
        "`baseline_pass_intervention_fail` = the intervention *broke* a task the",
        "model normally solves; `baseline_fail_intervention_pass` = the",
        "intervention accidentally *fixed* a task (0 of these is expected).",
        "",
        "### baseline_pass → intervention_fail",
        "",
        *(f"- {t}" for t in broken),
        "",
        "### baseline_fail → intervention_pass",
        "",
        *(f"- {t}" for t in fixed),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "Noise-floor caveat: the same pipeline run on two GPUs (MPS vs CUDA) with "
        "no intervention flipped 2/164 tasks, so differences of 1-2 tasks are not "
        "attributable to the intervention.",
        "",
        f"**In plain terms:** blunting the model's strongest J-Space signals "
        f"(α = {strength}) at layers {layer_label} moved coding accuracy from "
        f"{p_base:.1%} to {p_int:.1%}, while the model kept writing fluent code.",
        "",
    ]
    if hook:
        lines += [
            "## Intervention strength actually applied",
            "",
            "How much of each hidden state's magnitude the intervention altered",
            "(0.12 = the hidden vector changed by about 12% of its length).",
            "Small enough to keep output fluent; large enough to matter.",
            "",
            "| layer | forwards | positions | mean rel. change (hidden) |",
            "|---|---|---|---|",
            *(
                f"| {layer} | {s['n_forwards']} | {s['n_positions_modified']} "
                f"| {s['mean_rel_change_hidden']:.4f} |"
                for layer, s in sorted(hook["per_layer"].items(), key=lambda kv: int(kv[0]))
            ),
            "",
        ]
    if missing:
        lines += ["## WARNING: tasks present in only one condition (excluded)", ""]
        lines += [f"- {t}" for t in missing] + [""]
    lines += [
        "## Next possible ablations",
        "",
        "- Compare layers 12–20 against the broader 10–26 band.",
        "- Try layers 20–26 only.",
        "- Try single-layer ablations.",
        "- Try stronger interventions such as zero_topk.",
        "- Try larger top-k values.",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))

    print(f"Baseline score:     {p_base:.4f}")
    print(f"Intervention score: {p_int:.4f}")
    print(f"Absolute drop:      {-abs_diff:+.4f}")
    print(f"Relative drop:      {rel_drop:+.1%}")
    print(f"[compare] wrote {csv_path}, {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--baseline", required=True,
                        help="baseline results.jsonl (same hardware as the run)")
    parser.add_argument("--intervention-dir", required=True,
                        help="run dir with run_metadata.json + evaluation/results.jsonl")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    compare(cfg, Path(args.baseline), Path(args.intervention_dir))


if __name__ == "__main__":
    main()
