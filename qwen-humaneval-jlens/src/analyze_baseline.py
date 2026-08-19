"""Baseline report: pass@1 + J-Space capture summary.

Reads outputs/completions, outputs/evaluation, outputs/activations and writes
outputs/reports/baseline_report.md. No model required.

Usage:
    python -m src.analyze_baseline --config config.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import yaml


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def _summarize_activations(act_dir: Path) -> dict:
    """Stream all capture files; per-layer mean norms + one example readout."""
    per_layer: dict[int, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "hidden_norm": 0.0, "jspace_norm": 0.0}
    )
    example = None
    n_files = 0
    layers_seen: set[int] = set()
    for path in sorted(act_dir.glob("*.jsonl.gz")):
        n_files += 1
        with gzip.open(path, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                layer = rec["layer"]
                layers_seen.add(layer)
                s = per_layer[layer]
                s["n"] += 1
                s["hidden_norm"] += rec["hidden_norm"]
                s["jspace_norm"] += rec["jspace_norm"]
        if example is None:
            # top-5 readout at the final generated position, last hooked layer
            with gzip.open(path, "rt") as fh:
                recs = [json.loads(line) for line in fh]
            last_layer = max(r["layer"] for r in recs)
            final = [r for r in recs if r["layer"] == last_layer][-1]
            example = {
                "task_file": path.name,
                "layer": last_layer,
                "position": final["position"],
                "token": final.get("token"),
                "top5": final["top_jspace_tokens"][:5],
            }
    for s in per_layer.values():
        s["hidden_norm"] /= max(s["n"], 1)
        s["jspace_norm"] /= max(s["n"], 1)
    return {
        "n_files": n_files,
        "layers": sorted(layers_seen),
        "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
        "example": example,
    }


def _plot_layer_norms(capture: dict, out_path: Path) -> bool:
    """Bar/line chart of mean |h| and |J·h| per captured layer. Returns True
    if the chart was written (matplotlib is an optional dependency)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(layer) for layer in capture["per_layer"]]
    hidden = [capture["per_layer"][str(layer)]["hidden_norm"] for layer in layers]
    jspace = [capture["per_layer"][str(layer)]["jspace_norm"] for layer in layers]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, hidden, marker="o", label="mean |h| (residual stream)")
    ax.plot(layers, jspace, marker="s", label="mean |J·h| (J-Space)")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean L2 norm")
    ax.set_title("J-Space capture: activation magnitude by layer")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def analyze(cfg: dict) -> Path:
    out_base = Path(cfg["outputs"]["base_dir"])
    completions = _read_jsonl(out_base / "completions" / "completions.jsonl")
    results = {r["task_id"]: r for r in _read_jsonl(out_base / "evaluation" / "results.jsonl")}
    summary = json.loads((out_base / "evaluation" / "summary.json").read_text())

    passed = [t for t in completions if results.get(t["task_id"], {}).get("passed")]
    failed = [t for t in completions if not results.get(t["task_id"], {}).get("passed")]

    jlens_status = completions[0].get("jlens_status", "unknown") if completions else "unknown"
    capture = _summarize_activations(out_base / "activations")
    decoding = completions[0]["decoding"] if completions else {}

    chart_file = "jspace_layer_norms.png"
    chart_written = _plot_layer_norms(capture, out_base / "reports" / chart_file)

    lines = [
        "# Phase-1 baseline report — HumanEval × J-Space observation",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Tasks run: {summary['n_tasks']}",
        f"- Decoding: greedy (do_sample=false), max_new_tokens="
        f"{decoding.get('max_new_tokens')}, seed={decoding.get('seed')}",
        f"- **pass@1: {summary['pass_at_1']:.4f}** "
        f"({summary['n_passed']}/{summary['n_tasks']})",
        f"- J-Lens status: `{jlens_status}`"
        + (" — fitted checkpoint loaded, J-Space readouts valid"
           if jlens_status == "fitted"
           else " — WARNING: placeholder or disabled; readouts are controls only"),
        "",
        "## Passed tasks",
        "",
        *(f"- {t['task_id']}" for t in passed),
        "",
        "## Failed tasks",
        "",
        *(f"- {t['task_id']} ({results[t['task_id']]['status']})" for t in failed),
        "",
        "## J-Space capture summary",
        "",
        "How the capture works (observation-only — activations are never modified):",
        "",
        "```",
        "prompt token(s) ──> transformer layer l ──> residual h ──┬──> layer l+1 (untouched)",
        "                                                       │",
        "                                                z = J_l · h  (J-Lens projection)",
        "                                                       │",
        "                                     record |h|, |z|, top-k tokens of unembed(z)",
        "```",
        "",
        f"- Activation files: {capture['n_files']} tasks",
        f"- Layers observed: {capture['layers']}",
        "",
    ]
    if chart_written:
        lines += [
            f"![Mean activation norms by layer]({chart_file})",
            "",
        ]
    lines += [
        "| layer | positions | mean |h| | mean |J·h| |",
        "|---|---|---|---|",
        *(
            f"| {layer} | {int(s['n'])} | {s['hidden_norm']:.2f} | {s['jspace_norm']:.2f} |"
            for layer, s in capture["per_layer"].items()
        ),
        "",
    ]
    if capture["example"]:
        ex = capture["example"]
        lines += [
            "### Example J-Space readout",
            "",
            f"File `{ex['task_file']}`, layer {ex['layer']}, final generated "
            f"position {ex['position']} (token `{ex['token']}`), top-5 lens tokens:",
            "",
            *(f"- `{tok}` ({logit:.2f})" for tok, logit in ex["top5"]),
            "",
        ]
    lines += [
        "## Phase-2 TODOs",
        "",
        "- `JLens.project_from_jspace` (pinv back-projection)",
        "- intervention hooks: zero_topk / mean_replace / subtract_mean",
        "- intervention HumanEval rerun + baseline-vs-intervention comparison",
        "",
    ]

    report_path = out_base / "reports" / "baseline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"[analyze] pass@1={summary['pass_at_1']:.4f}; wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-dir", default=None,
                        help="override outputs.base_dir (e.g. outputs-remote-5090)")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.base_dir is not None:
        cfg["outputs"]["base_dir"] = args.base_dir
    analyze(cfg)


if __name__ == "__main__":
    main()
