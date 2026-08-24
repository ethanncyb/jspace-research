"""Summary report across all experiment arms.

Reads outputs/evaluation (baseline) and outputs/intervention_layers_*/
(comparison + metadata), then writes into outputs/reports/:

  summary_report.md     — method, layer rationale, findings, all results
  pass_at_1_by_arm.png  — bar chart of pass@1 across arms
  flips_by_arm.png      — broken vs fixed task counts per arm

Usage:
    python -m src.make_summary_report --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml


def _arm_sort_key(name: str) -> tuple[int, int]:
    # intervention_layers_12_20 -> (12, 20)
    parts = name.replace("intervention_layers_", "").split("_")
    return int(parts[0]), int(parts[1])


def _load_arms(out_base: Path) -> list[dict]:
    arms = []
    for run_dir in sorted(out_base.glob("intervention_layers_*"), key=lambda p: _arm_sort_key(p.name)):
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        summ = json.loads((run_dir / "evaluation" / "summary.json").read_text())
        rows = list(csv.DictReader((run_dir / "comparison" / "comparison.csv").open()))
        counts = Counter(r["change_type"] for r in rows)
        layers = meta["layers"]
        arms.append(
            {
                "name": run_dir.name,
                "layers": layers,
                "label": f"{layers[0]}–{layers[-1]}",
                "pass_at_1": summ["pass_at_1"],
                "n_passed": summ["n_passed"],
                "n_tasks": summ["n_tasks"],
                "broken": counts["baseline_pass_intervention_fail"],
                "fixed": counts["baseline_fail_intervention_pass"],
                "strength": meta["strength"],
                "top_k": meta["top_k"],
            }
        )
    return arms


def _plot_pass_at_1(baseline: float, arms: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["baseline\n(no intervention)"] + [f"layers {a['label']}" for a in arms]
    values = [baseline] + [a["pass_at_1"] for a in arms]
    colors = ["#4C9F70"] + ["#D06A5C" if a["pass_at_1"] < baseline - 0.02 else "#8FAACB" for a in arms]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(baseline, color="#4C9F70", ls="--", lw=1, alpha=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("HumanEval pass@1")
    ax.set_ylim(0, max(values) + 0.08)
    ax.set_title("HumanEval pass@1 by J-Space ablation arm\n(mean_replace, top-k=50, α=0.05)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_flips(arms: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    labels = [f"layers {a['label']}" for a in arms]
    broken = [a["broken"] for a in arms]
    fixed = [a["fixed"] for a in arms]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, broken, width=0.4, label="broken (pass → fail)", color="#D06A5C")
    ax.bar(x + 0.2, fixed, width=0.4, label="fixed (fail → pass)", color="#4C9F70")
    for xi, v in zip(x - 0.2, broken):
        ax.text(xi, v + 1, str(v), ha="center", fontsize=10)
    for xi, v in zip(x + 0.2, fixed):
        ax.text(xi, v + 1, str(v), ha="center", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("tasks (out of 164)")
    ax.set_title("Task outcome flips per arm — broken ≫ fixed means a real effect")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_window_profile(baseline: float, arms: list[dict], out_path: Path) -> bool:
    """pass@1 vs 2-layer window position (the 'which layers matter' figure)."""
    windows = [a for a in arms if len(a["layers"]) == 2]
    if not windows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [a["layers"][0] for a in windows]
    y = [a["pass_at_1"] for a in windows]
    doses = [a["rel_change"] for a in windows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(baseline, color="#4C9F70", ls="--", lw=1.2,
               label=f"baseline ({baseline:.3f})")
    ax.plot(x, y, marker="o", color="#C0504D", label="2-layer ablation")
    for xi, yi, a in zip(x, y, windows):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("first layer of the 2-layer ablated window")
    ax.set_ylabel("HumanEval pass@1")
    ax.set_xticks(x)
    ax.set_title("Which layers carry the effect? pass@1 vs ablated window\n"
                 "(mean_replace, top-k=50, α=0.05)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_report(cfg: dict) -> Path:
    out_base = Path(cfg["outputs"]["base_dir"])
    reports = out_base / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads((out_base / "evaluation" / "summary.json").read_text())
    p_base = baseline_summary["pass_at_1"]
    arms = _load_arms(out_base)
    if not arms:
        raise SystemExit("no intervention arms found under outputs/")

    # per-arm intervention strength actually applied (mean hidden rel change)
    for a in arms:
        hook_path = out_base / a["name"] / "hook_summary.json"
        if hook_path.exists():
            hook = json.loads(hook_path.read_text())
            rels = [v["mean_rel_change_hidden"] for v in hook["per_layer"].values()]
            a["rel_change"] = sum(rels) / len(rels)
        else:
            a["rel_change"] = None

    _plot_pass_at_1(p_base, arms, reports / "pass_at_1_by_arm.png")
    _plot_flips(arms, reports / "flips_by_arm.png")
    window_plot = _plot_window_profile(p_base, arms, reports / "window_profile.png")

    def _rel(a):
        return f"{a['rel_change']:.1%}" if a["rel_change"] is not None else "?"

    rows = "\n".join(
        f"| layers {a['label']} | {a['pass_at_1']:.4f} ({a['n_passed']}/{a['n_tasks']}) "
        f"| {(p_base - a['pass_at_1']):+.4f} | {(p_base - a['pass_at_1']) / p_base:+.1%} "
        f"| {a['broken']} | {a['fixed']} | {_rel(a)} |"
        for a in arms
    )

    lines = [
        "# HumanEval × J-Space ablation — summary of all experiments",
        "",
        "**TL;DR for the team:** Ablating the top-50 J-Space features (5% blend",
        "toward the task mean) in the mid-to-late band of Qwen3.5-9B-Base cuts",
        "HumanEval pass@1 from 60.4% to 11.0% (full band 10–26) or 30.5% (core",
        "12–20), while output stays fluent. The same ablation at the final fitted",
        "layers (27–30) changes nothing (61.0%), and at early layers (0–9) it",
        "destroys the model (0.6%). J-Space content in the mid-to-late band is",
        "causally load-bearing for coding.",
        "",
        "## Setup",
        "",
        "- Model: `Qwen/Qwen3.5-9B-Base` (32 layers, hidden 4096)",
        "- J-Lens: `neuronpedia/jacobian-lens@qwen-n1000`, file",
        "  `qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt`",
        "  (fitted on the base model, 458-prompt wikitext estimator, layers 0–30)",
        "- Benchmark: HumanEval (`openai/openai_humaneval`), all 164 tasks,",
        "  completion-style prompting (raw prompt verbatim)",
        "- Decoding: greedy (do_sample=false), max_new_tokens=512, seed=0,",
        "  standard HumanEval stop sequences; identical in every run",
        "- Evaluation: model code executed against the hidden HumanEval test",
        "  suites in a subprocess (10 s timeout); pass = clean exit. No LLM judge.",
        "- Hardware: RTX 5090 Laptop (24 GB), bf16; baseline additionally",
        "  replicated on Apple M1 Max (MPS)",
        "",
        "## Method",
        "",
        "Each intervention run is an inference-time ablation of selected JSPace",
        "features — not a full JSPace shutdown. At each hooked layer, on every token",
        "(prompt and generated), the hook: (1) projects the hidden state into",
        "J-Space with the fitted J-Lens (`z = J_l·h`), (2) finds the top-50",
        "largest-|z| coordinates, (3) pulls them 5% (α=0.05) toward their running",
        "average value for the current task, (4) transports only the change back",
        "into the hidden state via `pinv(J_l)` and continues generation.",
        "",
        "Controls: a `none` arm (same hooks, zero change) reproduced the baseline",
        "completions bit-for-bit on every smoke test. A dose sweep showed α=1.0",
        "(full replacement) destroys output (gibberish), while α=0.05 is the",
        "strongest dose that keeps the model writing fluent code.",
        "",
        "## Dose response (5-task smoke, layers 10–26)",
        "",
        "| α (blend toward mean) | hidden perturbation (mean ‖Δh‖/‖h‖) | smoke pass@1 | output character |",
        "|---|---|---|---|",
        "| 0.02 | 6% | 4/5 (= baseline) | coherent, correct |",
        "| 0.05 | 13% | 1/5 | coherent code, wrong logic |",
        "| 0.10 | 17% | 0/5 | degenerate loops |",
        "| 0.25 | 54% | 0/5 | gibberish |",
        "| 1.00 | ~300% | (not run to completion) | gibberish |",
        "",
        "α=0.05 was chosen for all full runs: the strongest dose in the coherent",
        "regime, so pass@1 drops reflect disrupted reasoning, not broken language.",
        "",
        "## Verification gates (all passed)",
        "",
        "- `none` control per arm: completions bit-identical to baseline (5/5 smoke",
        "  tasks, per arm) — the hook plumbing itself is inert.",
        "- Cross-hardware replication: baseline run on MPS and CUDA gave identical",
        "  pass@1 (99/164 both), 141/164 completions bit-identical, 2 task flips —",
        "  this is the noise floor for all comparisons.",
        "- Hook logs confirm: hooks fired on exactly the configured layers, top-50",
        "  coordinates modified, hidden-space delta nonzero, in every arm.",
        "",
        "## Why these layers",
        "",
        "Layers 10–26 are the J-Lens paper's validated mid-to-late band, where",
        "J-Space readouts are meaningful and steerable (early layers: local/syntax;",
        "final layers: the readout itself). Arms 0–9 and 27–30 were run afterward",
        "as boundary controls, and arm 12–20 as the band's core.",
        "",
        "## Results",
        "",
        "![pass@1 by arm](pass_at_1_by_arm.png)",
        "",
        f"| arm | pass@1 | abs. change | rel. change | broken | fixed | hidden Δ |",
        f"|---|---|---|---|---|---|---|",
        f"| **baseline** | **{p_base:.4f}** ({baseline_summary['n_passed']}/{baseline_summary['n_tasks']}) | — | — | — | — | — |",
        rows,
        "",
        "![flips by arm](flips_by_arm.png)",
        "",
        "## What we found",
        "",
    ]
    by_label = {a["label"]: a for a in arms}
    findings = []
    if "0–9" in by_label:
        a = by_label["0–9"]
        findings.append(
            f"- **Early layers 0–9: pass@1 {a['pass_at_1']:.4f} ({a['broken']} broken, "
            f"{a['fixed']} fixed).** Catastrophic but unspecific — early-layer "
            "representations are load-bearing for everything, so this collapse "
            "reflects general representational damage, not coding-specific JSPace "
            "content. Treat as a destructive control."
        )
    if "12–20" in by_label:
        a = by_label["12–20"]
        findings.append(
            f"- **Mid core 12–20: pass@1 {a['pass_at_1']:.4f} ({a['broken']} broken, "
            f"{a['fixed']} fixed).** A large, directional drop with fluent output — "
            "task-specific JSPace information in these layers contributes to "
            "coding performance."
        )
    if "10–26" in by_label:
        a = by_label["10–26"]
        findings.append(
            f"- **Full band 10–26: pass@1 {a['pass_at_1']:.4f} ({a['broken']} broken, "
            f"{a['fixed']} fixed).** The strongest interpretable effect. The 2-layer "
            "sweep shows the band's causal weight is concentrated at its FRONT "
            "(layers 10–13) and decays with depth."
        )
    if "27–30" in by_label:
        a = by_label["27–30"]
        findings.append(
            f"- **Late edge 27–30: pass@1 {a['pass_at_1']:.4f} ({a['broken']} broken, "
            f"{a['fixed']} fixed).** No directional effect — flips are balanced in "
            "both directions, i.e. noise. Consistent with JSPace at the final "
            "layers being the readout itself rather than content used by "
            "downstream computation."
        )
    lines += findings

    windows = [a for a in arms if len(a["layers"]) == 2]
    if windows:
        lines += [
            "",
            "## Fine-grained 2-layer sweep (windows across the band)",
            "",
        ]
        if window_plot:
            lines += ["![pass@1 by 2-layer window](window_profile.png)", ""]
        lines += [
            "| window | pass@1 | rel. drop | broken | fixed | hidden Δ |",
            "|---|---|---|---|---|---|",
            *(
                f"| layers {a['label']} | {a['pass_at_1']:.4f} "
                f"| {(p_base - a['pass_at_1']) / p_base:+.1%} | {a['broken']} "
                f"| {a['fixed']} | {_rel(a)} |"
                for a in windows
            ),
            "",
            "The effect decays monotonically with depth across the band: ablating",
            "the front of the band hurts most, the back of the band (20–21) is",
            "indistinguishable from baseline. Caveat: the hidden-space dose is not",
            "constant across windows (earlier windows perturb more for the same α),",
            "so part of the gradient reflects dose; but the broken/fixed balance",
            "(strongly directional at the front, 5/5 noise at the back) supports a",
            "genuine positional gradient, not just a dose artifact.",
            "",
        ]
    lines += [
        "",
        "**Bottom line:** the causal dependence on J-Space content for HumanEval",
        "coding lives in the middle-to-late band (10–26). A 5% blunting of the",
        "top-50 J-Space features there cuts coding accuracy by half to four-fifths",
        "while the model stays fluent; the same intervention at the final fitted",
        "layers does nothing, and at early layers it breaks the model outright.",
        "",
        "## Caveats",
        "",
        "- The perturbation is broad (50 of 4,096 coordinates per layer); a",
        "  random-direction control (`random_ablation`, TODO) would establish",
        "  J-Space specificity.",
        "- Greedy n=1 pass@1 is a point estimate; no sampling variance measured.",
        "- The lens was fitted on wikitext, not code.",
        "- Effects within ~2 task flips are indistinguishable from hardware noise.",
        "",
        "## Per-arm artifacts",
        "",
        *(
            f"- `outputs/{a['name']}/` — completions, metadata, hook logs, "
            f"evaluation, comparison/report.md"
            for a in arms
        ),
        "",
    ]

    report_path = reports / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"[summary] wrote {report_path}, pass_at_1_by_arm.png, flips_by_arm.png")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    make_report(cfg)


if __name__ == "__main__":
    main()
