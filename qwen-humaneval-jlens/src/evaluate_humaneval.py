"""HumanEval pass@1 evaluation over saved completions.

Builds ``prompt + extracted_code + test + check(entry_point)`` per task and
executes it in a subprocess with a 10 s timeout. pass@1 (greedy, n=1) is the
fraction of tasks whose program exits cleanly. Rerunnable any time on the
saved completions file — no model needed.

WARNING: executes model-generated code. Run in an environment you trust.

Usage:
    python -m src.evaluate_humaneval --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from src.humaneval_data import load_humaneval

TIMEOUT_S = 10


def build_program(task: dict, extracted_code: str) -> str:
    return (
        task["prompt"]
        + extracted_code
        + "\n\n"
        + task["test"]
        + f"\ncheck({task['entry_point']})\n"
    )


def run_program(program: str) -> tuple[bool, str]:
    """Returns (passed, status); status in pass | fail | timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)
    return (proc.returncode == 0), ("pass" if proc.returncode == 0 else "fail")


def evaluate(cfg: dict, completions_path: Path | None = None,
             eval_dir: Path | None = None) -> Path:
    out_base = Path(cfg["outputs"]["base_dir"])
    if completions_path is None:
        completions_path = out_base / "completions" / "completions.jsonl"
    if eval_dir is None:
        eval_dir = out_base / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tasks = {t["task_id"]: t for t in load_humaneval(cfg)}
    records = [json.loads(line) for line in completions_path.open()]

    results_path = eval_dir / "results.jsonl"
    n_passed = 0
    with results_path.open("w") as out:
        for rec in records:
            task = tasks[rec["task_id"]]
            passed, status = run_program(build_program(task, rec["extracted_code"]))
            n_passed += int(passed)
            out.write(json.dumps({"task_id": rec["task_id"], "passed": passed,
                                  "status": status}) + "\n")

    summary = {
        "model": records[0]["model"] if records else None,
        "n_tasks": len(records),
        "n_passed": n_passed,
        "pass_at_1": n_passed / len(records) if records else 0.0,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[evaluate] {summary}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--completions", default=None,
                        help="completions.jsonl path (default: <base_dir>/completions/)")
    parser.add_argument("--out-dir", default=None,
                        help="where results.jsonl/summary.json go "
                             "(default: <base_dir>/evaluation/)")
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    evaluate(
        cfg,
        completions_path=Path(args.completions) if args.completions else None,
        eval_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
