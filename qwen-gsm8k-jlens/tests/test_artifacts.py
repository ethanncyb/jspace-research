from __future__ import annotations

from pathlib import Path

from gsm8k_jspace.artifacts.manifest import (
    ResumeError,
    create_run_dir,
    load_completed_ids,
)
from gsm8k_jspace.artifacts.writer import append_jsonl
from gsm8k_jspace.config import load_config
from gsm8k_jspace.evaluation.evaluator import evaluate_run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_create_and_resume_fingerprint(tmp_path: Path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg.outputs.root_dir = str(tmp_path)
    cfg.outputs.run_id = "run-a"
    selection = [{"example_id": "gsm8k_test_000000", "source_index": 0, "gold_answer": "1"}]
    env = {"backend": {"resolved": "cpu"}}
    run_dir = create_run_dir(cfg, selection, environment=env)
    append_jsonl(
        run_dir / "completions.jsonl",
        {"example_id": "gsm8k_test_000000", "generated_text": "#### 1"},
    )
    assert load_completed_ids(run_dir) == {"gsm8k_test_000000"}
    cfg.outputs.on_existing = "resume"
    again = create_run_dir(cfg, selection, environment=env)
    assert again == run_dir
    cfg.generation.seed = 99
    cfg.outputs.on_existing = "resume"
    try:
        create_run_dir(cfg, selection, environment=env)
    except ResumeError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("expected ResumeError")


def test_evaluate_from_saved_completions(tmp_path: Path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg.outputs.root_dir = str(tmp_path)
    cfg.outputs.run_id = "eval-run"
    selection = [
        {
            "example_id": "gsm8k_test_000000",
            "source_index": 0,
            "gold_answer": "18",
        }
    ]
    run_dir = create_run_dir(cfg, selection, environment={})
    append_jsonl(
        run_dir / "completions.jsonl",
        {
            "run_id": "eval-run",
            "example_id": "gsm8k_test_000000",
            "model": "tiny",
            "condition": "baseline",
            "generated_text": "working... #### 18",
            "n_generated_tokens": 4,
            "finish_reason": "eos",
        },
    )
    summary = evaluate_run(run_dir, cfg)
    assert summary["accuracy"] == 1.0
    assert summary["n_correct"] == 1
