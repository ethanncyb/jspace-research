from __future__ import annotations

import json
from pathlib import Path

from gsm8k_jspace.config import load_config
from gsm8k_jspace.datasets.gsm8k import load_gsm8k_examples
from gsm8k_jspace.evaluation.evaluator import evaluate_run
from gsm8k_jspace.runner.experiment import run_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k_rows.json"


def _tiny_cfg(tmp_path: Path, condition: str = "baseline", capture: bool = False):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg.outputs.root_dir = str(tmp_path)
    cfg.outputs.run_id = f"tiny-{condition}"
    cfg.outputs.on_existing = "resume"
    cfg.model.name = "tiny"
    cfg.jlens.source = "identity"
    cfg.jlens.required = False
    cfg.capture.enabled = capture
    cfg.capture.layers.mode = "explicit"
    cfg.capture.layers.values = [1, 2]
    cfg.generation.max_new_tokens = 4
    cfg.experiment.condition = condition
    cfg.intervention.enabled = condition == "intervention"
    if condition == "intervention":
        cfg.intervention.layers.mode = "explicit"
        cfg.intervention.layers.values = [1]
        cfg.intervention.strength = 0.05
    if condition == "no_op":
        cfg.intervention.layers.mode = "explicit"
        cfg.intervention.layers.values = [1]
        cfg.intervention.method = "none"
    cfg.benchmark.subset_size = 3
    return cfg


def test_tiny_baseline_and_resume(tmp_path, tiny_model):
    rows = json.loads(FIXTURE.read_text())
    cfg = _tiny_cfg(tmp_path, "baseline", capture=False)
    examples = load_gsm8k_examples(cfg, rows=rows)
    run_dir = run_experiment(
        cfg, hf_model=tiny_model, tokenizer=tiny_model.tokenizer, examples=examples[:2]
    )
    completions = (run_dir / "completions.jsonl").read_text().strip().splitlines()
    assert len(completions) == 2
    cfg.outputs.on_existing = "resume"
    run_experiment(
        cfg, hf_model=tiny_model, tokenizer=tiny_model.tokenizer, examples=examples[:2]
    )
    completions2 = (run_dir / "completions.jsonl").read_text().strip().splitlines()
    assert len(completions2) == 2
    summary = evaluate_run(run_dir, cfg)
    assert summary["n_evaluated"] == 2


def test_tiny_capture_and_noop_token_ids(tmp_path, tiny_model):
    rows = json.loads(FIXTURE.read_text())
    base_cfg = _tiny_cfg(tmp_path, "baseline", capture=False)
    examples = load_gsm8k_examples(base_cfg, rows=rows)[:1]
    base_dir = run_experiment(
        base_cfg,
        hf_model=tiny_model,
        tokenizer=tiny_model.tokenizer,
        examples=examples,
    )
    noop_cfg = _tiny_cfg(tmp_path, "no_op", capture=True)
    noop_cfg.outputs.run_id = "tiny-no-op"
    noop_dir = run_experiment(
        noop_cfg,
        hf_model=tiny_model,
        tokenizer=tiny_model.tokenizer,
        examples=examples,
    )
    base = json.loads((base_dir / "completions.jsonl").read_text().splitlines()[0])
    noop = json.loads((noop_dir / "completions.jsonl").read_text().splitlines()[0])
    assert base["generated_token_ids"] == noop["generated_token_ids"]
    assert list((noop_dir / "captures").glob("*.jsonl.gz"))
