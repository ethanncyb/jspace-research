"""Phase-2 intervention runner: HumanEval with J-Space mean_replace active.

Identical to the baseline run except for the intervention hooks: same model,
dataset, prompt format, decoding settings, and generation code path (shared
with src.run_baseline by import).

Usage:
    python -m src.run_intervention --config config.yaml --limit 5   # smoke
    python -m src.run_intervention --config config.yaml             # full run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import StoppingCriteriaList

from src.capture_jspace import resolve_layers
from src.humaneval_data import load_humaneval
from src.jspace_intervention import JSPaceIntervention
from src.load_jlens import load_jlens
from src.load_model import deterministic_gen_config, load_model
from src.run_baseline import _StopOnStrings, extract_code

BASELINE_SCORE = 0.6037  # measured Phase-1 baseline pass@1 (99/164, both machines)


def run(cfg: dict) -> None:
    gen_cfg = deterministic_gen_config(cfg)
    iv_cfg = cfg["intervention"]
    out_base = Path(cfg["outputs"]["base_dir"])
    out_dir = out_base / "intervention"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_humaneval(cfg)
    hf_model, tokenizer = load_model(cfg)
    input_device = hf_model.get_input_embeddings().weight.device
    jlens_iface, lens_model = load_jlens(cfg, hf_model, tokenizer)

    layers = resolve_layers(
        iv_cfg.get("layers", cfg["jspace"].get("layers", "late")),
        lens_model.n_layers,
        jlens_iface.get_supported_layers(),
    )
    intervention = JSPaceIntervention(
        jlens_iface,
        lens_model,
        layers=layers,
        method=iv_cfg.get("method", "mean_replace"),
        top_k=iv_cfg.get("top_k", 50),
        strength=iv_cfg.get("strength", 1.0),
        token_strategy=iv_cfg.get("token_strategy", "all_generated_tokens"),
        log_path=out_dir / "hook_log.jsonl",
    )
    print(f"[run_intervention] method={intervention.method} "
          f"top_k={intervention.top_k} strength={intervention.strength} "
          f"layers={layers}")

    out_path = out_dir / "completions.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as fh:
            done = {json.loads(line)["task_id"] for line in fh}
        print(f"[run_intervention] resuming: {len(done)} tasks already completed")

    with intervention, out_path.open("a") as out:
        for i, task in enumerate(tasks):
            if task["task_id"] in done:
                continue
            intervention.reset()  # task-local running mean
            torch.manual_seed(gen_cfg["seed"])
            encoded = tokenizer(task["prompt"], return_tensors="pt").to(input_device)
            prompt_len = encoded.input_ids.shape[1]

            output_ids = hf_model.generate(
                **encoded,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=False,
                stopping_criteria=StoppingCriteriaList(
                    [_StopOnStrings(tokenizer, prompt_len)]
                ),
                pad_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            )

            out.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "model": cfg["model"]["name"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "decoding": gen_cfg,
                        "prompt": task["prompt"],
                        "raw_completion": completion,
                        "extracted_code": extract_code(completion),
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"[run_intervention] {i + 1}/{len(tasks)} {task['task_id']} done")

    (out_dir / "hook_summary.json").write_text(
        json.dumps(intervention.summary(), indent=2)
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": cfg["model"]["name"],
                "benchmark": "HumanEval",
                "condition": "jspace_intervention",
                "baseline_score": BASELINE_SCORE,
                "intervention_method": intervention.method,
                "layers": layers,
                "top_k": intervention.top_k,
                "strength": intervention.strength,
                "token_strategy": intervention.token_strategy,
                "decoding_settings": gen_cfg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    print(f"[run_intervention] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="override benchmark.subset_size (smoke tests)")
    parser.add_argument("--method", default=None,
                        help="override intervention.method (e.g. none for the control run)")
    parser.add_argument("--strength", type=float, default=None,
                        help="override intervention.strength (blend factor α)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.limit is not None:
        cfg["benchmark"]["subset_size"] = args.limit
        cfg["benchmark"]["full_run"] = False
    if args.method is not None:
        cfg["intervention"]["method"] = args.method
    if args.strength is not None:
        cfg["intervention"]["strength"] = args.strength

    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
