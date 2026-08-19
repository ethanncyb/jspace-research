"""Phase-1 baseline runner: Qwen3.5-9B-Base on HumanEval, no intervention.

Generates greedily with the read-only JSpaceCapture hooks active (when
``jspace.enabled``), saving raw completions and J-Space activation records.

Usage:
    python -m src.run_baseline --config config.yaml            # subset per config
    python -m src.run_baseline --config config.yaml --limit 5  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import StoppingCriteria, StoppingCriteriaList

from src.capture_jspace import JSpaceCapture, resolve_layers
from src.humaneval_data import load_humaneval
from src.load_jlens import load_jlens
from src.load_model import deterministic_gen_config, load_model

# Standard HumanEval stop sequences for base-model completion prompting.
STOP_STRINGS = ["\nclass", "\ndef ", "\n#", "\nif __name__", "\nprint"]


class _StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self._tokenizer.decode(
            input_ids[0, self._prompt_len:], skip_special_tokens=True
        )
        return any(s in text for s in STOP_STRINGS)


def extract_code(raw_completion: str) -> str:
    """Truncate the raw completion at the first HumanEval stop sequence."""
    cut = len(raw_completion)
    for stop in STOP_STRINGS:
        idx = raw_completion.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return raw_completion[:cut]


def run(cfg: dict) -> None:
    if cfg["jspace"].get("intervention_enabled", False):
        raise ValueError(
            "jspace.intervention_enabled is a Phase-2 placeholder and must "
            "stay false in Phase 1"
        )
    gen_cfg = deterministic_gen_config(cfg)
    out_base = Path(cfg["outputs"]["base_dir"])
    completions_dir = out_base / "completions"
    activations_dir = out_base / "activations"
    completions_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_humaneval(cfg)
    hf_model, tokenizer = load_model(cfg)
    input_device = hf_model.get_input_embeddings().weight.device

    jlens_iface = None
    lens_model = None
    capture_layers: list[int] = []
    if cfg["jspace"].get("enabled", True):
        jlens_iface, lens_model = load_jlens(cfg, hf_model, tokenizer)
        capture_layers = resolve_layers(
            cfg["jspace"].get("layers", "late"),
            lens_model.n_layers,
            jlens_iface.get_supported_layers(),
        )
        print(f"[run_baseline] capturing J-Space at layers {capture_layers}")

    out_path = completions_dir / "completions.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as fh:
            done = {json.loads(line)["task_id"] for line in fh}
        print(f"[run_baseline] resuming: {len(done)} tasks already completed")

    with out_path.open("a") as out:
        for i, task in enumerate(tasks):
            if task["task_id"] in done:
                continue
            torch.manual_seed(gen_cfg["seed"])
            encoded = tokenizer(task["prompt"], return_tensors="pt").to(input_device)
            prompt_len = encoded.input_ids.shape[1]

            capture = None
            if jlens_iface is not None and cfg["jspace"].get("save_activations", True):
                capture = JSpaceCapture(
                    lens_model,
                    jlens_iface,
                    layers=capture_layers,
                    top_k=cfg["jspace"].get("top_k_features", 20),
                    prompt_len=prompt_len,
                )

            ctx = capture if capture is not None else _nullcontext()
            with ctx:
                output_ids = hf_model.generate(
                    **encoded,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList(
                        [_StopOnStrings(tokenizer, prompt_len)]
                    ),
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, prompt_len:].tolist()
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if capture is not None:
                capture.attach_tokens(generated_ids, tokenizer)
                capture.save(activations_dir / f"{task['task_id'].replace('/', '_')}.jsonl")

            out.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "model": cfg["model"]["name"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "decoding": gen_cfg,
                        "jlens_status": (
                            jlens_iface.get_metadata()["status"]
                            if jlens_iface is not None
                            else "disabled"
                        ),
                        "prompt": task["prompt"],
                        "raw_completion": completion,
                        "extracted_code": extract_code(completion),
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"[run_baseline] {i + 1}/{len(tasks)} {task['task_id']} done")

    print(f"[run_baseline] wrote {out_path}")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="override benchmark.subset_size (smoke tests)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.limit is not None:
        cfg["benchmark"]["subset_size"] = args.limit
        cfg["benchmark"]["full_run"] = False

    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
