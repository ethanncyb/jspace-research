from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-run",
        description="Run or resume the J-space experiment through Phase 2.",
    )
    parser.add_argument("--config", required=True, help="Shared experiment YAML configuration")
    parser.add_argument("--run-dir", required=True, help="Root containing phase1/ and phase2/")
    parser.add_argument("--bipia-root", help="Override the BIPIA benchmark directory")
    parser.add_argument("--webqa-train", help="BIPIA-format WebQA train.jsonl")
    parser.add_argument("--summarization-train", help="BIPIA-format Summarization train.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .phase1.artifacts import load_selected_layer
    from .phase1.config import load_config as load_phase1_config
    from .phase1.pipeline import run as run_phase1
    from .phase2.config import load_config as load_phase2_config
    from .phase2.pipeline import run as run_phase2

    run_dir = Path(args.run_dir).expanduser().resolve()
    phase1_dir = run_dir / "phase1"
    phase2_dir = run_dir / "phase2"
    phase1_config = load_phase1_config(
        args.config,
        bipia_root=args.bipia_root,
        webqa_train_path=args.webqa_train,
        summarization_train_path=args.summarization_train,
        output_dir=phase1_dir,
    )
    selected_path = phase1_dir / "selected_layer.json"
    if selected_path.exists():
        selected, _ = load_selected_layer(selected_path)
        if selected.get("config_sha256") != phase1_config.identity_hash():
            raise RuntimeError("Existing Phase 1 handoff does not match the supplied config")
        print(f"Using frozen Phase 1 handoff: {selected_path}")
    else:
        run_phase1(phase1_config, "all")

    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    phase2_config = load_phase2_config(
        args.config,
        phase1_selected_path=selected_path,
        output_dir=phase2_dir,
    )
    run_phase2(phase2_config, "all")


if __name__ == "__main__":
    main()
