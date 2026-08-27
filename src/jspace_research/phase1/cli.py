from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-phase1",
        description="Select the fitted J-lens layer with the strongest BIPIA validation signal.",
    )
    parser.add_argument("--config", required=True, help="Phase 1 YAML configuration")
    parser.add_argument(
        "--stage",
        choices=("prepare", "capture", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--bipia-root", help="Override the BIPIA benchmark directory")
    parser.add_argument("--webqa-train", help="BIPIA-format WebQA train.jsonl")
    parser.add_argument("--summarization-train", help="BIPIA-format Summarization train.jsonl")
    parser.add_argument("--output-dir", help="Override the output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import run

    config = load_config(
        args.config,
        bipia_root=args.bipia_root,
        webqa_train_path=args.webqa_train,
        summarization_train_path=args.summarization_train,
        output_dir=args.output_dir,
    )
    run(config, args.stage)


if __name__ == "__main__":
    main()
