from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-phase2",
        description="Measure behavior while removing the selected J-space reconstruction.",
    )
    parser.add_argument("--config", required=True, help="Shared experiment YAML configuration")
    parser.add_argument(
        "--phase1",
        required=True,
        help="Frozen Phase 1 selected_layer.json",
    )
    parser.add_argument("--output-dir", required=True, help="Phase 2 output directory")
    parser.add_argument(
        "--stage",
        choices=("generate", "analyze", "all"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import run

    config = load_config(
        args.config,
        phase1_selected_path=args.phase1,
        output_dir=args.output_dir,
    )
    run(config, args.stage)


if __name__ == "__main__":
    main()
