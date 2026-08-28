from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-phase3",
        description="Construct and freeze two selected-layer J-space detectors.",
    )
    parser.add_argument("--config", required=True, help="Shared experiment YAML configuration")
    parser.add_argument("--phase1", required=True, help="Frozen Phase 1 selected_layer.json")
    parser.add_argument("--output-dir", required=True, help="Phase 3 output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import run

    run(
        load_config(
            args.config,
            phase1_selected_path=args.phase1,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
