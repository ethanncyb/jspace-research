from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-phase1-robustness",
        description="Run the post-hoc Phase 1 reconstruction and metric robustness study.",
    )
    parser.add_argument("--config", required=True, help="Original Phase 1 YAML configuration")
    parser.add_argument(
        "--phase1", required=True, help="Completed Phase 1 selected_layer.json"
    )
    parser.add_argument("--output-dir", required=True, help="Separate robustness output directory")
    parser.add_argument(
        "--stage", choices=("metrics", "reconstruction", "all"), default="all"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .robustness import run

    phase1_path = Path(args.phase1).expanduser().resolve()
    config = load_config(args.config, output_dir=phase1_path.parent)
    run(config, phase1_path, args.output_dir, args.stage)


if __name__ == "__main__":
    main()
