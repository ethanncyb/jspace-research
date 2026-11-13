from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import load_config


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="jspace-format-grid",
        description="Score the frozen detectors on a controlled format x wrapper x injection grid.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1", required=True, help="Frozen Phase 1 selected_layer.json")
    parser.add_argument("--phase3", required=True, help="Frozen Phase 3 output directory")
    parser.add_argument("--bipia-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records", type=int, default=60, help="Base records per cell")
    parser.add_argument("--seed", type=int, default=0)
    args: argparse.Namespace = parser.parse_args(argv)
    from .format_grid import run

    config = load_config(
        args.config,
        phase1_selected_path=args.phase1,
        phase3_dir=args.phase3,
        bipia_root=args.bipia_root,
        agentdojo_root=args.output_dir,
        injecagent_root=args.output_dir,
        output_dir=args.output_dir,
    )
    run(config, args.records, Path(args.output_dir).expanduser().resolve(), args.seed)


if __name__ == "__main__":
    main()
