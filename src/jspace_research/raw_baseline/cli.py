from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-raw-baseline",
        description="Compare same-layer raw-residual and J-space detectors.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--jspace-phase3", required=True)
    parser.add_argument("--jspace-phase4", required=True)
    parser.add_argument("--agentdojo-root", required=True)
    parser.add_argument("--injecagent-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("fit", "transfer", "analyze", "all"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import run

    run(
        load_config(
            args.config,
            phase1_selected_path=args.phase1,
            jspace_phase3_dir=args.jspace_phase3,
            jspace_phase4_dir=args.jspace_phase4,
            agentdojo_root=args.agentdojo_root,
            injecagent_root=args.injecagent_root,
            output_dir=args.output_dir,
        ),
        args.stage,
    )


if __name__ == "__main__":
    main()
