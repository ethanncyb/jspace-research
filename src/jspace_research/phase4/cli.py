from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-phase4",
        description="Evaluate the frozen Phase 3 detectors on held-out benchmarks.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1", required=True, help="Frozen Phase 1 selected_layer.json")
    parser.add_argument("--phase3", required=True, help="Frozen Phase 3 output directory")
    parser.add_argument("--bipia-root", required=True, help="BIPIA benchmark directory")
    parser.add_argument("--agentdojo-root", required=True, help="Pinned AgentDojo checkout")
    parser.add_argument("--injecagent-root", required=True, help="Pinned InjecAgent checkout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("generate", "analyze", "all"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import run

    run(
        load_config(
            args.config,
            phase1_selected_path=args.phase1,
            phase3_dir=args.phase3,
            bipia_root=args.bipia_root,
            agentdojo_root=args.agentdojo_root,
            injecagent_root=args.injecagent_root,
            output_dir=args.output_dir,
        ),
        args.stage,
    )


if __name__ == "__main__":
    main()
