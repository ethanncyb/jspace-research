"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gsm8k_jspace.config import (
    apply_cli_overrides,
    config_from_mapping,
    deep_merge,
    dump_resolved_yaml,
    load_yaml_mapping,
)
from gsm8k_jspace.evaluation.compare import compare_runs
from gsm8k_jspace.evaluation.evaluator import evaluate_run
from gsm8k_jspace.platform.capabilities import inspect_backend
from gsm8k_jspace.platform.host import detect_host_profile
from gsm8k_jspace.platform.memory import estimate_run_memory_gb


def _load(args) -> AppConfig:
    merged = load_yaml_mapping(args.config)
    if getattr(args, "host_config", None):
        merged = deep_merge(merged, load_yaml_mapping(args.host_config))
    for overlay in getattr(args, "overlay", None) or []:
        merged = deep_merge(merged, load_yaml_mapping(overlay))
    cfg = config_from_mapping(merged)
    return apply_cli_overrides(
        cfg,
        limit=getattr(args, "limit", None),
        run_id=getattr(args, "run_id", None),
        resume=getattr(args, "resume", False),
        condition=getattr(args, "condition", None),
    )


def cmd_inspect_config(args) -> int:
    cfg = _load(args)
    print(dump_resolved_yaml(cfg))
    return 0


def cmd_diagnose(args) -> int:
    cfg = _load(args) if getattr(args, "config", None) else None
    profile = detect_host_profile(None if cfg is None else cfg.runtime.host_profile)
    info = inspect_backend(
        "auto" if cfg is None else cfg.runtime.backend,
        "auto" if cfg is None else cfg.model.dtype,
    )
    print(
        json.dumps(
            {
                "host_profile": profile.name,
                "venv_dir": profile.venv_dir,
                "backend_requested": None if cfg is None else cfg.runtime.backend,
                "backend_resolved": info.name,
                "dtype": info.dtype_name,
                "device_name": info.device_name,
                "warnings": info.warnings,
                "memory_estimate": (
                    estimate_run_memory_gb(cfg).__dict__ if cfg is not None else None
                ),
            },
            indent=2,
        )
    )
    return 0


def cmd_run(args) -> int:
    from gsm8k_jspace.runner.experiment import run_experiment

    cfg = _load(args)
    run_dir = run_experiment(cfg, project_root=Path(__file__).resolve().parents[2])
    if args.evaluate:
        evaluate_run(run_dir, cfg)
    return 0


def cmd_evaluate(args) -> int:
    evaluate_run(args.run)
    return 0


def cmd_compare(args) -> int:
    compare_runs(args.baseline, args.candidate)
    return 0


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--host-config", default=None)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--condition", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsm8k-jspace")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-config")
    _add_config_args(inspect)
    inspect.set_defaults(func=cmd_inspect_config)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--config", default=None)
    diagnose.add_argument("--host-config", default=None)
    diagnose.add_argument("--overlay", action="append", default=[])
    diagnose.set_defaults(func=cmd_diagnose)

    run = sub.add_parser("run")
    _add_config_args(run)
    run.add_argument("--evaluate", action="store_true")
    run.set_defaults(func=cmd_run)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
