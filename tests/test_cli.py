from jspace_research.phase1.cli import build_parser
from jspace_research.phase1.robustness_cli import build_parser as build_robustness_parser
from jspace_research.phase2.cli import build_parser as build_phase2_parser
from jspace_research.phase3.cli import build_parser as build_phase3_parser
from jspace_research.phase4.cli import build_parser as build_phase4_parser
from jspace_research.raw_baseline.cli import build_parser as build_raw_baseline_parser


def test_cli_exposes_only_phase1_stages_and_path_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--stage",
            "prepare",
            "--bipia-root",
            "/data/BIPIA/benchmark",
            "--output-dir",
            "/output",
        ]
    )
    assert args.stage == "prepare"
    assert args.bipia_root == "/data/BIPIA/benchmark"
    assert args.output_dir == "/output"


def test_phase1_robustness_cli_uses_frozen_handoff_and_separate_output() -> None:
    args = build_robustness_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--phase1",
            "/run/phase1/selected_layer.json",
            "--output-dir",
            "/run/phase1_robustness",
        ]
    )
    assert args.phase1.endswith("selected_layer.json")
    assert args.output_dir.endswith("phase1_robustness")


def test_phase2_cli_exposes_only_generation_analysis_stages() -> None:
    args = build_phase2_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--phase1",
            "/run/phase1/selected_layer.json",
            "--output-dir",
            "/run/phase2",
            "--stage",
            "generate",
        ]
    )
    assert args.stage == "generate"
    assert args.phase1.endswith("selected_layer.json")


def test_phase3_cli_is_one_cpu_command_without_stages() -> None:
    args = build_phase3_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--phase1",
            "/run/phase1/selected_layer.json",
            "--output-dir",
            "/run/phase3",
        ]
    )
    assert args.phase1.endswith("selected_layer.json")
    assert args.output_dir == "/run/phase3"
    assert not hasattr(args, "stage")


def test_phase4_cli_keeps_generation_and_analysis_separate() -> None:
    args = build_phase4_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--phase1",
            "/run/phase1/selected_layer.json",
            "--phase3",
            "/run/phase3",
            "--bipia-root",
            "/benchmarks/BIPIA/benchmark",
            "--agentdojo-root",
            "/benchmarks/agentdojo",
            "--injecagent-root",
            "/benchmarks/InjecAgent",
            "--output-dir",
            "/run/phase4",
            "--stage",
            "analyze",
        ]
    )
    assert args.stage == "analyze"
    assert args.phase3 == "/run/phase3"


def test_raw_baseline_cli_keeps_fit_transfer_and_analysis_separate() -> None:
    args = build_raw_baseline_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--phase1",
            "/run/phase1/selected_layer.json",
            "--jspace-phase3",
            "/run/phase3",
            "--jspace-phase4",
            "/run/phase4",
            "--agentdojo-root",
            "/benchmarks/agentdojo",
            "--injecagent-root",
            "/benchmarks/InjecAgent",
            "--output-dir",
            "/run/raw_baseline",
            "--stage",
            "fit",
        ]
    )
    assert args.stage == "fit"
    assert args.jspace_phase4 == "/run/phase4"
