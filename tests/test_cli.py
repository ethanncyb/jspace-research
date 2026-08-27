from jspace_research.phase1.cli import build_parser


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
