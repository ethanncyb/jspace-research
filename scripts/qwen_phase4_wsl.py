"""Prepare and resume the pinned Qwen Phase 4 AgentDojo rerun on WSL.

Extract the numbered Drive archives into ``artifacts/qwen/shared`` first. Run this
script with the project's virtual environment, for example::

    .venv/bin/python scripts/qwen_phase4_wsl.py --stage prepare
    .venv/bin/python scripts/qwen_phase4_wsl.py --stage generate
    .venv/bin/python scripts/qwen_phase4_wsl.py --stage analyze
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "artifacts/qwen/shared"
RUN = ROOT / "artifacts/qwen/agentdojo-rerun"
PHASE1 = SHARED / "phase1/selected_layer_k25.json"
PHASE3 = SHARED / "phase3_implementation/results/k25"
PHASE4_SOURCE = SHARED / "phase4_implementation (1)/results/k25"
BENCHMARKS = SHARED / "phase4_implementation (1)/benchmarks"
PHASE4 = RUN / "phase4"
CONFIG = ROOT / "configs/phase1_qwen3_5_9b_full.yaml"
REUSED_FILES = (
    "bipia_test_manifest.jsonl",
    "bipia_records.jsonl",
    "bipia_judgments.jsonl",
    "injecagent_records.jsonl",
)
FROZEN_PROVENANCE_KEYS = (
    "schema_version",
    "phase",
    "run_id",
    "phase4_config_sha256",
    "bipia_test_manifest_sha256",
    "phase1_run_id",
    "phase3_config_sha256",
    "mean_detector_sha256",
    "logistic_detector_sha256",
    "model",
    "lens",
    "selected_layer",
    "decomposition",
    "detectors",
    "resolved_config",
    "judge",
)


def _merge_archive_folders() -> None:
    """Link the disjoint Drive ZIP parts into the original run layout."""

    archives = sorted(SHARED.glob("drive-download-*"))
    linked = 0
    for archive in archives:
        if not archive.is_dir():
            continue
        for source in archive.rglob("*"):
            if not source.is_file() or source.name.endswith(":Zone.Identifier"):
                continue
            relative = source.relative_to(archive)
            if relative.parts[0] not in {
                "phase1",
                "phase3_implementation",
                "phase4_implementation (1)",
            }:
                continue
            destination = SHARED / relative
            if destination.exists():
                if not destination.samefile(source):
                    raise RuntimeError(f"Conflicting ZIP contents: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            linked += 1
    if linked:
        print(f"Merged {linked} files from the Drive ZIP folders without duplicating data")


def _require_inputs() -> None:
    required = [
        PHASE1,
        PHASE3 / "mean_detector.pt",
        PHASE3 / "logistic_detector.pt",
        PHASE4_SOURCE / "provenance.json",
        *(PHASE4_SOURCE / name for name in REUSED_FILES),
        BENCHMARKS / "BIPIA/benchmark",
        BENCHMARKS / "agentdojo",
        BENCHMARKS / "InjecAgent",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing extracted files:\n" + "\n".join(missing))


def _copy_unchanged(source: Path, destination: Path) -> None:
    from jspace_research.runtime import sha256_file

    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Existing staged file differs: {destination}")
        return
    shutil.copy2(source, destination)


def prepare() -> None:
    from jspace_research.phase1.artifacts import load_phase1_handoff
    from jspace_research.phase1.data import verify_bipia_revision
    from jspace_research.phase4.agentdojo import HARNESS_VERSION
    from jspace_research.phase4.common import verify_checkout
    from jspace_research.phase4.config import load_config
    from jspace_research.phase4.detectors import FrozenDetectors
    from jspace_research.runtime import sha256_file

    _merge_archive_folders()
    _require_inputs()
    config = load_config(
        CONFIG,
        phase1_selected_path=PHASE1,
        phase3_dir=PHASE3,
        bipia_root=BENCHMARKS / "BIPIA/benchmark",
        agentdojo_root=BENCHMARKS / "agentdojo",
        injecagent_root=BENCHMARKS / "InjecAgent",
        output_dir=PHASE4,
    )
    verify_bipia_revision(config.phase1)
    verify_checkout(config.agentdojo_root, config.agentdojo_revision, "AgentDojo")
    verify_checkout(config.injecagent_root, config.injecagent_revision, "InjecAgent")
    handoff = load_phase1_handoff(PHASE1, config.phase1)
    detectors = FrozenDetectors.load(PHASE3, handoff.metadata)
    provenance = json.loads((PHASE4_SOURCE / "provenance.json").read_text())
    if config.identity_hash() != provenance["phase4_config_sha256"]:
        raise RuntimeError("The local config differs from the frozen Qwen Phase 4 run")
    if handoff.metadata["run_id"] != provenance["phase1_run_id"]:
        raise RuntimeError("The Phase 1 handoff differs from Phase 4 provenance")
    for name in ("mean", "logistic"):
        expected = provenance["detectors"][f"{name}_sha256"]
        if sha256_file(PHASE3 / f"{name}_detector.pt") != expected:
            raise RuntimeError(f"The frozen {name} detector hash differs")
    if detectors.mean["selected_layer"] != handoff.selected_layer:
        raise RuntimeError("The selected layer differs from the frozen detectors")
    if sha256_file(PHASE4_SOURCE / "bipia_test_manifest.jsonl") != provenance[
        "bipia_test_manifest_sha256"
    ]:
        raise RuntimeError("The BIPIA manifest hash differs from Phase 4 provenance")

    PHASE4.mkdir(parents=True, exist_ok=True)
    local_provenance = PHASE4 / "provenance.json"
    if local_provenance.exists():
        staged = json.loads(local_provenance.read_text())
        for key in FROZEN_PROVENANCE_KEYS:
            if staged.get(key) != provenance.get(key):
                raise RuntimeError(f"Staged Phase 4 provenance differs in frozen field: {key}")
    else:
        _copy_unchanged(PHASE4_SOURCE / "provenance.json", local_provenance)
    for name in REUSED_FILES:
        _copy_unchanged(PHASE4_SOURCE / name, PHASE4 / name)
    old_agentdojo = PHASE4_SOURCE / "agentdojo_records.jsonl"
    if old_agentdojo.is_file():
        _copy_unchanged(old_agentdojo, PHASE4 / "agentdojo_records_before_fix.jsonl")
    active_records = PHASE4 / "agentdojo_records.jsonl"
    active_trajectories = PHASE4 / "agentdojo_trajectories.jsonl"
    if active_records.exists() != active_trajectories.exists():
        raise RuntimeError("Active AgentDojo records and trajectories must be present together")
    if active_trajectories.exists():
        with active_trajectories.open() as stream:
            first_line = stream.readline()
        version = json.loads(first_line).get("harness_version") if first_line else None
        if version != HARNESS_VERSION:
            raise RuntimeError(
                f"Active AgentDojo trajectories use harness {version}, expected {HARNESS_VERSION}"
            )
    print(f"Qwen Phase 4 inputs verified; resumable output: {PHASE4}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "generate", "analyze", "all"), required=True)
    args = parser.parse_args()
    prepare()
    if args.stage == "prepare":
        return
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / "artifacts/qwen/hf-home"))
    env.setdefault("HF_HUB_CACHE", str(ROOT / "artifacts/qwen/hf-cache"))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    base = [
        sys.executable,
        "-m",
        "jspace_research.phase4.cli",
        "--config",
        str(CONFIG),
        "--phase1",
        str(PHASE1),
        "--phase3",
        str(PHASE3),
        "--bipia-root",
        str(BENCHMARKS / "BIPIA/benchmark"),
        "--agentdojo-root",
        str(BENCHMARKS / "agentdojo"),
        "--injecagent-root",
        str(BENCHMARKS / "InjecAgent"),
        "--output-dir",
        str(PHASE4),
    ]
    stages = ("generate", "analyze") if args.stage == "all" else (args.stage,)
    for stage in stages:
        subprocess.run([*base, "--stage", stage], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
