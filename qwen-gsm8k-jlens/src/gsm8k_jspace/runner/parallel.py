"""Data-parallel GSM8K runs across multiple CUDA/ROCm GPUs."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from gsm8k_jspace.artifacts.writer import read_jsonl, write_json, write_jsonl
from gsm8k_jspace.config import AppConfig, ConfigError, config_from_mapping
from gsm8k_jspace.platform.device import (
    pin_visible_gpu,
    prepare_worker_device_config,
    resolve_gpus,
)
from gsm8k_jspace.types import GSM8KExample


def partition_examples(
    examples: Sequence[GSM8KExample], n_workers: int
) -> list[list[GSM8KExample]]:
    """Deterministic round-robin shard of examples across workers."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    shards: list[list[GSM8KExample]] = [[] for _ in range(n_workers)]
    for index, example in enumerate(examples):
        shards[index % n_workers].append(example)
    return shards


def completions_shard_path(run_dir: Path, worker_id: int) -> Path:
    return run_dir / f"completions.shard-{worker_id}.jsonl"


def capture_index_shard_path(run_dir: Path, worker_id: int) -> Path:
    return run_dir / "captures" / f"index.shard-{worker_id}.jsonl"


def merge_jsonl_by_example_id(paths: Sequence[Path], out_path: Path) -> int:
    """Merge JSONL shards, keeping first row per example_id in path order."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            example_id = row.get("example_id")
            if example_id is None:
                continue
            key = str(example_id)
            if key in by_id:
                continue
            order.append(key)
            by_id[key] = row
    write_jsonl(out_path, [by_id[example_id] for example_id in order])
    return len(order)


def merge_parallel_shards(run_dir: Path, n_workers: int) -> int:
    """Rewrite canonical completions/capture index from merged + shard files."""
    completion_paths = [run_dir / "completions.jsonl"] + [
        completions_shard_path(run_dir, worker_id) for worker_id in range(n_workers)
    ]
    n_completions = merge_jsonl_by_example_id(
        completion_paths, run_dir / "completions.jsonl"
    )
    capture_paths = [run_dir / "captures" / "index.jsonl"] + [
        capture_index_shard_path(run_dir, worker_id) for worker_id in range(n_workers)
    ]
    merge_jsonl_by_example_id(capture_paths, run_dir / "captures" / "index.jsonl")
    return n_completions


def _example_to_dict(example: GSM8KExample) -> dict[str, Any]:
    return asdict(example)


def _example_from_dict(payload: dict[str, Any]) -> GSM8KExample:
    return GSM8KExample(**payload)


def _worker_entry(
    cfg_dict: dict[str, Any],
    run_dir_str: str,
    examples_payload: list[dict[str, Any]],
    worker_id: int,
    gpu_id: int,
    total_examples: int,
) -> dict[str, Any]:
    """Spawn target: pin one GPU, then run a shard of examples."""
    pin_visible_gpu(gpu_id)
    from gsm8k_jspace.runner.experiment import run_example_shard

    cfg = prepare_worker_device_config(config_from_mapping(cfg_dict))
    examples = [_example_from_dict(row) for row in examples_payload]
    run_dir = Path(run_dir_str)
    written = run_example_shard(
        cfg,
        examples=examples,
        run_dir=run_dir,
        worker_id=worker_id,
        total_examples=total_examples,
    )
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "written": written,
        "n_examples": len(examples),
    }


def run_parallel_experiment(
    cfg: AppConfig,
    *,
    examples: list[GSM8KExample],
    run_dir: Path,
    project_root: Path | None = None,
) -> Path:
    """Shard incomplete examples across GPUs and merge shard artifacts."""
    from gsm8k_jspace.artifacts.manifest import finalize_manifest, load_completed_ids
    from gsm8k_jspace.platform.capabilities import inspect_backend

    del project_root  # reserved for parity with run_experiment signature
    gpus = resolve_gpus(cfg)
    info = inspect_backend(cfg.runtime.backend, cfg.model.dtype)
    if info.name not in {"cuda", "rocm"}:
        raise ConfigError(
            f"runtime.parallel requires cuda/rocm backend, resolved {info.name!r}"
        )

    n_workers = len(gpus)
    merge_parallel_shards(run_dir, n_workers)
    done = load_completed_ids(run_dir)
    pending = [example for example in examples if example.example_id not in done]
    if done:
        print(f"[run] resuming: {len(done)} examples already completed")
    if not pending:
        finalize_manifest(run_dir, completed_examples=len(done), status="complete")
        print(f"[run] wrote {run_dir}")
        return run_dir

    shards = partition_examples(pending, n_workers)
    print(
        f"[run] parallel gpus={gpus} pending={len(pending)} "
        f"shard_sizes={[len(shard) for shard in shards]}"
    )

    for worker_id in range(n_workers):
        completions_shard_path(run_dir, worker_id).unlink(missing_ok=True)
        capture_index_shard_path(run_dir, worker_id).unlink(missing_ok=True)

    cfg_dict = cfg.to_dict()
    ctx = mp.get_context("spawn")
    jobs: list[tuple[Any, list[GSM8KExample]]] = []
    for worker_id, (gpu_id, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        process = ctx.Process(
            target=_worker_entry,
            args=(
                cfg_dict,
                str(run_dir),
                [_example_to_dict(example) for example in shard],
                worker_id,
                gpu_id,
                len(examples),
            ),
        )
        jobs.append((process, shard))
        process.start()

    failures: list[str] = []
    try:
        for process, shard in jobs:
            process.join()
            if process.exitcode != 0:
                failures.append(
                    f"worker pid={process.pid} exit={process.exitcode} "
                    f"n_examples={len(shard)}"
                )
    finally:
        n_written = merge_parallel_shards(run_dir, n_workers)
        write_json(
            run_dir / "intervention" / "summary.json",
            {"method": "parallel_baseline", "workers": n_workers, "gpus": gpus},
        )
        if failures:
            finalize_manifest(run_dir, completed_examples=n_written, status="interrupted")
            raise RuntimeError("parallel workers failed: " + "; ".join(failures))
        finalize_manifest(run_dir, completed_examples=n_written, status="complete")

    print(f"[run] wrote {run_dir}")
    return run_dir
