"""HumanEval dataset loading.

Loads ``openai/openai_humaneval`` and returns plain dicts so the rest of the
pipeline never touches the datasets library. ``benchmark.subset_size`` bounds
the subset for smoke tests; ``benchmark.full_run: true`` runs all 164 tasks.
"""

from __future__ import annotations

from datasets import load_dataset


def load_humaneval(cfg: dict) -> list[dict]:
    bench = cfg["benchmark"]
    ds = load_dataset(bench["dataset"])[bench.get("split", "test")]

    tasks = [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]
    if not bench.get("full_run", False):
        tasks = tasks[: int(bench.get("subset_size", 10))]
    print(f"[humaneval_data] loaded {len(tasks)} tasks "
          f"(full_run={bench.get('full_run', False)})")
    return tasks
