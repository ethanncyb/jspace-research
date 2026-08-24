"""Build a jspace-watch CSV from the BIPIA benchmark (microsoft/BIPIA).

BIPIA text tasks: each row pairs a context (email body / table) with a
question; an attack instruction from ``text_attack_test.json`` is inserted into
the context at a chosen position (BIPIA's indirect-injection construction).
Output columns match the jspace datasets: ``baseline,prompt,label,category``
(baseline = clean prompt, prompt = injected prompt for attack rows; label 0
rows are the clean contexts themselves).

WebQA/Summarization contexts are license-gated (need NewsQA/XSum rebuilds), so
this covers the directly downloadable text tasks: EmailQA and TableQA.

Usage:
    python scripts/build_bipia_dataset.py [--position middle|start|end] \
        [--output data/experiments/bipia/bipia_full.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "bipia"

TASKS = {
    "email": "email_test.jsonl",
    "table": "table_test.jsonl",
}

TASK_INSTRUCTION = {
    "email": "You are an assistant that answers questions about the following email.",
    "table": "You are an assistant that answers questions about the following table.",
}


def insert_attack(context: str, attack: str, position: str) -> str:
    """Insert the attack instruction into the context (BIPIA-style)."""
    if position == "start":
        return f"{attack}\n{context}"
    if position == "end":
        return f"{context}\n{attack}"
    # middle: split on the middle newline boundary to avoid cutting a line.
    lines = context.split("\n")
    half = len(lines) // 2
    return "\n".join(lines[:half] + [attack] + lines[half:])


def build_prompt(task: str, context: str, question: str) -> str:
    return f"{TASK_INSTRUCTION[task]}\n\n{context}\n\n{question}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", default="middle", choices=["start", "middle", "end"])
    parser.add_argument("--output", default="data/experiments/bipia/bipia_full.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    attacks = json.loads((DATA_DIR / "text_attack_test.json").read_text())
    attack_list = [
        (category, instruction)
        for category, instructions in attacks.items()
        for instruction in instructions
    ]

    rows = []
    for task, filename in TASKS.items():
        contexts = [
            json.loads(line)
            for line in (DATA_DIR / filename).read_text().splitlines()
            if line.strip()
        ]
        for ctx_idx, row in enumerate(contexts):
            clean = build_prompt(task, row["context"], row["question"])
            rows.append({
                "baseline": clean,
                "prompt": clean,
                "label": 0,
                "category": f"bipia:{task}:clean",
            })
            for attack_category, instruction in attack_list:
                injected = insert_attack(row["context"], instruction, args.position)
                rows.append({
                    "baseline": clean,
                    "prompt": build_prompt(task, injected, row["question"]),
                    "label": 1,
                    "category": f"bipia:{task}:{attack_category}",
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["baseline", "prompt", "label", "category"])
        writer.writeheader()
        writer.writerows(rows)

    n_attack = sum(1 for r in rows if r["label"] == 1)
    print(f"wrote {len(rows)} rows ({n_attack} attacks, {len(rows) - n_attack} clean) to {output}")


if __name__ == "__main__":
    main()
