"""Tag Phase 4 cases by content shape and report detector behaviour per tag.

Usage:
    .venv/bin/python tools_local/tag_content_shape.py \
        --predictions results/phase4-fixed/hub/jspace-full/phase4/phase4_predictions.parquet \
        --injecagent-root /path/to/InjecAgent \
        --out content_shape_report.csv

Tag sources:
    bipia       the BIPIA task name (email, qa, table, abstract, code)
    injecagent  shape of the item's "Tool Response" text (json, code, text)
    agentdojo   suite name only. Phase 4 records keep prompt hashes, not the
                tool-output text, so a shape tag needs the agent loop re-run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_PATTERN: re.Pattern[str] = re.compile(
    r"(```|^\s*(def|class|import|from|function|const|let|var|return)\b|;\s*$|\{\s*$|=>|</?\w+>)",
    re.MULTILINE,
)
DETECTORS: tuple[str, ...] = ("mean", "logistic")


def classify_shape(text: str) -> str:
    """Return json, pyrepr, code, or text for a tool-response string.

    InjecAgent stores responses as a Python-repr dict, sometimes wrapped in an
    extra pair of quotes, so strict JSON alone would tag them all as text.
    """
    stripped: str = text.strip()
    for candidate in (stripped, stripped.strip("\"'")):
        try:
            parsed: Any = json.loads(candidate)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict | list):
            return "json"
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            parsed = None
        if isinstance(parsed, dict | list):
            return "pyrepr"
    if CODE_PATTERN.search(stripped) is not None:
        return "code"
    return "text"


def injecagent_shapes(root: Path) -> dict[str, str]:
    """Map injecagent case ids to the shape of their tool response."""
    shapes: dict[str, str] = {}
    for subgroup, filename in (
        ("direct_harm", "test_cases_dh_base.json"),
        ("data_stealing", "test_cases_ds_base.json"),
    ):
        items: list[dict[str, Any]] = json.loads(
            (root / "data" / filename).read_text(encoding="utf-8")
        )
        for index, item in enumerate(items):
            shapes[f"injecagent:{subgroup}:{index:04d}"] = classify_shape(
                str(item["Tool Response"])
            )
    return shapes


def add_tags(frame: pd.DataFrame, injecagent_root: Path | None) -> pd.DataFrame:
    """Add a 'tag' column. Rows without a source for a shape tag get 'untagged'."""
    tagged: pd.DataFrame = frame.copy()
    tags: pd.Series = pd.Series("untagged", index=tagged.index, dtype="object")
    bipia: pd.Series = tagged.benchmark == "bipia"
    tags[bipia] = tagged.loc[bipia, "task"].astype(str)
    agentdojo: pd.Series = tagged.benchmark == "agentdojo"
    tags[agentdojo] = tagged.loc[agentdojo, "subgroup"].astype(str)
    if injecagent_root is not None:
        shapes: dict[str, str] = injecagent_shapes(injecagent_root)
        injecagent: pd.Series = tagged.benchmark == "injecagent"
        tags[injecagent] = tagged.loc[injecagent, "case_id"].map(shapes).fillna("untagged")
    tagged["tag"] = tags
    return tagged


def summarize(group: pd.DataFrame, detector: str) -> dict[str, Any]:
    """Rates and threshold-free separation for one benchmark/tag group.

    Rows without a score (AgentDojo attacks whose injection never entered the
    context) have no decision point and are dropped; their count is reported.
    """
    scored: pd.DataFrame = group[group[f"{detector}_score"].notna()]
    n_unscored: int = len(group) - len(scored)
    group = scored
    attack: pd.DataFrame = group[group.condition == "attack"]
    clean: pd.DataFrame = group[group.condition == "control"]
    score: str = f"{detector}_score"
    prediction: str = f"{detector}_prediction"
    auroc: float = float("nan")
    if len(attack) > 0 and len(clean) > 0:
        labels: np.ndarray = (group.condition == "attack").to_numpy(dtype=int)
        auroc = float(roc_auc_score(labels, group[score].to_numpy(dtype=float)))
    return {
        "detector": detector,
        "n_attack": len(attack),
        "n_clean": len(clean),
        "n_unscored": n_unscored,
        "tpr": float(attack[prediction].mean()) if len(attack) else float("nan"),
        "fpr": float(clean[prediction].mean()) if len(clean) else float("nan"),
        "auroc": auroc,
        "median_score_attack": float(attack[score].median()) if len(attack) else float("nan"),
        "median_score_clean": float(clean[score].median()) if len(clean) else float("nan"),
    }


def report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (benchmark, tag), group in frame.groupby(["benchmark", "tag"]):
        for detector in DETECTORS:
            rows.append({"benchmark": benchmark, "tag": tag, **summarize(group, detector)})
    return pd.DataFrame(rows)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--injecagent-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args: argparse.Namespace = parser.parse_args()

    frame: pd.DataFrame = add_tags(pd.read_parquet(args.predictions), args.injecagent_root)
    result: pd.DataFrame = report(frame)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(result.to_string(index=False))
    if args.out is not None:
        result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
