"""Tabular summaries for notebooks and reports."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Literal

from gsm8k_jspace.analysis.loaders import RunArtifacts, iter_capture_rows

TopkSource = Literal["jspace", "logit", "model"]
PositionSel = int | Literal["last", "all"] | None

_SOURCE_FIELDS: dict[str, str] = {
    "jspace": "top_jspace_tokens",
    "logit": "top_logit_tokens",
    "model": "top_model_tokens",
}


def accuracy_table(run: RunArtifacts) -> list[dict[str, Any]]:
    if run.summary:
        return [
            {
                "run_id": run.summary.get("run_id"),
                "condition": run.summary.get("condition"),
                "accuracy": run.summary.get("accuracy"),
                "n_correct": run.summary.get("n_correct"),
                "n_evaluated": run.summary.get("n_evaluated"),
                "extraction_rate": run.summary.get("extraction_rate"),
            }
        ]
    return []


def capture_layer_table(
    run: RunArtifacts, *, max_rows: int = 250_000
) -> list[dict[str, Any]]:
    acc: dict[int, dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "hidden_norm": 0.0, "jspace_norm": 0.0}
    )
    for row in iter_capture_rows(run.run_dir, max_rows=max_rows):
        layer = int(row["layer"])
        acc[layer]["n"] += 1
        acc[layer]["hidden_norm"] += float(row.get("hidden_norm") or 0.0)
        acc[layer]["jspace_norm"] += float(row.get("jspace_norm") or 0.0)
    table = []
    for layer in sorted(acc):
        n = max(acc[layer]["n"], 1.0)
        table.append(
            {
                "layer": layer,
                "n": int(acc[layer]["n"]),
                "mean_hidden_norm": acc[layer]["hidden_norm"] / n,
                "mean_jspace_norm": acc[layer]["jspace_norm"] / n,
            }
        )
    return table


def capture_readout_table(
    run: RunArtifacts,
    *,
    example_id: str | None = None,
    max_rows: int = 20_000,
) -> list[dict[str, Any]]:
    """Layer × token J-lens vs logit-lens vs model top-1 for notebooks."""
    table = []
    for row in iter_capture_rows(
        run.run_dir, example_id=example_id, max_rows=max_rows
    ):
        jspace = row.get("top_jspace_tokens") or []
        logit = row.get("top_logit_tokens") or []
        model = row.get("top_model_tokens") or []
        table.append(
            {
                "example_id": row.get("example_id"),
                "layer": row.get("layer"),
                "position": row.get("absolute_position"),
                "segment": row.get("segment"),
                "token": row.get("token_text"),
                "special": row.get("is_special"),
                "jspace_top": (jspace[0]["text"] if jspace else None),
                "jspace_logit": (jspace[0]["logit"] if jspace else None),
                "logit_lens_top": (logit[0]["text"] if logit else None),
                "logit_lens_logit": (logit[0]["logit"] if logit else None),
                "model_top": (model[0]["text"] if model else None),
                "hidden_norm": row.get("hidden_norm"),
                "jspace_norm": row.get("jspace_norm"),
            }
        )
    return table


def format_token_text(text: Any) -> str:
    """Quote token text so leading spaces and newlines stay visible."""
    if text is None:
        return ""
    return json.dumps(str(text), ensure_ascii=False)


def _source_field(source: str) -> str:
    try:
        return _SOURCE_FIELDS[source]
    except KeyError as exc:
        raise ValueError(
            f"source must be one of {tuple(_SOURCE_FIELDS)}, got {source!r}"
        ) from exc


def _filter_position(
    rows: list[dict[str, Any]], position: PositionSel
) -> list[dict[str, Any]]:
    if position == "all":
        return rows
    by_example: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[row.get("example_id")].append(row)
    kept: list[dict[str, Any]] = []
    target = None if position in {None, "last"} else int(position)
    for example_rows in by_example.values():
        if target is None:
            generated = [
                row
                for row in example_rows
                if row.get("generated_position") is not None
            ]
            pool = generated or example_rows
            last_pos = max(int(row["absolute_position"]) for row in pool)
            kept.extend(
                row
                for row in example_rows
                if int(row.get("absolute_position", -1)) == last_pos
            )
        else:
            kept.extend(
                row
                for row in example_rows
                if int(row.get("absolute_position", -1)) == target
            )
    return kept


def _capture_rows_for(
    run: RunArtifacts,
    *,
    example_id: str | None,
    position: PositionSel,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows = list(
        iter_capture_rows(run.run_dir, example_id=example_id, max_rows=max_rows)
    )
    return _filter_position(rows, position)


def _topk_hits(
    rows: list[dict[str, Any]],
    *,
    source: str,
    max_rank: int | None,
) -> list[dict[str, Any]]:
    field = _source_field(source)
    hits: list[dict[str, Any]] = []
    for row in rows:
        items = row.get(field) or []
        if max_rank is not None:
            items = items[: int(max_rank)]
        for rank, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            hits.append(
                {
                    "example_id": row.get("example_id"),
                    "layer": int(row["layer"]),
                    "absolute_position": row.get("absolute_position"),
                    "generated_position": row.get("generated_position"),
                    "segment": row.get("segment"),
                    "state_token": row.get("token_text"),
                    "source": source,
                    "rank": rank,
                    "token_id": item.get("token_id"),
                    "token": item.get("text"),
                    "token_display": format_token_text(item.get("text")),
                    "logit": item.get("logit"),
                }
            )
    return hits


def _hit_cell(hit: dict[str, Any] | None) -> str:
    if hit is None:
        return ""
    token = hit["token_display"]
    logit = hit.get("logit")
    if logit is None:
        return token
    return f"{token}  ({float(logit):.2f})"


def topk_token_table(
    run: RunArtifacts,
    *,
    example_id: str | None = None,
    position: PositionSel = "last",
    source: TopkSource = "jspace",
    max_rank: int | None = None,
    max_rows: int = 50_000,
) -> list[dict[str, Any]]:
    """One row per (layer, rank) top-k hit at the selected token position."""
    rows = _capture_rows_for(
        run, example_id=example_id, position=position, max_rows=max_rows
    )
    return _topk_hits(rows, source=source, max_rank=max_rank)


def topk_by_layer_table(
    run: RunArtifacts,
    *,
    example_id: str | None = None,
    position: PositionSel = "last",
    source: TopkSource = "jspace",
    max_rank: int = 10,
    max_rows: int = 50_000,
) -> list[dict[str, Any]]:
    """Wide table: one row per layer, columns rank_1 .. rank_k with token + logit."""
    hits = topk_token_table(
        run,
        example_id=example_id,
        position=position,
        source=source,
        max_rank=max_rank,
        max_rows=max_rows,
    )
    by_layer: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    meta: dict[int, dict[str, Any]] = {}
    for hit in hits:
        layer = int(hit["layer"])
        by_layer[layer][int(hit["rank"])] = hit
        meta[layer] = hit
    table: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        row: dict[str, Any] = {
            "layer": layer,
            "position": meta[layer].get("absolute_position"),
            "state_token": format_token_text(meta[layer].get("state_token")),
        }
        for rank in range(1, int(max_rank) + 1):
            row[f"rank_{rank}"] = _hit_cell(by_layer[layer].get(rank))
        table.append(row)
    return table


def token_layer_presence_table(
    run: RunArtifacts,
    *,
    example_id: str | None = None,
    position: PositionSel = "last",
    source: TopkSource = "jspace",
    max_rank: int | None = None,
    max_rows: int = 50_000,
) -> list[dict[str, Any]]:
    """One row per predicted token: which layers it appears in as a top-k hit."""
    hits = topk_token_table(
        run,
        example_id=example_id,
        position=position,
        source=source,
        max_rank=max_rank,
        max_rows=max_rows,
    )
    grouped: dict[tuple[Any, Any], dict[str, Any]] = {}
    for hit in hits:
        key = (hit.get("token_id"), hit.get("token"))
        group = grouped.setdefault(
            key,
            {
                "token": hit["token_display"],
                "token_id": hit.get("token_id"),
                "layers": set(),
                "ranks": [],
                "logits": [],
            },
        )
        group["layers"].add(int(hit["layer"]))
        group["ranks"].append(int(hit["rank"]))
        if hit.get("logit") is not None:
            group["logits"].append(float(hit["logit"]))
    table: list[dict[str, Any]] = []
    for group in grouped.values():
        layers = sorted(group["layers"])
        logits = group["logits"]
        table.append(
            {
                "token": group["token"],
                "token_id": group["token_id"],
                "n_layers": len(layers),
                "layers": ", ".join(str(layer) for layer in layers),
                "best_rank": min(group["ranks"]) if group["ranks"] else None,
                "mean_logit": (sum(logits) / len(logits)) if logits else None,
                "n_hits": len(group["ranks"]),
            }
        )
    table.sort(
        key=lambda row: (
            row["best_rank"] if row["best_rank"] is not None else 10**9,
            -int(row["n_layers"]),
            -(row["mean_logit"] or 0.0),
        )
    )
    return table


def token_layer_rank_grid(
    run: RunArtifacts,
    *,
    example_id: str | None = None,
    position: PositionSel = "last",
    source: TopkSource = "jspace",
    max_rank: int | None = None,
    max_rows: int = 50_000,
) -> list[dict[str, Any]]:
    """Token × layer rank matrix. Missing cells mean the token was not in top-k."""
    hits = topk_token_table(
        run,
        example_id=example_id,
        position=position,
        source=source,
        max_rank=max_rank,
        max_rows=max_rows,
    )
    layers = sorted({int(hit["layer"]) for hit in hits})
    by_token: dict[tuple[Any, Any], dict[str, Any]] = {}
    for hit in hits:
        key = (hit.get("token_id"), hit.get("token"))
        entry = by_token.setdefault(
            key,
            {
                "token": hit["token_display"],
                "token_id": hit.get("token_id"),
                "ranks": {},
                "best_rank": hit["rank"],
            },
        )
        layer = int(hit["layer"])
        rank = int(hit["rank"])
        previous = entry["ranks"].get(layer)
        if previous is None or rank < previous:
            entry["ranks"][layer] = rank
        if rank < entry["best_rank"]:
            entry["best_rank"] = rank
    ordered = sorted(
        by_token.values(),
        key=lambda entry: (
            int(entry["best_rank"]),
            -len(entry["ranks"]),
            str(entry["token"]),
        ),
    )
    table: list[dict[str, Any]] = []
    for entry in ordered:
        row: dict[str, Any] = {
            "token": entry["token"],
            "token_id": entry["token_id"],
            "best_rank": entry["best_rank"],
            "n_layers": len(entry["ranks"]),
        }
        for layer in layers:
            row[f"L{layer}"] = entry["ranks"].get(layer)
        table.append(row)
    return table
