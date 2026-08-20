"""Optional matplotlib plots. Importing this module does not require a GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_accuracy_summary(rows: list[dict[str, Any]], path: str | Path | None = None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc
    labels = [str(row.get("condition") or row.get("run_id")) for row in rows]
    values = [float(row.get("accuracy") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("exact-answer accuracy")
    ax.set_title("GSM8K accuracy")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_token_layer_heatmap(
    grid_rows: list[dict[str, Any]],
    *,
    title: str = "Top-k token rank by layer",
    max_tokens: int = 30,
    path: str | Path | None = None,
):
    """Heatmap of rank (1 = strongest) for tokens vs layers. Blank = not in top-k."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc
    if not grid_rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_axis_off()
        ax.set_title("No top-k capture rows")
        return fig
    layer_cols = sorted(
        (key for key in grid_rows[0] if key.startswith("L") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )
    rows = grid_rows[: int(max_tokens)]
    tokens = [str(row.get("token") or "") for row in rows]
    data = np.array(
        [[row.get(col) for col in layer_cols] for row in rows],
        dtype=float,
    )
    masked = np.ma.masked_invalid(data)
    height = max(3.0, 0.35 * len(tokens) + 1.4)
    width = max(6.0, 0.7 * len(layer_cols) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad(color="#f3f3f3")
    vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=1, vmax=max(vmax, 1))
    ax.set_xticks(range(len(layer_cols)))
    ax.set_xticklabels([col[1:] for col in layer_cols], rotation=0)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("layer")
    ax.set_ylabel("top-k token")
    ax.set_title(title)
    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(layer_cols):
            value = row.get(col)
            if value is None:
                continue
            ax.text(
                col_i,
                row_i,
                str(int(value)),
                ha="center",
                va="center",
                color="white" if float(value) <= max(vmax / 2, 1) else "#1a1a1a",
                fontsize=8,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rank (1 = strongest)")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
