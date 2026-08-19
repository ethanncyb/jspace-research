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
