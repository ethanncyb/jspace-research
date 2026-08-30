"""Discover saved experiment runs and pick one from a notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from gsm8k_jspace.artifacts.writer import read_json


PickerMode = Literal["single", "pair", "multi"]


@dataclass(frozen=True)
class RunCatalogEntry:
    run_id: str
    path: Path
    started_at: str | None
    finished_at: str | None
    status: str
    condition: str | None
    model: str | None
    backend: str | None
    host_profile: str | None
    accuracy: float | None
    n_evaluated: int | None

    @property
    def label(self) -> str:
        bits = [self.run_id, f"[{self.status}]"]
        if self.condition:
            bits.append(self.condition)
        if self.backend:
            bits.append(str(self.backend))
        if self.accuracy is not None:
            n = self.n_evaluated if self.n_evaluated is not None else "?"
            bits.append(f"metric={self.accuracy:.3f} n={n}")
        if self.model:
            bits.append(str(self.model).split("/")[-1])
        return "  |  ".join(bits)


def default_outputs_root() -> Path:
    here = Path(__file__).resolve()
    package_root = here.parents[3]
    candidates = [
        Path.cwd() / "outputs" / "gsm8k",
        Path.cwd().parent / "outputs" / "gsm8k",
        package_root / "outputs" / "gsm8k",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def default_outputs_roots() -> list[Path]:
    here = Path(__file__).resolve()
    package_root = here.parents[3]
    bases = [
        Path.cwd() / "outputs",
        Path.cwd().parent / "outputs",
        package_root / "outputs",
    ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            resolved = child.resolve()
            if child.is_dir() and resolved not in seen:
                seen.add(resolved)
                roots.append(child)
    return roots


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sort_key(entry: RunCatalogEntry) -> tuple[str, str]:
    stamp = entry.started_at or ""
    return (stamp, entry.run_id)


def list_runs(root: str | Path | None = None) -> list[RunCatalogEntry]:
    """Return saved runs newest-first. A run is a directory with manifest.json."""
    if root is not None:
        roots = [Path(root)]
    else:
        roots = default_outputs_roots() or [default_outputs_root()]
    entries: list[RunCatalogEntry] = []
    seen: set[Path] = set()
    for root_path in roots:
        if not root_path.is_dir():
            continue
        for child in root_path.iterdir():
            resolved = child.resolve()
            if resolved in seen:
                continue
            manifest_path = child / "manifest.json"
            if not child.is_dir() or not manifest_path.exists():
                continue
            seen.add(resolved)
            entries.append(_entry_from_run_dir(child))
    entries.sort(key=_sort_key, reverse=True)
    return entries


def _entry_from_run_dir(child: Path) -> RunCatalogEntry:
    manifest = _optional_json(child / "manifest.json")
    environment = _optional_json(child / "environment.json")
    summary = _optional_json(child / "evaluation" / "summary.json")
    backend = environment.get("backend")
    backend_name = None
    if isinstance(backend, dict):
        backend_name = backend.get("resolved") or backend.get("device_name")
    model = None
    model_block = manifest.get("model")
    if isinstance(model_block, dict):
        model = model_block.get("name")
    model = summary.get("model") or model
    accuracy = summary.get("asr", summary.get("accuracy"))
    n_evaluated = summary.get("n_evaluated")
    return RunCatalogEntry(
        run_id=str(manifest.get("run_id") or child.name),
        path=child.resolve(),
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        status=str(manifest.get("status") or "unknown"),
        condition=manifest.get("condition") or summary.get("condition"),
        model=model,
        backend=backend_name,
        host_profile=environment.get("host_profile"),
        accuracy=float(accuracy) if isinstance(accuracy, (int, float)) else None,
        n_evaluated=int(n_evaluated) if isinstance(n_evaluated, int) else None,
    )


def catalog_rows(entries: Iterable[RunCatalogEntry] | None = None) -> list[dict[str, Any]]:
    if entries is None:
        entries = list_runs()
    rows = []
    for entry in entries:
        rows.append(
            {
                "run_id": entry.run_id,
                "started_at": entry.started_at,
                "status": entry.status,
                "condition": entry.condition,
                "backend": entry.backend,
                "host": entry.host_profile,
                "model": entry.model,
                "accuracy": entry.accuracy,
                "n_evaluated": entry.n_evaluated,
                "path": str(entry.path),
            }
        )
    return rows


def resolve_run(
    query: str | Path,
    *,
    entries: list[RunCatalogEntry] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Resolve a path, exact run_id, unique suffix, or 'latest'/'latest_complete'."""
    if entries is None:
        entries = list_runs(root)
    text = str(query).strip()
    if not text:
        raise FileNotFoundError("empty run id")
    if text in {"latest", "latest_complete"}:
        chosen = _default_entry(entries, complete_only=text == "latest_complete")
        if chosen is None:
            raise FileNotFoundError(f"no saved runs matching {text!r}")
        return chosen.path
    as_path = Path(text).expanduser()
    if as_path.exists() and (as_path / "manifest.json").exists():
        return as_path.resolve()
    exact = [entry for entry in entries if entry.run_id == text]
    if len(exact) == 1:
        return exact[0].path
    prefix = [
        entry
        for entry in entries
        if entry.run_id.startswith(text.rstrip("_") + "_") or entry.run_id.endswith("_" + text)
    ]
    if len(prefix) == 1:
        return prefix[0].path
    raise FileNotFoundError(
        f"could not resolve run {text!r}; known runs: "
        + ", ".join(entry.run_id for entry in entries[:8])
    )


def _default_entry(
    entries: list[RunCatalogEntry],
    *,
    complete_only: bool = True,
    condition: str | None = None,
) -> RunCatalogEntry | None:
    filtered = entries
    if condition:
        filtered = [entry for entry in filtered if entry.condition == condition]
    if complete_only:
        complete = [entry for entry in filtered if entry.status == "complete"]
        if complete:
            return complete[0]
    return filtered[0] if filtered else None


class RunPicker:
    """Notebook control: dropdown if ipywidgets is available, else a printed catalog."""

    def __init__(
        self,
        entries: list[RunCatalogEntry],
        *,
        mode: PickerMode = "single",
        widget=None,
        baseline_widget=None,
        candidate_widget=None,
        fallback: Path | None = None,
        fallback_paths: list[Path] | None = None,
    ) -> None:
        self.entries = entries
        self.mode = mode
        self._widget = widget
        self._baseline_widget = baseline_widget
        self._candidate_widget = candidate_widget
        self._fallback = fallback
        self._fallback_paths = fallback_paths or ([] if fallback is None else [fallback])

    @property
    def selected(self) -> Path:
        if self.mode == "pair":
            return self.selected_pair[0]
        paths = self.selected_paths
        if not paths:
            raise FileNotFoundError("no run selected")
        return paths[0]

    @property
    def selected_paths(self) -> list[Path]:
        if self._widget is not None:
            value = self._widget.value
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [Path(item) for item in value]
            return [Path(value)]
        if self.mode == "pair":
            baseline, candidate = self.selected_pair
            return [baseline, candidate]
        return list(self._fallback_paths)

    @property
    def selected_pair(self) -> tuple[Path, Path]:
        if self._baseline_widget is not None and self._candidate_widget is not None:
            return Path(self._baseline_widget.value), Path(self._candidate_widget.value)
        if len(self._fallback_paths) >= 2:
            return self._fallback_paths[0], self._fallback_paths[1]
        if self._fallback is not None:
            return self._fallback, self._fallback
        raise FileNotFoundError("no baseline/candidate runs selected")

    def _ipython_display_(self) -> None:
        try:
            from IPython.display import display
        except Exception:
            print(self._text_catalog())
            return
        widget = self._ipython_widget()
        if widget is not None:
            display(widget)
            return
        display(self._html_catalog())

    def _ipython_widget(self):
        if self._widget is not None:
            return self._widget
        if self._baseline_widget is not None:
            try:
                from ipywidgets import VBox
            except Exception:
                return None
            return VBox([self._baseline_widget, self._candidate_widget])
        return None

    def _text_catalog(self) -> str:
        if not self.entries:
            return "No saved runs in outputs/."
        lines = ["Saved experiments (newest first):"]
        for index, entry in enumerate(self.entries):
            marker = "*" if self._fallback is not None and entry.path == self._fallback else " "
            lines.append(f"{marker}{index:3d}  {entry.label}")
        if self._fallback is not None:
            lines.append(f"Using: {self._fallback}")
        return "\n".join(lines)

    def _html_catalog(self):
        from IPython.display import HTML

        rows = catalog_rows(self.entries)
        if not rows:
            return HTML("<p>No saved runs in <code>outputs/</code>.</p>")
        try:
            import pandas as pd

            frame = pd.DataFrame(rows).drop(columns=["path"])
            return frame
        except Exception:
            body = "".join(
                f"<li><code>{row['run_id']}</code> {row['status']} {row['condition']}</li>"
                for row in rows
            )
            return HTML(f"<ol>{body}</ol>")


def run_picker(
    root: str | Path | None = None,
    *,
    mode: PickerMode = "single",
    default: str = "latest_complete",
) -> RunPicker:
    """Build a notebook picker over saved experiment folders."""
    entries = list_runs(root)
    if not entries:
        return RunPicker(entries, mode=mode)

    options = [(entry.label, str(entry.path)) for entry in entries]
    latest = _default_entry(entries, complete_only=default == "latest_complete")
    if latest is None:
        latest = entries[0]

    if mode == "pair":
        baseline = _default_entry(entries, condition="baseline") or latest
        candidate = (
            _default_entry(entries, condition="intervention")
            or _default_entry(entries, condition="no_op")
            or latest
        )
        baseline_widget = _dropdown("Baseline", options, str(baseline.path))
        candidate_widget = _dropdown("Candidate", options, str(candidate.path))
        return RunPicker(
            entries,
            mode=mode,
            baseline_widget=baseline_widget,
            candidate_widget=candidate_widget,
            fallback=baseline.path,
            fallback_paths=[baseline.path, candidate.path],
        )

    if mode == "multi":
        widget = _select_multiple(options, [str(latest.path)])
        return RunPicker(
            entries,
            mode=mode,
            widget=widget,
            fallback=latest.path,
            fallback_paths=[latest.path],
        )

    widget = _dropdown("Experiment", options, str(latest.path))
    return RunPicker(
        entries,
        mode=mode,
        widget=widget,
        fallback=latest.path,
        fallback_paths=[latest.path],
    )


def _dropdown(description: str, options: list[tuple[str, str]], value: str):
    try:
        import ipywidgets as widgets
    except Exception:
        return None
    return widgets.Dropdown(
        options=options,
        value=value,
        description=description,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "90px"},
    )


def _select_multiple(options: list[tuple[str, str]], value: list[str]):
    try:
        import ipywidgets as widgets
    except Exception:
        return None
    return widgets.SelectMultiple(
        options=options,
        value=tuple(value),
        description="Runs",
        layout=widgets.Layout(width="100%", height="160px"),
        style={"description_width": "90px"},
    )
