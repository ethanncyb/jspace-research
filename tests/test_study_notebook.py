# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path


def test_colab_notebook_is_clean_parseable_and_hotswappable():
    path = Path(__file__).parents[1] / "qwen_size_study.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for key in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"):
        assert key in source
    for directory in ("benchmarks", "steering", "jspace", "security"):
        assert f"'{directory}'" in source
    assert "ACTIVE_MODEL = 'qwen3-4b'" in source
    assert "RUN_PROFILE = 'smoke'" in source
    assert "NF4" in source and "private thoughts" in source
    assert "git', 'clone', '--branch', REPO_BRANCH" in source
    assert "jlens import OK" in source
    assert "Commit and push the branch first" in source
    assert "%pip install -q -e $BIPIA_ROOT" not in source
    assert "BIPIA_COMMIT = 'a004b69ec0dd446e0afd461d98cb5e96e120a5d0'" in source
    assert "'checkout', '--detach', BIPIA_COMMIT" in source
    assert "visual_complete = heatmap_path.exists()" in source
    assert "globals().pop(name, None)" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs", []) == []
        code = "".join(cell["source"])
        # IPython magics are not Python syntax, but the rest of those cells is.
        code = "\n".join(
            "" if line.lstrip().startswith("%") else line
            for line in code.splitlines()
        )
        try:
            ast.parse(code)
        except SyntaxError as exc:  # pragma: no cover - failure diagnostic
            raise AssertionError(f"syntax error in notebook cell {index}: {exc}") from exc
