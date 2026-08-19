from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
SRC = ROOT / "src"

for path in (SRC, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_tiny_decoder():
    path = PARENT / "tests" / "tiny.py"
    spec = importlib.util.spec_from_file_location("jlens_tiny_decoder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TinyDecoder


@pytest.fixture
def tiny_model():
    TinyDecoder = load_tiny_decoder()
    return TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0)
