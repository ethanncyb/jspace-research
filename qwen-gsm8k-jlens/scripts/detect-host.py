#!/usr/bin/env python3
"""Detect host GPU profile without importing project torch."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_PY = ROOT / "src" / "gsm8k_jspace" / "platform" / "host.py"


def _load_host():
    spec = importlib.util.spec_from_file_location("gsm8k_jspace_host", HOST_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HOST_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    profile = _load_host().detect_host_profile(explicit)
    print(
        json.dumps(
            {
                "name": profile.name,
                "backend": profile.backend,
                "venv_dir": profile.venv_dir,
                "torch_backend": profile.torch_backend,
                "uv_extra_args": list(profile.uv_extra_args),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
