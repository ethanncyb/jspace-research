#!/usr/bin/env python3
"""Verify that the selected uv environment's torch backend matches the host."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gsm8k_jspace.platform.capabilities import inspect_backend  # noqa: E402
from gsm8k_jspace.platform.host import detect_host_profile  # noqa: E402


def main() -> int:
    expected = detect_host_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    info = inspect_backend("auto")
    payload = {
        "expected_backend": expected.backend,
        "resolved_backend": info.name,
        "device_name": info.device_name,
        "dtype": info.dtype_name,
    }
    print(json.dumps(payload, indent=2))
    apple = {"mlx", "mps"}
    if expected.backend != "cpu" and info.name != expected.backend:
        if not (expected.backend in apple and info.name in apple):
            print("ERROR: requested/resolved backend mismatch", file=sys.stderr)
            return 2
        if expected.backend == "mlx" and info.name != "mlx":
            print("ERROR: Apple host expects MLX; mlx is not importable in this env", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
