#!/usr/bin/env python3
"""Detect host GPU profile without importing project torch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gsm8k_jspace.platform.host import detect_host_profile  # noqa: E402


def main() -> int:
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    profile = detect_host_profile(explicit)
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
