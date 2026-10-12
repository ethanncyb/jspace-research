"""Durable supplemental InjecAgent responses without modifying pinned data."""
from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from typing import Any

from .common import content_hash


class MissingSimulatorResponse(RuntimeError):
    """A native request was persisted and requires a supplemental result."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o777)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_response(root: Path, output: Path, pinned: dict, tools: dict,
                     tool: str, parameters: str) -> tuple[str, str]:
    key = f"({tool},{parameters})"
    if key in pinned:
        return pinned[key], "pinned"
    values = runpy.run_path(str(root / "src/prompts/generation_prompts.py"))
    request = {
        "model": "gpt-4-0613",
        "messages": [
            {"role": "system", "content": values["SYSTEM_MESSAGE"]},
            {"role": "user", "content": values["DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE"].format(
                example=values["EXAMPLE"],
                attacker_tool=json.dumps(tools[tool], indent=2),
                tool_parameters=parameters,
            )},
        ],
    }
    request_hash = content_hash(request)
    cache_path = output / "injecagent_simulator_responses.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if key in cache:
        entry = cache[key]
        response = entry.get("response")
        if (entry.get("request_sha256") != request_hash or
                not isinstance(response, str) or not response.strip() or
                entry.get("response_sha256") != content_hash(response) or
                not entry.get("source")):
            raise RuntimeError(f"Invalid supplemental InjecAgent response for {key}")
        return response, "supplemental"
    pending = output / "injecagent_simulator_pending.json"
    atomic_json(pending, {"key": key, "request": request, "request_sha256": request_hash})
    raise MissingSimulatorResponse(
        f"Missing InjecAgent simulated response for {key}. Native simulator request saved "
        f"to {pending}; supply a validated supplemental response before resuming."
    )
