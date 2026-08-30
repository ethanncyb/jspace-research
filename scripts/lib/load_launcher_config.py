#!/usr/bin/env python3
"""Load launcher YAML files and emit shell export statements."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - pre-setup fallback
    yaml = None


def _export(name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    print(f"export {name}={shlex.quote(text)}")


def _export_if_unset(name: str, value: Any) -> None:
    if name in os.environ and os.environ[name].strip():
        return
    _export(name, value)


def _expand_path(repo_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "~"}:
        return None
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return str(path)


def _read_secret_file(path: str | None) -> str | None:
    if path is None:
        return None
    secret_path = Path(path)
    if not secret_path.is_file():
        return None
    return secret_path.read_text(encoding="utf-8").strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Run ./scripts/setup.sh first or pip install pyyaml."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping at root of {path}")
    return data


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RuntimeError(f"Expected mapping for {key}")
    return section


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--env-config", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_data: dict[str, Any] = {}
    env_data: dict[str, Any] = {}

    if args.run_config is not None:
        if not args.run_config.is_file():
            raise RuntimeError(f"run config not found: {args.run_config}")
        run_data = _load_yaml(args.run_config.resolve())

    if args.env_config is not None:
        if not args.env_config.is_file():
            raise RuntimeError(f"env config not found: {args.env_config}")
        env_data = _load_yaml(args.env_config.resolve())

    experiment = _section(run_data, "experiment")
    pipeline = _section(run_data, "pipeline")
    runtime = _section(env_data, "runtime")
    paths = _section(env_data, "paths")
    credentials = _section(env_data, "credentials")

    model_key = experiment.get("model_key", "qwen35_9b")
    run_mode = experiment.get("run_mode", "smoke")
    physical_gpu_index = experiment.get("physical_gpu_index", 0)

    config_path = _expand_path(repo_root, experiment.get("config"))
    if config_path is None:
        config_path = str((repo_root / "configs" / f"phase1_{model_key}_{run_mode}.yaml").resolve())

    run_root = _expand_path(repo_root, experiment.get("run_root"))
    if run_root is None:
        run_name = experiment.get("run_name")
        if run_name:
            run_root = str((repo_root / "artifacts" / str(run_name)).resolve())
        else:
            run_root = str(
                (
                    repo_root
                    / "artifacts"
                    / f"jspace-{model_key}-{run_mode}-gpu{physical_gpu_index}"
                ).resolve()
            )

    benchmarks_root = _expand_path(
        repo_root,
        paths.get("benchmarks_root", "../jspace-benchmarks"),
    )
    bipia_checkout = _expand_path(repo_root, paths.get("bipia_checkout", "BIPIA"))
    agentdojo_checkout = _expand_path(repo_root, paths.get("agentdojo_checkout"))
    injecagent_checkout = _expand_path(repo_root, paths.get("injecagent_checkout"))
    if agentdojo_checkout is None and benchmarks_root is not None:
        agentdojo_checkout = str(Path(benchmarks_root) / "agentdojo")
    if injecagent_checkout is None and benchmarks_root is not None:
        injecagent_checkout = str(Path(benchmarks_root) / "InjecAgent")

    bipia_root = str(Path(bipia_checkout) / "benchmark") if bipia_checkout else None
    venv_dir = _expand_path(repo_root, runtime.get("venv_dir", ".venv"))
    python_path = _expand_path(repo_root, runtime.get("python"))

    hf_env = str(credentials.get("hf_token_env", "HF_TOKEN"))
    openrouter_env = str(credentials.get("openrouter_api_key_env", "OPENROUTER_API_KEY"))
    hf_token_file = _expand_path(repo_root, credentials.get("hf_token_file"))
    openrouter_file = _expand_path(repo_root, credentials.get("openrouter_api_key_file"))

    _export_if_unset("JSPACE_MODEL_KEY", model_key)
    _export_if_unset("JSPACE_RUN_MODE", run_mode)
    _export_if_unset("JSPACE_PHYSICAL_GPU_INDEX", physical_gpu_index)
    _export_if_unset("JSPACE_CONFIG_PATH", config_path)
    _export_if_unset("JSPACE_RUN_ROOT", run_root)
    _export_if_unset("JSPACE_BENCHMARKS_ROOT", benchmarks_root)
    _export_if_unset("JSPACE_BIPIA_CHECKOUT", bipia_checkout)
    _export_if_unset("JSPACE_AGENTDOJO_CHECKOUT", agentdojo_checkout)
    _export_if_unset("JSPACE_INJECAGENT_CHECKOUT", injecagent_checkout)
    _export_if_unset("JSPACE_BIPIA_ROOT", bipia_root)
    _export_if_unset("JSPACE_USE_PROJECT_VENV", int(_as_bool(runtime.get("use_project_venv"), True)))
    _export_if_unset("JSPACE_VENV_DIR", venv_dir)
    _export_if_unset("JSPACE_PYTHON", python_path)
    _export_if_unset("WEBQA_TRAIN_PATH", _expand_path(repo_root, paths.get("webqa_train_path")))
    _export_if_unset(
        "SUMMARIZATION_TRAIN_PATH",
        _expand_path(repo_root, paths.get("summarization_train_path")),
    )
    _export_if_unset("SKIP_SETUP", int(_as_bool(pipeline.get("skip_setup"), False)))
    _export_if_unset("SKIP_GPU_CHECK", int(_as_bool(pipeline.get("skip_gpu_check"), False)))
    if pipeline.get("stage") is not None:
        _export_if_unset("STAGE", pipeline.get("stage"))

    hf_token = _read_secret_file(hf_token_file)
    openrouter_key = _read_secret_file(openrouter_file)
    if hf_token:
        _export_if_unset(hf_env, hf_token)
        if hf_env != "HF_TOKEN":
            _export_if_unset("HF_TOKEN", hf_token)
        if hf_env != "HUGGING_FACE_HUB_TOKEN":
            _export_if_unset("HUGGING_FACE_HUB_TOKEN", hf_token)
    if openrouter_key:
        _export_if_unset(openrouter_env, openrouter_key)
        if openrouter_env != "OPENROUTER_API_KEY":
            _export_if_unset("OPENROUTER_API_KEY", openrouter_key)

    phase1_dir = str(Path(run_root) / "phase1")
    _export_if_unset("JSPACE_PHASE1_DIR", phase1_dir)
    _export_if_unset("JSPACE_PHASE2_DIR", str(Path(run_root) / "phase2"))
    _export_if_unset("JSPACE_PHASE3_DIR", str(Path(run_root) / "phase3"))
    _export_if_unset("JSPACE_PHASE4_DIR", str(Path(run_root) / "phase4"))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - surfaced to shell
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
