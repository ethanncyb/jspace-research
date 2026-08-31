#!/usr/bin/env python3
"""Load launcher YAML files for shell scripts and notebooks."""

from __future__ import annotations

import argparse
import json
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


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_config_data(
    unified: dict[str, Any],
    run_data: dict[str, Any],
    env_data: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(unified)
    if run_data:
        merged = _deep_merge(merged, run_data)
    if env_data:
        merged = _deep_merge(merged, env_data)
    return merged


def _derive_run_name(config_path: Path, physical_gpu_index: int) -> str:
    stem = config_path.stem
    if stem.startswith("phase1_"):
        stem = stem[len("phase1_") :]
    if "_" in stem:
        model_part, mode_part = stem.rsplit("_", 1)
        name_part = f"{model_part}-{mode_part}"
    else:
        name_part = stem
    return f"jspace-{name_part}-gpu{physical_gpu_index}"


def _resolve_config_path(
    repo_root: Path,
    experiment: dict[str, Any],
    model_key: str,
    run_mode: str,
) -> Path:
    config_path = _expand_path(repo_root, experiment.get("config"))
    if config_path is None:
        config_path = str(
            (repo_root / "configs" / f"phase1_{model_key}_{run_mode}.yaml").resolve()
        )
    return Path(config_path)


def _resolve_run_root(
    repo_root: Path,
    experiment: dict[str, Any],
    output: dict[str, Any],
    config_path: Path,
    physical_gpu_index: int,
    model_key: str,
    run_mode: str,
) -> str:
    run_root = _expand_path(repo_root, output.get("run_root"))
    if run_root is None:
        run_root = _expand_path(repo_root, experiment.get("run_root"))
    if run_root is None:
        run_name = output.get("run_name") or experiment.get("run_name")
        if run_name:
            run_root = str((repo_root / "artifacts" / str(run_name)).resolve())
        else:
            run_root = str(
                (
                    repo_root
                    / "artifacts"
                    / _derive_run_name(config_path, physical_gpu_index)
                ).resolve()
            )
    return run_root


def _load_merged_launcher_data(
    repo_root: Path,
    *,
    config: Path | None = None,
    local_config: Path | None = None,
    run_config: Path | None = None,
    env_config: Path | None = None,
) -> dict[str, Any]:
    unified_data: dict[str, Any] = {}
    run_data: dict[str, Any] = {}
    env_data: dict[str, Any] = {}

    if config is not None:
        if not config.is_file():
            raise RuntimeError(f"config not found: {config}")
        unified_data = _load_yaml(config.resolve())
        if local_config is None:
            local_candidate = config.with_name(f"{config.stem}.local.yaml")
            if local_candidate.is_file():
                local_config = local_candidate
        if local_config is not None:
            if not local_config.is_file():
                raise RuntimeError(f"local config not found: {local_config}")
            unified_data = _deep_merge(unified_data, _load_yaml(local_config.resolve()))

    if run_config is not None:
        if not run_config.is_file():
            raise RuntimeError(f"run config not found: {run_config}")
        run_data = _load_yaml(run_config.resolve())

    if env_config is not None:
        if not env_config.is_file():
            raise RuntimeError(f"env config not found: {env_config}")
        env_data = _load_yaml(env_config.resolve())

    return _merge_config_data(unified_data, run_data, env_data)


def resolve_launcher_config(
    repo_root: str | Path,
    *,
    config: str | Path | None = None,
    local_config: str | Path | None = None,
    run_config: str | Path | None = None,
    env_config: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path_arg = Path(config).resolve() if config is not None else None
    local_config_arg = Path(local_config).resolve() if local_config is not None else None
    run_config_arg = Path(run_config).resolve() if run_config is not None else None
    env_config_arg = Path(env_config).resolve() if env_config is not None else None

    data = _load_merged_launcher_data(
        repo_root,
        config=config_path_arg,
        local_config=local_config_arg,
        run_config=run_config_arg,
        env_config=env_config_arg,
    )

    experiment = _section(data, "experiment")
    hardware = _section(data, "hardware")
    output = _section(data, "output")
    pipeline = _section(data, "pipeline")
    runtime = _section(data, "runtime")
    paths = _section(data, "paths")
    credentials = _section(data, "credentials")

    model_key = experiment.get("model_key", "qwen35_4b")
    run_mode = experiment.get("run_mode", "smoke")
    physical_gpu_index = int(
        hardware.get("physical_gpu_index", experiment.get("physical_gpu_index", 0))
    )

    config_path = _resolve_config_path(repo_root, experiment, model_key, run_mode)
    run_root = _resolve_run_root(
        repo_root,
        experiment,
        output,
        config_path,
        physical_gpu_index,
        model_key,
        run_mode,
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

    hf_token = _read_secret_file(hf_token_file)
    openrouter_key = _read_secret_file(openrouter_file)

    phase1_dir = str(Path(run_root) / "phase1")
    return {
        "repo_root": str(repo_root),
        "launcher_config": str(config_path_arg) if config_path_arg is not None else None,
        "model_key": model_key,
        "run_mode": run_mode,
        "physical_gpu_index": physical_gpu_index,
        "config_path": str(config_path),
        "run_root": run_root,
        "benchmarks_root": benchmarks_root,
        "bipia_checkout": bipia_checkout,
        "agentdojo_checkout": agentdojo_checkout,
        "injecagent_checkout": injecagent_checkout,
        "bipia_root": bipia_root,
        "phase1_dir": phase1_dir,
        "phase2_dir": str(Path(run_root) / "phase2"),
        "phase3_dir": str(Path(run_root) / "phase3"),
        "phase4_dir": str(Path(run_root) / "phase4"),
        "webqa_train_path": _expand_path(repo_root, paths.get("webqa_train_path")),
        "summarization_train_path": _expand_path(
            repo_root, paths.get("summarization_train_path")
        ),
        "newsqa_dir": _expand_path(repo_root, paths.get("newsqa_dir")),
        "skip_setup": _as_bool(pipeline.get("skip_setup"), False),
        "skip_gpu_check": _as_bool(pipeline.get("skip_gpu_check"), False),
        "use_project_venv": _as_bool(runtime.get("use_project_venv"), True),
        "venv_dir": venv_dir,
        "python_path": python_path,
        "hf_token_env": hf_env,
        "openrouter_api_key_env": openrouter_env,
        "hf_token_file": hf_token_file,
        "openrouter_api_key_file": openrouter_file,
        "hf_token": hf_token,
        "openrouter_api_key": openrouter_key,
    }


def emit_shell_exports(resolved: dict[str, Any]) -> None:
    repo_root = resolved["repo_root"]
    hf_env = resolved["hf_token_env"]
    openrouter_env = resolved["openrouter_api_key_env"]

    _export_if_unset("JSPACE_REPO_ROOT", repo_root)
    _export_if_unset("JSPACE_MODEL_KEY", resolved["model_key"])
    _export_if_unset("JSPACE_RUN_MODE", resolved["run_mode"])
    _export_if_unset("JSPACE_PHYSICAL_GPU_INDEX", resolved["physical_gpu_index"])
    _export_if_unset("JSPACE_CONFIG_PATH", resolved["config_path"])
    _export_if_unset("JSPACE_RUN_ROOT", resolved["run_root"])
    _export_if_unset("JSPACE_BENCHMARKS_ROOT", resolved["benchmarks_root"])
    _export_if_unset("JSPACE_BIPIA_CHECKOUT", resolved["bipia_checkout"])
    _export_if_unset("JSPACE_AGENTDOJO_CHECKOUT", resolved["agentdojo_checkout"])
    _export_if_unset("JSPACE_INJECAGENT_CHECKOUT", resolved["injecagent_checkout"])
    _export_if_unset("JSPACE_BIPIA_ROOT", resolved["bipia_root"])
    _export_if_unset(
        "JSPACE_USE_PROJECT_VENV", int(resolved["use_project_venv"])
    )
    _export_if_unset("JSPACE_VENV_DIR", resolved["venv_dir"])
    _export_if_unset("JSPACE_PYTHON", resolved["python_path"])
    _export_if_unset("WEBQA_TRAIN_PATH", resolved["webqa_train_path"])
    _export_if_unset("SUMMARIZATION_TRAIN_PATH", resolved["summarization_train_path"])
    _export_if_unset("JSPACE_NEWSQA_DIR", resolved["newsqa_dir"])
    _export_if_unset("SKIP_SETUP", int(resolved["skip_setup"]))
    _export_if_unset("SKIP_GPU_CHECK", int(resolved["skip_gpu_check"]))

    hf_token = resolved["hf_token"]
    openrouter_key = resolved["openrouter_api_key"]
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

    _export_if_unset("JSPACE_PHASE1_DIR", resolved["phase1_dir"])
    _export_if_unset("JSPACE_PHASE2_DIR", resolved["phase2_dir"])
    _export_if_unset("JSPACE_PHASE3_DIR", resolved["phase3_dir"])
    _export_if_unset("JSPACE_PHASE4_DIR", resolved["phase4_dir"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--local-config", type=Path, default=None)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--env-config", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("shell", "json"),
        default="shell",
        help="Output shell export statements (default) or JSON for notebooks.",
    )
    args = parser.parse_args()

    resolved = resolve_launcher_config(
        args.repo_root.resolve(),
        config=args.config,
        local_config=args.local_config,
        run_config=args.run_config,
        env_config=args.env_config,
    )

    if args.format == "json":
        print(json.dumps(resolved, indent=2))
    else:
        emit_shell_exports(resolved)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - surfaced to shell
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
