# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Reproducible orchestration for the Qwen3 model-size study.

The notebook is intentionally thin.  This module owns model selection,
hardware checks, append-only artifacts, benchmark prompts/scoring, local
Jacobian sweeps, and the compact HTML views used in Colab.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import torch

from jlens.hf import HFLensModel, from_hf
from jlens.local import (
    DEFAULT_LOCAL_STRENGTHS,
    LocalJacobianResult,
    compute_local_jacobian,
    relative_layers,
    run_local_steering_sweep,
)

SEED = 17
EXPECTED_HUMANEVAL_CASES = 164
EXPECTED_GSM8K_CASES = 1319
HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reason privately, then put only "
    "executable Python code in the final answer."
)
GSM8K_INSTRUCTION = (
    "Solve this problem step by step and put the final numeric answer within "
    "\\boxed{}."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    parameters_b: float
    recommended_gpu: Literal["L4", "A100-40GB"]
    minimum_vram_gb: float


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen3-4b": ModelSpec(
        "qwen3-4b", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 4.0, "L4", 20.0
    ),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", 8.0, "L4", 20.0
    ),
    "qwen3-14b": ModelSpec(
        "qwen3-14b", "Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", 14.0, "L4", 20.0
    ),
    "qwen3-32b": ModelSpec(
        "qwen3-32b",
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
        32.0,
        "A100-40GB",
        38.0,
    ),
}


@dataclass(frozen=True)
class RunProfile:
    name: Literal["smoke", "full"]
    humaneval_cases: int | None
    gsm8k_cases: int | None
    steering_cases_per_dataset: int
    security_cases: int | None
    heatmap_cases_per_dataset: int


RUN_PROFILES: dict[str, RunProfile] = {
    "smoke": RunProfile("smoke", 2, 2, 2, 4, 1),
    "full": RunProfile("full", None, None, 64, 64, 4),
}


@dataclass(frozen=True)
class GenerationConfig:
    enable_thinking: bool = True
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = SEED
    humaneval_max_new_tokens: int = 4096
    gsm8k_max_new_tokens: int = 2048
    security_max_new_tokens: int = 512


@dataclass(frozen=True)
class StudyConfig:
    active_model: str
    profile: str = "smoke"
    output_root: str = "/content/drive/MyDrive/jspace-runs"
    seed: int = SEED
    model_revision: str | None = None
    humaneval_revision: str = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
    gsm8k_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    strengths: tuple[float, ...] = DEFAULT_LOCAL_STRENGTHS
    depth_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if self.active_model not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown model {self.active_model!r}; choose from "
                f"{sorted(MODEL_REGISTRY)}"
            )
        if self.profile not in RUN_PROFILES:
            raise ValueError(
                f"unknown profile {self.profile!r}; choose from {sorted(RUN_PROFILES)}"
            )
        if self.model_revision is None:
            object.__setattr__(self, "model_revision", self.model.revision)
        if tuple(self.strengths) != DEFAULT_LOCAL_STRENGTHS:
            raise ValueError(
                "the registered study uses strengths "
                f"{DEFAULT_LOCAL_STRENGTHS}; create a distinct study for another sweep"
            )

    @property
    def model(self) -> ModelSpec:
        return MODEL_REGISTRY[self.active_model]

    @property
    def run_profile(self) -> RunProfile:
        return RUN_PROFILES[self.profile]

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("output_root")
        return payload

    def experiment_payload(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload.pop("active_model")
        payload.pop("model_revision")
        return payload

    @property
    def config_hash(self) -> str:
        raw = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def experiment_hash(self) -> str:
        raw = json.dumps(
            self.experiment_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_hash / self.active_model


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return None


def hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "datasets": _version("datasets"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        snapshot.update(
            {
                "gpu": properties.name,
                "vram_gb": properties.total_memory / 1024**3,
                "cuda": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return snapshot


def validate_gpu(spec: ModelSpec, snapshot: Mapping[str, Any]) -> list[str]:
    """Validate hard requirements and return non-fatal recommendation notes."""
    if not snapshot.get("cuda_available"):
        raise RuntimeError("CUDA is required for the registered Qwen3 study")
    if not snapshot.get("bf16_supported"):
        raise RuntimeError("the selected GPU must support BF16 compute")
    vram = float(snapshot.get("vram_gb", 0))
    if vram < spec.minimum_vram_gb:
        raise RuntimeError(
            f"{spec.key} requires at least {spec.minimum_vram_gb:.0f} GB VRAM; "
            f"found {vram:.1f} GB"
        )
    name = str(snapshot.get("gpu", ""))
    notes: list[str] = []
    if spec.recommended_gpu == "A100-40GB" and "A100" not in name.upper():
        raise RuntimeError(f"{spec.key} is registered for an A100; found {name!r}")
    if spec.recommended_gpu == "L4" and "L4" not in name.upper():
        notes.append(f"registered target is L4; continuing on {name}")
    return notes


def preflight(config: StudyConfig, *, minimum_free_disk_gb: float = 8.0) -> dict[str, Any]:
    snapshot = hardware_snapshot()
    notes = validate_gpu(config.model, snapshot)
    output = Path(config.output_root)
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free_disk = stat.f_bavail * stat.f_frsize / 1024**3
    if free_disk < minimum_free_disk_gb:
        raise RuntimeError(
            f"at least {minimum_free_disk_gb:.1f} GB free disk is required; "
            f"found {free_disk:.1f} GB"
        )
    snapshot["free_disk_gb"] = free_disk
    snapshot["notes"] = notes
    snapshot["config_hash"] = config.config_hash
    return snapshot


def load_quantized_model(
    config: StudyConfig,
) -> tuple[Any, Any, HFLensModel]:
    """Load the active checkpoint in frozen NF4 with BF16 compute."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id, revision=config.model_revision
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    hf_model.eval()
    for parameter in hf_model.parameters():
        parameter.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return hf_model, tokenizer, from_hf(hf_model, tokenizer)


@dataclass(frozen=True)
class RunManifest:
    format_version: int
    created_at: str
    config: dict[str, Any]
    config_hash: str
    git_commit: str | None
    environment: dict[str, Any]
    model_commit: str | None = None
    tokenizer_commit: str | None = None
    dataset_revisions: dict[str, str] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: StudyConfig,
        *,
        environment: Mapping[str, Any] | None = None,
        model_commit: str | None = None,
        tokenizer_commit: str | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        protocol: Mapping[str, Any] | None = None,
    ) -> RunManifest:
        return cls(
            format_version=1,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config=asdict(config),
            config_hash=config.config_hash,
            git_commit=_git_commit(),
            environment=dict(environment or hardware_snapshot()),
            model_commit=model_commit,
            tokenizer_commit=tokenizer_commit,
            dataset_revisions=dict(dataset_revisions or {}),
            protocol={
                "model": asdict(config.model),
                "quantization": {
                    "weights": "NF4",
                    "double_quantization": True,
                    "compute_dtype": "bfloat16",
                    "parameters_frozen": True,
                },
                "benchmark_prompts": {
                    "humaneval": HUMANEVAL_INSTRUCTION,
                    "gsm8k": GSM8K_INSTRUCTION,
                },
                "local_jacobian": {
                    "strengths": list(config.strengths),
                    "depth_fractions": list(config.depth_fractions),
                    "position": "final teacher-forced prefix token",
                    "control": "seeded matched-norm random direction",
                },
                "security": {
                    "dataset": "Microsoft BIPIA email/table/code",
                    "probe_rank": 128,
                    "checkpoint_depth_fractions": [
                        0.125,
                        0.25,
                        0.375,
                        0.5,
                        0.625,
                        0.75,
                        0.875,
                        1.0,
                    ],
                    "mitigation_enabled": False,
                },
                **dict(protocol or {}),
            },
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config_hash:
                raise ValueError("existing manifest belongs to another configuration")
            return path
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


class JsonlStore:
    """Append-only, fsynced result store with config and ID validation."""

    def __init__(
        self,
        path: str | Path,
        *,
        config_hash: str,
        id_field: str = "case_id",
    ) -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.config_hash = config_hash
        self.id_field = id_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "config_hash": config_hash, "id_field": id_field}
        if self.meta_path.exists():
            existing = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if existing != metadata:
                raise ValueError("result store metadata does not match this run")
        else:
            self.meta_path.write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
        self._ids: set[str] = set()
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                try:
                    row = json.loads(line)
                    self._ids.add(str(row[id_field]))
                except Exception as exc:
                    raise ValueError(
                        f"malformed result row at {self.path}:{line_number}"
                    ) from exc

    def contains(self, identifier: str) -> bool:
        return str(identifier) in self._ids

    def append(self, row: Mapping[str, Any]) -> bool:
        if self.id_field not in row:
            raise ValueError(f"result row is missing {self.id_field!r}")
        identifier = str(row[self.id_field])
        if identifier in self._ids:
            return False
        encoded = json.dumps(dict(row), sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._ids.add(identifier)
        return True

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


@dataclass(frozen=True)
class BehaviorCase:
    dataset: Literal["humaneval", "gsm8k"]
    case_id: str
    prompt: str
    reference: str
    test: str | None = None
    entry_point: str | None = None


def load_behavior_cases(
    profile: RunProfile,
    *,
    humaneval_revision: str = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    gsm8k_revision: str = "740312add88f781978c0658806c59bc2815b9866",
) -> list[BehaviorCase]:
    from datasets import load_dataset

    humaneval = load_dataset(
        "openai/openai_humaneval", split="test", revision=humaneval_revision
    )
    gsm8k = load_dataset(
        "openai/gsm8k", "main", split="test", revision=gsm8k_revision
    )
    if profile.name == "full":
        if len(humaneval) != EXPECTED_HUMANEVAL_CASES:
            raise RuntimeError(
                f"HumanEval test split changed: expected {EXPECTED_HUMANEVAL_CASES}, "
                f"found {len(humaneval)}"
            )
        if len(gsm8k) != EXPECTED_GSM8K_CASES:
            raise RuntimeError(
                f"GSM8K test split changed: expected {EXPECTED_GSM8K_CASES}, "
                f"found {len(gsm8k)}"
            )
    human_limit = profile.humaneval_cases or len(humaneval)
    gsm_limit = profile.gsm8k_cases or len(gsm8k)
    rows = [
        BehaviorCase(
            "humaneval",
            str(row["task_id"]),
            str(row["prompt"]),
            str(row["canonical_solution"]),
            str(row["test"]),
            str(row["entry_point"]),
        )
        for row in humaneval.select(range(min(human_limit, len(humaneval))))
    ]
    rows.extend(
        BehaviorCase(
            "gsm8k",
            f"gsm8k-{index}",
            str(row["question"]),
            str(row["answer"]),
        )
        for index, row in enumerate(
            gsm8k.select(range(min(gsm_limit, len(gsm8k))))
        )
    )
    return rows


def case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def benchmark_messages(case: BehaviorCase) -> list[dict[str, str]]:
    if case.dataset == "humaneval":
        content = HUMANEVAL_INSTRUCTION + "\n\n" + case.prompt
    else:
        content = GSM8K_INSTRUCTION + "\n\n" + case.prompt
    return [{"role": "user", "content": content}]


def split_thinking(text: str) -> tuple[str, str]:
    """Return Qwen3 thinking and final content without conflating the two."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return "", re.sub(r"<\|[^|>]+\|>", "", text).strip()
    thinking = match.group(1).strip()
    final = (text[: match.start()] + text[match.end() :]).strip()
    return thinking, re.sub(r"<\|[^|>]+\|>", "", final).strip()


def generate_case(
    hf_model: Any,
    tokenizer: Any,
    case: BehaviorCase,
    generation: GenerationConfig,
) -> dict[str, Any]:
    messages = benchmark_messages(case)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=generation.enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(hf_model.device)
    seed = case_seed(generation.seed, case.case_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    max_new = (
        generation.humaneval_max_new_tokens
        if case.dataset == "humaneval"
        else generation.gsm8k_max_new_tokens
    )
    with torch.no_grad():
        output = hf_model.generate(
            **encoded,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0, encoded.input_ids.shape[1] :]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    thinking, final = split_thinking(raw)
    eos = tokenizer.eos_token_id
    truncated = bool(len(generated_ids) >= max_new and int(generated_ids[-1]) != eos)
    return {
        "raw_output": raw,
        "thinking": thinking,
        "final_answer": final,
        "generated_tokens": int(len(generated_ids)),
        "truncated": truncated,
        "seed": seed,
    }


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(final_answer: str) -> str:
    matches = _CODE_FENCE.findall(final_answer)
    return max(matches, key=len).strip() if matches else final_answer.strip()


def _limit_child() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except Exception:
        pass


def score_humaneval(
    case: BehaviorCase,
    final_answer: str,
    *,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    if not case.test or not case.entry_point:
        raise ValueError("HumanEval case requires test and entry_point")
    code = extract_python_code(final_answer)
    if re.search(rf"\bdef\s+{re.escape(case.entry_point)}\s*\(", code):
        candidate = code
    else:
        candidate = case.prompt + code
    script = (
        candidate
        + "\n\n"
        + case.test
        + f"\n\ncheck({case.entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="jlens-humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_limit_child if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error_type": "timeout", "stderr": ""}
    stderr = completed.stderr[-2000:]
    error_type = "" if completed.returncode == 0 else "execution_error"
    if "AssertionError" in stderr:
        error_type = "assertion_error"
    elif "SyntaxError" in stderr:
        error_type = "syntax_error"
    return {
        "passed": completed.returncode == 0,
        "error_type": error_type,
        "stderr": stderr,
    }


_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*)?")


def extract_numeric_answer(text: str) -> str | None:
    boxed = _BOXED.findall(text)
    candidates = boxed or _NUMBER.findall(text)
    if not candidates:
        return None
    return candidates[-1].strip().replace("$", "").replace(",", "")


def _numeric_value(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(value)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def score_gsm8k(reference: str, final_answer: str) -> dict[str, Any]:
    gold_source = reference.rsplit("####", 1)[-1]
    gold = extract_numeric_answer(gold_source)
    predicted = extract_numeric_answer(final_answer)
    gold_value = _numeric_value(gold)
    predicted_value = _numeric_value(predicted)
    parsed = predicted_value is not None
    return {
        "passed": bool(parsed and gold_value == predicted_value),
        "parsed": parsed,
        "predicted_answer": predicted,
        "gold_answer": gold,
    }


def run_behavioral_benchmark(
    hf_model: Any,
    tokenizer: Any,
    cases: Sequence[BehaviorCase],
    store: JsonlStore,
    generation: GenerationConfig,
    *,
    progress: Callable[[int, int, BehaviorCase], None] | None = None,
) -> list[dict[str, Any]]:
    total = len(cases)
    for index, case in enumerate(cases):
        identifier = f"{case.dataset}:{case.case_id}"
        if store.contains(identifier):
            continue
        if progress:
            progress(index, total, case)
        started = time.perf_counter()
        generated = generate_case(hf_model, tokenizer, case, generation)
        if case.dataset == "humaneval":
            score = score_humaneval(case, generated["final_answer"])
        else:
            score = score_gsm8k(case.reference, generated["final_answer"])
        store.append(
            {
                "case_id": identifier,
                "dataset": case.dataset,
                **generated,
                **score,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return store.rows()


def run_steering_case(
    model: HFLensModel,
    *,
    dataset: str,
    case_id: str,
    input_ids: Sequence[int],
    target_token_id: int,
    target_text: str,
    strengths: Sequence[float] = DEFAULT_LOCAL_STRENGTHS,
    depth_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    random_seed: int = SEED,
) -> tuple[LocalJacobianResult, list[dict[str, Any]]]:
    layers = relative_layers(model.n_layers, depth_fractions)
    tensor = torch.tensor(
        [list(input_ids)], dtype=torch.long, device=model.input_device
    )
    local = compute_local_jacobian(
        model,
        tensor,
        target_token_id=target_token_id,
        layers=list(range(model.n_layers)),
    )
    sweep = run_local_steering_sweep(
        model,
        local,
        layers=layers,
        strengths=strengths,
        random_seed=case_seed(random_seed, f"{dataset}:{case_id}"),
    )
    rows = [
        {
            "observation_id": (
                f"{dataset}:{case_id}:{row.layer}:{row.strength}:{row.control}"
            ),
            "dataset": dataset,
            "case_id": case_id,
            "target_text": target_text,
            **row.to_dict(),
        }
        for row in sweep
    ]
    return local, rows


def append_steering_rows(store: JsonlStore, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        store.append(row)


def _cell_color(value: float, maximum: float, *, diverging: bool = False) -> str:
    if diverging:
        scale = max(-1.0, min(1.0, value / maximum if maximum else 0.0))
        if scale >= 0:
            return f"rgb({255-int(80*scale)},{255-int(150*scale)},{255-int(210*scale)})"
        scale = -scale
        return f"rgb({255-int(210*scale)},{255-int(130*scale)},{255-int(70*scale)})"
    scale = max(0.0, min(1.0, value / maximum if maximum else 0.0))
    return f"rgb({245-int(170*scale)},{248-int(95*scale)},{255-int(35*scale)})"


def build_local_jspace_page(
    local: LocalJacobianResult,
    steering_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    title: str = "Targeted local J-space",
) -> str:
    """Render aligned sensitivity and causal rank-improvement heatmaps."""
    sensitivities = torch.log1p(local.sensitivity()).numpy()
    maximum = float(sensitivities.max()) if sensitivities.size else 1.0
    token_ids = local.input_ids[0].detach().cpu().tolist()
    token_labels = [
        tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in token_ids
    ]

    sensitivity_rows: list[str] = []
    for layer_index, layer in enumerate(local.layers):
        cells = "".join(
            f'<td style="background:{_cell_color(float(value), maximum)}" '
            f'title="layer {layer}, position {position}, token {html.escape(token_labels[position])}, '
            f'log1p sensitivity {float(value):.5f}"></td>'
            for position, value in enumerate(sensitivities[layer_index])
        )
        sensitivity_rows.append(f"<tr><th>L{layer}</th>{cells}</tr>")

    causal = [row for row in steering_rows if float(row["strength"]) > 0]
    layers = sorted({int(row["layer"]) for row in causal})
    strengths = sorted({float(row["strength"]) for row in causal})
    values: dict[tuple[int, float], float] = {}
    for layer in layers:
        for strength in strengths:
            local_row = next(
                row
                for row in causal
                if int(row["layer"]) == layer
                and float(row["strength"]) == strength
                and row["control"] == "local_jacobian"
            )
            random_row = next(
                row
                for row in causal
                if int(row["layer"]) == layer
                and float(row["strength"]) == strength
                and row["control"] == "random"
            )
            local_gain = int(local_row["clean_rank"]) - int(local_row["steered_rank"])
            random_gain = int(random_row["clean_rank"]) - int(random_row["steered_rank"])
            values[layer, strength] = float(local_gain - random_gain)
    causal_max = max((abs(value) for value in values.values()), default=1.0)
    causal_rows = []
    for layer in layers:
        cells = "".join(
            f'<td style="background:{_cell_color(values[layer, strength], causal_max, diverging=True)}" '
            f'title="layer {layer}, alpha {strength:g}, random-adjusted rank gain '
            f'{values[layer, strength]:g}">{values[layer, strength]:g}</td>'
            for strength in strengths
        )
        causal_rows.append(f"<tr><th>L{layer}</th>{cells}</tr>")

    token_header = "".join(
        f"<th title='token id {token}'>{html.escape(label) or '␠'}</th>"
        for token, label in zip(token_ids, token_labels, strict=True)
    )
    strength_header = "".join(f"<th>{value:g}</th>" for value in strengths)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font: 13px system-ui, sans-serif; margin: 20px; color: #172033 }}
h1 {{ font-size: 20px }} h2 {{ font-size: 15px; margin-top: 24px }}
.note {{ max-width: 900px; color: #4b5563 }} .scroll {{ overflow-x: auto }}
table {{ border-collapse: collapse }} th {{ font-size: 10px; font-weight: 500; padding: 2px }}
td {{ min-width: 18px; height: 16px; text-align: center; font-size: 9px; border: 1px solid #fff }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Prompt-local target-logit sensitivity, not a decoder of private thoughts.
Target token ID: {local.target_token_id}. Hover cells for exact values.</p>
<h2>Token × layer log(1 + ||∂ target logit / ∂ residual||₂)</h2>
<div class="scroll"><table><thead><tr><th></th>{token_header}</tr></thead>
<tbody>{''.join(sensitivity_rows)}</tbody></table></div>
<h2>Layer × strength random-adjusted target-rank gain</h2>
<table><thead><tr><th></th>{strength_header}</tr></thead>
<tbody>{''.join(causal_rows)}</tbody></table>
</body></html>"""


def write_local_jspace_page(
    path: str | Path,
    local: LocalJacobianResult,
    steering_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    title: str = "Targeted local J-space",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_local_jspace_page(local, steering_rows, tokenizer, title=title),
        encoding="utf-8",
    )
    return path


def write_csv_summary(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)
    return path
