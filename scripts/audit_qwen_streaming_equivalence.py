"""Compare full-GPU and streamed dictionary scoring on frozen Qwen activations."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from jspace_research.phase1.adapters import JacobianLensAdapter
from jspace_research.phase1.artifacts import load_phase1_handoff
from jspace_research.phase1.config import load_config
from jspace_research.phase1.jspace import build_normalized_dictionary
from jspace_research.phase4.detectors import FrozenDetectors

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "artifacts/qwen/shared"
PHASE1 = SHARED / "phase1"
PHASE3 = SHARED / "phase3_implementation/results/k25"
INDICES = (0, 17, 500, 1000, 2500, 5000, 7499)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "artifacts/qwen/hf-cache"))
    config = load_config(
        ROOT / "configs/phase1_qwen3_5_9b_full.yaml",
        bipia_root=SHARED / "phase4_implementation (1)/benchmarks/BIPIA/benchmark",
    )
    handoff = load_phase1_handoff(PHASE1 / "selected_layer_k25.json", config)
    detectors = FrozenDetectors.load(PHASE3, handoff.metadata)
    lens = JacobianLensAdapter.load(config)
    unembedding = torch.load(
        PHASE1 / "cache/unembedding_weight_bf16.pt", map_location="cpu", weights_only=True
    )
    dictionary_cpu = build_normalized_dictionary(
        jacobian=lens.jacobian(handoff.selected_layer),
        unembedding=unembedding,
        layer=handoff.selected_layer,
        device=torch.device("cuda"),
        chunk_size=config.dictionary_chunk_size,
        storage_device=torch.device("cpu"),
    )
    dictionary_gpu = dictionary_cpu.to("cuda")
    streamed = detectors.with_dictionary(
        dictionary_cpu,
        scoring_device=torch.device("cuda"),
        screen_chunk_size=config.dictionary_chunk_size,
    )
    full = detectors.with_dictionary(dictionary_gpu)
    cache = np.memmap(
        PHASE1 / "cache/activations_bf16.dat",
        dtype=np.uint16,
        mode="r",
        shape=(7500, 31, 4096),
    )
    max_mean_delta = 0.0
    max_logistic_delta = 0.0
    for index in INDICES:
        bits = np.array(cache[index, handoff.selected_layer], copy=True)
        residual = torch.from_numpy(bits).view(torch.bfloat16).float()
        expected = full.score(residual, dictionary_gpu)
        observed = streamed.score(residual, dictionary_cpu)
        max_mean_delta = max(
            max_mean_delta, abs(expected["mean_score"] - observed["mean_score"])
        )
        max_logistic_delta = max(
            max_logistic_delta, abs(expected["logistic_score"] - observed["logistic_score"])
        )
        if expected["mean_prediction"] != observed["mean_prediction"] or expected[
            "logistic_prediction"
        ] != observed["logistic_prediction"]:
            raise AssertionError(f"Detector prediction changed for cached activation {index}")
    print(
        f"Compared {len(INDICES)} frozen activations: "
        f"max mean score delta={max_mean_delta:.9g}, "
        f"max logistic score delta={max_logistic_delta:.9g}; "
        "all predictions match"
    )


if __name__ == "__main__":
    main()
