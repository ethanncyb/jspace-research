from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jspace_research.phase1.cache import (
    ensure_cache_metadata,
    load_done,
    open_memmap,
    open_uint16_memmap,
    save_done,
)
from jspace_research.phase1.config import (
    EXPECTED_BIPIA_REVISION,
    EXPECTED_JLENS_REVISION,
    load_config,
)


def write_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
model:
  id: google/gemma-4-12B-it
  revision: 5926caa4ec0cac5cbfadaf4077420520de1d5205
  precision: bfloat16
lens:
  repository: solarkyle/jspace-lenses
  revision: 1d95a2fc8a5c5a26c75a8c01c145173353e5fb65
  filename: gemma-4-12b-it/lens.pt
  sha256: 214ba70486c648d97cccb3c88d05cfb17adf9467c93b5d1f268fc4902e360048
dependencies:
  jacobian_lens_revision: {EXPECTED_JLENS_REVISION}
  bipia_revision: {EXPECTED_BIPIA_REVISION}
data:
  bipia_root: /tmp/BIPIA/benchmark
  webqa_train_path: null
  summarization_train_path: null
output_dir: {output_dir}
seed: 42
tasks: [email]
train_pairs_per_task: 12
validation_pairs_per_task: 6
max_input_tokens: 4096
token_match_tolerance: 1
sparsity_k: 25
screen_candidates: 512
decomposition_batch_size: 8
dictionary_chunk_size: 4096
smoke_layer_count: 6
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_config_and_path_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "original")
    config = load_config(
        path,
        bipia_root=tmp_path / "BIPIA" / "benchmark",
        output_dir=tmp_path / "override",
    )
    assert config.output_dir == (tmp_path / "override").resolve()
    assert config.data.bipia_root == (tmp_path / "BIPIA" / "benchmark").resolve()
    assert config.sparsity_k == 25
    assert len(config.identity_hash()) == 64

    relocated = load_config(
        path,
        bipia_root=tmp_path / "elsewhere" / "BIPIA" / "benchmark",
        output_dir=tmp_path / "relocated-output",
    )
    assert relocated.identity_hash() == config.identity_hash()


def test_config_rejects_wrong_jlens_revision(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "output")
    text = path.read_text(encoding="utf-8").replace(EXPECTED_JLENS_REVISION, "0" * 40)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Jacobian-lens"):
        load_config(path)


def test_config_enforces_fixed_scientific_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "output")
    config = load_config(path)
    with pytest.raises(ValueError, match="seed 42"):
        replace(config, seed=41).validate()
    with pytest.raises(ValueError, match="sparsity_k=25"):
        replace(config, sparsity_k=24).validate()
    with pytest.raises(ValueError, match="does not permit quantization"):
        replace(config, model=replace(config.model, quantization="int8")).validate()
    with pytest.raises(ValueError, match="all five tasks"):
        replace(config, smoke_layer_count=None).validate()


def test_cache_metadata_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    ensure_cache_metadata(path, {"shape": [2, 3], "identity": "a"})
    ensure_cache_metadata(path, {"shape": [2, 3], "identity": "a"})
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        ensure_cache_metadata(path, {"shape": [2, 3], "identity": "b"})
    assert json.loads(path.read_text()) == {"identity": "a", "shape": [2, 3]}


def test_completion_bitmap_and_memmap_resume(tmp_path: Path) -> None:
    done_path = tmp_path / "done.npy"
    done = load_done(done_path, 3)
    done[1] = True
    save_done(done_path, done)
    np.testing.assert_array_equal(load_done(done_path, 3), [False, True, False])

    data_path = tmp_path / "cache.dat"
    first = open_uint16_memmap(data_path, (2, 3))
    first[0] = [1, 2, 3]
    first.flush()
    second = open_uint16_memmap(data_path, (2, 3))
    np.testing.assert_array_equal(second[0], [1, 2, 3])
    with pytest.raises(RuntimeError, match="shape mismatch"):
        open_uint16_memmap(data_path, (3, 3))


def test_typed_sparse_memmaps_have_safe_initial_values(tmp_path: Path) -> None:
    support_path = tmp_path / "support.dat"
    coefficient_path = tmp_path / "coefficient.dat"
    support = open_memmap(support_path, (2, 3), dtype=np.int32, fill_value=-1)
    coefficients = open_memmap(coefficient_path, (2, 3), dtype=np.float32, fill_value=0.0)
    np.testing.assert_array_equal(support, np.full((2, 3), -1, dtype=np.int32))
    np.testing.assert_array_equal(coefficients, np.zeros((2, 3), dtype=np.float32))

    support[0, 0] = 7
    coefficients[0, 0] = 1.5
    support.flush()
    coefficients.flush()
    reopened_support = open_memmap(support_path, (2, 3), dtype=np.int32, fill_value=-1)
    reopened_coefficients = open_memmap(coefficient_path, (2, 3), dtype=np.float32, fill_value=0.0)
    assert reopened_support[0, 0] == 7
    assert reopened_coefficients[0, 0] == pytest.approx(1.5)
