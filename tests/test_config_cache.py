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
from jspace_research.phase2.config import FIXED_ALPHAS
from jspace_research.phase2.config import load_config as load_phase2_config
from jspace_research.phase3.config import load_config as load_phase3_config


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
phase2:
  alphas: [0.0, 0.5, 1.0]
  max_new_tokens: 512
  do_sample: false
  generation_batch_size: 1
  judge_model: openai/gpt-4.1-mini
phase3:
  penalty: l2
  regularization_c: 1.0
  solver: liblinear
  fit_intercept: true
  class_weight: null
  random_state: 42
  max_iter: 1000
  tol: 0.0001
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
    replace(config, sparsity_k=24).validate()
    with pytest.raises(ValueError, match="screen_candidates=512"):
        replace(config, screen_candidates=511).validate()
    with pytest.raises(ValueError, match="does not permit quantization"):
        replace(config, model=replace(config.model, quantization="int8")).validate()
    with pytest.raises(ValueError, match="all five tasks"):
        replace(config, smoke_layer_count=None).validate()


def test_phase2_config_uses_fixed_sweep_and_shared_phase1_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    config = load_phase2_config(
        path,
        phase1_selected_path=tmp_path / "phase1" / "selected_layer.json",
        output_dir=tmp_path / "phase2",
    )
    assert config.alphas == FIXED_ALPHAS
    assert config.max_new_tokens == 512
    assert config.phase1.identity_hash() == load_config(path).identity_hash()

    text = path.read_text().replace("0.5, 1.0", "0.6, 1.0")
    path.write_text(text)
    changed = load_phase2_config(
        path,
        phase1_selected_path=tmp_path / "phase1" / "selected_layer.json",
        output_dir=tmp_path / "phase2-other",
    )
    assert changed.alphas == (0.0, 0.6, 1.0)
    assert changed.identity_hash() != config.identity_hash()


def test_phase3_config_pins_logistic_settings_and_phase1_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    config = load_phase3_config(
        path,
        phase1_selected_path=tmp_path / "phase1" / "selected_layer.json",
        output_dir=tmp_path / "phase3",
    )
    assert config.solver == "liblinear"
    assert config.regularization_c == 1.0
    assert config.phase1.identity_hash() == load_config(path).identity_hash()

    path.write_text(path.read_text().replace("regularization_c: 1.0", "regularization_c: 0.5"))
    with pytest.raises(ValueError, match="fixed logistic settings"):
        load_phase3_config(
            path,
            phase1_selected_path=tmp_path / "phase1" / "selected_layer.json",
            output_dir=tmp_path / "phase3-other",
        )


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("sparsity_k_values", "[]"),
        ("sparsity_k_values", "[20, 20]"),
        ("sparsity_k_values", "[true]"),
        ("sparsity_k_values", "[20.5]"),
        ("sparsity_k_values", "[513]"),
        ("output_token_windows", "[0]"),
        ("output_token_windows", "[512]"),
        ("output_token_windows", "[1, 1]"),
    ],
)
def test_sweep_lists_reject_invalid_values(tmp_path, field, value):
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    text = path.read_text().replace(
        "sparsity_k: 25", "sparsity_k_values: [20, 25, 30]\noutput_token_windows: [1, 5]"
    )
    original = (
        "sparsity_k_values: [20, 25, 30]"
        if field == "sparsity_k_values"
        else "output_token_windows: [1, 5]"
    )
    path.write_text(text.replace(original, f"{field}: {value}"))
    with pytest.raises(ValueError):
        load_phase2_config(
            path, phase1_selected_path=tmp_path / "selected_layers.json", output_dir=tmp_path / "p2"
        )


def test_k_decomposition_and_capture_identities_are_separate(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    path.write_text(
        path.read_text().replace(
            "sparsity_k: 25", "sparsity_k_values: [20, 25, 30]\noutput_token_windows: [1, 5]"
        )
    )
    config = load_config(path)
    assert config.k_values == (20, 25, 30)
    k20 = replace(config, sparsity_k=20)
    assert k20.identity_hash() != config.identity_hash()
    assert k20.capture_identity_hash() == config.capture_identity_hash()
    assert (
        replace(config, max_input_tokens=1024).capture_identity_hash()
        != config.capture_identity_hash()
    )
    assert replace(config, output_dir=tmp_path / "moved").identity_hash() == config.identity_hash()
    path.write_text(path.read_text() + "\nsparsity_k: 25\n")
    with pytest.raises(ValueError, match="not both"):
        load_config(path)


@pytest.mark.parametrize(
    "alphas", [(), (0.5, 1.0), (0.0, 0.0), (0.0, float("nan")), (0.0, float("inf")), (0.0, -1.0)]
)
def test_invalid_alpha_sweeps(tmp_path, alphas):
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    config = load_phase2_config(
        path, phase1_selected_path=tmp_path / "selection.json", output_dir=tmp_path / "p2"
    )
    with pytest.raises(ValueError, match="alphas"):
        replace(config, alphas=alphas).validate()
    replace(config, alphas=(0.0, 0.1, 0.8, 1.5)).validate()


def test_descriptive_sweep_names_preserve_legacy_run_identities(tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    legacy = path.read_text().replace("sparsity_k: 25", "K: [20, 25, 30]\nW: [1, 5]")
    kwargs = dict(
        phase1_selected_path=tmp_path / "selected_layers.json", output_dir=tmp_path / "p2"
    )
    path.write_text(legacy)
    old = load_phase2_config(path, **kwargs)
    path.write_text(
        legacy.replace("K:", "sparsity_k_values:").replace("W:", "output_token_windows:")
    )
    renamed = load_phase2_config(path, **kwargs)
    assert renamed.phase1.k_values == (20, 25, 30)
    assert renamed.windows == (1, 5)
    assert renamed.phase1.identity_hash() == old.phase1.identity_hash()
    assert renamed.phase1.capture_identity_hash() == old.phase1.capture_identity_hash()
    assert renamed.identity_hash() == old.identity_hash()


@pytest.mark.parametrize("duplicate", ["K: [20, 25]", "sparsity_k: 25", "W: [1, 5]"])
def test_duplicate_sweep_aliases_are_rejected(tmp_path, duplicate):
    path = tmp_path / "config.yaml"
    write_config(path, tmp_path / "phase1")
    path.write_text(
        path.read_text().replace(
            "sparsity_k: 25", "sparsity_k_values: [20, 25]\noutput_token_windows: [1, 5]"
        )
        + f"\n{duplicate}\n"
    )
    with pytest.raises(ValueError, match="not both"):
        load_phase2_config(
            path, phase1_selected_path=tmp_path / "selected_layers.json", output_dir=tmp_path / "p2"
        )


@pytest.mark.parametrize("path", sorted(Path("configs").glob("phase1*.yaml")))
def test_checked_in_configs_use_descriptive_sweep_keys(path, tmp_path):
    import yaml

    raw = yaml.safe_load(path.read_text())
    assert raw["sparsity_k_values"] == [20, 25, 30, 35, 40]
    assert raw["output_token_windows"] == [1, 5, 10, 15]
    assert "K" not in raw and "W" not in raw
    config = load_phase2_config(
        path, phase1_selected_path=tmp_path / "selected_layers.json", output_dir=tmp_path / "p2"
    )
    assert len(config.phase1.k_values) * len(config.windows) == 20
