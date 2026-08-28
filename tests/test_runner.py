from __future__ import annotations

from pathlib import Path

from jspace_research.run import main


def write_config(path: Path) -> None:
    path.write_text(
        """
model:
  id: google/gemma-4-12B-it
  revision: 5926caa4ec0cac5cbfadaf4077420520de1d5205
  precision: bfloat16
  quantization: none
lens:
  repository: solarkyle/jspace-lenses
  revision: 1d95a2fc8a5c5a26c75a8c01c145173353e5fb65
  filename: gemma-4-12b-it/lens.pt
  sha256: 214ba70486c648d97cccb3c88d05cfb17adf9467c93b5d1f268fc4902e360048
dependencies:
  jacobian_lens_revision: 581d398613e5602a5af361e1c34d3a92ea82ba8e
  bipia_revision: a004b69ec0dd446e0afd461d98cb5e96e120a5d0
data:
  bipia_root: /tmp/BIPIA/benchmark
  webqa_train_path: null
  summarization_train_path: null
output_dir: /tmp/unused
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
  alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
  max_new_tokens: 512
  do_sample: false
  generation_batch_size: 1
  judge_model: gpt-4.1-mini-2025-04-14
""".strip()
        + "\n"
    )


def test_dispatcher_wires_phase_subdirectories(
    tmp_path: Path, monkeypatch
) -> None:
    from jspace_research.phase1 import pipeline as phase1_pipeline
    from jspace_research.phase2 import pipeline as phase2_pipeline

    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    calls = []

    def fake_phase1(config, stage: str) -> None:
        calls.append((1, config.output_dir, stage))
        config.output_dir.mkdir(parents=True)
        (config.output_dir / "selected_layer.json").write_text("{}\n")

    def fake_phase2(config, stage: str) -> None:
        calls.append((2, config.output_dir, stage, config.phase1_selected_path))

    monkeypatch.setattr(phase1_pipeline, "run", fake_phase1)
    monkeypatch.setattr(phase2_pipeline, "run", fake_phase2)
    run_dir = tmp_path / "run"
    main(["--config", str(config_path), "--run-dir", str(run_dir)])

    assert calls == [
        (1, run_dir / "phase1", "all"),
        (2, run_dir / "phase2", "all", run_dir / "phase1" / "selected_layer.json"),
    ]
