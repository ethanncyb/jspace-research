"""CPU-only coverage for promptguard's hooks, learning, and evolution logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from promptguard.config import (
    GuardConfig,
    InterventionConfig,
    ProbeConfig,
    ResearchConfig,
)
from promptguard.drift_probe import DriftProbe, pool_activations
from promptguard.evolution_loop import (
    EvolutionRunner,
    active_family_evaluate,
    cross_stage_evaluate,
    evaluation_attack_strategies,
    seeded_prompt_order,
)
from promptguard.evolving_attacker import (
    AttackRecord,
    DetectionOutcome,
    EvolvingAttacker,
    MutationProposal,
    continuous_attack_reward,
    starter_strategies,
)
from promptguard.evolving_guard import EvolvingGuard, SelfLabeledExample
from promptguard.intervention import InterventionController
from promptguard.model_hooks import ActivationHooks, HookedModel, HookMode


class _Block(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class _TupleBlock(nn.Module):
    def forward(self, hidden):
        return hidden + 1, "cache"


def test_read_and_write_hooks_preserve_block_output_shape():
    layers = nn.ModuleList([_Block(), _TupleBlock()])
    hidden = torch.zeros(1, 2, 3)
    with ActivationHooks(layers, [0], writable_layers=[0]) as hooks:
        observed = layers[0](hidden)
    assert torch.equal(hooks.activations[0], observed)

    with ActivationHooks(
        layers,
        [1],
        mode=HookMode.WRITE,
        writer=lambda _layer, value: value * 2,
        writable_layers=[1],
    ) as hooks:
        changed, cache = layers[1](hidden)
    assert cache == "cache"
    assert torch.equal(changed, torch.full_like(hidden, 2))
    assert 1 in hooks.modified_activations


def test_write_hooks_reject_read_only_gdn_layer():
    with pytest.raises(ValueError, match="full-attention"):
        ActivationHooks(
            nn.ModuleList([_Block()]),
            [0],
            mode=HookMode.WRITE,
            writer=lambda _layer, value: value,
            writable_layers=[],
        )


def test_circuit_breaker_projects_probe_direction():
    probe = DriftProbe([0], 2)
    with torch.no_grad():
        probe.classifiers["0"].weight[:] = torch.tensor([[1.0, 0.0]])
    controller = InterventionController(
        probe, InterventionConfig(mode="circuit_breaker", beta=0.5)
    )
    hidden = torch.tensor([[[2.0, 3.0]]])
    changed = controller.writer()(0, hidden)
    assert torch.allclose(changed, torch.tensor([[[1.0, 3.0]]]))


def test_hard_stop_never_starts_generation():
    class FakeModel:
        def capture(self, text):
            value = 1.0 if "\n" in text else 0.0
            return {0: torch.tensor([[[value, 0.0]]])}, torch.ones(1, 1)

        def generate(self, *_args, **_kwargs):
            raise AssertionError("hard stop must not generate")

    probe = DriftProbe([0], 2)
    with torch.no_grad():
        probe.classifiers["0"].weight[:] = torch.tensor([[2.0, 0.0]])
        probe.classifiers["0"].bias.zero_()
    controller = InterventionController(
        probe,
        InterventionConfig(mode="hard_stop", threshold=0.5, refusal="fixed refusal"),
    )
    result = controller.run(FakeModel(), baseline_text="base", prompt="segment")
    assert result.triggered
    assert result.text == "fixed refusal"
    assert result.mode == "hard_stop"


def test_last_token_pooling_handles_left_and_right_padding():
    hidden = torch.arange(2 * 4 * 2).reshape(2, 4, 2).float()
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    pooled = pool_activations({3: hidden}, mask)[3]
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 3])


def test_drift_probe_trains_saves_and_loads(tmp_path):
    generator = torch.Generator().manual_seed(4)
    examples = []
    labels = []
    for label in [0, 1] * 40:
        sign = -1 if label == 0 else 1
        examples.append(
            {
                1: torch.randn(6, generator=generator) * 0.2
                + torch.tensor([sign * 2.0, 0, 0, 0, 0, 0]),
                3: torch.randn(6, generator=generator) * 0.2
                + torch.tensor([0, sign * 2.0, 0, 0, 0, 0]),
            }
        )
        labels.append(label)
    probe = DriftProbe([1, 3], 6)
    probe.fit(examples, labels, epochs=20, learning_rate=0.03, batch_size=16)
    metrics = probe.evaluate(examples, labels)
    assert metrics.auc > 0.99
    checkpoint = tmp_path / "probe.pt"
    probe.save(checkpoint)
    restored = DriftProbe.load(checkpoint)
    assert torch.allclose(probe.score(examples[0])[0], restored.score(examples[0])[0])


class _Config:
    hidden_size = 4
    layer_types = ["linear_attention", "full_attention"]

    def get_text_config(self):
        return self


class _MockHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(), _Block()])
        self.embed = nn.Embedding(10, 4)
        self.config = _Config()

    def get_input_embeddings(self):
        return self.embed


def test_qwen_layer_types_override_fallback_indices():
    adapter = HookedModel(_MockHF(), SimpleNamespace(), layer_indices=[0], hidden_dim=4)
    assert adapter.full_attention_layers == [1]


class _Reflector:
    def propose(self, failed, strategy):
        return MutationProposal(
            name="admitted_mutation",
            family="mutated",
            template="ALLOW {payload}",
        )


def test_attacker_admits_only_validated_mutation_and_normalizes_transition():
    attacker = EvolvingAttacker(
        reflector=_Reflector(), sandbox_tau=0.5, attempts_per_prompt=1, seed=2
    )

    def evaluator(prompt):
        flagged = not prompt.startswith("ALLOW ")
        return DetectionOutcome(0.9 if flagged else 0.1, flagged)

    result = attacker.run_round(
        round_index=1,
        payloads=["unsafe seed"],
        evaluator=evaluator,
        validation_payloads=["one", "two"],
        heldout_evaluator=evaluator,
    )
    assert result.admitted_strategies == ["admitted_mutation"]
    assert len(attacker.pool) == len(starter_strategies()) + 1
    assert np.allclose(attacker.transitions.sum(axis=1), 1.0)


def test_attacker_admits_meaningful_score_reduction_without_threshold_crossing():
    class PartialProgressReflector:
        def propose(self, failed, strategy):
            return MutationProposal(
                name="partial_progress",
                family="mutated",
                template="IMPROVE {payload}",
            )

    attacker = EvolvingAttacker(
        reflector=PartialProgressReflector(),
        sandbox_tau=0.25,
        sandbox_min_score_reduction=0.10,
        attempts_per_prompt=1,
        seed=2,
    )

    def evaluator(prompt):
        score = 0.65 if prompt.startswith("IMPROVE ") else 0.9
        return DetectionOutcome(score, flagged=True)

    result = attacker.run_round(
        round_index=1,
        payloads=["unsafe seed"],
        evaluator=evaluator,
        validation_payloads=["one", "two"],
        heldout_evaluator=evaluator,
    )

    assert result.admitted_strategies == ["partial_progress"]
    assert all(record.flagged for record in result.records)


def test_continuous_reward_credits_score_improvement_and_gates_semantics():
    improved = continuous_attack_reward(0.9, 0.6)
    regressed = continuous_attack_reward(0.6, 0.9)
    evaded = continuous_attack_reward(0.6, 0.4)

    assert improved > regressed
    assert evaded > continuous_attack_reward(0.6, 0.51)
    assert continuous_attack_reward(0.9, 0.1, semantic_valid=False) == -1.0


def test_frozen_guard_run_skips_every_guard_update(tmp_path):
    config = ResearchConfig()
    config.evolution.rounds = 1
    config.evolution.output_dir = str(tmp_path / "evolution")
    config.evolution.checkpoint_dir = str(tmp_path / "new-checkpoints")
    config.guard.review_queue_path = str(tmp_path / "review.csv")
    config.attacker.attempts_per_prompt = 1
    config.attacker.max_reflections_per_round = 1
    config.semantic.enabled = False
    config.data.benign_prompts = ["benign"]
    config.data.malicious_prompts = ["unsafe"]
    config.data.heldout_malicious_prompts = ["final unsafe"]

    probe = DriftProbe([0], 2)
    with torch.no_grad():
        probe.classifiers["0"].weight.zero_()
        probe.classifiers["0"].bias.fill_(-10.0)
    frozen_checkpoint = tmp_path / "guard_010.pt"
    probe.save(frozen_checkpoint, round_index=10)
    checkpoint_before = frozen_checkpoint.read_bytes()

    model = SimpleNamespace(full_attention_layers=[0], hidden_dim=2)
    runner = EvolutionRunner(
        config,
        model,
        probe=probe,
        frozen_guard_checkpoint=frozen_checkpoint,
    )
    runner.extractor = lambda _baseline, _text: {0: torch.zeros(1, 2)}

    def unexpected_guard_update(**_kwargs):
        raise AssertionError("frozen evaluation must not update the guard")

    runner.guard.run_round = unexpected_guard_update
    summaries = runner.run()

    assert summaries[0]["guard_update_enabled"] is False
    assert frozen_checkpoint.read_bytes() == checkpoint_before
    assert not (tmp_path / "new-checkpoints").exists()


def test_guard_confidence_gate_and_continual_checkpoint(tmp_path):
    probe = DriftProbe([1], 3)
    pool = EvolvingAttacker(seed=1).pool
    guard = EvolvingGuard(
        probe,
        strategies=pool,
        probe_config=ProbeConfig(epochs=2, batch_size=2),
        guard_config=GuardConfig(
            confidence_gate=0.9,
            review_queue_path=str(tmp_path / "review.csv"),
        ),
    )
    attack = AttackRecord(
        1,
        0,
        0,
        "unsafe raw",
        pool[0].transform("unsafe raw"),
        pool[0].name,
        pool[0].family,
        0.1,
        False,
        True,
    )

    def extractor(_baseline, text):
        value = 1.0 if "unsafe" in text else -1.0
        return {1: torch.tensor([value, 0.0, 0.0])}

    result = guard.run_round(
        round_index=1,
        successful_attacks=[attack],
        benign_prompts=["ordinary request"],
        baseline="baseline",
        extractor=extractor,
        checkpoint_path=tmp_path / "guard.pt",
        self_labeled=[
            SelfLabeledExample("uncertain", 1, 0.6),
            SelfLabeledExample("unsafe certain", 1, 0.99),
        ],
    )
    assert result.accepted_count == 1
    assert result.review_count == 1
    assert result.checkpoint_path.exists()
    assert (tmp_path / "review.csv").exists()


def test_cross_stage_matrix_includes_heldout(tmp_path):
    guard0 = DriftProbe([0], 2)
    guard1 = DriftProbe([0], 2)
    with torch.no_grad():
        guard0.classifiers["0"].weight[:] = torch.tensor([[1.0, 0.0]])
        guard0.classifiers["0"].bias.zero_()
        guard1.load_state_dict(guard0.state_dict())
    first = tmp_path / "g0.pt"
    second = tmp_path / "g1.pt"
    guard0.save(first)
    guard1.save(second)
    rows = cross_stage_evaluate(
        {1: [{0: torch.tensor([[-1.0, 0.0]])}]},
        {0: first, 1: second},
        threshold=0.5,
        heldout=[{0: torch.tensor([[1.0, 0.0]])}],
        csv_path=tmp_path / "matrix.csv",
    )
    assert len(rows) == 4
    assert (tmp_path / "matrix.csv").exists()


def test_family_reporting_keeps_hard_heldout_categories_distinct(tmp_path):
    families = {strategy.family for strategy in evaluation_attack_strategies()}
    assert {"translated", "acrostic_encoding", "roleplay"} <= families

    records = [
        AttackRecord(1, 0, 0, "a", "wrapped a", "one", "roleplay", 0.1, False, True),
        AttackRecord(1, 1, 0, "b", "wrapped b", "two", "encoding", 0.9, True, False),
    ]
    rows = active_family_evaluate(
        records,
        round_index=1,
        payload_count=2,
        csv_path=tmp_path / "active.csv",
    )
    by_family = {row.attack_family: row.asr for row in rows}
    assert by_family == {"encoding": 0.0, "roleplay": 1.0, "all_active": 0.5}


def test_seeded_prompt_order_is_reproducible_and_nonmutating():
    prompts = [str(index) for index in range(12)]
    first = seeded_prompt_order(prompts, seed=7)
    assert first == seeded_prompt_order(prompts, seed=7)
    assert first != seeded_prompt_order(prompts, seed=19)
    assert prompts == [str(index) for index in range(12)]
