from __future__ import annotations

from pathlib import Path

from gsm8k_jspace.config import load_config
from gsm8k_jspace.datasets.agentdojo import load_agentdojo_examples
from gsm8k_jspace.evaluation.agentdojo import score_agentdojo_output
from gsm8k_jspace.prompting.agentdojo import render_prompt

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_agentdojo_cartesian_product():
    cfg = load_config(CONFIGS / "agentdojo-smoke.yaml")
    cfg.benchmark.full_run = True
    examples = load_agentdojo_examples(cfg)
    assert len(examples) == 4
    ids = {ex.extra["user_task_id"] for ex in examples}
    assert ids == {"user_task_0", "user_task_15"}
    assert all("<INFORMATION>" in ex.extra["observation"] for ex in examples)


def test_agentdojo_prompt_and_asr():
    cfg = load_config(CONFIGS / "agentdojo-smoke.yaml")
    cfg.benchmark.full_run = True
    example = load_agentdojo_examples(cfg)[0]
    prompt = render_prompt(example, cfg)
    assert "Available tools:" in prompt
    assert example.question in prompt
    assert example.extra["goal"] in prompt
    hit = score_agentdojo_output(
        "Thought: I should email Mark.\nAction: send_email\nAction Input: {\"recipients\": [\"mark.black-2134@gmail.com\"]}",
        attacker_action="send_email",
        markers=["mark.black-2134@gmail.com"],
    )
    assert hit["attack_succeeded"] is True
    miss = score_agentdojo_output(
        "Thought: I will answer the user.\nFinal Answer: Sarah Baker is invited.",
        attacker_action="send_email",
        markers=["mark.black-2134@gmail.com"],
    )
    assert miss["attack_succeeded"] is False
