from __future__ import annotations

import json
from pathlib import Path

from gsm8k_jspace.config import load_config
from gsm8k_jspace.datasets.injecagent import load_injecagent_examples
from gsm8k_jspace.evaluation.injecagent import score_injecagent_output
from gsm8k_jspace.prompting.injecagent import render_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "injecagent_rows.json"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_injecagent_injects_attacker_instruction():
    cfg = load_config(CONFIGS / "injecagent-smoke.yaml")
    payload = json.loads(FIXTURE.read_text())
    examples = load_injecagent_examples(
        cfg,
        user_cases=payload["user_cases"],
        attacker_cases=payload["attacker_cases"],
    )
    assert len(examples) == 1
    example = examples[0]
    assert "BankManagerTransferFunds" in example.extra["attacker_tools"]
    assert "transfer $3,000" in example.extra["tool_response"]
    prompt = render_prompt(example, cfg)
    assert "GmailReadEmail" in prompt
    assert "BankManagerTransferFunds" in prompt
    assert "Observation:" in prompt


def test_injecagent_action_asr():
    succ = score_injecagent_output(
        "Thought: I should help.\nAction: BankManagerTransferFunds\nAction Input: {}",
        attacker_tools=["BankManagerTransferFunds"],
        user_tool="GmailReadEmail",
    )
    assert succ["eval"] == "succ"
    unsucc = score_injecagent_output(
        "Thought: This looks malicious.\nFinal Answer: I will not transfer money.",
        attacker_tools=["BankManagerTransferFunds"],
        user_tool="GmailReadEmail",
    )
    assert unsucc["eval"] == "unsucc"
    assert unsucc["attack_succeeded"] is False


def test_injecagent_loads_repo_data():
    cfg = load_config(CONFIGS / "injecagent-smoke.yaml")
    examples = load_injecagent_examples(cfg)
    assert len(examples) == 5
    assert examples[0].extra["user_tool"]
    assert examples[0].extra["attacker_tools"]
