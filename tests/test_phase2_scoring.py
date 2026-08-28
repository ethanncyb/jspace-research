from __future__ import annotations

import pandas as pd
import pytest

from jspace_research.phase2.scoring import (
    OpenAIAttackJudge,
    is_refusal,
    qualitative_examples,
    rouge_reference_overlap,
    summarize_results,
)


def test_rouge_reference_overlap_and_refusal_detection() -> None:
    scores = rouge_reference_overlap("Answer: expected.", "Answer: expected.")
    assert scores == {
        "rouge1_recall": 1.0,
        "rouge2_recall": 1.0,
        "rougeL_recall": 1.0,
        "rougeLsum_recall": 1.0,
    }
    assert is_refusal("I'm sorry, but I cannot help with that.")
    assert is_refusal("I cannot provide that response.")
    assert not is_refusal("Answer: expected.")


def test_attack_judge_uses_structured_yes_no_unknown_output() -> None:
    class Responses:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            return type("Response", (), {"output_text": '{"label":"YES"}'})()

    responses = Responses()
    client = type("Client", (), {"responses": responses})()
    judge = OpenAIAttackJudge("fixed-model", client=client)
    assert judge.judge("Injected request", "Compliant response") == "YES"
    assert responses.kwargs is not None
    schema = responses.kwargs["text"]["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == ["YES", "NO", "UNKNOWN"]


def make_results() -> pd.DataFrame:
    rows = []
    for task, baseline_utility in (("email", 0.8), ("code", 0.4)):
        for condition in ("attack", "control"):
            for alpha in (0.0, 1.0):
                utility = baseline_utility if alpha == 0.0 else baseline_utility / 2
                rows.append(
                    {
                        "example_id": f"{task}:{condition}",
                        "pair_id": f"{task}:validation:00000",
                        "task": task,
                        "condition": condition,
                        "alpha": alpha,
                        "generation": f"{task}-{condition}-{alpha}",
                        "attack_success": condition == "attack" and alpha == 0.0,
                        "task_score": utility,
                        "rouge1_recall": utility,
                        "rouge2_recall": utility,
                        "rougeL_recall": utility,
                        "rougeLsum_recall": utility,
                        "refusal": alpha == 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_reports_asr_delta_and_per_task_utility_without_raw_average() -> None:
    summary = summarize_results(make_results())
    overall_asr = summary[(summary.metric == "asr") & (summary.scope == "overall")]
    assert overall_asr[overall_asr.alpha == 0.0].iloc[0].value == pytest.approx(1.0)
    assert overall_asr[overall_asr.alpha == 1.0].iloc[0].delta == pytest.approx(-1.0)

    utility = summary[summary.metric == "rougeL_recall"]
    assert set(utility.scope) == {"task"}
    assert set(utility.task) == {"email", "code"}
    retained = utility[utility.alpha == 1.0]
    assert retained.retention.tolist() == pytest.approx([0.5, 0.5])


def test_qualitative_table_uses_first_pair_and_alpha_endpoints() -> None:
    results = make_results()
    extra = results.copy()
    extra["pair_id"] = extra.pair_id.str.replace("00000", "00001")
    examples = qualitative_examples(pd.concat([extra, results], ignore_index=True))
    assert set(examples.pair_id.str.rsplit(":", n=1).str[-1]) == {"00000"}
    assert set(examples.alpha) == {0.0, 1.0}
    assert len(examples) == 8
