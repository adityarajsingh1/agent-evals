import json

import pytest

from agent_evals.llm import FunctionModel
from agent_evals.models import AgentResult, Step, TestCase, ToolCall, Trace
from agent_evals.scorers import get_scorer
from agent_evals.scorers.judge import RUBRICS, Rubric

CASE = TestCase(id="c", input="How long do refunds take?", expected_output="Up to 5 business days.")
ANSWER = AgentResult(output="Refunds usually arrive within five business days.")


def replies(*texts):
    it = iter(texts)
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return next(it)

    model = FunctionModel(complete)
    model.prompts = prompts  # type: ignore[attr-defined]
    return model


def scores(**kw):
    return json.dumps({"scores": kw, "reasoning": "Matches the reference."})


def test_weighted_score_and_pass():
    model = replies(scores(correct=5, complete=4))
    result = get_scorer("judge", model=model)(CASE, ANSWER)
    # correct has weight 2: (5*2 + 4) / 3 = 4.67 on 1-5, rescaled to 0.92
    assert result.value == pytest.approx((14 / 3 - 1) / 4)
    assert result.passed
    assert result.reason == "correctness: correct 5/5, complete 4/5. Matches the reference."


def test_prompt_has_question_reference_answer_and_tools():
    model = replies(scores(correct=3, complete=3))
    trace = Trace(
        steps=[
            Step(
                type="tool",
                tool_call=ToolCall(
                    name="policy_lookup", arguments={"topic": "refunds"}, result="5 business days"
                ),
            )
        ]
    )
    get_scorer("judge", model=model)(CASE, AgentResult(output="Five days.", trace=trace))
    prompt = model.prompts[0]
    assert "How long do refunds take?" in prompt and "Up to 5 business days." in prompt
    assert "policy_lookup({'topic': 'refunds'}) -> 5 business days" in prompt
    assert '"correct": <1-5>' in prompt and "Five days." in prompt


def test_low_scores_fail():
    result = get_scorer("judge", model=replies(scores(correct=2, complete=3)))(CASE, ANSWER)
    assert not result.passed and result.value == pytest.approx((7 / 3 - 1) / 4)


def test_samples_are_averaged():
    model = replies(scores(correct=5, complete=5), scores(correct=3, complete=3))
    result = get_scorer("judge", model=model, samples=2)(CASE, ANSWER)
    assert result.value == pytest.approx(0.75)
    assert "(spread 0.50 over 2 samples)" in result.reason


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("Looks good to me!", "judge reply isn't JSON"),
        (json.dumps({"score": 5}), "judge reply has no 'scores' object"),
        (scores(correct=5), "judge gave no score for 'complete'"),
        (scores(correct=9, complete=4), "judge score 9 for 'correct' is outside 1-5"),
    ],
)
def test_bad_judge_replies_fail_the_case(reply, reason):
    result = get_scorer("judge", model=replies(reply))(CASE, ANSWER)
    assert not result.passed and result.reason.startswith(reason)


def test_custom_rubric_from_dict():
    rubric = {
        "name": "tone",
        "scale": 3,
        "criteria": [{"name": "polite", "description": "Friendly and polite."}],
    }
    result = get_scorer("judge", model=replies(scores(polite=3)), rubric=rubric)(CASE, ANSWER)
    assert result.value == 1.0 and result.reason.startswith("tone: polite 3/3")


def test_rubric_validation():
    assert set(RUBRICS) == {"correctness", "helpfulness", "safety"}
    with pytest.raises(KeyError, match="unknown rubric"):
        get_scorer("judge", model=replies(), rubric="vibes")
    with pytest.raises(ValueError):
        Rubric(name="empty", criteria=[])
    with pytest.raises(ValueError):
        get_scorer("judge", model=replies(), samples=0)


def test_no_answer():
    assert (
        get_scorer("judge", model=replies())(CASE, AgentResult(output=None)).reason == "agent gave no answer"
    )
