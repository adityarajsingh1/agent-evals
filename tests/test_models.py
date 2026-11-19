import json

import pytest
from pydantic import ValidationError

from agent_evals.models import (
    AgentResult,
    CaseResult,
    Run,
    Score,
    Step,
    TestCase,
    ToolCall,
    Trace,
)


def make_trace() -> Trace:
    return Trace(
        steps=[
            Step(type="llm", model="m", input_tokens=120, output_tokens=30, cost_usd=0.002, duration_ms=400),
            Step(
                type="tool",
                tool_call=ToolCall(name="search", arguments={"q": "berlin weather"}, result="12°C"),
                duration_ms=150,
            ),
            Step(
                type="llm", content="It's 12°C in Berlin.", input_tokens=160, output_tokens=12, cost_usd=0.003
            ),
        ]
    )


def test_test_case_from_json():
    case = TestCase.model_validate_json(
        json.dumps(
            {
                "id": "weather-1",
                "input": "What's the weather in Berlin?",
                "expected_tools": [{"name": "search", "arguments": {"q": "berlin weather"}}],
                "sources": [{"id": "s1", "text": "Berlin: 12°C, cloudy."}],
                "max_steps": 5,
                "tags": ["weather"],
            }
        )
    )
    assert case.expected_tools[0].name == "search"
    assert case.expected_output is None
    assert not case.tools_in_order


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="expected_toools"):
        TestCase.model_validate({"id": "x", "input": "hi", "expected_toools": []})


@pytest.mark.parametrize("field", [{"max_steps": 0}, {"max_cost_usd": -1}])
def test_limits_must_be_sensible(field):
    with pytest.raises(ValidationError):
        TestCase(id="x", input="hi", **field)


def test_step_type_matches_tool_call():
    with pytest.raises(ValidationError, match="needs a tool_call"):
        Step(type="tool")
    with pytest.raises(ValidationError, match="own 'tool' step"):
        Step(type="llm", tool_call=ToolCall(name="search"))


def test_trace_totals():
    trace = make_trace()
    assert [c.name for c in trace.tool_calls] == ["search"]
    assert trace.cost_usd == pytest.approx(0.005)
    assert trace.tokens == 322
    assert trace.duration_ms == 550


def test_case_result_passed():
    ok = Score(name="match", value=1.0, passed=True)
    bad = Score(name="tools", value=0.0, passed=False, reason="search was never called")
    result = AgentResult(output="12°C", trace=make_trace())

    assert CaseResult(case_id="a", result=result, scores=[ok]).passed
    assert not CaseResult(case_id="a", result=result, scores=[ok, bad]).passed
    assert not CaseResult(case_id="a", result=result).passed  # nothing was checked
    errored = AgentResult(error="TimeoutError: agent took longer than 30s")
    assert not CaseResult(case_id="a", result=errored, scores=[ok]).passed


def test_score_value_is_bounded():
    with pytest.raises(ValidationError):
        Score(name="x", value=1.5, passed=True)


def test_run_summary_and_round_trip():
    result = AgentResult(output="12°C", trace=make_trace())
    run = Run(
        agent="weather-bot",
        dataset="weather.jsonl",
        git_commit="abc1234",
        cases=[
            CaseResult(case_id="a", result=result, scores=[Score(name="match", value=1.0, passed=True)]),
            CaseResult(
                case_id="b",
                result=result,
                scores=[
                    Score(name="match", value=0.0, passed=False),
                    Score(name="tools", value=0.5, passed=False),
                ],
            ),
        ],
    )
    assert run.pass_rate == 0.5
    assert run.mean_score("match") == 0.5
    assert run.mean_score("tools") == 0.5
    assert run.mean_score("missing") is None
    assert run.cost_usd == pytest.approx(0.010)

    again = Run.model_validate_json(run.model_dump_json())
    assert again == run
    assert again.started_at.tzinfo is not None


def test_empty_run():
    run = Run(agent="a", dataset="d")
    assert run.pass_rate == 0.0
    assert len(run.id) == 12
