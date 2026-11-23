import pytest

from agent_evals.agents import FunctionAgent
from agent_evals.dataset import load_dataset
from agent_evals.models import AgentResult, Step, TestCase, ToolCall, Trace
from agent_evals.runner import run_eval_sync
from agent_evals.scorers import available_scorers, default_scorers, get_scorer


def result(steps=3, cost=0.0, ms=0.0):
    return AgentResult(
        output="ok",
        trace=Trace(
            steps=[Step(type="llm", cost_usd=cost / steps, duration_ms=ms / steps) for _ in range(steps)]
        ),
    )


def test_step_limit():
    case = TestCase(id="c", input="q", max_steps=4)
    assert get_scorer("max_steps")(case, result(steps=4)).passed
    over = get_scorer("max_steps")(case, result(steps=8))
    assert not over.passed and over.value == 0.5 and over.reason == "used 8 steps, limit 4 steps"


def test_option_is_a_default_the_case_can_override():
    scorer = get_scorer("max_steps", limit=2)
    assert scorer.applies_to(TestCase(id="c", input="q"))
    assert not scorer(TestCase(id="c", input="q"), result(steps=3)).passed
    assert scorer(TestCase(id="c", input="q", max_steps=5), result(steps=3)).passed
    assert not get_scorer("max_steps").applies_to(TestCase(id="c", input="q"))


def test_cost_limit():
    case = TestCase(id="c", input="q", max_cost_usd=0.01)
    ok = get_scorer("max_cost")(case, result(cost=0.004))
    assert ok.passed and ok.reason == "$0.0040 of $0.0100"
    assert get_scorer("max_cost")(case, result(cost=0.04)).value == pytest.approx(0.25)


def test_latency_limit_from_metadata():
    case = TestCase(id="c", input="q", metadata={"max_latency_ms": 500})
    assert get_scorer("max_latency")(case, result(ms=300)).passed
    assert get_scorer("max_latency")(case, result(ms=900)).reason == "used 900 ms, limit 500 ms"


def test_all_builtins_registered():
    assert {
        "exact",
        "contains",
        "numeric",
        "regex",
        "json_schema",
        "tools",
        "tool_args",
        "max_steps",
        "max_cost",
        "max_latency",
    } <= set(available_scorers())


def test_default_scorers_on_the_example_dataset():
    def weather_agent(question, rec):
        rec.llm(model="fake", input_tokens=50, output_tokens=10)
        if "Lisbon" in question:
            rec.tool("get_weather", {"city": "Lisbon"}, result={"temp_c": 19})
            rec.tool("get_weather", {"city": "Oslo"}, result={"temp_c": 2})
            return "Lisbon is warmer (19°C vs 2°C)."
        if "Berlin" in question:
            rec.tool("get_weather", {"city": "berlin"}, result={"temp_c": 12})
            return "12°C and cloudy."
        return "Overcast means the sky is fully covered by clouds."

    run = run_eval_sync(
        FunctionAgent(weather_agent), load_dataset("examples/weather.jsonl"), default_scorers(), commit=""
    )
    scored = {c.case_id: sorted(s.name for s in c.scores) for c in run.cases}
    assert scored == {
        "berlin-now": ["max_steps", "tool_args"],
        "compare": ["contains", "tool_args"],
        "no-tool": ["max_steps", "tool_args"],
    }
    assert run.pass_rate == 1.0


def test_tool_call_counts_as_step():
    trace = Trace(steps=[Step(type="tool", tool_call=ToolCall(name="t"))])
    assert get_scorer("max_steps", limit=1)(
        TestCase(id="c", input="q"), AgentResult(output="x", trace=trace)
    ).passed
