import pytest

from agent_evals.models import AgentResult, ExpectedToolCall, Step, TestCase, ToolCall, Trace
from agent_evals.scorers import get_scorer
from agent_evals.scorers.trajectory import match_calls


def trace(*calls):
    return AgentResult(
        output="ok",
        trace=Trace(steps=[Step(type="tool", tool_call=ToolCall(name=n, arguments=a)) for n, a in calls]),
    )


def case(*expected, ordered=False):
    return TestCase(
        id="c",
        input="q",
        tools_in_order=ordered,
        expected_tools=[ExpectedToolCall(name=n, arguments=a) for n, a in expected],
    )


def test_match_calls_unordered_and_ordered():
    exp = [ExpectedToolCall(name="b"), ExpectedToolCall(name="a")]
    actual = [ToolCall(name="a"), ToolCall(name="b")]
    assert match_calls(exp, actual, ordered=False) == [1, 0]
    assert match_calls(exp, actual, ordered=True) == [1, None]


def test_each_actual_call_matches_once():
    exp = [ExpectedToolCall(name="search"), ExpectedToolCall(name="search")]
    assert match_calls(exp, [ToolCall(name="search")], ordered=False) == [0, None]


def test_tools_scorer():
    scorer = get_scorer("tools")
    c = case(("search", {}), ("summarize", {}))
    assert scorer(c, trace(("summarize", {}), ("search", {}))).passed

    half = scorer(c, trace(("search", {})))
    assert half.value == 0.5 and not half.passed and half.reason == "missing summarize()"

    ordered = case(("search", {}), ("summarize", {}), ordered=True)
    wrong_order = scorer(ordered, trace(("summarize", {}), ("search", {})))
    assert not wrong_order.passed and "(in this order)" in wrong_order.reason


def test_extra_calls():
    c = case(("search", {}))
    lenient = get_scorer("tools")(c, trace(("search", {}), ("delete_all", {"confirm": True})))
    assert lenient.passed and lenient.reason == "also called delete_all(confirm=True)"

    strict = get_scorer("tools", allow_extra=False)(c, trace(("search", {}), ("delete_all", {})))
    assert not strict.passed and strict.value == 0.5 and strict.reason.startswith("unexpected delete_all")


def test_expected_no_tools():
    no_tools = TestCase.model_validate({"id": "c", "input": "define overcast", "expected_tools": []})
    scorer = get_scorer("tools")
    assert scorer.applies_to(no_tools)
    assert scorer(no_tools, trace()).passed
    assert (
        scorer(no_tools, trace(("search", {"q": "overcast"}))).reason
        == "expected no tool calls, got search(q='overcast')"
    )


def test_not_applied_without_expectations():
    assert not get_scorer("tools").applies_to(TestCase(id="c", input="q"))


@pytest.mark.parametrize("ordered", [True, False])
def test_empty_trace(ordered):
    result = get_scorer("tools")(case(("search", {}), ordered=ordered), trace())
    assert result.value == 0.0 and not result.passed
