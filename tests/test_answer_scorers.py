import pytest

from agent_evals.models import AgentResult, TestCase
from agent_evals.scorers import get_scorer
from agent_evals.scorers.answer import extract_numbers, normalize


def check(name, expected, output, **opts):
    case = TestCase(id="c", input="q", expected_output=expected)
    return get_scorer(name, **opts)(case, AgentResult(output=output))


def test_normalize():
    assert normalize("  The Eiffel Tower! ") == "eiffel tower"
    assert normalize("Zürich, café") == "zurich cafe"


@pytest.mark.parametrize(
    "text, numbers",
    [
        ("It costs 1,234.50 EUR", [1234.5]),
        ("between -3 and +4.5, or .5", [-3, 4.5, 0.5]),
        ("1e-3 and 12,34", [0.001, 12, 34]),
        ("gpt-4o costs nothing", []),  # the 4 is part of a word
        ("Version 2.0.1", []),
    ],
)
def test_extract_numbers(text, numbers):
    assert extract_numbers(text) == pytest.approx(numbers)


def test_exact():
    assert check("exact", "Paris", "paris.").passed
    assert not check("exact", "Paris", "It's Paris.").passed
    assert not check("exact", "Paris", "paris.", normalized=False).passed


def test_contains():
    assert check("contains", "Lisbon", "It's warmer in Lisbon today.").passed
    assert not check("contains", "Lis", "It's warmer in Lisbon today.").passed  # whole words only
    assert check("contains", "New York", "Flights to new-york are cheap").passed


def test_numeric():
    assert check("numeric", "1234.5", "That comes to about 1,234.50 EUR.").passed
    partial = check("numeric", "min 3, max 9", "The minimum is 3.")
    assert partial.value == 0.5 and not partial.passed and "missing 9" in partial.reason
    assert check("numeric", "100", "roughly 101", rel_tol=0.02).passed
    assert not check("numeric", "100", "roughly 101").passed


def test_missing_answer_or_expectation():
    assert check("exact", "x", None).reason == "agent gave no answer"
    scorer = get_scorer("contains")
    case = TestCase(id="c", input="q")
    assert not scorer.applies_to(case)
    assert not scorer(case, AgentResult(output="x")).passed
