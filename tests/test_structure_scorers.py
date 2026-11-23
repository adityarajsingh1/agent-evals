import pytest

from agent_evals.models import AgentResult, TestCase
from agent_evals.scorers import get_scorer
from agent_evals.scorers.structure import extract_json

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "pattern": "^ORD-\\d+$"},
        "qty": {"type": "integer", "minimum": 1},
    },
    "required": ["order_id", "qty"],
}


def run(scorer, output, metadata=None):
    return scorer(TestCase(id="c", input="q", metadata=metadata or {}), AgentResult(output=output))


def test_regex_from_options_and_metadata():
    iso_date = get_scorer("regex", pattern=r"\b\d{4}-\d{2}-\d{2}\b")
    assert run(iso_date, "It ships on 2026-12-01.").passed
    assert not run(iso_date, "It ships on Dec 1st.").passed

    per_case = get_scorer("regex", ignore_case=True)
    assert per_case.applies_to(TestCase(id="c", input="q", metadata={"regex": "^yes"}))
    assert not per_case.applies_to(TestCase(id="c", input="q"))
    assert run(per_case, "Yes, it does.", {"regex": "^yes"}).passed


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        'Here you go:\n```json\n{"a": 1}\n```\nAnything else?',
        'Sure! {"a": 1} is the result.',
    ],
)
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_fails_clearly():
    with pytest.raises(ValueError, match="no JSON"):
        extract_json("no braces here")


def test_json_schema():
    scorer = get_scorer("json_schema", schema=ORDER_SCHEMA)
    assert run(scorer, 'Created: {"order_id": "ORD-17", "qty": 2}').passed

    bad = run(scorer, '{"order_id": "17", "qty": 0}')
    assert not bad.passed
    assert bad.reason.startswith("order_id: '17' does not match") and bad.reason.endswith("(+1 more)")

    missing = run(scorer, '{"qty": 1}')
    assert missing.reason == "(root): 'order_id' is a required property"
    assert run(scorer, "I couldn't create the order.").reason == "no JSON found in the answer"


def test_json_schema_per_case():
    scorer = get_scorer("json_schema")
    assert run(scorer, '{"order_id": "ORD-1", "qty": 1}', {"json_schema": ORDER_SCHEMA}).passed
