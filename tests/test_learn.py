import pytest

from agent_evals.agents import FunctionAgent
from agent_evals.learn import (
    Example,
    PromptRegistry,
    PromptVersion,
    evaluate_candidate,
    examples_from_failures,
    record,
)
from agent_evals.models import AgentResult, CaseResult, Run, Score, TestCase
from agent_evals.scorers import get_scorer

TEMPS = {"Berlin": 12, "Lisbon": 19, "Oslo": 2, "Madrid": 21}


def fake_model(prompt: str) -> str:
    """Stands in for an LLM that picks up a pattern from its instructions or examples.

    It knows the temperatures, but reads every comparison as "which is warmer" unless the prompt
    mentions "colder" somewhere before the question.
    """
    if "always answer oslo" in prompt.lower():
        return "Oslo"
    head, _, question = prompt.rpartition("Q: ")
    cities = [c for c in TEMPS if c in question]
    if not cities:
        return "I don't know."
    pick = min if "colder" in question and "colder" in head else max
    return pick(cities, key=TEMPS.__getitem__)


def make_agent(version: PromptVersion):
    return FunctionAgent(lambda q: fake_model(version.render(q)), name="weather-llm")


def case(cid, q, answer, **kw):
    return TestCase(id=cid, input=q, expected_output=answer, **kw)


CASES = [
    case("warm-1", "Is it warmer in Lisbon or Oslo?", "Lisbon"),
    case("warm-2", "Is it warmer in Berlin or Madrid?", "Madrid"),
    case("cold-1", "Is it colder in Berlin or Oslo?", "Oslo", metadata={"reports": 3}),
    case("cold-2", "Is it colder in Lisbon or Berlin?", "Berlin"),
    case("cold-3", "Is it colder in Madrid or Lisbon?", "Lisbon", metadata={"reports": 1}),
]
SCORERS = [get_scorer("exact")]
BASE = PromptVersion(system="Answer with the name of one city.")


def run_of(results):
    return Run(
        agent="a",
        dataset="d",
        cases=[
            CaseResult(
                case_id=cid, result=AgentResult(output="x"), scores=[Score(name="s", value=1, passed=ok)]
            )
            for cid, ok in results.items()
        ],
    )


def test_messages_and_render():
    v = PromptVersion(system="Be brief.", examples=[Example(input="2+2?", output="4")])
    assert v.messages("3+3?") == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]
    assert v.render("3+3?") == "Be brief.\n\nExamples:\n\nQ: 2+2?\nA: 4\n\nQ: 3+3?\nA:"
    assert PromptVersion(system="Be brief.").render("hi") == "Be brief.\n\nQ: hi\nA:"


def test_with_examples_makes_a_child_and_replaces_same_inputs():
    v = BASE.with_examples([Example(input="a", output="1", case_id="c1")])
    w = v.with_examples([Example(input="a", output="2"), Example(input="b", output="3")])
    assert w.parent == v.id and v.parent == BASE.id
    assert [(e.input, e.output) for e in w.examples] == [("a", "2"), ("b", "3")]
    assert v.held_out() == {"c1"} and w.held_out() == set()
    assert BASE.examples == []  # the parent isn't changed


def test_examples_from_failures():
    cases = [*CASES, TestCase(id="tools-only", input="Weather in Oslo?")]
    run = run_of({"warm-1": True, "cold-1": False, "cold-2": False, "cold-3": False, "tools-only": False})
    examples = examples_from_failures(run, cases, limit=2)
    # most reported first; the case with nothing to teach is skipped
    assert [e.case_id for e in examples] == ["cold-1", "cold-3"]
    assert examples[0].output == "Oslo"
    assert [e.case_id for e in examples_from_failures(run, cases, exclude={"cold-1"})] == ["cold-3", "cold-2"]


async def test_candidate_that_generalises_is_promoted():
    candidate = BASE.with_examples([Example(input=CASES[2].input, output="Oslo", case_id="cold-1")])
    decision = await evaluate_candidate(make_agent, BASE, candidate, CASES, SCORERS, commit="")
    assert decision.held_out == ["cold-1"]
    assert len(decision.current_run.cases) == 4  # cold-1 isn't counted
    assert decision.current_run.pass_rate == 0.5
    assert decision.candidate_run.pass_rate == 1.0
    assert decision.promoted, decision.reasons
    assert [c.case_id for c in decision.comparison.fixes] == ["cold-2", "cold-3"]
    assert "promoted" in decision.summary() and "held out 1 case(s)" in decision.summary()


async def test_a_tie_keeps_the_current_version():
    candidate = PromptVersion(system="Answer with the name of one city. Be concise.", parent=BASE.id)
    decision = await evaluate_candidate(make_agent, BASE, candidate, CASES, SCORERS, commit="")
    assert not decision.promoted
    assert decision.reasons == ["not better: 40.0% -> 40.0%"]


async def test_a_regression_blocks_promotion_even_when_the_pass_rate_goes_up():
    # gets the "colder" cases right by always answering Oslo, and breaks the "warmer" one
    candidate = PromptVersion(system="Always answer Oslo. Colder.", parent=BASE.id)
    cases = [CASES[0], CASES[2], case("cold-4", "Is it colder in Madrid or Oslo?", "Oslo")]
    decision = await evaluate_candidate(make_agent, BASE, candidate, cases, SCORERS, commit="")
    assert decision.candidate_run.pass_rate > decision.current_run.pass_rate
    assert not decision.promoted
    assert decision.reasons == ["1 case(s) that passed before now fail"]
    assert "broke: warm-1" in decision.summary()
    decision = await evaluate_candidate(
        make_agent, BASE, candidate, cases, SCORERS, commit="", max_regressions=1
    )
    assert decision.promoted


async def test_nothing_left_to_evaluate():
    candidate = BASE.with_examples([Example(input=c.input, output="x", case_id=c.id) for c in CASES])
    with pytest.raises(ValueError, match="nothing left"):
        await evaluate_candidate(make_agent, BASE, candidate, CASES, SCORERS)


async def test_registry_records_promotions_and_rolls_back(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    assert reg.current() is None and reg.history() == []
    first = reg.init("Answer with the name of one city.")
    assert reg.init("something else") == first  # init only seeds an empty registry

    good = first.with_examples([Example(input=CASES[2].input, output="Oslo", case_id="cold-1")])
    record(reg, await evaluate_candidate(make_agent, first, good, CASES, SCORERS, commit=""))
    assert reg.current() == good
    bad = PromptVersion(system="Always answer Oslo.", parent=good.id)
    record(reg, await evaluate_candidate(make_agent, good, bad, CASES, SCORERS, commit=""))
    assert reg.current() == good  # the bad one is saved but not used
    assert {v.id for v in reg.versions()} == {first.id, good.id, bad.id}

    history = reg.history()
    assert [h["promoted"] for h in history] == [True, False]
    assert history[0]["fixed"] == ["cold-2", "cold-3"] and history[0]["held_out"] == ["cold-1"]
    assert history[1]["broke"]

    assert reg.rollback() == first
    assert reg.current() == first
    with pytest.raises(KeyError):
        reg.get("nope")


def test_rollback_with_nothing_promoted(tmp_path):
    reg = PromptRegistry(tmp_path)
    reg.init("x")
    with pytest.raises(ValueError, match="nothing has been promoted"):
        reg.rollback()
