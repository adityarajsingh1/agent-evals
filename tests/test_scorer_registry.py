from types import SimpleNamespace

import pytest

from agent_evals.agents import FunctionAgent
from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.runner import run_eval_sync
from agent_evals.scorers import BaseScorer, available_scorers, base, build_scorers, get_scorer, register


@register
class LengthAtMost(BaseScorer):
    name = "test_length_at_most"

    def __init__(self, limit: int = 10) -> None:
        self.limit = limit

    def applies_to(self, case: TestCase) -> bool:
        return "short" in case.tags

    def score(self, case: TestCase, result: AgentResult) -> Score:
        n = len(result.output or "")
        return self.result(1.0 if n <= self.limit else 0.0, n <= self.limit, f"{n} characters")


def test_get_and_build_by_name():
    assert "test_length_at_most" in available_scorers()
    assert get_scorer("test_length_at_most", limit=3).limit == 3
    scorers = build_scorers(["test_length_at_most", {"name": "test_length_at_most", "limit": 1}])
    assert [s.limit for s in scorers] == [10, 1]
    assert repr(scorers[1]) == "LengthAtMost(limit=1)"


def test_unknown_and_bad_specs():
    with pytest.raises(KeyError, match="unknown scorer 'nope'"):
        get_scorer("nope")
    with pytest.raises(ValueError, match="needs a 'name'"):
        build_scorers([{"limit": 1}])


def test_names_must_be_unique():
    class Other(BaseScorer):
        name = "test_length_at_most"

        def score(self, case, result):
            raise NotImplementedError

    with pytest.raises(ValueError, match="already used"):
        register(Other)


def test_runner_skips_scorers_that_dont_apply():
    cases = [TestCase(id="a", input="hi", tags=["short"]), TestCase(id="b", input="hi")]
    run = run_eval_sync(FunctionAgent(lambda q: "a long answer"), cases, [LengthAtMost()], commit="")
    assert [len(c.scores) for c in run.cases] == [1, 0]
    assert run.cases[0].scores[0].reason == "13 characters"


def test_plugins_from_entry_points(monkeypatch):
    class FromPlugin(LengthAtMost):
        pass

    ep = SimpleNamespace(name="test_plugin", value="my_pkg:FromPlugin", load=lambda: FromPlugin)
    monkeypatch.setattr(base, "_plugins_loaded", False)
    monkeypatch.setattr(base, "entry_points", lambda group: [ep])
    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))
    assert isinstance(get_scorer("test_plugin"), FromPlugin)


def test_plugin_must_be_a_scorer(monkeypatch):
    ep = SimpleNamespace(name="bad", value="my_pkg:thing", load=lambda: object)
    monkeypatch.setattr(base, "_plugins_loaded", False)
    monkeypatch.setattr(base, "entry_points", lambda group: [ep])
    with pytest.raises(TypeError, match="not a BaseScorer subclass"):
        available_scorers()
