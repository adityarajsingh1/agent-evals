import asyncio

import pytest

from agent_evals.agents import FunctionAgent
from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.runner import git_commit, run_case, run_eval, run_eval_sync

CASES = [TestCase(id=f"c{i}", input=str(i)) for i in range(10)]


def exact(case: TestCase, result: AgentResult) -> Score:
    ok = result.output == case.input
    return Score(name="exact", value=float(ok), passed=ok)


async def test_runs_in_parallel_up_to_the_limit():
    running = peak = 0

    async def slow_echo(text):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return text

    run = await run_eval(FunctionAgent(slow_echo), CASES, [exact], concurrency=3, commit="abc")
    assert peak == 3
    assert [c.case_id for c in run.cases] == [c.id for c in CASES]  # dataset order, not finish order
    assert run.pass_rate == 1.0
    assert run.git_commit == "abc"
    assert run.config["concurrency"] == 3
    assert run.finished_at is not None and run.finished_at >= run.started_at


async def test_errors_and_timeouts_are_recorded_not_raised():
    async def flaky(text):
        if text == "3":
            raise ValueError("bad input")
        if text == "4":
            await asyncio.sleep(1)
        return text

    run = await run_eval(FunctionAgent(flaky), CASES, [exact], timeout_s=0.1, commit="")
    by_id = {c.case_id: c for c in run.cases}
    assert by_id["c3"].result.error == "ValueError: bad input"
    assert by_id["c4"].result.error == "timed out after 0.1s"
    assert not by_id["c3"].scores  # nothing to score when the agent failed
    assert run.pass_rate == pytest.approx(0.8)


async def test_a_crashing_scorer_fails_only_its_case():
    def picky(case, result):
        if case.id == "c0":
            raise KeyError("missing field")
        return Score(name="picky", value=1.0, passed=True)

    results = [await run_case(FunctionAgent(lambda t: t), c, [picky]) for c in CASES[:2]]
    assert results[0].scores[0].reason == "scorer crashed: KeyError: 'missing field'"
    assert not results[0].passed and results[1].passed


async def test_async_scorers_and_progress_callback():
    async def async_exact(case, result):
        return exact(case, result)

    seen = []
    run = await run_eval(
        FunctionAgent(lambda t: t), CASES[:4], [async_exact], on_result=seen.append, commit=""
    )
    assert sorted(r.case_id for r in seen) == ["c0", "c1", "c2", "c3"]
    assert run.mean_score("exact") == 1.0


def test_sync_wrapper():
    run = run_eval_sync(FunctionAgent(lambda t: "wrong"), CASES[:2], [exact], dataset="d.jsonl", commit="")
    assert run.pass_rate == 0.0 and run.dataset == "d.jsonl"


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        run_eval_sync(FunctionAgent(lambda t: t), CASES, concurrency=0)


def test_git_commit_outside_a_repo(tmp_path):
    assert git_commit(tmp_path) is None
