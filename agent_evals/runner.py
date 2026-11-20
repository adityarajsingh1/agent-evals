"""
Running an agent over a dataset.

Cases run concurrently (up to `concurrency` at a time), each with a timeout. A case whose agent
raises or times out is recorded with the error instead of stopping the run, because one broken
case shouldn't hide how the other 199 did. Results come back in dataset order.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_evals.agents import Agent
from agent_evals.models import AgentResult, CaseResult, Run, Score, TestCase

# Takes the case and what the agent produced, returns a verdict. The built-in scorers come with
# their own module; any function with this shape works.
Scorer = Callable[[TestCase, AgentResult], "Score | Awaitable[Score]"]


def git_commit(cwd: str | Path | None = None) -> str | None:
    """The current commit of the repo at `cwd`, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


async def _score(scorer: Scorer, case: TestCase, result: AgentResult) -> Score:
    score = scorer(case, result)
    if inspect.isawaitable(score):
        score = await score
    return score


async def run_case(
    agent: Agent, case: TestCase, scorers: Sequence[Scorer] = (), timeout_s: float | None = 60.0
) -> CaseResult:
    try:
        result = await asyncio.wait_for(agent.run(case), timeout=timeout_s)
    except asyncio.TimeoutError:
        result = AgentResult(error=f"timed out after {timeout_s:g}s")
    except Exception as e:
        result = AgentResult(error=f"{type(e).__name__}: {e}")

    scores = []
    # an agent that errored has nothing to score; the case fails on the error alone
    if result.error is None:
        for scorer in scorers:
            try:
                scores.append(await _score(scorer, case, result))
            except Exception as e:
                name = getattr(scorer, "name", None) or getattr(scorer, "__name__", "scorer")
                scores.append(
                    Score(
                        name=name, value=0.0, passed=False, reason=f"scorer crashed: {type(e).__name__}: {e}"
                    )
                )
    return CaseResult(case_id=case.id, result=result, scores=scores)


async def run_eval(
    agent: Agent,
    cases: Sequence[TestCase],
    scorers: Sequence[Scorer] = (),
    *,
    dataset: str = "",
    concurrency: int = 8,
    timeout_s: float | None = 60.0,
    commit: str | None = None,
    config: dict[str, Any] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Run:
    """Run `agent` on every case and score it. `on_result` is called as each case finishes."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    run = Run(
        agent=agent.name,
        dataset=dataset,
        git_commit=commit if commit is not None else git_commit(),
        config={"concurrency": concurrency, "timeout_s": timeout_s, **(config or {})},
    )
    sem = asyncio.Semaphore(concurrency)

    async def one(case: TestCase) -> CaseResult:
        async with sem:
            res = await run_case(agent, case, scorers, timeout_s)
        if on_result is not None:
            on_result(res)
        return res

    run.cases = list(await asyncio.gather(*(one(c) for c in cases)))
    run.finished_at = datetime.now(timezone.utc)
    return run


def run_eval_sync(
    agent: Agent, cases: Sequence[TestCase], scorers: Sequence[Scorer] = (), **kwargs: Any
) -> Run:
    """`run_eval` for scripts and notebooks that aren't async themselves."""
    return asyncio.run(run_eval(agent, cases, scorers, **kwargs))
