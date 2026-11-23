"""Budget scorers: the agent got there without too many steps, too much money or too much time.

Limits come from the case (`max_steps`, `max_cost_usd`) or from the scorer's options, which act
as a default for cases that don't set their own.
"""

from __future__ import annotations

from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.scorers.base import BaseScorer, register


class _Limit(BaseScorer):
    unit = ""

    def __init__(self, limit: float | None = None) -> None:
        self.limit = limit

    def case_limit(self, case: TestCase) -> float | None:
        return None

    def measure(self, result: AgentResult) -> float:
        raise NotImplementedError

    def _limit(self, case: TestCase) -> float | None:
        own = self.case_limit(case)
        return own if own is not None else self.limit

    def applies_to(self, case: TestCase) -> bool:
        return self._limit(case) is not None

    def fmt(self, x: float) -> str:
        return f"{x:g}{self.unit}"

    def score(self, case: TestCase, result: AgentResult) -> Score:
        limit = self._limit(case)
        if limit is None:
            return self.result(0.0, False, "no limit set")
        used = self.measure(result)
        if used <= limit:
            return self.result(1.0, True, f"{self.fmt(used)} of {self.fmt(limit)}")
        # over budget: the score says how far over, so 2x the limit scores 0.5
        return self.result(limit / used, False, f"used {self.fmt(used)}, limit {self.fmt(limit)}")


@register
class StepLimit(_Limit):
    """At most `case.max_steps` steps (model calls plus tool calls)."""

    name = "max_steps"
    unit = " steps"

    def case_limit(self, case: TestCase) -> float | None:
        return case.max_steps

    def measure(self, result: AgentResult) -> float:
        return len(result.trace.steps)


@register
class CostLimit(_Limit):
    """At most `case.max_cost_usd`, as recorded in the trace."""

    name = "max_cost"

    def case_limit(self, case: TestCase) -> float | None:
        return case.max_cost_usd

    def measure(self, result: AgentResult) -> float:
        return result.trace.cost_usd

    def fmt(self, x: float) -> str:
        return f"${x:.4f}"


@register
class LatencyLimit(_Limit):
    """Total time in the trace's steps, in milliseconds. Set per case with `metadata.max_latency_ms`."""

    name = "max_latency"
    unit = " ms"

    def case_limit(self, case: TestCase) -> float | None:
        value = case.metadata.get("max_latency_ms")
        return float(value) if value is not None else None

    def measure(self, result: AgentResult) -> float:
        return result.trace.duration_ms

    def fmt(self, x: float) -> str:
        return f"{x:.0f}{self.unit}"
