"""Scorers for how the agent got to its answer: which tools it called and with what.

A trajectory is the list of tool calls in the trace. Each expected call is matched to a different
actual call. With `case.tools_in_order` the matches also have to keep their order (the expected
calls must appear as a subsequence, other calls in between are fine).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_evals.models import AgentResult, ExpectedToolCall, Score, TestCase, ToolCall
from agent_evals.scorers.base import BaseScorer, register

CallMatcher = Callable[[ExpectedToolCall, ToolCall], bool]


def same_name(expected: ExpectedToolCall, actual: ToolCall) -> bool:
    return expected.name == actual.name


def match_calls(
    expected: Sequence[ExpectedToolCall],
    actual: Sequence[ToolCall],
    *,
    ordered: bool,
    matches: CallMatcher = same_name,
) -> list[int | None]:
    """For each expected call, the index of the actual call it matched, or None.

    Unordered matching is greedy in expected order, which is good enough here: a call can only
    match one expectation, and expectations for the same tool are usually told apart by arguments.
    """
    used: set[int] = set()
    out: list[int | None] = []
    start = 0
    for exp in expected:
        hit = None
        for i in range(start if ordered else 0, len(actual)):
            if i not in used and matches(exp, actual[i]):
                hit = i
                break
        out.append(hit)
        if hit is not None:
            used.add(hit)
            if ordered:
                start = hit + 1
    return out


def _describe(call: ExpectedToolCall | ToolCall) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return f"{call.name}({args})"


class _TrajectoryScorer(BaseScorer):
    matches: CallMatcher = staticmethod(same_name)

    def __init__(self, allow_extra: bool = True) -> None:
        """`allow_extra=False` also fails the case when the agent made calls nobody expected."""
        self.allow_extra = allow_extra

    def applies_to(self, case: TestCase) -> bool:
        # an explicit "expected_tools": [] means "no tools", which is worth checking too;
        # a case that never mentions expected_tools doesn't get this scorer
        return "expected_tools" in case.model_fields_set

    def score(self, case: TestCase, result: AgentResult) -> Score:
        expected, actual = case.expected_tools, result.trace.tool_calls
        if not expected:
            if not actual:
                return self.result(1.0, True)
            return self.result(0.0, False, "expected no tool calls, got " + ", ".join(map(_describe, actual)))

        hits = match_calls(expected, actual, ordered=case.tools_in_order, matches=type(self).matches)
        missing = [e for e, h in zip(expected, hits, strict=True) if h is None]
        extra = [a for i, a in enumerate(actual) if i not in set(hits)]
        found = len(expected) - len(missing)
        # recall over the expected calls; extra calls only count against it when they're not allowed
        denominator = len(expected) + (0 if self.allow_extra else len(extra))
        passed = not missing and (self.allow_extra or not extra)

        reasons = []
        if missing:
            reasons.append("missing " + ", ".join(map(_describe, missing)))
            if case.tools_in_order:
                reasons.append("(in this order)")
        if extra:
            reasons.append(
                ("unexpected " if not self.allow_extra else "also called ") + ", ".join(map(_describe, extra))
            )
        return self.result(found / denominator, passed, "; ".join(reasons))


@register
class ToolSelection(_TrajectoryScorer):
    """The agent called the expected tools, judged by name only."""

    name = "tools"
