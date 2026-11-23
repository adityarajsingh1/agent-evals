"""Scorers for how the agent got to its answer: which tools it called and with what.

A trajectory is the list of tool calls in the trace. Each expected call is matched to a different
actual call. With `case.tools_in_order` the matches also have to keep their order (the expected
calls must appear as a subsequence, other calls in between are fine).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

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


def values_match(expected: Any, actual: Any, *, rel_tol: float = 1e-6, case_sensitive: bool = False) -> bool:
    """Compare one expected argument value with what the agent passed.

    - dicts: every expected key must be there and match; extra keys are fine
    - lists: same length, matched element by element
    - numbers: within `rel_tol`, and "3" matches 3 because models often pass numbers as strings
    - strings: compared after stripping whitespace, case-insensitively unless `case_sensitive`
    """

    def again(e: Any, a: Any) -> bool:
        return values_match(e, a, rel_tol=rel_tol, case_sensitive=case_sensitive)

    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and again(v, actual[k]) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(again(e, a) for e, a in zip(expected, actual, strict=True))
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=rel_tol)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        e, a = expected.strip(), actual.strip()
        return e == a if case_sensitive else e.casefold() == a.casefold()
    return bool(expected == actual)


class _TrajectoryScorer(BaseScorer):
    show_near_misses = False

    def _matches(self, expected: ExpectedToolCall, actual: ToolCall) -> bool:
        return same_name(expected, actual)

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

        hits = match_calls(expected, actual, ordered=case.tools_in_order, matches=self._matches)
        missing = [e for e, h in zip(expected, hits, strict=True) if h is None]
        extra = [a for i, a in enumerate(actual) if i not in set(hits)]
        found = len(expected) - len(missing)
        # recall over the expected calls; extra calls only count against it when they're not allowed
        denominator = len(expected) + (0 if self.allow_extra else len(extra))
        passed = not missing and (self.allow_extra or not extra)

        reasons = []
        if missing:
            notes = []
            for exp in missing:
                # right tool, wrong arguments is the most common failure, so show the near miss
                near = next((a for a in extra if a.name == exp.name), None) if self.show_near_misses else None
                if near is not None:
                    extra.remove(near)
                    notes.append(f"{_describe(exp)} (closest call: {_describe(near)})")
                else:
                    notes.append(_describe(exp))
            reasons.append("missing " + ", ".join(notes))
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


@register
class ToolArguments(_TrajectoryScorer):
    """The agent called the expected tools with the expected arguments.

    Only the arguments listed in the expectation are checked, so a case can pin down
    `{"city": "Berlin"}` without caring about `units` or `lang`. See `values_match` for how values
    are compared.
    """

    name = "tool_args"
    show_near_misses = True

    def __init__(self, allow_extra: bool = True, rel_tol: float = 1e-6, case_sensitive: bool = False) -> None:
        super().__init__(allow_extra)
        self.rel_tol = rel_tol
        self.case_sensitive = case_sensitive

    def _matches(self, expected: ExpectedToolCall, actual: ToolCall) -> bool:
        return same_name(expected, actual) and values_match(
            expected.arguments, actual.arguments, rel_tol=self.rel_tol, case_sensitive=self.case_sensitive
        )
