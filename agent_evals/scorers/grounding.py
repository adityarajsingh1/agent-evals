"""The groundedness scorer: what share of the answer's claims the sources back up."""

from __future__ import annotations

import json

from agent_evals.grounding import ClaimChecker, LexicalChecker, check_grounding
from agent_evals.models import AgentResult, Score, Source, TestCase
from agent_evals.scorers.base import BaseScorer, register


def tool_sources(result: AgentResult) -> list[Source]:
    """Tool results as sources: an agent that looked something up may state what it found."""
    out = []
    for i, call in enumerate(result.trace.tool_calls):
        if call.error is None and call.result is not None:
            text = call.result if isinstance(call.result, str) else json.dumps(call.result, default=str)
            out.append(Source(id=f"tool:{call.name}#{i}", text=text))
    return out


@register
class Groundedness(BaseScorer):
    """Fails when too many of the answer's claims aren't backed by `case.sources` (or tool results)."""

    name = "groundedness"

    def __init__(
        self,
        threshold: float = 0.8,
        min_coverage: float = 0.7,
        include_tool_results: bool = True,
        checker: ClaimChecker | None = None,
    ) -> None:
        """`threshold` is the share of claims that must be supported for the case to pass."""
        self.threshold = threshold
        self.include_tool_results = include_tool_results
        self.checker = checker or LexicalChecker(min_coverage=min_coverage)

    def applies_to(self, case: TestCase) -> bool:
        return bool(case.sources)

    def score(self, case: TestCase, result: AgentResult) -> Score:
        sources = list(case.sources)
        if self.include_tool_results:
            sources += tool_sources(result)
        if not result.output:
            return self.result(0.0, False, "agent gave no answer")
        if not sources:
            return self.result(0.0, False, "no sources to check against")

        report = check_grounding(result.output, sources, self.checker)
        value = report.groundedness
        if not report.verdicts:
            return self.result(1.0, True, "no factual claims")
        bad = report.unsupported
        total = len(report.verdicts)
        reason = f"{total - len(bad)} of {total} claims supported"
        if bad:
            shown = "; ".join(f"{v.claim!r} ({v.reason})" if v.reason else repr(v.claim) for v in bad[:3])
            reason += f". Unsupported: {shown}" + (f" and {len(bad) - 3} more" if len(bad) > 3 else "")
        return self.result(value, value >= self.threshold, reason)
