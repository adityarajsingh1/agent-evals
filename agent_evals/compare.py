"""Comparing two runs of the same dataset, case by case, to find regressions.

A regression is a case that passed in the base run and fails in the new one. The pass rate alone
can hide them: fix two cases and break two others, and it doesn't move. So the comparison works
per case, and the CI gate fails on any new failure by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_evals.models import Run
from agent_evals.report import failure_reason, score_names


@dataclass
class CaseChange:
    case_id: str
    reason: str = ""  # why it fails now, for regressions


@dataclass
class Comparison:
    base: Run
    head: Run
    regressions: list[CaseChange] = field(default_factory=list)
    fixes: list[CaseChange] = field(default_factory=list)
    still_failing: list[CaseChange] = field(default_factory=list)
    added: list[str] = field(default_factory=list)  # cases only in the new run
    removed: list[str] = field(default_factory=list)  # cases only in the base run
    # scorer name -> (base mean, head mean); None where a run has no such scores
    scores: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    @property
    def pass_rate_delta(self) -> float:
        return self.head.pass_rate - self.base.pass_rate

    @property
    def cost_delta(self) -> float:
        return self.head.cost_usd - self.base.cost_usd

    def score_delta(self, name: str) -> float | None:
        base, head = self.scores.get(name, (None, None))
        return None if base is None or head is None else head - base

    def gate(
        self,
        max_regressions: int = 0,
        max_score_drop: float | None = None,
        max_cost_increase: float | None = None,
    ) -> list[str]:
        """Reasons the new run should fail CI. An empty list means it's fine to merge."""
        problems = []
        if len(self.regressions) > max_regressions:
            problems.append(f"{len(self.regressions)} case(s) that passed before now fail")
        if max_score_drop is not None:
            for name in self.scores:
                delta = self.score_delta(name)
                if delta is not None and -delta > max_score_drop:
                    problems.append(f"mean {name} dropped by {-delta:.3f} (allowed {max_score_drop:g})")
        if max_cost_increase is not None and self.base.cost_usd > 0:
            increase = self.cost_delta / self.base.cost_usd
            if increase > max_cost_increase:
                problems.append(f"cost went up {increase:.0%} (allowed {max_cost_increase:.0%})")
        return problems


def compare_runs(base: Run, head: Run) -> Comparison:
    # the datasets may have different names (files get moved); cases are matched by id
    cmp = Comparison(base=base, head=head)
    before = {c.case_id: c for c in base.cases}
    after = {c.case_id: c for c in head.cases}
    for case_id, new in after.items():
        old = before.get(case_id)
        if old is None:
            cmp.added.append(case_id)
        elif old.passed and not new.passed:
            cmp.regressions.append(CaseChange(case_id, failure_reason(new)))
        elif not old.passed and new.passed:
            cmp.fixes.append(CaseChange(case_id))
        elif not new.passed:
            cmp.still_failing.append(CaseChange(case_id, failure_reason(new)))
    cmp.removed = [cid for cid in before if cid not in after]
    for name in dict.fromkeys(score_names(base) + score_names(head)):
        cmp.scores[name] = (base.mean_score(name), head.mean_score(name))
    return cmp
