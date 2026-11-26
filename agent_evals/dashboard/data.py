"""
The numbers behind the dashboard, as plain rows. Kept apart from the Streamlit code so it can be
tested without a browser and reused (e.g. for a weekly report).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_evals.models import CaseResult, Run
from agent_evals.report import failure_reason, score_names
from agent_evals.store import ResultStore


def hallucination_rate(run: Run) -> float | None:
    """Share of grounding-checked cases whose answer failed the groundedness check."""
    checked = [s for c in run.cases if (s := c.score("groundedness")) is not None]
    if not checked:
        return None
    return sum(not s.passed for s in checked) / len(checked)


@dataclass
class TrendPoint:
    run_id: str
    started_at: datetime
    git_commit: str | None
    pass_rate: float
    cost_usd: float
    hallucination_rate: float | None
    scores: dict[str, float]  # scorer name -> mean


def trend(store: ResultStore, agent: str, dataset: str, limit: int = 100) -> list[TrendPoint]:
    """One point per run, oldest first."""
    points = []
    for summary in reversed(store.list_runs(agent=agent, dataset=dataset, limit=limit)):
        run = store.load_run(summary.id)
        means = {name: m for name in score_names(run) if (m := run.mean_score(name)) is not None}
        points.append(
            TrendPoint(
                run.id,
                run.started_at,
                run.git_commit,
                run.pass_rate,
                run.cost_usd,
                hallucination_rate(run),
                means,
            )
        )
    return points


def trend_table(points: list[TrendPoint]) -> list[dict[str, Any]]:
    """Flat rows for charts: one column per metric, `None` where a run has no value."""
    names = sorted({n for p in points for n in p.scores})
    rows = []
    for p in points:
        row: dict[str, Any] = {
            "run": p.run_id,
            "started_at": p.started_at,
            "commit": (p.git_commit or "")[:10],
            "pass_rate": p.pass_rate,
            "hallucination_rate": p.hallucination_rate,
            "cost_usd": p.cost_usd,
        }
        row.update({f"score:{n}": p.scores.get(n) for n in names})
        rows.append(row)
    return rows


def case_rows(run: Run, only_failing: bool = True) -> list[dict[str, Any]]:
    rows = []
    for c in run.cases:
        if only_failing and c.passed:
            continue
        rows.append(
            {
                "case": c.case_id,
                "passed": c.passed,
                "why": "" if c.passed else failure_reason(c),
                "steps": len(c.result.trace.steps),
                "cost_usd": c.result.trace.cost_usd,
                "output": c.result.output or "",
            }
        )
    return rows


def trace_rows(case: CaseResult) -> list[dict[str, Any]]:
    """The trace as a table: one row per model or tool call."""
    rows = []
    for i, s in enumerate(case.result.trace.steps, start=1):
        if s.tool_call is not None:
            what = f"{s.tool_call.name}({s.tool_call.arguments})"
            result = s.tool_call.error or s.tool_call.result
        else:
            what = s.model or "model"
            result = s.content
        rows.append(
            {
                "#": i,
                "type": s.type,
                "call": what,
                "result": "" if result is None else str(result),
                "tokens": s.input_tokens + s.output_tokens,
                "cost_usd": s.cost_usd,
                "ms": round(s.duration_ms, 1),
            }
        )
    return rows
