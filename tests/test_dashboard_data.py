from datetime import datetime, timedelta, timezone

import pytest

from agent_evals.dashboard.data import case_rows, hallucination_rate, trace_rows, trend, trend_table
from agent_evals.models import AgentResult, CaseResult, Run, Score, Step, ToolCall, Trace
from agent_evals.store import ResultStore

T0 = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)
TRACE = Trace(
    steps=[
        Step(type="llm", model="m", input_tokens=10, output_tokens=5, cost_usd=0.001, duration_ms=120.04),
        Step(type="tool", tool_call=ToolCall(name="search", arguments={"q": "refunds"}, result="30 days")),
        Step(type="tool", tool_call=ToolCall(name="lookup", error="TimeoutError: slow")),
    ]
)


def make_run(rid, hours, grounded, commit):
    cases = [
        CaseResult(
            case_id=f"c{i}",
            result=AgentResult(output=f"answer {i}", trace=TRACE),
            scores=[
                Score(
                    name="groundedness",
                    value=1.0 if g else 0.4,
                    passed=g,
                    reason="" if g else "1 of 2 claims supported",
                ),
            ],
        )
        for i, g in enumerate(grounded)
    ]
    cases.append(
        CaseResult(
            case_id="no-sources",
            result=AgentResult(output="x", trace=TRACE),
            scores=[Score(name="contains", value=1.0, passed=True)],
        )
    )
    return Run(
        id=rid,
        agent="bot",
        dataset="d.jsonl",
        git_commit=commit,
        started_at=T0 + timedelta(hours=hours),
        cases=cases,
    )


@pytest.fixture
def store(tmp_path):
    with ResultStore(tmp_path / "r.sqlite") as s:
        s.save(make_run("r1", 0, [True, True, True, True], "aaaaaaaaaaaa1"))
        s.save(make_run("r2", 1, [True, False, True, False], "bbbbbbbbbbbb2"))
        s.save(make_run("other", 2, [True], "c"))
        yield s


def test_hallucination_rate():
    assert hallucination_rate(make_run("x", 0, [True, False, False, False], None)) == 0.75
    assert hallucination_rate(Run(agent="a", dataset="d")) is None


def test_trend_oldest_first(store):
    points = trend(store, "bot", "d.jsonl")
    assert [p.run_id for p in points] == ["r1", "r2", "other"]
    assert [p.hallucination_rate for p in points[:2]] == [0.0, 0.5]
    assert points[1].scores == {"groundedness": pytest.approx(0.7), "contains": 1.0}
    assert points[1].pass_rate == pytest.approx(3 / 5)


def test_trend_table(store):
    rows = trend_table(trend(store, "bot", "d.jsonl"))
    assert rows[0]["commit"] == "aaaaaaaaaa"
    assert set(rows[0]) >= {
        "pass_rate",
        "hallucination_rate",
        "cost_usd",
        "score:groundedness",
        "score:contains",
    }


def test_groups(store):
    assert store.groups() == [("bot", "d.jsonl", 3)]


def test_case_and_trace_rows(store):
    run = store.load_run("r2")
    rows = case_rows(run)
    assert [r["case"] for r in rows] == ["c1", "c3"]
    assert rows[0]["why"] == "groundedness: 1 of 2 claims supported" and rows[0]["steps"] == 3
    assert len(case_rows(run, only_failing=False)) == 5

    steps = trace_rows(run.cases[0])
    assert steps[0] == {
        "#": 1,
        "type": "llm",
        "call": "m",
        "result": "",
        "tokens": 15,
        "cost_usd": 0.001,
        "ms": 120.0,
    }
    assert steps[1]["call"] == "search({'q': 'refunds'})" and steps[1]["result"] == "30 days"
    assert steps[2]["result"] == "TimeoutError: slow"
