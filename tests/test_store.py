from datetime import datetime, timedelta, timezone

import pytest

from agent_evals.models import AgentResult, CaseResult, Run, Score, Step, ToolCall, Trace
from agent_evals.store import ResultStore

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


def make_run(run_id, hours, passes, commit="abc", agent="bot", dataset="d.jsonl"):
    trace = Trace(
        steps=[
            Step(type="llm", cost_usd=0.001),
            Step(type="tool", tool_call=ToolCall(name="search", arguments={"q": "x"}, result="y")),
        ]
    )
    cases = [
        CaseResult(
            case_id=f"c{i}",
            result=AgentResult(output=f"answer {i}", trace=trace),
            scores=[
                Score(name="contains", value=float(ok), passed=ok, reason="" if ok else "not found"),
                Score(name="groundedness", value=0.5 + 0.5 * ok, passed=True),
            ],
        )
        for i, ok in enumerate(passes)
    ]
    return Run(
        id=run_id,
        agent=agent,
        dataset=dataset,
        git_commit=commit,
        started_at=T0 + timedelta(hours=hours),
        finished_at=T0 + timedelta(hours=hours, minutes=1),
        cases=cases,
    )


@pytest.fixture
def store(tmp_path):
    with ResultStore(tmp_path / "evals" / "results.sqlite") as s:
        yield s


def test_round_trip_keeps_everything(store):
    run = make_run("run-1", 0, [True, False])
    store.save(run)
    assert store.load_run("run-1") == run
    assert store.load_run("run-1").cases[0].result.trace.tool_calls[0].arguments == {"q": "x"}


def test_list_runs_newest_first_with_filters(store):
    store.save(make_run("old", 0, [True, False], commit="c1"))
    store.save(make_run("new", 2, [True, True], commit="c2"))
    store.save(make_run("other", 1, [False], agent="other-bot"))

    runs = store.list_runs(agent="bot")
    assert [r.id for r in runs] == ["new", "old"]
    assert runs[0].pass_rate == 1.0 and runs[1].pass_rate == 0.5
    assert runs[0].git_commit == "c2" and runs[0].n_cases == 2
    assert runs[0].cost_usd == pytest.approx(0.002)
    assert len(store.list_runs()) == 3
    assert store.list_runs(dataset="missing.jsonl") == []


def test_latest_and_previous_run(store):
    for i, rid in enumerate(["r1", "r2", "r3"]):
        store.save(make_run(rid, i, [True]))
    assert store.latest_run("bot", "d.jsonl").id == "r3"
    assert store.latest_run("bot", "d.jsonl", before="r3").id == "r2"
    assert store.latest_run("bot", "d.jsonl", before="r1") is None
    assert store.latest_run("nobody", "d.jsonl") is None


def test_score_and_case_history(store):
    store.save(make_run("r1", 0, [True, True], commit="c1"))
    store.save(make_run("r2", 1, [True, False], commit="c2"))
    history = store.score_history("groundedness", "bot", "d.jsonl")
    assert [v for _, v in history] == [1.0, 0.75]  # oldest first
    case = store.case_history("c1", "bot", "d.jsonl")
    assert [(h.git_commit, h.passed) for h in case] == [("c2", False), ("c1", True)]


def test_prefix_lookup_and_errors(store):
    store.save(make_run("5f2a9c01aa", 0, [True]))
    store.save(make_run("5f2b0000bb", 1, [True]))
    assert store.load_run("5f2a").id == "5f2a9c01aa"
    with pytest.raises(KeyError, match="matches 2 runs"):
        store.load_run("5f2")
    with pytest.raises(KeyError, match="no run"):
        store.load_run("zzz")


def test_saving_again_replaces_and_delete_cascades(store):
    store.save(make_run("r1", 0, [True, True]))
    store.save(make_run("r1", 0, [False]))
    assert store.list_runs()[0].n_cases == 1
    store.delete_run("r1")
    assert store.list_runs() == []
    assert store._conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0


def test_in_memory_store():
    with ResultStore(":memory:") as s:
        s.save(make_run("r1", 0, [True]))
        assert s.load_run("r1").pass_rate == 1.0
