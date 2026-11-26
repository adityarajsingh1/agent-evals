from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from agent_evals.models import AgentResult, CaseResult, Run, Score, Step, ToolCall, Trace  # noqa: E402
from agent_evals.store import ResultStore  # noqa: E402

APP = str(Path(__file__).parent.parent / "agent_evals" / "dashboard" / "app.py")
T0 = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)
TRACE = Trace(
    steps=[
        Step(type="llm", model="m", input_tokens=10, output_tokens=5, cost_usd=0.001),
        Step(type="tool", tool_call=ToolCall(name="search", arguments={"q": "refunds"}, result="30 days")),
    ]
)


def make_run(rid: str, hours: int, refund_passes: bool) -> Run:
    def case(cid: str, passed: bool) -> CaseResult:
        reason = "" if passed else "expected '30 days' in the answer"
        return CaseResult(
            case_id=cid,
            result=AgentResult(output="Refunds take 30 days." if passed else "No idea.", trace=TRACE),
            scores=[Score(name="contains", value=float(passed), passed=passed, reason=reason)],
        )

    return Run(
        id=rid,
        agent="bot",
        dataset="d.jsonl",
        git_commit=f"{rid}commit",
        started_at=T0 + timedelta(hours=hours),
        cases=[case("refund", refund_passes), case("greeting", True)],
    )


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "r.sqlite"
    with ResultStore(path) as s:
        s.save(make_run("r1", 0, True))
        s.save(make_run("r2", 1, False))
        s.save(make_run("r3", 2, True))
        s.save(make_run("r4", 3, False))
    monkeypatch.setenv("AGENT_EVALS_DB", str(path))
    return path


def test_empty_store_says_how_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_DB", str(tmp_path / "empty.sqlite"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "No runs" in at.info[0].value


def test_trends_view(db):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Pass rate"] == "50.0%"
    assert metrics["Runs"] == "4"
    # no groundedness scores, so no hallucination metric
    assert "Hallucination rate" not in metrics


def test_run_view_shows_failing_cases_with_reasons(db):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Run").run()
    assert not at.exception
    assert at.header[0].value == "Run r4"
    assert [e.label for e in at.expander] == ["❌ refund"]
    assert "30 days" in at.error[0].value

    at.toggle[0].set_value(False).run()
    assert len(at.expander) == 2


def test_run_view_when_everything_passed(db):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Run").run()
    at.sidebar.selectbox[1].set_value("r3").run()
    assert not at.exception
    assert "Every case passed" in at.success[0].value


def test_case_view_flags_flaky_cases(db):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Case").run()
    at.sidebar.selectbox[1].set_value("refund").run()
    assert not at.exception
    assert at.header[0].value == "Case refund"
    assert "Passed 2 of the last 4 runs" in at.caption[0].value
    assert "flaky" in at.warning[0].value
