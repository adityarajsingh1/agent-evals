import json
import threading
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agent_evals.dataset import load_dataset
from agent_evals.feedback import Feedback, FeedbackLog, case_id_for, to_test_cases, write_cases
from agent_evals.models import AgentResult, ExpectedToolCall, Source, Step, TestCase, ToolCall, Trace

T0 = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)


def fb(input, kind, minutes=0, **kw):
    return Feedback(input=input, kind=kind, created_at=T0 + timedelta(minutes=minutes), **kw)


def test_a_correction_needs_the_right_answer():
    with pytest.raises(ValidationError, match="corrected answer"):
        Feedback(input="q", kind="correction")
    with pytest.raises(ValidationError):
        Feedback(input="q", kind="angry")


def test_log_round_trip(tmp_path):
    log = FeedbackLog(tmp_path / "logs" / "feedback.jsonl")
    assert log.read() == []
    log.record("What's the refund window?", "correction", output="14 days", correction="30 days")
    trace = Trace(steps=[Step(type="tool", tool_call=ToolCall(name="search", error="timeout"))])
    log.record_result("Track order 42", AgentResult(error="TimeoutError: slow", trace=trace))
    log.record_result("Hi", AgentResult(output="Bye"), comment="rude")

    items = log.read()
    assert [f.kind for f in items] == ["correction", "error", "thumbs_down"]
    assert items[1].trace == trace
    assert items[2].comment == "rude"
    # defaults aren't written, so lines stay short
    first = json.loads((tmp_path / "logs" / "feedback.jsonl").read_text().splitlines()[0])
    assert "sources" not in first


def test_concurrent_writers_dont_mix_lines(tmp_path):
    log = FeedbackLog(tmp_path / "f.jsonl")

    def write(i):
        for j in range(25):
            log.record(f"question {i}-{j} " + "x" * 500, "thumbs_down")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.read()) == 200


def test_bad_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"input": "q", "kind": "error"}\n{"input": "q"}\n')
    with pytest.raises(ValueError, match="f.jsonl:2"):
        FeedbackLog(path).read()


def test_corrections_become_ready_cases_and_the_rest_drafts():
    out = to_test_cases(
        [
            fb("What's the refund window?", "thumbs_down", 0, output="14 days"),
            fb("what's the refund window", "correction", 5, output="14 days", correction="Thirty days"),
            fb("What's the refund window?", "correction", 9, output="14 days", correction="30 days"),
            fb("Where is order 42?", "error", 1, error="TimeoutError: slow"),
            fb("Do you ship to Norway?", "thumbs_down", 2, output="No.", comment="we do"),
            fb("Do you ship to Norway?", "thumbs_down", 3, output="No."),
        ]
    )
    [ready] = out.ready
    assert ready.expected_output == "30 days"  # the newest correction wins
    assert ready.input == "What's the refund window?"
    assert ready.metadata["reports"] == 3
    assert ready.metadata["bad_output"] == "14 days"
    assert ready.metadata["first_seen"] == T0.isoformat()
    assert "from-feedback" in ready.tags and "correction" in ready.tags

    # the most reported draft first
    assert [d.input for d in out.drafts] == ["Do you ship to Norway?", "Where is order 42?"]
    assert out.drafts[1].metadata["error"] == "TimeoutError: slow"
    assert all(d.expected_output is None for d in out.drafts)


def test_expected_tools_make_a_case_ready_and_sources_are_kept():
    src = Source(id="policy", text="Refunds within 30 days.")
    out = to_test_cases(
        [
            fb("Where is order 42?", "error", 0, sources=[src]),
            fb("Where is order 42?", "thumbs_down", 1, expected_tools=[ExpectedToolCall(name="track_order")]),
        ]
    )
    [case] = out.ready
    assert case.expected_tools[0].name == "track_order"
    assert case.sources == [src]


def test_inputs_already_in_the_dataset_are_skipped():
    existing = [TestCase(id="refund", input="What is the refund window")]
    out = to_test_cases([fb("What is the refund window?", "correction", correction="30 days")], existing)
    assert out.ready == [] and out.duplicates == [case_id_for("What is the refund window?")]


def test_ids_are_stable():
    assert case_id_for("Where is order 42?") == case_id_for("where is order 42")
    assert case_id_for("a") != case_id_for("b")


def test_written_cases_load_as_a_dataset(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "old", "input": "hi", "expected_output": "hello"}')  # no trailing newline
    out = to_test_cases([fb("What's the refund window?", "correction", correction="30 days")])
    assert write_cases(path, out.ready) == 1
    cases = load_dataset(path)
    assert [c.id for c in cases] == ["old", out.ready[0].id]
    assert cases[1].metadata["reports"] == 1
