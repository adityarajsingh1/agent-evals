"""
Capturing failures from production and turning them into test cases.

The cases that matter most are the ones real users hit. So the agent in production logs what went
wrong: a thumbs-down, a user correcting the answer, or the agent crashing. Each one is a line in a
JSONL file (append-only, so several workers can write to it and nothing needs a server)::

    log = FeedbackLog("feedback.jsonl")
    log.record("What's the refund window?", output=answer, kind="correction", correction="30 days")

Later, `to_test_cases` turns the log into new test cases. A correction already says what the right
answer is, so it becomes a ready case. A thumbs-down or a crash only says the answer was wrong, so
it becomes a draft that a person has to finish before it can be scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agent_evals.models import AgentResult, ExpectedToolCall, Source, TestCase, Trace, _Model
from agent_evals.text import normalize

Kind = Literal["thumbs_down", "correction", "error"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Feedback(_Model):
    """One thing that went wrong in production."""

    input: str
    kind: Kind
    output: str | None = None
    # what the answer should have been, for kind="correction"
    correction: str | None = None
    # free text from the user or whoever triaged it
    comment: str = ""
    error: str | None = None
    trace: Trace | None = None
    # the documents the agent was given, so the case can check groundedness too
    sources: list[Source] = Field(default_factory=list)
    # tool calls a reviewer says should have happened
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _corrections_say_what_is_right(self) -> Feedback:
        if self.kind == "correction" and not (self.correction and self.correction.strip()):
            raise ValueError("a 'correction' needs the corrected answer in `correction`")
        return self


class FeedbackLog:
    """An append-only JSONL file of `Feedback`. Safe to share between threads of one process; lines
    from several processes don't interleave because each record is a single small append."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, feedback: Feedback) -> Feedback:
        line = feedback.model_dump_json(exclude_defaults=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # one write call per record, opened in append mode, so concurrent writers don't mix lines
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        return feedback

    def record(self, input: str, kind: Kind, **fields: Any) -> Feedback:
        return self.add(Feedback(input=input, kind=kind, **fields))

    def record_result(
        self, input: str, result: AgentResult, kind: Kind | None = None, **fields: Any
    ) -> Feedback:
        """Log an `AgentResult` from production. With no `kind`, an errored result is logged as
        an error and anything else as a thumbs-down."""
        if kind is None:
            kind = "error" if result.error else "thumbs_down"
        return self.record(
            input, kind, output=result.output, error=result.error, trace=result.trace, **fields
        )

    def __iter__(self) -> Iterator[Feedback]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield Feedback.model_validate_json(line)
                except ValueError as e:
                    raise ValueError(f"{self.path}:{lineno}: bad feedback record ({e})") from e

    def read(self) -> list[Feedback]:
        return list(self)


def case_id_for(text: str) -> str:
    """A stable id from the input, so the same question asked twice maps to one case."""
    return "fb-" + hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:10]


@dataclass
class Export:
    """The result of `to_test_cases`."""

    # cases that can be scored as they are
    ready: list[TestCase] = field(default_factory=list)
    # cases with no expectation yet; someone has to fill in expected_output or expected_tools
    drafts: list[TestCase] = field(default_factory=list)
    # ids skipped because the dataset already has a case with the same input
    duplicates: list[str] = field(default_factory=list)


def _to_case(fb: Feedback, case_id: str, count: int) -> TestCase:
    tags = list(dict.fromkeys(["from-feedback", fb.kind, *fb.tags]))
    meta: dict[str, Any] = {
        "feedback_kind": fb.kind,
        "first_seen": fb.created_at.isoformat(),
        "reports": count,
    }
    if fb.output is not None:
        meta["bad_output"] = fb.output
    if fb.error:
        meta["error"] = fb.error
    if fb.comment:
        meta["comment"] = fb.comment
    if fb.metadata:
        meta["production"] = fb.metadata
    return TestCase(
        id=case_id,
        input=fb.input,
        expected_output=fb.correction,
        expected_tools=fb.expected_tools,
        sources=fb.sources,
        tags=tags,
        metadata=meta,
    )


def to_test_cases(feedback: Iterable[Feedback], existing: Iterable[TestCase] = ()) -> Export:
    """Turn feedback into test cases, one per distinct input.

    Feedback on the same input (after normalising case and punctuation) is merged: the newest
    correction wins, and `metadata.reports` counts how many times it came up, which is a fair
    guide to which drafts to finish first. Inputs the existing dataset already covers are skipped.
    """
    known = {normalize(c.input) for c in existing}
    groups: dict[str, list[Feedback]] = {}
    for fb in feedback:
        groups.setdefault(normalize(fb.input), []).append(fb)

    out = Export()
    for key, items in groups.items():
        items.sort(key=lambda f: f.created_at)
        case_id = case_id_for(items[0].input)
        if key in known:
            out.duplicates.append(case_id)
            continue
        corrections = [f for f in items if f.kind == "correction"]
        best = corrections[-1] if corrections else items[-1]
        # sources and expected tools from any report are worth keeping
        if not best.sources:
            best = best.model_copy(update={"sources": next((f.sources for f in items if f.sources), [])})
        if not best.expected_tools:
            tools = next((f.expected_tools for f in items if f.expected_tools), [])
            best = best.model_copy(update={"expected_tools": tools})
        case = _to_case(best, case_id, len(items))
        case.metadata["first_seen"] = items[0].created_at.isoformat()
        if case.expected_output is not None or case.expected_tools:
            out.ready.append(case)
        else:
            out.drafts.append(case)
    # the most reported problems first
    out.ready.sort(key=lambda c: -c.metadata["reports"])
    out.drafts.sort(key=lambda c: -c.metadata["reports"])
    return out


def write_cases(path: str | Path, cases: Iterable[TestCase], append: bool = True) -> int:
    """Write cases as JSONL, appending by default so new cases land at the end of a dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # don't glue the first new case onto a last line that has no newline
    needs_newline = (
        append and path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    )
    with path.open("a" if append else "w", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for case in cases:
            f.write(
                json.dumps(case.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False) + "\n"
            )
            n += 1
    return n
