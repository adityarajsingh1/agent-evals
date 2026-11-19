"""
The data agent-evals works with.

A `TestCase` is one input plus what a good outcome looks like. Running an agent on it produces
an `AgentResult`: the final answer and a `Trace` of what the agent did on the way (model calls,
tool calls, tokens, cost). Scorers turn that into `Score`s, which are collected per case in a
`CaseResult` and per run in a `Run`.

Everything is a pydantic model, so test cases load from JSON/JSONL and runs can be stored and
compared as plain JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Model(BaseModel):
    # typos in a dataset ("expected_toools") should fail loudly, not be ignored
    model_config = ConfigDict(extra="forbid")


# --- test cases -------------------------------------------------------------------------------


class Source(_Model):
    """A document the agent is allowed to base its answer on (used for groundedness checks)."""

    id: str
    text: str
    title: str | None = None
    url: str | None = None


class ExpectedToolCall(_Model):
    """A tool call a good trajectory should contain.

    `arguments` only lists the arguments that matter: a call matches if it has these keys with
    these values, and any extra arguments the agent passed are ignored.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TestCase(_Model):
    """One input to the agent and what a good outcome looks like.

    Every expectation is optional, so a case can check only the answer, only the tools, or only
    that the agent stays within budget.
    """

    __test__ = False  # stop pytest from collecting this class because of its name

    id: str
    input: str
    expected_output: str | None = None
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    # True means the order of expected_tools matters
    tools_in_order: bool = False
    sources: list[Source] = Field(default_factory=list)
    max_steps: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- what the agent did -----------------------------------------------------------------------


class ToolCall(_Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class Step(_Model):
    """One thing the agent did: a model call or a tool call."""

    type: Literal["llm", "tool"]
    # for "llm" steps: the text the model produced, if any
    content: str | None = None
    # for "tool" steps
    tool_call: ToolCall | None = None
    model: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _tool_steps_have_a_call(self) -> Step:
        if self.type == "tool" and self.tool_call is None:
            raise ValueError("a 'tool' step needs a tool_call")
        if self.type == "llm" and self.tool_call is not None:
            raise ValueError("an 'llm' step can't have a tool_call; record the call as its own 'tool' step")
        return self


class Trace(_Model):
    steps: list[Step] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)


class AgentResult(_Model):
    """What came back from running the agent on one test case."""

    output: str | None = None
    trace: Trace = Field(default_factory=Trace)
    # set when the agent raised or timed out; output is then usually None
    error: str | None = None


# --- scores and runs --------------------------------------------------------------------------


class Score(_Model):
    """One scorer's verdict on one case. `value` is 0-1 so different scorers can be averaged."""

    name: str
    value: float = Field(ge=0, le=1)
    passed: bool
    reason: str = ""


class CaseResult(_Model):
    case_id: str
    result: AgentResult
    scores: list[Score] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        # a case with no scores hasn't passed anything; an agent error always fails it
        return self.result.error is None and bool(self.scores) and all(s.passed for s in self.scores)

    def score(self, name: str) -> Score | None:
        return next((s for s in self.scores if s.name == name), None)


class Run(_Model):
    """One evaluation run: an agent at a given version over a dataset."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str
    dataset: str
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    # the git commit of the agent under test, so a regression can be traced to a change
    git_commit: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    def mean_score(self, name: str) -> float | None:
        """Average of one scorer over the cases it scored, or None if it scored none."""
        values = [s.value for c in self.cases if (s := c.score(name)) is not None]
        return sum(values) / len(values) if values else None

    @property
    def cost_usd(self) -> float:
        return sum(c.result.trace.cost_usd for c in self.cases)
