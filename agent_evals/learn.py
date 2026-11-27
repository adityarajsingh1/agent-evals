"""
Learning from failures without making things worse.

The loop: run the current prompt, take the cases it fails, and build a candidate prompt from them
(their corrected answers as few-shot examples, and optionally a model rewriting the instructions
from the failure reasons). Then run both versions and only promote the candidate if it beats the
current one with no case regressing. Every version and every decision is kept on disk, so a bad
promotion can be traced and rolled back.

The one rule that matters: a candidate is judged on cases it hasn't seen. A case whose answer was
copied into the prompt as an example will pass, and counting it would make every candidate look
like an improvement. So cases used as examples are held out of the comparison.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from agent_evals.agents import Agent
from agent_evals.compare import Comparison, compare_runs
from agent_evals.llm import TextModel
from agent_evals.models import CaseResult, Run, TestCase, _Model
from agent_evals.report import failure_reason
from agent_evals.runner import Scorer, run_eval


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Example(_Model):
    """A worked example shown to the model: an input and the answer it should give."""

    input: str
    output: str
    # the test case it came from, so it can be held out of evaluation
    case_id: str | None = None


class PromptVersion(_Model):
    """One version of an agent's prompt: instructions plus few-shot examples."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    system: str
    examples: list[Example] = Field(default_factory=list)
    parent: str | None = None
    # cases whose answers this version (or an ancestor) has seen, e.g. shown to a model rewriting
    # the instructions; they stay out of evaluation for good
    seen_cases: list[str] = Field(default_factory=list)
    # how this version was made, e.g. "+2 examples from failures"
    note: str = ""
    created_at: datetime = Field(default_factory=_now)

    def messages(self, input: str) -> list[dict[str, str]]:
        """Chat messages for this prompt: the system prompt, each example as a user/assistant
        turn, then the input. The format the OpenAI-style and most local chat APIs take."""
        msgs = [{"role": "system", "content": self.system}]
        for ex in self.examples:
            msgs += [{"role": "user", "content": ex.input}, {"role": "assistant", "content": ex.output}]
        msgs.append({"role": "user", "content": input})
        return msgs

    def render(self, input: str) -> str:
        """The same prompt as one string, for models that take plain text."""
        parts = [self.system.strip()]
        if self.examples:
            parts.append("Examples:\n\n" + "\n\n".join(f"Q: {e.input}\nA: {e.output}" for e in self.examples))
        parts.append(f"Q: {input}\nA:")
        return "\n\n".join(parts)

    def held_out(self) -> set[str]:
        """Case ids this version can't be fairly tested on."""
        return set(self.seen_cases) | {e.case_id for e in self.examples if e.case_id}

    def with_examples(self, examples: Sequence[Example], note: str = "") -> PromptVersion:
        """A child version with these examples added (an input already in the prompt is replaced)."""
        new_inputs = {e.input for e in examples}
        kept = [e for e in self.examples if e.input not in new_inputs]
        return PromptVersion(
            system=self.system,
            examples=[*kept, *examples],
            parent=self.id,
            # a replaced example's case was still seen by the lineage
            seen_cases=sorted(self.held_out() - {e.case_id for e in examples if e.case_id}),
            note=note or f"+{len(examples)} example(s)",
        )


def examples_from_failures(
    run: Run, cases: Sequence[TestCase], limit: int = 4, exclude: set[str] | None = None
) -> list[Example]:
    """Few-shot examples from the cases `run` failed that say what the right answer is.

    Only cases with an `expected_output` can teach anything, so cases that only check tools or
    limits are skipped. Cases that came from production feedback and were reported most often
    go first, since they're the problems users actually hit.
    """
    by_id = {c.id: c for c in cases}
    exclude = exclude or set()
    failed = [
        by_id[r.case_id]
        for r in run.cases
        if not r.passed and r.case_id in by_id and r.case_id not in exclude
    ]
    teachable = [c for c in failed if c.expected_output]
    teachable.sort(key=lambda c: -int(c.metadata.get("reports", 0)))
    return [Example(input=c.input, output=c.expected_output or "", case_id=c.id) for c in teachable[:limit]]


class PromptRegistry:
    """Prompt versions in a directory: one JSON file per version, plus a log of promotions.

        prompts/
          versions/<id>.json
          history.jsonl      one line per decision, promoted or not
          CURRENT            the id of the version in use

    Plain files, so the whole thing can be committed next to the agent and reviewed in a PR.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        (self.path / "versions").mkdir(parents=True, exist_ok=True)

    def save(self, version: PromptVersion) -> PromptVersion:
        file = self.path / "versions" / f"{version.id}.json"
        file.write_text(version.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return version

    def get(self, version_id: str) -> PromptVersion:
        file = self.path / "versions" / f"{version_id}.json"
        if not file.exists():
            raise KeyError(f"no prompt version {version_id!r} in {self.path}")
        return PromptVersion.model_validate_json(file.read_text(encoding="utf-8"))

    def versions(self) -> list[PromptVersion]:
        found = [
            PromptVersion.model_validate_json(f.read_text()) for f in (self.path / "versions").glob("*.json")
        ]
        return sorted(found, key=lambda v: v.created_at)

    def current(self) -> PromptVersion | None:
        pointer = self.path / "CURRENT"
        return self.get(pointer.read_text().strip()) if pointer.exists() else None

    def set_current(self, version_id: str) -> None:
        self.get(version_id)  # must exist
        (self.path / "CURRENT").write_text(version_id + "\n")

    def init(self, system: str) -> PromptVersion:
        """Start the registry with a first version, if it doesn't have one yet."""
        current = self.current()
        if current is not None:
            return current
        first = self.save(PromptVersion(system=system, note="initial"))
        self.set_current(first.id)
        return first

    def log(self, entry: dict[str, Any]) -> None:
        with (self.path / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": _now().isoformat(), **entry}) + "\n")

    def history(self) -> list[dict[str, Any]]:
        file = self.path / "history.jsonl"
        if not file.exists():
            return []
        return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def rollback(self) -> PromptVersion:
        """Go back to the version that was current before the last promotion."""
        promoted = [h for h in self.history() if h.get("promoted")]
        if not promoted:
            raise ValueError("nothing has been promoted, so there's nothing to roll back")
        previous = promoted[-1]["current"]
        self.set_current(previous)
        self.log(
            {"rollback": True, "current": promoted[-1]["candidate"], "candidate": previous, "promoted": False}
        )
        return self.get(previous)


@dataclass
class Decision:
    """The outcome of pitting a candidate against the current version."""

    current: PromptVersion
    candidate: PromptVersion
    current_run: Run
    candidate_run: Run
    comparison: Comparison
    # case ids left out because the candidate has their answers as examples
    held_out: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # why it wasn't promoted; empty if it was

    @property
    def promoted(self) -> bool:
        return not self.reasons

    def summary(self) -> str:
        cmp = self.comparison
        verdict = "promoted" if self.promoted else "kept the current version"
        lines = [
            f"{self.current.id} -> {self.candidate.id} ({self.candidate.note}): {verdict}",
            f"  pass rate on {len(self.current_run.cases)} unseen case(s): "
            f"{self.current_run.pass_rate:.1%} -> {self.candidate_run.pass_rate:.1%}",
        ]
        if self.held_out:
            lines.append(f"  held out {len(self.held_out)} case(s) the prompt has seen")
        if cmp.fixes:
            lines.append("  fixed: " + ", ".join(c.case_id for c in cmp.fixes))
        if cmp.regressions:
            lines.append("  broke: " + ", ".join(c.case_id for c in cmp.regressions))
        lines += [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


async def evaluate_candidate(
    make_agent: Callable[[PromptVersion], Agent],
    current: PromptVersion,
    candidate: PromptVersion,
    cases: Sequence[TestCase],
    scorers: Sequence[Scorer] = (),
    *,
    min_gain: float = 0.0,
    max_regressions: int = 0,
    **run_kwargs: Any,
) -> Decision:
    """Run both versions on the cases the candidate hasn't seen and decide.

    The candidate is promoted only if its pass rate is more than `min_gain` above the current
    one's (a tie keeps the current version) and no more than `max_regressions` cases that passed
    now fail.
    """
    held_out = candidate.held_out() | current.held_out()
    unseen = [c for c in cases if c.id not in held_out]
    if not unseen:
        raise ValueError("the prompt has seen every case, so there's nothing left to evaluate on")
    base = await run_eval(make_agent(current), unseen, scorers, config={"prompt": current.id}, **run_kwargs)
    head = await run_eval(
        make_agent(candidate), unseen, scorers, config={"prompt": candidate.id}, **run_kwargs
    )
    cmp = compare_runs(base, head)

    reasons = cmp.gate(max_regressions=max_regressions)
    gain = head.pass_rate - base.pass_rate
    if gain <= min_gain:
        need = f"more than {min_gain:.1%} better" if min_gain else "better"
        reasons.append(f"not {need}: {base.pass_rate:.1%} -> {head.pass_rate:.1%}")
    return Decision(
        current=current,
        candidate=candidate,
        current_run=base,
        candidate_run=head,
        comparison=cmp,
        held_out=sorted(i for i in held_out if any(c.id == i for c in cases)),
        reasons=reasons,
    )


def record(registry: PromptRegistry, decision: Decision) -> None:
    """Save the candidate, log the decision, and make the candidate current if it won."""
    registry.save(decision.candidate)
    if decision.promoted:
        registry.set_current(decision.candidate.id)
    registry.log(
        {
            "current": decision.current.id,
            "candidate": decision.candidate.id,
            "promoted": decision.promoted,
            "note": decision.candidate.note,
            "pass_rate": [decision.current_run.pass_rate, decision.candidate_run.pass_rate],
            "fixed": [c.case_id for c in decision.comparison.fixes],
            "broke": [c.case_id for c in decision.comparison.regressions],
            "held_out": decision.held_out,
            "reasons": decision.reasons,
        }
    )


REWRITE_PROMPT = """You are improving the instructions (system prompt) of an AI agent.

Current instructions:
<instructions>
{system}
</instructions>

The agent got these cases wrong:

{failures}

Write new instructions that fix the general problem behind these failures without breaking what
already works. Keep everything in the current instructions that is still right. Don't copy the
specific questions or answers into the instructions: describe the rule instead.

Reply with only the new instructions inside <instructions></instructions> tags."""


class PromptRewriter:
    """Asks a model to rewrite the instructions from the failures, as a second way (besides
    few-shot examples) to make a candidate. The cases it was shown are held out afterwards."""

    def __init__(self, model: TextModel, max_failures: int = 8) -> None:
        self.model = model
        self.max_failures = max_failures

    def prompt(self, system: str, failures: Sequence[tuple[TestCase, CaseResult]]) -> str:
        blocks = []
        for i, (case, result) in enumerate(failures[: self.max_failures], start=1):
            lines = [f"{i}. Input: {case.input}", f"   Agent answered: {result.result.output or '(nothing)'}"]
            if case.expected_output:
                lines.append(f"   Expected: {case.expected_output}")
            lines.append(f"   Why it failed: {failure_reason(result)}")
            blocks.append("\n".join(lines))
        return REWRITE_PROMPT.format(system=system.strip(), failures="\n\n".join(blocks))

    def rewrite(self, system: str, failures: Sequence[tuple[TestCase, CaseResult]]) -> str:
        reply = self.model.complete(self.prompt(system, failures))
        match = re.search(r"<instructions>(.*?)</instructions>", reply, re.S)
        new = (match.group(1) if match else reply).strip()
        if not new:
            raise ValueError("the rewriter returned empty instructions")
        return new


class NothingToLearn(Exception):
    """The current version passes everything it can be tested on, or its failures can't teach a
    candidate anything (no expected answers and no rewriter)."""


async def improve(
    registry: PromptRegistry,
    make_agent: Callable[[PromptVersion], Agent],
    cases: Sequence[TestCase],
    scorers: Sequence[Scorer] = (),
    *,
    rewriter: PromptRewriter | None = None,
    max_examples: int = 4,
    min_gain: float = 0.0,
    max_regressions: int = 0,
    **run_kwargs: Any,
) -> Decision:
    """One turn of the loop: find what the current version fails, build a candidate from it,
    evaluate the two on unseen cases, and record the decision (promoting the candidate if it won).
    """
    current = registry.current()
    if current is None:
        raise ValueError(f"{registry.path} has no current prompt; start it with PromptRegistry.init()")
    unseen = [c for c in cases if c.id not in current.held_out()]
    run = await run_eval(make_agent(current), unseen, scorers, config={"prompt": current.id}, **run_kwargs)
    by_id = {c.id: c for c in unseen}
    failures = [(by_id[r.case_id], r) for r in run.cases if not r.passed]
    if not failures:
        raise NothingToLearn(f"{current.id} passes all {len(run.cases)} unseen case(s)")

    examples = examples_from_failures(run, unseen, limit=max_examples)
    if not examples and rewriter is None:
        raise NothingToLearn(
            f"{len(failures)} failing case(s), but none has an expected_output to learn from; "
            "pass a rewriter to learn from the failure reasons"
        )
    candidate = current.with_examples(examples) if examples else current.with_examples([])
    notes = [f"+{len(examples)} example(s) from failures"] if examples else []
    if rewriter is not None:
        shown = failures[: rewriter.max_failures]
        candidate.system = rewriter.rewrite(current.system, shown)
        candidate.seen_cases = sorted(set(candidate.seen_cases) | {c.id for c, _ in shown})
        notes.append(f"instructions rewritten from {len(shown)} failure(s)")
    candidate.note = ", ".join(notes)

    decision = await evaluate_candidate(
        make_agent,
        current,
        candidate,
        cases,
        scorers,
        min_gain=min_gain,
        max_regressions=max_regressions,
        **run_kwargs,
    )
    record(registry, decision)
    return decision
