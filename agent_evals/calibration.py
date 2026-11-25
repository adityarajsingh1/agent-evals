"""
How far to trust a judge: run it on answers people have already labelled, and measure agreement.

A judge that passes answers a person would fail is worse than no judge, because it makes a
broken agent look fine. So besides plain accuracy this reports Cohen's kappa (agreement beyond
chance, which matters when most answers are good) and the two error rates separately:

- false pass rate: of the answers people failed, how many the judge passed
- false fail rate: of the answers people passed, how many the judge failed

Labelled examples are JSONL, one per line:

    {"case": {"id": "r1", "input": "..."}, "output": "the agent's answer", "human_passed": false,
     "human_score": 0.2, "note": "says 60 days, policy says 30"}

`output` is a shortcut for `"result": {"output": ...}`; a full `result` with a trace also works.
`human_score` (0-1) is optional and adds the mean absolute error between the scores.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.runner import Scorer


class LabelledExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: TestCase
    result: AgentResult
    human_passed: bool
    human_score: float | None = Field(default=None, ge=0, le=1)
    note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _output_shortcut(cls, data: Any) -> Any:
        if isinstance(data, dict) and "output" in data:
            data = dict(data)
            if "result" in data:
                raise ValueError("give either 'output' or 'result', not both")
            data["result"] = {"output": data.pop("output")}
        return data


def load_labels(path: str | Path) -> list[LabelledExample]:
    examples = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            examples.append(LabelledExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
    return examples


@dataclass
class Disagreement:
    case_id: str
    human_passed: bool
    judge: Score
    note: str = ""


@dataclass
class AgreementReport:
    # "positive" = the answer passed
    both_pass: int = 0
    both_fail: int = 0
    false_pass: int = 0  # judge passed, human failed
    false_fail: int = 0  # judge failed, human passed
    abs_errors: list[float] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.both_pass + self.both_fail + self.false_pass + self.false_fail

    @property
    def accuracy(self) -> float:
        return (self.both_pass + self.both_fail) / self.n if self.n else 0.0

    @property
    def kappa(self) -> float:
        """Cohen's kappa: 1 is perfect agreement, 0 is what guessing with the same pass rates gives."""
        if not self.n:
            return 0.0
        judge_pass = (self.both_pass + self.false_pass) / self.n
        human_pass = (self.both_pass + self.false_fail) / self.n
        chance = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
        if chance == 1.0:
            return 1.0 if self.accuracy == 1.0 else 0.0
        return (self.accuracy - chance) / (1 - chance)

    @property
    def false_pass_rate(self) -> float | None:
        human_failed = self.both_fail + self.false_pass
        return self.false_pass / human_failed if human_failed else None

    @property
    def false_fail_rate(self) -> float | None:
        human_passed = self.both_pass + self.false_fail
        return self.false_fail / human_passed if human_passed else None

    @property
    def mean_abs_error(self) -> float | None:
        return sum(self.abs_errors) / len(self.abs_errors) if self.abs_errors else None

    def trustworthy(self, min_kappa: float = 0.6, max_false_pass_rate: float = 0.1) -> bool:
        """A rule of thumb before gating CI on a judge: substantial agreement, few bad answers let through."""
        fpr = self.false_pass_rate
        return self.kappa >= min_kappa and (fpr is None or fpr <= max_false_pass_rate)

    def summary(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.0%}"

        lines = [
            f"{self.n} labelled answers: accuracy {pct(self.accuracy)}, kappa {self.kappa:.2f}",
            f"false pass rate {pct(self.false_pass_rate)} ({self.false_pass} bad answers passed), "
            f"false fail rate {pct(self.false_fail_rate)} ({self.false_fail} good answers failed)",
        ]
        if self.mean_abs_error is not None:
            lines.append(f"mean absolute score error {self.mean_abs_error:.2f}")
        return "\n".join(lines)


def calibrate(scorer: Scorer, examples: Sequence[LabelledExample]) -> AgreementReport:
    """Score each labelled answer with `scorer` (sync) and compare with the human verdicts."""
    report = AgreementReport()
    for ex in examples:
        score = scorer(ex.case, ex.result)
        if not isinstance(score, Score):
            raise TypeError("calibrate needs a sync scorer that returns a Score")
        if score.passed and ex.human_passed:
            report.both_pass += 1
        elif not score.passed and not ex.human_passed:
            report.both_fail += 1
        elif score.passed:
            report.false_pass += 1
        else:
            report.false_fail += 1
        if score.passed != ex.human_passed:
            report.disagreements.append(Disagreement(ex.case.id, ex.human_passed, score, ex.note))
        if ex.human_score is not None:
            report.abs_errors.append(abs(score.value - ex.human_score))
    return report
