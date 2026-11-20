"""Loading test cases from JSONL: one case per line, blank lines and `#` comments allowed."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agent_evals.models import TestCase


class DatasetError(ValueError):
    """A dataset file has a problem. The message names the file and line."""


def load_dataset(path: str | Path) -> list[TestCase]:
    path = Path(path)
    cases: list[TestCase] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path}:{lineno}"
            try:
                case = TestCase.model_validate(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{where}: not valid JSON ({e.msg})") from e
            except ValidationError as e:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'case'}: {err['msg']}" for err in e.errors()
                )
                raise DatasetError(f"{where}: {problems}") from e
            if case.id in seen:
                raise DatasetError(f"{where}: duplicate id {case.id!r} (first used on line {seen[case.id]})")
            seen[case.id] = lineno
            cases.append(case)
    if not cases:
        raise DatasetError(f"{path}: no test cases")
    return cases
