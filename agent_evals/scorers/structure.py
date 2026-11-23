"""Scorers that check the shape of an answer rather than its content: a regex, or a JSON schema.

Both take what to check from their options (the same rule for every case) or from the case's
`metadata` (`"regex"` / `"json_schema"`), so a dataset can mix cases with different formats.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.scorers.base import BaseScorer, register

_FENCED = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


@register
class Regex(BaseScorer):
    """The answer matches a regular expression (searched, not anchored; use ^...$ for a full match)."""

    name = "regex"

    def __init__(self, pattern: str | None = None, ignore_case: bool = False) -> None:
        self.pattern = pattern
        self.ignore_case = ignore_case

    def _pattern(self, case: TestCase) -> str | None:
        return self.pattern if self.pattern is not None else case.metadata.get("regex")

    def applies_to(self, case: TestCase) -> bool:
        return self._pattern(case) is not None

    def score(self, case: TestCase, result: AgentResult) -> Score:
        pattern = self._pattern(case)
        if pattern is None:
            return self.result(0.0, False, "no pattern given")
        flags = re.IGNORECASE if self.ignore_case else 0
        if re.search(pattern, result.output or "", flags):
            return self.result(1.0, True)
        return self.result(0.0, False, f"answer doesn't match /{pattern}/")


def extract_json(text: str) -> Any:
    """Parse JSON from an answer: the whole text, a ```json fenced block, or the first {...} / [...]."""
    candidates = [text.strip()]
    candidates += [m.group(1) for m in _FENCED.finditer(text)]
    for open_, close in ("{}", "[]"):
        start, end = text.find(open_), text.rfind(close)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    raise ValueError("no JSON found in the answer")


@register
class JsonSchema(BaseScorer):
    """The answer contains JSON that validates against a schema. Needs `pip install agent-evals[schema]`."""

    name = "json_schema"

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema

    def _schema(self, case: TestCase) -> dict[str, Any] | None:
        return self.schema if self.schema is not None else case.metadata.get("json_schema")

    def applies_to(self, case: TestCase) -> bool:
        return self._schema(case) is not None

    def score(self, case: TestCase, result: AgentResult) -> Score:
        try:
            import jsonschema
        except ImportError as e:
            raise ImportError("the json_schema scorer needs `pip install agent-evals[schema]`") from e

        schema = self._schema(case)
        if schema is None:
            return self.result(0.0, False, "no schema given")
        try:
            data = extract_json(result.output or "")
        except ValueError as e:
            return self.result(0.0, False, str(e))
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda err: list(err.absolute_path))
        if not errors:
            return self.result(1.0, True)
        first = errors[0]
        where = "/".join(map(str, first.absolute_path)) or "(root)"
        more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        return self.result(0.0, False, f"{where}: {first.message}{more}")
