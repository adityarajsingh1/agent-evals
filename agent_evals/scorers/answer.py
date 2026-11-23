"""Scorers that compare the agent's final answer with `case.expected_output`."""

from __future__ import annotations

import math
import re
import string
import unicodedata

from agent_evals.models import AgentResult, Score, TestCase
from agent_evals.scorers.base import BaseScorer, register

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})
# 1,234.5 / -3 / .5 / 1e-3; thousands separators only in groups of three
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|\.\d)")


def normalize(text: str) -> str:
    """Lowercase, strip accents, punctuation and articles, collapse whitespace.

    The usual answer normalisation from QA benchmarks (SQuAD-style), so "The Eiffel Tower." and
    "eiffel tower" compare equal.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = _ARTICLES.sub(" ", text.translate(_PUNCT))
    return " ".join(text.split())


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for m in _NUMBER.finditer(text):
        raw = m.group().replace(",", "")
        if raw in {"", "+", "-", "."}:
            continue
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


class _AnswerScorer(BaseScorer):
    def applies_to(self, case: TestCase) -> bool:
        return case.expected_output is not None

    def score(self, case: TestCase, result: AgentResult) -> Score:
        if case.expected_output is None:
            return self.result(0.0, False, "case has no expected_output")
        if not result.output:
            return self.result(0.0, False, "agent gave no answer")
        return self.compare(case.expected_output, result.output)

    def compare(self, expected: str, output: str) -> Score:
        raise NotImplementedError


@register
class ExactMatch(_AnswerScorer):
    """The whole answer equals the expected one, after `normalize` (unless `normalized=False`)."""

    name = "exact"

    def __init__(self, normalized: bool = True) -> None:
        self.normalized = normalized

    def compare(self, expected: str, output: str) -> Score:
        a, b = (normalize(expected), normalize(output)) if self.normalized else (expected, output)
        if a == b:
            return self.result(1.0, True)
        return self.result(0.0, False, f"expected {expected!r}, got {output!r}")


@register
class Contains(_AnswerScorer):
    """The expected answer appears in the agent's answer as whole words, e.g. "Lisbon" in a full sentence."""

    name = "contains"

    def compare(self, expected: str, output: str) -> Score:
        needle, hay = normalize(expected), normalize(output)
        if needle and re.search(rf"(?<!\S){re.escape(needle)}(?!\S)", hay):
            return self.result(1.0, True)
        return self.result(0.0, False, f"{expected!r} not found in the answer")


@register
class Numeric(_AnswerScorer):
    """Every number in the expected answer appears in the agent's answer, within a tolerance.

    Handy for "how much / how many" questions where the answer is a sentence: expected "1234.5"
    matches "That comes to about 1,234.50 EUR." The score is the fraction of expected numbers found.
    """

    name = "numeric"

    def __init__(self, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> None:
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def compare(self, expected: str, output: str) -> Score:
        want = extract_numbers(expected)
        if not want:
            return self.result(0.0, False, f"no number in expected_output {expected!r}")
        got = extract_numbers(output)
        missing = [
            w
            for w in want
            if not any(math.isclose(w, g, rel_tol=self.rel_tol, abs_tol=self.abs_tol) for g in got)
        ]
        found = len(want) - len(missing)
        if not missing:
            return self.result(1.0, True)
        shown = ", ".join(f"{m:g}" for m in missing)
        return self.result(found / len(want), False, f"missing {shown} (answer had {got or 'no numbers'})")
