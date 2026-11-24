"""
Checking that an answer sticks to its sources.

An answer is split into claims (roughly, its sentences), and a `ClaimChecker` decides for each
claim whether some source supports it. The share of supported claims is the groundedness score;
the unsupported claims are what to look at, because they're where the agent made something up.

The built-in `LexicalChecker` needs no model. It counts a claim as supported when a source
passage contains most of the claim's content words and *every* number in it. That catches the
common hallucinations in agent answers (an invented figure, date, name or product) and runs in
CI for free. It doesn't understand paraphrase or subtle contradictions; for those, plug in a
model-based checker.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agent_evals.models import Source
from agent_evals.text import extract_numbers

# words that carry no claim on their own
_STOPWORD_LIST = """
a an the and or but if then so of to in on at by for with from as into about over after before
is are was were be been being am do does did has have had having will would can could should may might
must shall it its this that these those there here they them their he she his her him we our us you your
i me my mine not no nor also just very really quite than too more most much many some any all each
which who whom whose what when where why how one ones get got
"""
_STOPWORDS = frozenset(_STOPWORD_LIST.split())
_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
# sentences that make no factual claim
_FILLER = re.compile(
    r"^(sure|okay|ok|of course|certainly|great question|happy to help|i hope (this|that) helps|"
    r"let me know|feel free|is there anything else|thanks|thank you)\b",
    re.IGNORECASE,
)


def split_claims(text: str) -> list[str]:
    """Split an answer into checkable claims: its sentences and list items, minus questions and filler."""
    claims = []
    for part in _SENTENCE_END.split(text):
        part = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", part).strip()  # list markers
        if not part or part.endswith("?") or _FILLER.match(part):
            continue
        if content_words(part) or extract_numbers(part):
            claims.append(part)
    return claims


def _stem(word: str) -> str:
    # crude, but enough to match "flights"/"flight" and "berlin's"/"berlin"
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and not w.isdigit()}


@dataclass
class Verdict:
    claim: str
    supported: bool
    score: float  # how well the best passage covers the claim, 0-1
    source_id: str | None = None
    evidence: str | None = None  # the passage that supports it, or the closest one
    reason: str = ""


class ClaimChecker(Protocol):
    def check(self, claim: str, sources: Sequence[Source]) -> Verdict: ...


def passages(source: Source) -> list[str]:
    """Single sentences plus pairs of neighbours, since one claim often draws on two sentences."""
    sents = [s.strip() for s in _SENTENCE_END.split(source.text) if s.strip()]
    return sents + [f"{a} {b}" for a, b in zip(sents, sents[1:], strict=False)]


class LexicalChecker:
    def __init__(self, min_coverage: float = 0.7) -> None:
        """`min_coverage` is the share of a claim's content words a passage must contain."""
        self.min_coverage = min_coverage

    def check(self, claim: str, sources: Sequence[Source]) -> Verdict:
        words = content_words(claim)
        numbers = extract_numbers(claim)
        best = Verdict(claim, False, 0.0, reason="no sources")
        for source in sources:
            for passage in passages(source):
                have = content_words(passage)
                coverage = len(words & have) / len(words) if words else 1.0
                passage_numbers = extract_numbers(passage)
                missing_numbers = [n for n in numbers if not any(abs(n - p) < 1e-9 for p in passage_numbers)]
                supported = coverage >= self.min_coverage and not missing_numbers
                # numbers that don't match pull the score down hard: a wrong figure is the
                # most damaging kind of hallucination
                score = coverage * (0.5 ** len(missing_numbers))
                if (supported, score) > (best.supported, best.score):
                    if missing_numbers:
                        reason = "number not in sources: " + ", ".join(f"{n:g}" for n in missing_numbers)
                    elif not supported:
                        missing = sorted(words - have)[:5]
                        reason = "not found in sources: " + ", ".join(missing)
                    else:
                        reason = ""
                    best = Verdict(claim, supported, score, source.id, passage, reason)
                    if supported and coverage == 1.0:
                        return best
        return best


@dataclass
class GroundingReport:
    verdicts: list[Verdict]

    @property
    def groundedness(self) -> float:
        """Share of claims supported. An answer with no claims has nothing ungrounded, so it's 1.0."""
        if not self.verdicts:
            return 1.0
        return sum(v.supported for v in self.verdicts) / len(self.verdicts)

    @property
    def unsupported(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.supported]


def check_grounding(
    answer: str, sources: Sequence[Source], checker: ClaimChecker | None = None
) -> GroundingReport:
    checker = checker or LexicalChecker()
    return GroundingReport([checker.check(claim, sources) for claim in split_claims(answer)])
