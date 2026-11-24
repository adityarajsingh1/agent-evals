"""
Checking that an answer sticks to its sources.

An answer is split into claims (roughly, its sentences), and a `ClaimChecker` decides for each
claim whether some source supports it. The share of supported claims is the groundedness score;
the unsupported claims are what to look at, because they're where the agent made something up.

The built-in `LexicalChecker` needs no model. It counts a claim as supported when a source
passage contains most of the claim's content words and *every* number in it. That catches the
common hallucinations in agent answers (an invented figure, date, name or product) and runs in
CI for free. It doesn't understand paraphrase or subtle contradictions; for those there's
`LLMChecker`, and `LLMClaimExtractor` splits answers into one-fact claims.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent_evals.llm import TextModel
from agent_evals.models import Source
from agent_evals.text import extract_numbers, normalize

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


def _parse_json(text: str) -> object:
    # imported here to keep grounding usable without pulling in the scorer modules
    from agent_evals.scorers.structure import extract_json

    return extract_json(text)


CHECK_PROMPT = """You are checking whether a claim is supported by the sources below.

Sources:
{sources}

Claim: {claim}

Answer with JSON only:
{{"verdict": "supported" | "contradicted" | "not_found",
  "source_id": "<id of the source that decides it, or null>",
  "quote": "<the exact sentence from that source, or null>"}}

"supported" means the sources state it (paraphrase is fine). "contradicted" means a source says
otherwise. "not_found" means the sources don't say. Don't use outside knowledge."""


class LLMChecker:
    """Asks a model whether the sources support a claim.

    The model has to quote the passage it relied on, and a quote that isn't actually in that source
    makes the verdict unsupported. That keeps the checker from hallucinating support itself.
    """

    def __init__(self, model: TextModel, prompt: str = CHECK_PROMPT) -> None:
        self.model = model
        self.prompt = prompt

    def check(self, claim: str, sources: Sequence[Source]) -> Verdict:
        listing = "\n\n".join(f"[{s.id}] {s.text}" for s in sources)
        reply = self.model.complete(self.prompt.format(sources=listing, claim=claim))
        try:
            data = _parse_json(reply)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            verdict = str(data.get("verdict", "")).lower()
        except ValueError:
            return Verdict(claim, False, 0.0, reason=f"couldn't read the checker's reply: {reply[:80]!r}")

        source_id = data.get("source_id") or None
        quote = data.get("quote") or None
        if verdict != "supported":
            label = {"contradicted": "contradicted by sources", "not_found": "not in sources"}
            return Verdict(claim, False, 0.0, source_id, quote, label.get(verdict, f"verdict {verdict!r}"))

        source = next((s for s in sources if s.id == source_id), None)
        if source is None:
            return Verdict(claim, False, 0.0, None, quote, f"checker cited unknown source {source_id!r}")
        if quote and normalize(quote) not in normalize(source.text):
            return Verdict(claim, False, 0.0, source.id, quote, "checker's quote isn't in the source")
        return Verdict(claim, True, 1.0, source.id, quote)


EXTRACT_PROMPT = """Break this answer into short, self-contained factual claims, one fact each.
Resolve pronouns ("it" -> what it refers to). Leave out greetings, questions, opinions and offers
to help. Reply with a JSON list of strings only.

Answer:
{answer}"""


class LLMClaimExtractor:
    """Splits an answer into atomic claims with a model, falling back to `split_claims` if the reply
    isn't a JSON list. Atomic claims make the score finer: "Refunds take 30 days and include
    shipping" is one sentence but two facts, one of which may be wrong."""

    def __init__(self, model: TextModel, prompt: str = EXTRACT_PROMPT) -> None:
        self.model = model
        self.prompt = prompt

    def __call__(self, answer: str) -> list[str]:
        reply = self.model.complete(self.prompt.format(answer=answer))
        try:
            claims = _parse_json(reply)
        except ValueError:
            return split_claims(answer)
        if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
            return split_claims(answer)
        return [c.strip() for c in claims if c.strip()]


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
    answer: str,
    sources: Sequence[Source],
    checker: ClaimChecker | None = None,
    extract: Callable[[str], list[str]] = split_claims,
) -> GroundingReport:
    checker = checker or LexicalChecker()
    return GroundingReport([checker.check(claim, sources) for claim in extract(answer)])
