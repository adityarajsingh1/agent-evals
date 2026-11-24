"""Text helpers shared by the scorers: answer normalisation and number extraction."""

from __future__ import annotations

import re
import string
import unicodedata

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
