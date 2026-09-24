import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import Chunk, SearchResult

TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "who",
    "with",
}


def _normalize(token: str) -> str:
    """Apply a deliberately small stemmer so the baseline remains easy to inspect."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_normalize(token) for token in TOKEN.findall(text.lower()) if token not in STOP_WORDS]


class BM25Index:
    """A small, inspectable BM25 baseline suitable for an offline portfolio demo."""

    def __init__(self, chunks: Iterable[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("At least one chunk is required")
        self.k1 = k1
        self.b = b
        self._terms = [Counter(tokenize(f"{chunk.heading} {chunk.text}")) for chunk in self.chunks]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            document_frequency.update(terms.keys())
        size = len(self.chunks)
        self._idf = {
            term: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        scored: list[SearchResult] = []
        for chunk, terms, length in zip(self.chunks, self._terms, self._lengths, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._average_length
                )
                score += self._idf[term] * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:limit]
