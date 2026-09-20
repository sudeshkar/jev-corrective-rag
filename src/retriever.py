"""Chunking + BM25 retrieval.

Deliberately boring. The point of this project is the decision layer, not the
retriever, so this stays dependency-free and deterministic: no embedding API
key needed to run the demo, and no vector-store noise in the benchmark numbers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with what how when
    where which who why can do does my me i you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, crudely de-pluralize.

    The de-pluralization matters more than it looks: without it "429s" misses
    "429" and "locked" misses "locks", which is enough to sink a retrieval.
    """
    out = []
    for token in _WORD.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 4 and token.endswith(("ed", "ing")):
            token = token.rstrip("g").removesuffix("in").removesuffix("ed") or token
        out.append(token)
    return out


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str


def load_chunks(kb_dir: Path) -> list[Chunk]:
    """One chunk per paragraph. Paragraphs in these docs are already topical."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = "\n".join(lines[1:])
        for i, para in enumerate(p.strip() for p in body.split("\n\n")):
            if not para:
                continue
            chunks.append(
                Chunk(
                    doc_id=path.stem,
                    chunk_id=f"{path.stem}#{i}",
                    title=title,
                    text=" ".join(para.split()),
                )
            )
    return chunks


class BM25:
    """Okapi BM25. ~40 lines beats a badly-tuned vector search often enough."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(c.text + " " + c.title) for c in chunks]
        self._lens = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        self._tf = [Counter(t) for t in self._tokens]

        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = len(chunks)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for idx, chunk in enumerate(self.chunks):
            tf, length = self._tf[idx], self._lens[idx]
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                score += self._idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
