"""LexicalEntityLinker: links a question to seed entities by matching the question's contiguous
word-spans against entity names (exact normalized match), scored by span length, with maximal-span
dedup and a content-token guard. Implements the SeedProvider seam (engine/graph/retriever.py).
The embedding-based linker (Phase 2) plugs into the same seam."""

from __future__ import annotations

import re
from collections import defaultdict

from engine.ports.graph_store import GraphStore

_WORD = re.compile(r"[a-z0-9]+")

# Stop / query-intent words that must not, alone, anchor a seed. A title made entirely of these
# (e.g. "The Theme") is not seedable. Deliberately excludes short content words so one-word titles
# (It / Us / Up) stay seedable.
_NON_CONTENT = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "with",
        "by",
        "at",
        "as",
        "is",
        "are",
        "what",
        "who",
        "which",
        "that",
        "this",
        "these",
        "those",
        "theme",
        "themes",
        "plot",
        "plots",
        "story",
        "stories",
        "about",
        "common",
        "motif",
        "motifs",
        "character",
        "characters",
        "relationship",
        "relationships",
        "related",
        "connect",
        "connection",
        "connects",
        "actor",
        "actors",
        "cast",
        "director",
        "directors",
        "directed",
        "between",
        "share",
        "shared",
        # common auxiliary / high-frequency verbs that should not anchor a 1-token seed
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "be",
        "been",
        "being",
        "get",
        "got",
        "make",
        "made",
        "go",
        "goes",
        "went",
        "see",
        "saw",
        "say",
        "said",
        "want",
        "tell",
        "give",
    }
)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class LexicalEntityLinker:
    def __init__(
        self,
        store: GraphStore,
        *,
        attribute_relations: frozenset[str] = frozenset(),
        top_k: int = 10,
        max_ngram: int = 6,
    ):
        non_seedable = {t.obj for t in store.triples() if t.relation in attribute_relations}
        self._by_norm: dict[str, list[str]] = defaultdict(list)
        for e in store.entities():
            if e in non_seedable:
                continue
            toks = _tokens(e)
            if not toks or all(t in _NON_CONTENT for t in toks):
                continue
            self._by_norm[" ".join(toks)].append(e)
        self._top_k = top_k
        self._max_ngram = max_ngram

    def seeds(self, question: str) -> list[str]:
        q = _tokens(question)
        n = len(q)
        if n == 0:
            return []
        matches: list[tuple[int, int, str, int]] = []
        for size in range(min(n, self._max_ngram), 0, -1):
            for i in range(0, n - size + 1):
                span = " ".join(q[i : i + size])
                if size == 1 and span in _NON_CONTENT:
                    continue
                for e in self._by_norm.get(span, ()):
                    matches.append((i, i + size, e, size))
        kept: list[tuple[str, int]] = []
        for s, e_, ent, size in matches:
            if any(s2 <= s and e_ <= e2 and (e2 - s2) > size for s2, e2, _o, _sz in matches):
                continue
            kept.append((ent, size))
        ranked: list[str] = []
        seen: set[str] = set()
        for ent, _size in sorted(kept, key=lambda x: (-x[1], len(x[0]), x[0])):
            if ent not in seen:
                seen.add(ent)
                ranked.append(ent)
        return ranked[: self._top_k]
