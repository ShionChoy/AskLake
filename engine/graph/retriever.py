from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engine.ports.graph_store import GraphStore, Triple

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


# Generic query-structure tokens that must not, on their own, anchor a seed match: common
# stopwords plus the graph query-intent words (mirrors GraphRagPath._GRAPH_HINTS). A film whose
# title is made entirely of these (e.g. "The Theme", "The Plot") would otherwise seed off the
# question's phrasing rather than its subject. Deliberately conservative — it excludes no
# pronoun-like short words, so one-word titles such as "It"/"Us"/"Up" stay seedable.
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
    }
)


@dataclass(frozen=True)
class RetrievedSubgraph:
    seeds: tuple[str, ...]
    triples: tuple[Triple, ...]


@runtime_checkable
class SeedProvider(Protocol):
    """Maps a question to seed entity names. LexicalSeedProvider ships now;
    EmbeddingSeedProvider (Phase 2) plugs into the same seam."""

    def seeds(self, question: str) -> list[str]: ...


class LexicalSeedProvider:
    """Token-overlap seeding over an inverted index, excluding attribute-value entities
    (objects of `attribute_relations`) and capping to the most specific `top_k` matches."""

    def __init__(
        self,
        store: GraphStore,
        *,
        attribute_relations: frozenset[str] = frozenset(),
        top_k: int = 10,
    ):
        self._top_k = top_k
        non_seedable = {t.obj for t in store.triples() if t.relation in attribute_relations}
        self._entity_tokens: dict[str, frozenset[str]] = {}
        self._index: dict[str, set[str]] = defaultdict(set)
        for e in store.entities():
            if e in non_seedable:
                continue
            toks = frozenset(_tokens(e))
            if not toks:
                continue
            self._entity_tokens[e] = toks
            for tok in toks:
                self._index[tok].add(e)

    def seeds(self, question: str) -> list[str]:
        qtok = _tokens(question)
        if not qtok:
            return []
        candidates: set[str] = set()
        for tok in qtok:  # union of posting lists of the question's tokens
            candidates |= self._index.get(tok, set())
        matched = [
            e
            for e in candidates
            if self._entity_tokens[e] <= qtok and not (self._entity_tokens[e] <= _NON_CONTENT)
        ]
        # keep only maximal matches: drop a candidate whose tokens are a strict subset of
        # another candidate's (so "The Dark"/"Dark" drop out when "The Dark Knight" matches)
        maximal = [
            e
            for e in matched
            if not any(self._entity_tokens[e] < self._entity_tokens[other] for other in matched)
        ]
        # specificity: more tokens first, then longer name (deterministic tiebreak)
        maximal.sort(key=lambda e: (len(self._entity_tokens[e]), len(e), e), reverse=True)
        return maximal[: self._top_k]


class GraphRetriever:
    """Multi-hop retrieval over a GraphStore: find seed entities named in the question, BFS-expand
    up to `max_hops`, and return the connected subgraph (triples carry source citations). Lexical
    seed-matching is the hermetic stand-in for an embedding-based entity linker."""

    def __init__(
        self,
        store: GraphStore,
        max_hops: int = 2,
        *,
        max_triples: int = 300,
        max_neighbors_per_node: int = 50,
        max_degree: int = 100,
        attribute_relations: frozenset[str] = frozenset(),
        top_k_seeds: int = 10,
        seed_provider: SeedProvider | None = None,
    ):
        self._store = store
        self._max_hops = max_hops
        self._max_triples = max_triples
        self._max_neighbors_per_node = max_neighbors_per_node
        self._max_degree = max_degree
        self._attribute_nodes = frozenset(
            t.obj for t in store.triples() if t.relation in attribute_relations
        )
        self._seed_provider = seed_provider or LexicalSeedProvider(
            store, attribute_relations=attribute_relations, top_k=top_k_seeds
        )

    def seeds(self, question: str) -> list[str]:
        return self._seed_provider.seeds(question)

    def retrieve(self, question: str) -> RetrievedSubgraph:
        seeds = self.seeds(question)
        seen_entities: set[str] = set(seeds)
        seen_triples: set[tuple[str, str, str, str]] = set()
        collected: list[Triple] = []
        frontier = list(seeds)
        for _ in range(self._max_hops):
            if len(collected) >= self._max_triples:
                break
            next_frontier: list[str] = []
            for ent in frontier:
                if len(collected) >= self._max_triples:
                    break
                nbrs = self._store.neighbors(ent)
                expandable = len(nbrs) <= self._max_degree  # don't traverse through hubs
                for t in nbrs[: self._max_neighbors_per_node]:  # cap per-node fan-out
                    key = (t.subject, t.relation, t.obj, t.source)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        collected.append(t)
                        if len(collected) >= self._max_triples:
                            break
                    if expandable:
                        for other in (t.subject, t.obj):
                            if other not in seen_entities and other not in self._attribute_nodes:
                                seen_entities.add(other)
                                next_frontier.append(other)
            frontier = next_frontier
        return RetrievedSubgraph(seeds=tuple(seeds), triples=tuple(collected))
