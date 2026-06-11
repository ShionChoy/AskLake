from __future__ import annotations

import math
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
    """Intent-aware multi-hop retrieval. Finds seeds via the SeedProvider, resolves a query intent
    (when an IntentResolver is supplied), runs the shape strategy, and returns a relevance-ranked
    subgraph. Without an intent resolver it falls back to open-shape bounded BFS (legacy behavior).
    PPR is a future swap behind _rank()."""

    def __init__(
        self,
        store: GraphStore,
        max_hops: int = 2,
        *,
        max_triples: int = 300,
        max_neighbors_per_node: int = 50,
        max_degree: int = 100,
        attribute_relations: frozenset[str] = frozenset(),
        connective_relations: frozenset[str] = frozenset(),
        top_k_seeds: int = 10,
        top_k: int = 200,
        seed_provider: SeedProvider | None = None,
        intent_resolver: object | None = None,
    ):
        self._store = store
        self._max_hops = max_hops
        self._max_triples = max_triples
        self._max_neighbors_per_node = max_neighbors_per_node
        self._max_degree = max_degree
        self._connective = frozenset(connective_relations)
        self._top_k = top_k
        self._attribute_nodes = frozenset(
            t.obj for t in store.triples() if t.relation in attribute_relations
        )
        self._degree = {e: len(store.neighbors(e)) for e in store.entities()}
        self._seed_provider = seed_provider or LexicalSeedProvider(
            store, attribute_relations=attribute_relations, top_k=top_k_seeds
        )
        self._intent_resolver = intent_resolver

    def seeds(self, question: str) -> list[str]:
        return self._seed_provider.seeds(question)

    def retrieve(self, question: str) -> RetrievedSubgraph:
        seeds = self.seeds(question)
        if not seeds:
            return RetrievedSubgraph(seeds=(), triples=())
        intent = self._intent_resolver.resolve(question) if self._intent_resolver else None
        shape = getattr(intent, "shape", "open")
        targets = getattr(intent, "target_relations", None)
        if shape == "entity_lookup":
            triples = self._entity_lookup(seeds, targets)
        elif shape == "cluster":
            triples = self._cluster(seeds, targets)
        elif shape == "pairwise":
            triples = self._pairwise(seeds, targets)
        else:
            triples = self._open(seeds)
        ranked = self._rank(triples, set(seeds), targets)
        return RetrievedSubgraph(seeds=tuple(seeds), triples=tuple(ranked[: self._top_k]))

    def _entity_lookup(self, seeds, targets):
        out = []
        for s in seeds:
            for t in self._store.neighbors(s)[: self._max_neighbors_per_node]:
                if targets is None or t.relation in targets:
                    out.append(t)
        return out

    def _cluster(self, seeds, targets):
        rels = targets or self._connective
        out, hubs = [], set()
        for s in seeds:
            for t in self._store.neighbors(s):
                if t.relation in rels:
                    out.append(t)
                    hubs.add(t.obj if t.subject == s else t.subject)
        for hub in hubs:
            nbrs = self._store.neighbors(hub)
            if len(nbrs) > self._max_degree:
                nbrs = nbrs[: self._max_neighbors_per_node]
            for t in nbrs:
                if t.relation in rels:
                    out.append(t)
        return out

    def _pairwise(self, seeds, targets):
        if len(seeds) < 2:
            return self._entity_lookup(seeds, targets)
        edges, objsets = {}, []
        for s in seeds:
            objs = set()
            for t in self._store.neighbors(s):
                if targets is None or t.relation in targets:
                    edges.setdefault(s, []).append(t)
                    objs.add(t.obj if t.subject == s else t.subject)
            objsets.append(objs)
        shared = set.intersection(*objsets) if objsets else set()
        out = []
        for s in seeds:
            for t in edges.get(s, []):
                if (t.obj if t.subject == s else t.subject) in shared:
                    out.append(t)
        return out

    def _open(self, seeds):
        seen_entities = set(seeds)
        seen_triples = set()
        collected = []
        frontier = list(seeds)
        for _ in range(self._max_hops):
            if len(collected) >= self._max_triples:
                break
            next_frontier = []
            for ent in frontier:
                if len(collected) >= self._max_triples:
                    break
                nbrs = self._store.neighbors(ent)
                expandable = len(nbrs) <= self._max_degree
                for t in nbrs[: self._max_neighbors_per_node]:
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
        return collected

    def _rank(self, triples, seedset, targets):
        def score(t):
            s = 0.0
            if targets and t.relation in targets:
                s += 3.0
            if t.subject in seedset or t.obj in seedset:
                s += 2.0
            deg = max(self._degree.get(t.subject, 1), self._degree.get(t.obj, 1))
            s += 1.0 / math.log(2 + deg)
            return s

        return sorted(dict.fromkeys(triples), key=score, reverse=True)
