"""Neo4jGraphRetriever: intent-aware retrieval over a Neo4jGraphStore. Mirrors GraphRetriever's
typed shapes but runs Cypher (through the store) instead of in-process traversal. Reuses the
IntentResolver, the lexical entity linker (built once from seedable names), and the shared
rank_triples scoring, so behavior matches the in-memory backend."""

from __future__ import annotations

from engine.graph.entity_linker import LexicalEntityLinker
from engine.graph.retriever import RetrievedSubgraph, rank_triples


class Neo4jGraphRetriever:
    def __init__(
        self,
        store,
        intent_resolver,
        *,
        attribute_relations=frozenset(),
        connective_relations=frozenset(),
        relation_roles=None,
        max_neighbors_per_node: int = 50,
        max_degree: int = 100,
        max_triples: int = 300,
        max_hops: int = 2,
        top_k: int = 200,
        top_k_seeds: int = 10,
        linker=None,
    ):
        self._store = store
        self._intent = intent_resolver
        self._attr_rel = frozenset(attribute_relations)
        self._connective = frozenset(connective_relations)
        roles = relation_roles or {}
        self._attr_types = [
            roles[r]["object"] for r in self._attr_rel if r in roles and roles[r].get("object")
        ]
        self._max_neighbors = max_neighbors_per_node
        self._max_degree = max_degree
        self._max_triples = max_triples
        self._max_hops = max_hops
        self._top_k = top_k
        self._linker = linker or LexicalEntityLinker.from_names(
            store.seedable_names(self._attr_types), top_k=top_k_seeds
        )

    def seeds(self, question: str) -> list[str]:
        return self._linker.seeds(question)

    def retrieve(self, question: str) -> RetrievedSubgraph:
        seeds = self.seeds(question)
        if not seeds:
            return RetrievedSubgraph(seeds=(), triples=())
        intent = self._intent.resolve(question)
        targets = intent.target_relations
        if intent.shape == "entity_lookup":
            triples = self._entity_lookup(seeds, targets)
        elif intent.shape == "cluster":
            triples = self._cluster(seeds, targets)
        elif intent.shape == "pairwise":
            triples = self._pairwise(seeds, targets)
        else:
            triples = self._open(seeds)
        names = {n for t in triples for n in (t.subject, t.obj)}
        degree_of = self._store.degrees(names)
        ranked = rank_triples(triples, set(seeds), targets, degree_of)
        return RetrievedSubgraph(seeds=tuple(seeds), triples=tuple(ranked[: self._top_k]))

    def _entity_lookup(self, seeds, targets):
        out = []
        for s in seeds:
            out.extend(self._store.neighbors(s, limit=self._max_neighbors, relations=targets))
        return out

    def _cluster(self, seeds, targets):
        rels = targets or self._connective
        out, hubs = [], set()
        for s in seeds:
            for t in self._store.neighbors(s, relations=rels):
                out.append(t)
                hubs.add(t.obj if t.subject == s else t.subject)
        for hub in hubs:
            lim = self._max_neighbors if self._store.degree(hub) > self._max_degree else None
            out.extend(self._store.neighbors(hub, limit=lim, relations=rels))
        return out

    def _pairwise(self, seeds, targets):
        if len(seeds) < 2:
            return self._entity_lookup(seeds, targets)
        return list(self._store.shared(seeds, targets))

    def _open(self, seeds):
        seen_entities = set(seeds)
        seen_triples = set()
        collected = []
        frontier = list(seeds)
        for _ in range(self._max_hops):
            if len(collected) >= self._max_triples:
                break
            nxt = []
            frontier_degrees = self._store.degrees(frontier)  # one batched query per hop
            for ent in frontier:
                if len(collected) >= self._max_triples:
                    break
                expandable = frontier_degrees.get(ent, 0) <= self._max_degree
                for t in self._store.neighbors(ent, limit=self._max_neighbors):
                    key = (t.subject, t.relation, t.obj, t.source)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        collected.append(t)
                        if len(collected) >= self._max_triples:
                            break
                    if expandable and t.relation not in self._attr_rel:
                        for other in (t.subject, t.obj):
                            if other not in seen_entities:
                                seen_entities.add(other)
                                nxt.append(other)
            frontier = nxt
        return collected
