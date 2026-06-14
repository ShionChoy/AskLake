"""Neo4jGraphStore: the AskLake knowledge graph backed by Neo4j (official driver, injected).

Typed property graph: nodes carry :Entity{name} plus a typed label (:Film/:Person/...) derived
from the ontology's relation_roles; relationships are typed from the relation string and carry a
`source` property for citation. Retrieval reads run Cypher; the whole graph is never materialized
into Python. Implements the GraphStore port (add/triples/neighbors/entities) plus retrieval
helpers (degree/degrees/seedable_names/shared) and a batched loader."""

from __future__ import annotations

import re
from collections.abc import Iterable

from engine.ports.graph_store import Triple

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe(ident: str) -> str:
    """Validate a label / relationship identifier before interpolating it into Cypher (Cypher
    cannot parameterize labels or relationship types). Identifiers come from the controlled
    ontology, so this is defense-in-depth, not user-input sanitization."""
    if not _IDENT.match(ident):
        raise ValueError(f"unsafe graph identifier: {ident!r}")
    return ident


def _triples(rows) -> tuple[Triple, ...]:
    return tuple(
        Triple(subject=r["s"], relation=r["rel"], obj=r["o"], source=r["src"] or "") for r in rows
    )


class Neo4jGraphStore:
    def __init__(self, driver, *, relation_roles: dict[str, dict[str, str]] | None = None):
        self._driver = driver
        self._roles = relation_roles or {}

    # --- low level ---
    def query(self, cypher: str, **params):
        with self._driver.session() as session:
            return list(session.run(cypher, **params))

    def close(self) -> None:
        self._driver.close()

    # --- schema ---
    def ensure_schema(self) -> None:
        self.query(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        self.query(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]"
        )

    def _labels(self, relation: str) -> tuple[str | None, str | None]:
        role = self._roles.get(relation, {})
        subj, obj = role.get("subject"), role.get("object")
        return (_safe(subj) if subj else None, _safe(obj) if obj else None)

    def _merge_clause(self, relation: str, *, row_prefix: str) -> str:
        """Build the MERGE body shared by add() and load_triples(). `row_prefix` is "$" (add,
        params s/o/src) or "row." (UNWIND, fields s/o/src)."""
        rel = _safe(relation)
        subj_label, obj_label = self._labels(relation)
        set_s = f" SET s:`{subj_label}`" if subj_label else ""
        set_o = f" SET o:`{obj_label}`" if obj_label else ""
        s, o, src = f"{row_prefix}s", f"{row_prefix}o", f"{row_prefix}src"
        return (
            f"MERGE (s:Entity {{name: {s}}}){set_s} "
            f"MERGE (o:Entity {{name: {o}}}){set_o} "
            f"MERGE (s)-[r:`{rel}`]->(o) SET r.source = {src}"
        )

    # --- writes ---
    def add(self, triple: Triple) -> None:
        self.query(
            self._merge_clause(triple.relation, row_prefix="$"),
            s=triple.subject,
            o=triple.obj,
            src=triple.source,
        )

    def load_triples(self, triples: Iterable[Triple], *, batch_size: int = 5000) -> int:
        """Bulk-load with batched UNWIND, grouping each batch by relation so the relation type +
        labels (which can't be parameterized) are interpolated once per group."""
        total, batch = 0, []
        for t in triples:
            batch.append(t)
            if len(batch) >= batch_size:
                total += self._flush(batch)
                batch = []
        if batch:
            total += self._flush(batch)
        return total

    def _flush(self, batch: list[Triple]) -> int:
        groups: dict[str, list[dict]] = {}
        for t in batch:
            groups.setdefault(t.relation, []).append({"s": t.subject, "o": t.obj, "src": t.source})
        n = 0
        for relation, rows in groups.items():
            self.query(
                "UNWIND $rows AS row " + self._merge_clause(relation, row_prefix="row."),
                rows=rows,
            )
            n += len(rows)
        return n

    # --- reads ---
    def neighbors(
        self,
        entity: str,
        *,
        limit: int | None = None,
        relations: Iterable[str] | None = None,
    ) -> tuple[Triple, ...]:
        where, params = "", {"name": entity}
        if relations is not None:
            where = "WHERE type(r) IN $relations "
            params["relations"] = list(relations)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit
        rows = self.query(
            f"MATCH (n:Entity {{name: $name}})-[r]-(m:Entity) {where}"
            f"RETURN startNode(r).name AS s, type(r) AS rel, endNode(r).name AS o, "
            f"r.source AS src {limit_clause}",
            **params,
        )
        return _triples(rows)

    def degree(self, entity: str) -> int:
        rows = self.query(
            "MATCH (n:Entity {name: $name}) RETURN COUNT { (n)--() } AS deg", name=entity
        )
        return rows[0]["deg"] if rows else 0

    def degrees(self, names: Iterable[str]) -> dict[str, int]:
        names = list(names)
        if not names:
            return {}
        rows = self.query(
            "MATCH (n:Entity) WHERE n.name IN $names "
            "RETURN n.name AS name, COUNT { (n)--() } AS deg",
            names=names,
        )
        return {r["name"]: r["deg"] for r in rows}

    def entities(self) -> frozenset[str]:
        rows = self.query("MATCH (e:Entity) RETURN e.name AS name")
        return frozenset(r["name"] for r in rows)

    def seedable_names(self, attr_types: Iterable[str]) -> list[str]:
        rows = self.query(
            "MATCH (e:Entity) WHERE NOT any(l IN labels(e) WHERE l IN $attr) RETURN e.name AS name",
            attr=list(attr_types),
        )
        return [r["name"] for r in rows]

    def triples(self) -> tuple[Triple, ...]:
        rows = self.query(
            "MATCH (s:Entity)-[r]->(o:Entity) "
            "RETURN s.name AS s, type(r) AS rel, o.name AS o, r.source AS src"
        )
        return _triples(rows)

    def shared(self, seeds: Iterable[str], relations: Iterable[str]) -> list[Triple]:
        """Pairwise intersection: edges from each seed to a neighbor adjacent to ALL seeds
        (via the target relations). Native Cypher."""
        rows = self.query(
            "MATCH (s:Entity)-[r]-(x:Entity) "
            "WHERE s.name IN $seeds AND type(r) IN $rels "
            "WITH x, count(DISTINCT s) AS c, "
            "collect(DISTINCT {s: startNode(r).name, rel: type(r), o: endNode(r).name, "
            "src: r.source}) AS edges "
            "WHERE c = size($seeds) "
            "UNWIND edges AS e RETURN DISTINCT e.s AS s, e.rel AS rel, e.o AS o, e.src AS src",
            seeds=list(seeds),
            rels=list(relations),
        )
        return list(_triples(rows))
