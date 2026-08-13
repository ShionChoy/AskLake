"""Live Neo4j cross-backend equivalence. Skipped unless NEO4J_TEST_URI is set (and a DB is up).
Run locally with:  NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_USER=neo4j \
NEO4J_TEST_PASSWORD=asklake-graph uv run pytest tests/unit/test_neo4j_live.py -v"""

from __future__ import annotations

import os

import pytest

NEO4J_TEST_URI = os.environ.get("NEO4J_TEST_URI")
pytestmark = pytest.mark.skipif(
    not NEO4J_TEST_URI, reason="set NEO4J_TEST_URI to run live Neo4j tests"
)

ROLES = {
    "ACTED_IN": {"subject": "Person", "object": "Film"},
    "HAS_THEME": {"subject": "Film", "object": "Theme"},
    "HAS_GENRE": {"subject": "Film", "object": "Genre"},
}
FIXTURE = [
    ("Leonardo DiCaprio", "ACTED_IN", "Inception", "imdb:1"),
    ("Inception", "HAS_THEME", "time", "wiki:1"),
    ("Inception", "HAS_GENRE", "Sci-Fi", "imdb:1"),
]

# All entity names this test writes; deleted in teardown so the test is self-contained.
# NEO4J_TEST_URI must point at a dedicated or ephemeral development Neo4j instance, never a
# shared/production graph — teardown deletes these names unconditionally.
FIXTURE_NAMES = sorted({n for s, _r, o, _src in FIXTURE for n in (s, o)})


def _store():
    from neo4j import GraphDatabase

    from engine.graph.neo4j_store import Neo4jGraphStore
    from engine.ports.graph_store import Triple

    driver = GraphDatabase.driver(
        NEO4J_TEST_URI,
        auth=(os.environ["NEO4J_TEST_USER"], os.environ["NEO4J_TEST_PASSWORD"]),
    )
    try:
        store = Neo4jGraphStore(driver, relation_roles=ROLES)
        store.ensure_schema()
        store.load_triples(Triple(s, r, o, src) for s, r, o, src in FIXTURE)
    except Exception:
        driver.close()
        raise
    return store, driver


def test_entity_lookup_live():
    from engine.graph.intent import IntentResolver
    from engine.graph.neo4j_retriever import Neo4jGraphRetriever
    from engine.graph.ontology import load_ontology

    store, driver = _store()
    try:
        r = Neo4jGraphRetriever(
            store,
            IntentResolver(load_ontology("datasets/imdb/graph/ontology.yaml")),
            attribute_relations=frozenset({"HAS_GENRE", "RELEASED_IN"}),
            connective_relations=frozenset({"HAS_THEME"}),
            relation_roles=ROLES,
        )
        sg = r.retrieve("who acted in Inception")
        rels = {t.relation for t in sg.triples}
        assert "ACTED_IN" in rels and "HAS_GENRE" not in rels
    finally:
        store.query("MATCH (n:Entity) WHERE n.name IN $names DETACH DELETE n", names=FIXTURE_NAMES)
        driver.close()
