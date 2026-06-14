"""Manual demo of the Neo4j GraphRAG path (needs a running Neo4j; NOT in CI). Loads a tiny
synthetic film graph and runs the three typed shapes. Use via `make demo-neo4j`."""

from __future__ import annotations

import os

from neo4j import GraphDatabase

from engine.graph.intent import IntentResolver
from engine.graph.neo4j_retriever import Neo4jGraphRetriever
from engine.graph.neo4j_store import Neo4jGraphStore
from engine.graph.ontology import load_ontology
from engine.ports.graph_store import Triple

ONTOLOGY_YAML = "datasets/imdb_cmu/graph/ontology.yaml"
TRIPLES = [
    Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "wiki:1"),
    Triple("Inception", "HAS_THEME", "time", "wiki:1"),
    Triple("Interstellar", "DIRECTED_BY", "Christopher Nolan", "wiki:2"),
    Triple("Interstellar", "HAS_THEME", "time", "wiki:2"),
    Triple("Leonardo DiCaprio", "ACTED_IN", "Inception", "imdb:1"),
    Triple("Inception", "HAS_GENRE", "Sci-Fi", "imdb:1"),
]


def main() -> None:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
    )
    ont = load_ontology(ONTOLOGY_YAML)
    store = Neo4jGraphStore(driver, relation_roles=ont.relation_roles)
    store.ensure_schema()
    store.load_triples(TRIPLES)
    retriever = Neo4jGraphRetriever(
        store,
        IntentResolver(ont),
        attribute_relations=frozenset(ont.attribute_relations),
        connective_relations=frozenset(ont.connective_relations),
        relation_roles=ont.relation_roles,
    )
    for q in [
        "who acted in Inception",
        "themes of Inception",
        "what do Inception and Interstellar share",
    ]:
        sg = retriever.retrieve(q)
        print(f"\nQ: {q}\n  seeds={sg.seeds}")
        for t in sg.triples:
            print(f"  {t.subject} {t.relation} {t.obj} [{t.source}]")
    store.close()


if __name__ == "__main__":
    main()
