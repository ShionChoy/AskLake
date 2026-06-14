"""Bulk-load data/imdb/graph/triples.jsonl into Neo4j: ensure schema, then batched UNWIND MERGE.
Run via `make graph-load-neo4j` (needs NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD and a running DB)."""

from __future__ import annotations

import os

from neo4j import GraphDatabase

from engine.graph.neo4j_store import Neo4jGraphStore
from engine.graph.ontology import load_ontology
from engine.graph.persistence import iter_triples

GRAPH_PATH = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
ONTOLOGY_YAML = os.environ.get("ASKLAKE_ONTOLOGY", "datasets/imdb_cmu/graph/ontology.yaml")


def main() -> None:
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
    roles = load_ontology(ONTOLOGY_YAML).relation_roles
    driver = GraphDatabase.driver(uri, auth=auth)
    store = Neo4jGraphStore(driver, relation_roles=roles)
    print(f"[load_neo4j] ensuring schema on {uri} …")
    store.ensure_schema()
    print(f"[load_neo4j] loading {GRAPH_PATH} …")
    n = store.load_triples(iter_triples(GRAPH_PATH), batch_size=5000)
    print(f"[load_neo4j] done: {n} triples loaded")
    store.close()


if __name__ == "__main__":
    main()
