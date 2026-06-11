from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Intent:
    """A query intent: trigger keywords map a question to a set of target relations and a
    retrieval shape (entity_lookup | cluster | pairwise | open)."""

    name: str
    triggers: frozenset[str]
    target_relations: frozenset[str]
    shape: str


@dataclass(frozen=True)
class GraphOntology:
    """Per-dataset GraphRAG ontology (loaded from datasets/<name>/graph/ontology.yaml): the entity
    and relation types the extractor is allowed to emit, plus a free-text extraction hint.
    `attribute_relations` lists relations whose *objects* are descriptive attribute values
    (genres, years, languages…) rather than nameable entities — the retriever excludes those
    objects from seed candidates."""

    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    attribute_relations: tuple[str, ...] = ()
    hint: str = ""
    empty_graph_hint: str = ""
    connective_relations: tuple[str, ...] = ()
    entity_relations: tuple[str, ...] = ()
    intents: tuple[Intent, ...] = ()


def load_ontology(path: str | Path) -> GraphOntology:
    data = yaml.safe_load(Path(path).read_text()) or {}
    intents = tuple(
        Intent(
            name=str(d.get("name", "")),
            triggers=frozenset(d.get("triggers", []) or []),
            target_relations=frozenset(d.get("target_relations", []) or []),
            shape=str(d.get("shape", "open")),
        )
        for d in (data.get("intents", []) or [])
    )
    return GraphOntology(
        entity_types=tuple(data.get("entity_types", []) or []),
        relation_types=tuple(data.get("relation_types", []) or []),
        attribute_relations=tuple(data.get("attribute_relations", []) or []),
        hint=(data.get("hint", "") or "").strip(),
        empty_graph_hint=(data.get("empty_graph_hint", "") or "").strip(),
        connective_relations=tuple(data.get("connective_relations", []) or []),
        entity_relations=tuple(data.get("entity_relations", []) or []),
        intents=intents,
    )
