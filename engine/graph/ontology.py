from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GraphOntology:
    """Per-dataset GraphRAG ontology (loaded from datasets/<name>/graph/ontology.yaml): the entity
    and relation types the extractor is allowed to emit, plus a free-text extraction hint."""

    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    hint: str = ""


def load_ontology(path: str | Path) -> GraphOntology:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return GraphOntology(
        entity_types=tuple(data.get("entity_types", []) or []),
        relation_types=tuple(data.get("relation_types", []) or []),
        hint=(data.get("hint", "") or "").strip(),
    )
