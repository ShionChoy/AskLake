"""IntentResolver: maps a question to an Intent (target relations + retrieval shape) by keyword
match against the ontology's declared intents. Deterministic; an LLM resolver can replace it
behind resolve()."""

from __future__ import annotations

import re

from engine.graph.ontology import GraphOntology, Intent

_WORD = re.compile(r"[a-z0-9]+")


class IntentResolver:
    def __init__(self, ontology: GraphOntology):
        self._intents = ontology.intents
        self._all = frozenset(ontology.relation_types)

    def resolve(self, question: str) -> Intent:
        q = set(_WORD.findall(question.lower()))
        best: tuple[int, Intent] | None = None
        for intent in self._intents:  # declaration order; most trigger hits wins, ties -> first
            hits = len(q & intent.triggers)
            if hits and (best is None or hits > best[0]):
                best = (hits, intent)
        if best is not None:
            return best[1]
        return Intent(name="open", triggers=frozenset(), target_relations=self._all, shape="open")
