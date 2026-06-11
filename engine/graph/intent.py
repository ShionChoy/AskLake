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
        scored = [
            (len(q & i.triggers), 1 if i.shape == "pairwise" else 0, -idx, i)
            for idx, i in enumerate(self._intents)
        ]
        scored = [s for s in scored if s[0] > 0]
        if scored:  # most trigger hits; tie -> prefer pairwise; tie -> earliest declared
            return max(scored, key=lambda s: (s[0], s[1], s[2]))[3]
        return Intent(name="open", triggers=frozenset(), target_relations=self._all, shape="open")
