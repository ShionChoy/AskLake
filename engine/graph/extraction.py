from __future__ import annotations

from dataclasses import dataclass

from engine.graph.ontology import GraphOntology
from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple
from engine.ports.llm import LLMProvider

EXTRACTION_SYSTEM = (
    "You are an information-extraction engine. Read a film plot summary and emit knowledge-graph "
    "triples. Use ONLY the allowed relation types. Output one triple per line as "
    "'subject | relation | object'. No prose, no numbering."
)

EXTRACTION_TEMPLATE = """Allowed relation types: {relations}
{hint}

Film: {title}
Plot:
{text}

Emit triples (subject | relation | object), one per line."""


@dataclass(frozen=True)
class PlotDoc:
    id: str
    title: str
    text: str


def _parse_triples(text: str, allowed: set[str], source: str) -> list[Triple]:
    out: list[Triple] = []
    for line in text.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        subj, rel, obj = parts
        if not (subj and rel and obj):
            continue
        if allowed and rel not in allowed:
            continue
        out.append(Triple(subject=subj, relation=rel, obj=obj, source=source))
    return out


def extract_triples(llm: LLMProvider, doc: PlotDoc, ontology: GraphOntology) -> list[Triple]:
    """LLM entity/relation extraction for one document, constrained to the ontology's relations."""
    prompt = EXTRACTION_TEMPLATE.format(
        relations=", ".join(ontology.relation_types),
        hint=ontology.hint,
        title=doc.title,
        text=doc.text,
    )
    raw = llm.complete(prompt, system=EXTRACTION_SYSTEM)
    return _parse_triples(raw, set(ontology.relation_types), source=doc.id)


def build_graph(
    llm: LLMProvider, docs: list[PlotDoc], ontology: GraphOntology
) -> InMemoryGraphStore:
    """Build an in-memory graph by extracting ontology-constrained triples from each document."""
    store = InMemoryGraphStore()
    for doc in docs:
        for t in extract_triples(llm, doc, ontology):
            store.add(t)
    return store
