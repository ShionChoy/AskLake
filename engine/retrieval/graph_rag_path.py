from __future__ import annotations

import re

from engine.graph.retriever import GraphRetriever
from engine.ports.graph_store import GraphStore, Triple
from engine.ports.llm import LLMProvider
from engine.ports.retrieval import RetrievalResult
from engine.ports.storage import QueryResult

_GRAPH_HINTS = frozenset(
    {
        "theme",
        "themes",
        "plot",
        "plots",
        "story",
        "stories",
        "about",
        "common",
        "motif",
        "motifs",
        "character",
        "characters",
        "relationship",
        "relationships",
        "related",
        "connect",
        "connection",
        "connects",
    }
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _narrative(
    seeds: tuple[str, ...],
    triples: tuple[Triple, ...],
    total: int,
    max_rows: int,
    empty_hint: str,
) -> str:
    if not triples:
        return empty_hint
    seed_str = ", ".join(seeds) if seeds else "the question"
    lines = [f"{t.subject} {t.relation} {t.obj} [{t.source}]" for t in triples]
    body = f"From the knowledge graph, starting at {seed_str}:\n" + "\n".join(lines)
    if total > len(triples):
        body += f"\n… (showing first {max_rows} of {total} facts)"
    return body


class GraphRagPath:
    """RetrievalPath (P4): answers from a knowledge graph via multi-hop retrieval with traceable
    citations. Additive sibling of SqlPath; selected by the Router. Reuses the generic
    RetrievalResult (facts ride in `result` as rows; citations in `narrative`)."""

    name = "graph"

    def __init__(
        self,
        store: GraphStore,
        max_hops: int = 2,
        *,
        attribute_relations: frozenset[str] = frozenset(),
        connective_relations: frozenset[str] = frozenset(),
        top_k_seeds: int = 10,
        max_rows: int = 200,
        empty_hint: str = "No matching facts found in the knowledge graph.",
        intent_resolver: object | None = None,
        seed_provider: object | None = None,
        retriever: object | None = None,
    ):
        self._store = store
        self._max_rows = max_rows
        self._empty_hint = empty_hint
        if retriever is not None:
            # Injected (e.g. Neo4jGraphRetriever). The default GraphRetriever is NOT built —
            # its constructor eagerly materializes store.triples()/entities().
            self._retriever = retriever
        else:
            self._retriever = GraphRetriever(
                store,
                max_hops=max_hops,
                attribute_relations=attribute_relations,
                connective_relations=connective_relations,
                top_k_seeds=top_k_seeds,
                intent_resolver=intent_resolver,
                seed_provider=seed_provider,
            )

    def can_handle(self, question: str) -> bool:
        w = _words(question)
        if w & _GRAPH_HINTS:
            return True
        return bool(self._retriever.seeds(question))  # fast: inverted index, no full scan

    def run(self, question: str) -> RetrievalResult:
        sg = self._retriever.retrieve(question)
        total = len(sg.triples)
        shown = sg.triples[: self._max_rows]
        rows = [(t.subject, t.relation, t.obj, t.source) for t in shown]
        result = (
            QueryResult(columns=["subject", "relation", "object", "source"], rows=rows)
            if rows
            else None
        )
        return RetrievalResult(
            path=self.name,
            sql=None,
            result=result,
            narrative=_narrative(sg.seeds, shown, total, self._max_rows, self._empty_hint),
            chart_spec=None,
        )


GROUNDED_SYSTEM = (
    "You answer questions strictly from the supplied knowledge-graph facts. Use ONLY those facts, "
    "cite each claim with its [source] tag, never invent, and if the facts don't answer the "
    "question say so plainly."
)


class GroundedGraphRagPath:
    """RetrievalPath: GraphRagPath + one LLM call that turns the retrieved, cited facts into a
    grounded natural-language answer. Additive sibling of the deterministic GraphRagPath (same
    name='graph'; only one active at a time, mirroring GroundedSqlPath vs SqlPath)."""

    name = "graph"

    def __init__(self, base: GraphRagPath, llm: LLMProvider, *, answer_budget: int = 40):
        self._base = base
        self._llm = llm
        self._budget = answer_budget

    def can_handle(self, question: str) -> bool:
        return self._base.can_handle(question)

    def run(self, question: str) -> RetrievalResult:
        rr = self._base.run(question)
        if rr.result is None or not rr.result.rows:
            return rr
        facts = "\n".join(
            f"{s} {rel} {o} [{src}]" for s, rel, o, src in rr.result.rows[: self._budget]
        )
        prompt = f"Question: {question}\n\nKnowledge-graph facts:\n{facts}\n\nGrounded answer:"
        answer = self._llm.complete(prompt, system=GROUNDED_SYSTEM)
        return RetrievalResult(
            path=self.name, sql=None, result=rr.result, narrative=answer.strip(), chart_spec=None
        )
