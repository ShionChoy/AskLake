from __future__ import annotations

from engine.governance.policy import GovernanceError, Policy
from engine.ports.retrieval import RetrievalPath, RetrievalResult
from engine.ports.storage import QueryResult


class GovernedGraphPath:
    """Apply role, relation, citation, and size policy before graph facts reach an LLM."""

    name = "graph"

    def __init__(
        self,
        inner: RetrievalPath,
        policy: Policy,
        role: str,
        *,
        denied_sources: frozenset[str] = frozenset(),
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._role = role
        self._denied_sources = denied_sources

    def can_handle(self, question: str) -> bool:
        return self._inner.can_handle(question)

    def run(self, question: str) -> RetrievalResult:
        if not self._policy.allows_action(self._role, "graph"):
            raise GovernanceError(
                f"role {self._role!r} is not permitted to query the graph",
                code="action_denied",
            )
        result = self._inner.run(question)
        if result.result is None:
            return result
        if result.result.columns != ["subject", "relation", "object", "source"]:
            raise GovernanceError("graph result schema is not governed", code="graph_schema_denied")

        allowed = self._policy.graph_relations_for(self._role)
        max_rows = self._policy.max_graph_triples_for(self._role)
        rows: list[tuple] = []
        lines: list[str] = []
        for row in result.result.rows:
            subject, relation, obj, source = row
            if str(source) in self._denied_sources:
                continue
            relation_policy = self._policy.graph_relations.get(str(relation))
            if relation_policy is None:
                if (
                    self._policy.graph_default_effect == "deny"
                    or "*" not in self._policy.role(self._role).graph_relations
                ):
                    continue
            elif str(relation) not in allowed:
                continue
            if relation_policy and relation_policy.citation_required and not source:
                continue
            rows.append(tuple(row))
            integrity = (
                relation_policy.classification.integrity if relation_policy is not None else ""
            )
            qualifier = " (model-inferred)" if integrity == "llm_inferred" else ""
            lines.append(f"{subject} {relation} {obj} [{source}]{qualifier}")
            if len(rows) >= max_rows:
                break

        if not rows:
            return RetrievalResult(
                path=result.path,
                sql=None,
                result=None,
                narrative="No graph facts are available under the active governance policy.",
                chart_spec=None,
            )
        narrative = "Governed knowledge-graph facts:\n" + "\n".join(lines)
        return RetrievalResult(
            path=result.path,
            sql=None,
            result=QueryResult(columns=list(result.result.columns), rows=rows),
            narrative=narrative,
            chart_spec=None,
        )
