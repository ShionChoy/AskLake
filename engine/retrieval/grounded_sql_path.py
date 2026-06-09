from __future__ import annotations

from engine.agents.grounded_graph import build_grounded_sql_graph
from engine.governance.passthrough import PassthroughGovernance
from engine.ports.governance import GovernanceHook
from engine.ports.llm import LLMProvider
from engine.ports.retrieval import RetrievalResult
from engine.ports.schema import SchemaProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.retrieval.sql_path import infer_chart_spec
from engine.semantic.value_index import ValueIndex


class GroundedSqlPath:
    """RetrievalPath: grounded, correctness-aware NL->SQL. Additive sibling of AgenticSqlPath.

    Schema- + value-linking, difficulty routing, Planner decomposition, K-candidate
    self-consistency, and a reflexion critic. The injected executor wraps governance +
    StorageBackend so the graph stays storage-agnostic."""

    name = "sql"  # same logical path name as AgenticSqlPath; only one is active at a time

    def __init__(
        self,
        llm: LLMProvider,
        schema_provider: SchemaProvider,
        backend: StorageBackend,
        governance: GovernanceHook | None = None,
        role: str = "analyst",
        value_index: ValueIndex | None = None,
        k_candidates: int = 3,
        max_retries: int = 2,
        use_plan: bool = True,
        use_critic: bool = True,
    ):
        self._backend = backend
        self._gov = governance or PassthroughGovernance()
        self._role = role

        def executor(sql: str) -> QueryResult:
            safe_sql = self._gov.before_query(sql, role=self._role)
            result = self._backend.run_sql(safe_sql)
            return self._gov.after_result(result, role=self._role)

        self._graph = build_grounded_sql_graph(
            llm,
            schema_provider,
            executor,
            value_index=value_index,
            k_candidates=k_candidates,
            max_retries=max_retries,
            use_plan=use_plan,
            use_critic=use_critic,
        )

    def can_handle(self, question: str) -> bool:
        return True

    def run(self, question: str) -> RetrievalResult:
        state = self._graph.invoke({"question": question})
        sql = state.get("sql", "")
        result = state.get("result")
        attempts = state.get("attempts", 0)
        if result is not None:
            return RetrievalResult(
                path=self.name,
                sql=sql,
                result=result,
                narrative=f"Returned {len(result.rows)} row(s) after {attempts} correction(s) "
                "(grounded + self-consistent).",
                chart_spec=infer_chart_spec(result),
            )
        return RetrievalResult(
            path=self.name,
            sql=sql,
            result=None,
            narrative=f"SQL failed after {attempts} correction(s): {state.get('error', '')}",
            chart_spec=None,
        )
