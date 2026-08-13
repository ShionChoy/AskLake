from __future__ import annotations

from engine.agents.agentic_graph import build_agentic_sql_graph
from engine.governance.passthrough import PassthroughGovernance
from engine.ports.governance import GovernanceHook
from engine.ports.llm import LLMProvider
from engine.ports.retrieval import RetrievalResult
from engine.ports.schema import SchemaProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.retrieval.sql_path import infer_chart_spec


class AgenticSqlPath:
    """Self-correcting NL-to-SQL retrieval path and additive sibling of SqlPath.

    The injected executor wraps governance + StorageBackend so the graph stays storage-agnostic.
    """

    name = "sql"

    def __init__(
        self,
        llm: LLMProvider,
        schema_provider: SchemaProvider,
        backend: StorageBackend,
        governance: GovernanceHook | None = None,
        role: str = "analyst",
        max_retries: int = 2,
    ):
        self._backend = backend
        self._gov = governance or PassthroughGovernance()
        self._role = role

        def executor(sql: str) -> QueryResult:
            safe_sql = self._gov.before_query(sql, role=self._role)
            result = self._backend.run_sql(safe_sql)
            return self._gov.after_result(result, role=self._role)

        self._graph = build_agentic_sql_graph(llm, schema_provider, executor, max_retries)

    def can_handle(self, question: str) -> bool:
        return True  # SQL is the general fallback; Router handles path selection.

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
                narrative=(
                    f"Returned {len(result.rows)} row(s) after {attempts} self-correction(s)."
                ),
                chart_spec=infer_chart_spec(result),
            )
        return RetrievalResult(
            path=self.name,
            sql=sql,
            result=None,
            narrative=f"SQL failed after {attempts} self-correction(s): {state.get('error', '')}",
            chart_spec=None,
        )
