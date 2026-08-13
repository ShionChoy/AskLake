from __future__ import annotations

from engine.agents.graph import build_sql_graph
from engine.governance.passthrough import PassthroughGovernance
from engine.ports.governance import GovernanceHook
from engine.ports.llm import LLMProvider
from engine.ports.retrieval import RetrievalResult
from engine.ports.schema import SchemaProvider
from engine.ports.storage import QueryResult, StorageBackend


def infer_chart_spec(result: QueryResult) -> dict | None:
    if len(result.columns) >= 2 and result.rows:
        last = result.rows[0][-1]
        if isinstance(last, (int, float)) and not isinstance(last, bool):
            return {"type": "bar", "x": result.columns[0], "y": result.columns[-1]}
    return None


class SqlPath:
    """RetrievalPath: NL question -> SQL (via graph) -> governed execution -> chart."""

    name = "sql"

    def __init__(
        self,
        llm: LLMProvider,
        schema_provider: SchemaProvider,
        backend: StorageBackend,
        governance: GovernanceHook | None = None,
        role: str = "analyst",
    ):
        self._graph = build_sql_graph(llm, schema_provider)
        self._backend = backend
        self._gov = governance or PassthroughGovernance()
        self._role = role

    def can_handle(self, question: str) -> bool:
        return True  # SQL is the general fallback; Router handles path selection.

    def run(self, question: str) -> RetrievalResult:
        state = self._graph.invoke({"question": question})
        sql = state.get("sql", "")
        if not sql:
            return RetrievalResult(
                path=self.name,
                sql=sql,
                result=None,
                narrative="SQL generation produced no query.",
                chart_spec=None,
            )
        try:
            safe_sql = self._gov.before_query(sql, role=self._role)
            result = self._backend.run_sql(safe_sql)
            result = self._gov.after_result(result, role=self._role)
        except Exception as exc:  # noqa: BLE001 - graceful degradation, surfaced to caller
            return RetrievalResult(
                path=self.name,
                sql=sql,
                result=None,
                narrative=f"SQL execution failed: {exc}",
                chart_spec=None,
            )
        return RetrievalResult(
            path=self.name,
            sql=sql,
            result=result,
            narrative=f"Returned {len(result.rows)} row(s).",
            chart_spec=infer_chart_spec(result),
        )
