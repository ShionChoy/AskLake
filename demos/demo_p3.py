"""Phase 3 demo: governance (RBAC + PII masking + row filtering + cost guardrail), hermetic.

Same question, two roles: `analyst` sees full data; `public` gets PII columns masked and
restricted rows filtered out. Then one cost-guardrail interception (a query without LIMIT).
Uses the semantic layer for grounding and PolicyGovernance for enforcement, via AgenticSqlPath."""

from __future__ import annotations

from engine.governance.policy import GovernanceError, Policy, PolicyGovernance, RowFilter
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import ColumnDef, SemanticLayer, TableDef

SEED = (
    "CREATE TABLE people AS SELECT * FROM (VALUES "
    "('Nolan', 1970, 'movie'), ('Hidden', 1980, 'adult')"
    ") t(primaryName, birthYear, titleType);"
)

_LAYER = SemanticLayer(
    tables=(
        TableDef(
            "people",
            "People with an associated title category.",
            (
                ColumnDef("primaryName", "Person name."),
                ColumnDef("birthYear", "Year of birth (PII).", "INTEGER"),
                ColumnDef("titleType", "Category of the associated title."),
            ),
        ),
    ),
)

_POLICY = Policy(
    pii_columns=("birthYear",),
    mask_roles=("public",),
    row_filters={"public": (RowFilter("titleType", ("adult",)),)},
    require_limit=True,
    forbid_writes=True,
)

_SQL = "SELECT primaryName, birthYear, titleType FROM people LIMIT 10"


def _run_as(role: str) -> dict:
    backend = DuckDBBackend()
    backend.setup(SEED)
    path = AgenticSqlPath(
        FakeLLMProvider(responses=[_SQL]),
        SemanticLayerProvider(_LAYER),
        backend,
        governance=PolicyGovernance(_POLICY),
        role=role,
    )
    rr = path.run("list people with birth years")
    return {"role": role, "rows": [list(r) for r in rr.result.rows] if rr.result else None}


def run_demo_p3() -> dict:
    analyst = _run_as("analyst")
    public = _run_as("public")
    blocked = None
    try:  # cost guardrail: a query without LIMIT is blocked before execution
        PolicyGovernance(_POLICY).before_query("SELECT primaryName FROM people", role="public")
    except GovernanceError as exc:
        blocked = str(exc)
    return {"analyst": analyst, "public": public, "cost_guardrail_block": blocked}


if __name__ == "__main__":
    out = run_demo_p3()
    print("analyst rows:", out["analyst"]["rows"])
    print("public rows: ", out["public"]["rows"])
    print("cost guardrail:", out["cost_guardrail_block"])
    print("demo-p3 OK")
