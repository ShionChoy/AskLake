from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.semantic.semantic_model import SemanticLayer
from eval.harness import EvalCase, SystemReport
from eval.real_run import run_real_eval

_SEED = """
CREATE TABLE movies AS SELECT * FROM (VALUES
    ('Alpha', 8.9), ('Beta', 7.5)
) t(title, averageRating);
"""

_CASES = [
    EvalCase(
        name="top",
        schema_sql="",
        question="highest rated movie",
        gold_sql="SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1",
    )
]


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(_SEED)
    return b


def test_run_real_eval_returns_three_reports():
    # FakeLLM emits the correct SQL for every system -> all three score 100% exec-acc.
    correct = "SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1"
    llm = FakeLLMProvider(responses=[correct])
    reports = run_real_eval(llm, _backend(), _CASES, SemanticLayer())
    assert isinstance(reports, list) and len(reports) == 3
    names = [r.name for r in reports]
    assert names == ["baseline", "agentic", "semantic"]
    for r in reports:
        assert isinstance(r, SystemReport)
        assert r.n == 1
        assert r.execution_accuracy == 1.0
        assert r.valid_sql_rate == 1.0
