from __future__ import annotations

from datasets.crm_demo.source import build_parquet
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.semantic.semantic_model import load_semantic_layer
from eval.crm_gold import CRM_GOLD


def test_crm_gold_shape():
    assert len(CRM_GOLD) >= 10
    names = [c.name for c in CRM_GOLD]
    assert len(names) == len(set(names))
    assert {c.tier for c in CRM_GOLD} == {"aggregation", "topn", "multihop"}


def test_crm_semantic_layer_has_link_flags_and_minimal_fewshots():
    layer = load_semantic_layer("datasets/crm_demo/semantic.yaml")
    links = {c.name: c.link for t in layer.tables for c in t.columns if c.link}
    assert links.get("status") == "categorical"
    assert links.get("region") == "categorical"
    assert len(layer.few_shots) <= 1  # generalization: not hand-tuned with examples


def test_crm_gold_executes_on_built_parquet(tmp_path):
    out = tmp_path / "crm"
    build_parquet(str(out))
    b = DuckDBBackend(parquet_dir=str(out))
    for c in CRM_GOLD:
        res = b.run_sql(c.gold_sql)
        assert res.rows, f"gold returned no rows for {c.name}: {c.gold_sql}"
    # top-N tie-safety: the ordered metric (last selected column) must be strictly decreasing with
    # no ties, else multiset-exact scoring is ambiguous.
    for c in CRM_GOLD:
        if c.tier == "topn":
            metric = [r[-1] for r in b.run_sql(c.gold_sql).rows]
            assert metric == sorted(metric, reverse=True), f"topn not descending in {c.name}"
            assert len(set(metric)) == len(metric), f"within-result tie in {c.name}: {metric}"
