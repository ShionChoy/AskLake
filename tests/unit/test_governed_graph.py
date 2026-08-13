from engine.governance.graph import GovernedGraphPath
from engine.governance.policy import (
    Classification,
    GraphRelationPolicy,
    Policy,
    RolePolicy,
)
from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple
from engine.retrieval.graph_rag_path import GraphRagPath


def test_graph_policy_filters_unknown_uncited_and_row_denied_sources():
    store = InMemoryGraphStore()
    store.add(Triple("Film", "HAS_THEME", "identity", "wiki:1"))
    store.add(Triple("Film", "SET_IN", "Tokyo", ""))
    store.add(Triple("Film", "UNCLASSIFIED", "secret", "internal:1"))
    store.add(Triple("Film", "HAS_THEME", "forbidden", "wiki:adult"))
    policy = Policy(
        version=2,
        roles=("public",),
        role_rules={
            "public": RolePolicy(
                actions=frozenset({"graph"}),
                graph_relations=("HAS_THEME", "SET_IN"),
                max_graph_triples=10,
            )
        },
        graph_default_effect="deny",
        graph_relations={
            "HAS_THEME": GraphRelationPolicy(
                Classification(integrity="llm_inferred"), citation_required=True
            ),
            "SET_IN": GraphRelationPolicy(
                Classification(integrity="llm_inferred"), citation_required=True
            ),
        },
    )
    path = GovernedGraphPath(
        GraphRagPath(store), policy, "public", denied_sources=frozenset({"wiki:adult"})
    )
    result = path.run("Film themes and setting")
    assert result.result is not None
    assert result.result.rows == [("Film", "HAS_THEME", "identity", "wiki:1")]
    assert "model-inferred" in result.narrative
