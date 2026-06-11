from engine.graph.store import InMemoryGraphStore
from engine.llm.fake import FakeLLMProvider
from engine.ports.graph_store import Triple
from engine.ports.retrieval import RetrievalPath
from engine.retrieval.graph_rag_path import GraphRagPath, GroundedGraphRagPath


def _store():
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_THEME", "dreams", "wiki:tt1"))
    return g


def test_is_retrieval_path_named_graph():
    p = GroundedGraphRagPath(GraphRagPath(_store()), FakeLLMProvider(responses=["x"]))
    assert isinstance(p, RetrievalPath) and p.name == "graph"


def test_narrative_is_llm_answer_result_keeps_triples():
    llm = FakeLLMProvider(responses=["Inception explores dreams [wiki:tt1]."])
    p = GroundedGraphRagPath(GraphRagPath(_store()), llm)
    rr = p.run("themes in Inception")
    assert rr.narrative == "Inception explores dreams [wiki:tt1]."
    assert rr.result is not None
    assert ("Inception", "HAS_THEME", "dreams", "wiki:tt1") in rr.result.rows


def test_empty_subgraph_skips_llm_and_returns_hint():
    llm = FakeLLMProvider(responses=["should not be used"])
    p = GroundedGraphRagPath(GraphRagPath(_store(), empty_hint="nothing in graph"), llm)
    rr = p.run("totally unrelated zzz")
    assert rr.result is None
    assert rr.narrative == "nothing in graph"
