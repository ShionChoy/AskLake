from engine.graph.retriever import RetrievedSubgraph
from engine.ports.graph_store import Triple
from engine.retrieval.graph_rag_path import GraphRagPath


class _FakeRetriever:
    def __init__(self, sg):
        self._sg = sg
        self.seeds_calls = []

    def seeds(self, q):
        self.seeds_calls.append(q)
        return list(self._sg.seeds)

    def retrieve(self, q):
        return self._sg


def test_injected_retriever_used_and_store_not_required():
    sg = RetrievedSubgraph(
        seeds=("Inception",),
        triples=(Triple("Inception", "HAS_THEME", "time", "wiki:tt1"),),
    )
    fake = _FakeRetriever(sg)
    path = GraphRagPath(store=None, retriever=fake)  # store=None must NOT crash
    rr = path.run("themes of inception")
    assert rr.result is not None
    assert rr.result.rows[0] == ("Inception", "HAS_THEME", "time", "wiki:tt1")


def test_injected_retriever_drives_can_handle_seeds():
    sg = RetrievedSubgraph(seeds=("Inception",), triples=())
    fake = _FakeRetriever(sg)
    path = GraphRagPath(store=None, retriever=fake)
    assert path.can_handle("inception") is True  # no hint word -> consults retriever.seeds
    assert fake.seeds_calls == ["inception"]
