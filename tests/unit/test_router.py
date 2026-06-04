from engine.ports.retrieval import RetrievalPath, RetrievalResult
from engine.ports.storage import QueryResult
from engine.retrieval.router import Router, RoutingDecision


class _SqlStub:
    name = "sql"

    def can_handle(self, question: str) -> bool:
        return True

    def run(self, question: str) -> RetrievalResult:
        return RetrievalResult(
            path="sql",
            sql="SELECT title FROM films LIMIT 1",
            result=QueryResult(columns=["title"], rows=[("Inception",)]),
            narrative="Returned 1 row(s).",
            chart_spec=None,
        )


class _GraphStub:
    name = "graph"

    def can_handle(self, question: str) -> bool:
        return True

    def run(self, question: str) -> RetrievalResult:
        return RetrievalResult(
            path="graph",
            sql=None,
            result=QueryResult(columns=["object"], rows=[("dreams",)]),
            narrative="Inception HAS_THEME dreams [plot_inception]",
            chart_spec=None,
        )


def _router() -> Router:
    return Router(_SqlStub(), _GraphStub(), entity_vocab={"Inception"})


def test_router_is_retrieval_path():
    assert isinstance(_router(), RetrievalPath)


def test_route_pure_sql_question():
    d = _router().route("How many films are rated above 8?")
    assert d == RoutingDecision(paths=("sql",), fuse=False)


def test_route_pure_graph_question():
    d = _router().route("What are the common themes across these plots?")
    assert d == RoutingDecision(paths=("graph",), fuse=False)


def test_route_fusion_question():
    d = _router().route("Highest-rated films before 2013 and their common plot themes")
    assert d == RoutingDecision(paths=("sql", "graph"), fuse=True)


def test_tie_with_named_entity_prefers_graph():
    # no sql/graph hint words at all -> tie; a named entity tips toward graph
    d = _router().route("Tell me Inception")
    assert d.paths == ("graph",) and d.fuse is False


def test_no_graph_path_always_sql():
    r = Router(_SqlStub(), graph_path=None)
    assert r.route("common themes in Inception") == RoutingDecision(paths=("sql",), fuse=False)


def test_run_fusion_combines_both_with_citation():
    rr = _router().run("Highest-rated films before 2013 and their common plot themes")
    assert rr.path == "sql+graph"
    assert rr.result is not None and rr.result.columns == ["title"]  # structured table from SQL
    assert "[sql]" in rr.narrative and "[graph]" in rr.narrative
    assert "[plot_inception]" in rr.narrative  # graph citation preserved


def test_run_single_path_delegates():
    rr = _router().run("How many films are rated above 8?")
    assert rr.path == "sql"
