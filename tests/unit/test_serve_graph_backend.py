# Uses monkeypatch.setattr on the module constants (auto-restored, no reload pollution) and
# points the graph-path fallback at a nonexistent file so the test never loads the real
# 170 MB triples.jsonl.


def test_graph_backend_constant_exists():
    import api.serve as serve

    assert hasattr(serve, "GRAPH_BACKEND")


def test_neo4j_backend_falls_back_when_unreachable(monkeypatch, tmp_path):
    import api.serve as serve

    monkeypatch.setattr(serve, "GRAPH_BACKEND", "neo4j")
    monkeypatch.setattr(serve, "NEO4J_URI", "bolt://127.0.0.1:1")  # nothing listening
    monkeypatch.setattr(serve, "NEO4J_PASSWORD", "x")
    monkeypatch.setattr(serve, "GRAPH_PATH", str(tmp_path / "nope.jsonl"))  # no fallback file

    app = serve.build_app()  # must NOT crash despite neo4j being unreachable
    assert app.state.graph_enabled is False
