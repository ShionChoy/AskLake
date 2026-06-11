from engine.graph.extraction import PlotDoc
from engine.graph.ontology import GraphOntology
from engine.graph.persistence import load_store
from engine.llm.fake import FakeLLMProvider
from scripts.build_graph import build_and_save


def test_build_and_save_writes_triples(tmp_path):
    docs = [
        PlotDoc(id="cmu:1", title="The Dark Knight", text="Batman vs the Joker."),
        PlotDoc(id="cmu:2", title="Inception", text="A thief in dreams."),
    ]
    ontology = GraphOntology(
        entity_types=("Film", "Person", "Theme"),
        relation_types=("HAS_THEME", "DIRECTED_BY"),
        hint="Extract director and themes.",
    )
    extraction = [
        "The Dark Knight | DIRECTED_BY | Christopher Nolan\nThe Dark Knight | HAS_THEME | chaos\n",
        "Inception | DIRECTED_BY | Christopher Nolan\nInception | HAS_THEME | dreams\n",
    ]
    out = tmp_path / "triples.jsonl"
    count = build_and_save(docs, FakeLLMProvider(extraction), ontology, out)
    assert count == 4
    store = load_store(out)
    assert "Christopher Nolan" in store.entities()
    rows = [(t.subject, t.relation, t.obj, t.source) for t in store.triples()]
    assert ("The Dark Knight", "HAS_THEME", "chaos", "cmu:1") in rows


def test_build_and_save_parallel_writes_structured_then_themes(tmp_path):
    from engine.ports.graph_store import Triple
    from scripts.build_graph import build_and_save_parallel

    docs = [
        PlotDoc(id="cmu:1", title="The Dark Knight", text="Batman vs the Joker."),
        PlotDoc(id="cmu:2", title="Inception", text="A thief in dreams."),
    ]
    ontology = GraphOntology(relation_types=("HAS_THEME",), hint="themes")
    extraction = [
        "The Dark Knight | HAS_THEME | chaos\n",
        "Inception | HAS_THEME | dreams\n",
    ]
    structured = [Triple("The Dark Knight", "HAS_GENRE", "Action", "cmu:1")]
    out = tmp_path / "triples.jsonl"
    count = build_and_save_parallel(
        structured, docs, FakeLLMProvider(extraction), ontology, out, workers=2
    )
    assert count == 2  # theme-triple count (structured are written separately)
    rows = {(t.subject, t.relation, t.obj) for t in load_store(out).triples()}
    assert ("The Dark Knight", "HAS_GENRE", "Action") in rows  # structured present
    assert ("Inception", "HAS_THEME", "dreams") in rows  # theme present


def test_theme_ontology_restricts_to_theme_and_setting():
    from engine.graph.ontology import load_ontology
    from scripts.build_graph import theme_ontology

    full = load_ontology("datasets/imdb_cmu/graph/ontology.yaml")
    restricted = theme_ontology(full)
    assert set(restricted.relation_types) == {"HAS_THEME", "SET_IN"}


def test_theme_extraction_drops_structured_relations():
    from engine.graph.extraction import PlotDoc, extract_triples
    from engine.graph.ontology import load_ontology
    from engine.llm.fake import FakeLLMProvider
    from scripts.build_graph import theme_ontology

    restricted = theme_ontology(load_ontology("datasets/imdb_cmu/graph/ontology.yaml"))
    # canned LLM output mixing a theme, a director, and a setting
    llm = FakeLLMProvider(
        responses=["X | HAS_THEME | chaos\nX | DIRECTED_BY | Someone\nX | SET_IN | Gotham\n"]
    )
    doc = PlotDoc(id="wiki:tt1", title="X", text="...")
    rels = {t.relation for t in extract_triples(llm, doc, restricted)}
    assert rels == {"HAS_THEME", "SET_IN"}  # DIRECTED_BY filtered out by restricted ontology


def test_build_parallel_skips_film_on_extraction_error(tmp_path, monkeypatch):
    import json

    import scripts.build_graph as bg
    from engine.graph.extraction import PlotDoc
    from engine.graph.ontology import GraphOntology

    monkeypatch.setattr(bg.time, "sleep", lambda *_a, **_k: None)  # no real backoff in tests

    class _FlakyLLM:
        def complete(self, prompt, system=None):
            if "BOOM" in prompt:
                raise RuntimeError("llm down")
            return "X | HAS_THEME | chaos\n"

    ont = GraphOntology(relation_types=("HAS_THEME",))
    docs = [PlotDoc("wiki:tt1", "Good", "ok"), PlotDoc("wiki:tt2", "Bad", "BOOM plot")]
    out = tmp_path / "g.jsonl"
    count = bg.build_and_save_parallel([], docs, _FlakyLLM(), ont, str(out), workers=2)
    rels = [json.loads(line)["relation"] for line in out.read_text().splitlines() if line.strip()]
    assert count == 1 and rels == ["HAS_THEME"]  # good film written, bad film skipped, no abort
