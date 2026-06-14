from engine.graph.entity_linker import LexicalEntityLinker


def test_from_names_exact_span_and_maximal():
    linker = LexicalEntityLinker.from_names(["The Dark Knight", "The Dark", "Drama Queen"])
    assert linker.seeds("who directed the dark knight") == ["The Dark Knight"]


def test_from_names_skips_all_stopword_titles():
    linker = LexicalEntityLinker.from_names(["The Theme"])
    assert linker.seeds("what is the theme of life") == []


def test_from_names_respects_top_k():
    linker = LexicalEntityLinker.from_names(["Inception", "Interstellar"], top_k=1)
    assert len(linker.seeds("inception interstellar")) == 1
