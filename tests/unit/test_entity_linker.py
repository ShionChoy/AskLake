from engine.graph.entity_linker import LexicalEntityLinker
from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple


def _store(*entities_as_subjects):
    g = InMemoryGraphStore()
    for s in entities_as_subjects:
        g.add(Triple(s, "HAS_THEME", "x", "src"))
    return g


def test_exact_span_links_full_title_not_fragment():
    g = _store("The Dark Knight", "The Dark", "Dark")
    linker = LexicalEntityLinker(g)
    assert linker.seeds("who directed the dark knight") == ["The Dark Knight"]


def test_fragment_links_only_when_named_alone():
    g = _store("The Dark Knight", "The Dark")
    linker = LexicalEntityLinker(g)
    assert linker.seeds("tell me about the dark") == ["The Dark"]


def test_two_distinct_entities_both_seed():
    g = _store("Inception", "The Dark Knight")
    assert set(LexicalEntityLinker(g).seeds("inception and the dark knight")) == {
        "Inception",
        "The Dark Knight",
    }


def test_structure_word_title_does_not_seed():
    g = _store("The Theme", "Inception")
    assert LexicalEntityLinker(g).seeds("the theme of inception") == ["Inception"]


def test_single_content_word_title_seeds():
    assert LexicalEntityLinker(_store("Up")).seeds("the theme of up") == ["Up"]


def test_attribute_objects_excluded():
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_GENRE", "Drama", "s"))
    g.add(Triple("Inception", "HAS_THEME", "dreams", "s"))
    linker = LexicalEntityLinker(g, attribute_relations=frozenset({"HAS_GENRE"}))
    assert "Drama" not in linker.seeds("drama")
    assert linker.seeds("inception") == ["Inception"]


def test_top_k_and_specificity_order():
    g = _store("Batman", "Batman Begins")
    assert LexicalEntityLinker(g, top_k=1).seeds("the batman begins story") == ["Batman Begins"]


def test_empty_question_no_seeds():
    assert LexicalEntityLinker(_store("Inception")).seeds("") == []


def test_common_verb_does_not_seed_short_title():
    g = _store("Do", "Inception", "Interstellar")
    linker = LexicalEntityLinker(g)
    seeds = linker.seeds("what themes do Inception and Interstellar share")
    assert "Do" not in seeds
    assert set(seeds) == {"Inception", "Interstellar"}


def test_content_one_word_title_still_seeds_after_guard():
    assert LexicalEntityLinker(_store("Up")).seeds("the theme of up") == ["Up"]
