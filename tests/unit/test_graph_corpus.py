from datasets.imdb_cmu.graph_corpus import _normalize, load_plot_docs


def test_normalize_strips_punct_and_case():
    assert _normalize("The Dark Knight!") == "the dark knight"
    assert _normalize("WALL·E (2008)") == "wall e 2008"


def _write_cmu(tmp_path):
    # movie.metadata.tsv: wiki_id, freebase, name, release_date, box, runtime,
    # langs, countries, genres
    (tmp_path / "movie.metadata.tsv").write_text(
        "1\t/m/a\tThe Dark Knight\t2008-07-18\t\t152.0\t{}\t{}\t{}\n"
        "2\t/m/b\tInception\t2010-07-16\t\t148.0\t{}\t{}\t{}\n"
        "3\t/m/c\tSome Obscure Film\t1975\t\t90.0\t{}\t{}\t{}\n"
    )
    (tmp_path / "plot_summaries.txt").write_text(
        "1\tBatman faces the Joker, an agent of chaos.\n"
        "2\tA thief enters dreams to plant an idea.\n"
        "3\tNobody has heard of this one.\n"
    )
    return str(tmp_path)


def test_load_plot_docs_aligns_to_imdb_titles(tmp_path):
    cmu = _write_cmu(tmp_path)
    imdb_titles = {
        "the dark knight": ("The Dark Knight", 2008),
        "inception": ("Inception", 2010),
    }
    docs = load_plot_docs(cmu, imdb_titles, max_films=10)
    by_title = {d.title: d for d in docs}
    assert set(by_title) == {"The Dark Knight", "Inception"}  # obscure film excluded
    assert "Joker" in by_title["The Dark Knight"].text
    assert by_title["The Dark Knight"].id == "cmu:1"  # citation id


def test_load_plot_docs_excludes_on_year_mismatch(tmp_path):
    cmu = _write_cmu(tmp_path)
    imdb_titles = {"inception": ("Inception", 1999)}  # >1 year off the CMU 2010 -> excluded
    assert load_plot_docs(cmu, imdb_titles, max_films=10) == []


def test_load_plot_docs_caps(tmp_path):
    cmu = _write_cmu(tmp_path)
    imdb_titles = {
        "the dark knight": ("The Dark Knight", 2008),
        "inception": ("Inception", 2010),
    }
    assert len(load_plot_docs(cmu, imdb_titles, max_films=1)) == 1


def test_load_plot_docs_accepts_when_cmu_year_missing(tmp_path):
    (tmp_path / "movie.metadata.tsv").write_text(
        "1\t/m/a\tThe Dark Knight\t\t\t152.0\t{}\t{}\t{}\n"  # empty release_date -> no CMU year
    )
    (tmp_path / "plot_summaries.txt").write_text("1\tBatman plot.\n")
    imdb_titles = {"the dark knight": ("The Dark Knight", 2008)}
    docs = load_plot_docs(str(tmp_path), imdb_titles, max_films=10)
    assert len(docs) == 1  # accepted despite missing CMU year
