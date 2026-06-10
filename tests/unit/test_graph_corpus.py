from datasets.imdb_cmu.graph_corpus import _normalize, aligned_films, select_plot_docs


def test_normalize_strips_punct_and_case():
    assert _normalize("The Dark Knight!") == "the dark knight"
    assert _normalize("WALL·E (2008)") == "wall e 2008"


def _write_cmu(tmp_path):
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


# index: {normtitle: (primaryTitle, year, numVotes, tconst)}
_INDEX = {
    "the dark knight": ("The Dark Knight", 2008, 2_700_000, "tt0468569"),
    "inception": ("Inception", 2010, 2_400_000, "tt1375666"),
}


def test_aligned_films_maps_wiki_to_imdb_title_tconst_votes(tmp_path):
    out = aligned_films(_write_cmu(tmp_path), _INDEX)
    assert out["1"] == ("The Dark Knight", "tt0468569", 2_700_000)
    assert out["2"] == ("Inception", "tt1375666", 2_400_000)
    assert "3" not in out  # obscure film not in the IMDb index


def test_aligned_films_excludes_on_year_mismatch(tmp_path):
    idx = {"inception": ("Inception", 1999, 10, "tt1375666")}  # >1yr off CMU 2010
    assert aligned_films(_write_cmu(tmp_path), idx) == {}


def test_select_plot_docs_ranks_by_votes_and_caps(tmp_path):
    cmu = _write_cmu(tmp_path)
    docs = select_plot_docs(cmu, _INDEX, max_films=1)
    assert [d.title for d in docs] == ["The Dark Knight"]  # higher votes wins the single slot
    assert docs[0].id == "cmu:1" and "Joker" in docs[0].text


def test_select_plot_docs_returns_all_when_under_cap(tmp_path):
    docs = select_plot_docs(_write_cmu(tmp_path), _INDEX, max_films=10)
    assert {d.title for d in docs} == {"The Dark Knight", "Inception"}
