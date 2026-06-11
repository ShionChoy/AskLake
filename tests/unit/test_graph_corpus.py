import duckdb

from datasets.imdb_cmu.graph_corpus import select_plot_docs


def _fixture_parquet(pq):
    pq.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "COPY (SELECT * FROM (VALUES "
        "('tt1','movie','The Matrix'),('tt2','movie','Inception')) "
        f"v(tconst,titleType,primaryTitle)) TO '{pq}/title_basics.parquet' (FORMAT PARQUET)"
    )
    con.execute(
        "COPY (SELECT * FROM (VALUES ('tt1',2000000),('tt2',2400000)) "
        f"v(tconst,numVotes)) TO '{pq}/title_ratings.parquet' (FORMAT PARQUET)"
    )
    return str(pq)


class _FakeWiki:
    def resolve(self, tconsts):
        return {"tt1": "The Matrix", "tt2": "Inception"}

    def plot(self, title):
        return {"The Matrix": "Neo learns the truth.", "Inception": "A thief enters dreams."}[title]


def test_select_plot_docs_ranks_by_votes_and_caps(tmp_path):
    docs = select_plot_docs(_fixture_parquet(tmp_path / "pq"), max_films=1, wiki=_FakeWiki())
    assert [d.title for d in docs] == ["Inception"]  # 2.4M > 2.0M votes
    assert docs[0].id == "wiki:tt2" and "dreams" in docs[0].text


def test_select_plot_docs_skips_films_without_plot(tmp_path):
    class _Partial(_FakeWiki):
        def plot(self, title):
            return "Neo learns the truth." if title == "The Matrix" else None

    docs = select_plot_docs(_fixture_parquet(tmp_path / "pq"), max_films=10, wiki=_Partial())
    assert {d.title for d in docs} == {"The Matrix"}  # Inception had no plot
