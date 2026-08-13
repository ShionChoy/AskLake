import duckdb

from datasets.imdb.graph_structured import structured_triples


def _fixture_parquet(pq):
    pq.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    def copy(select, name):
        con.execute(f"COPY ({select}) TO '{pq}/{name}.parquet' (FORMAT PARQUET)")

    copy(
        "SELECT * FROM (VALUES "
        "('tt1','movie','The Matrix',1999,'Action,Sci-Fi'),"
        "('tt2','movie','Obscure',2001,'Drama')) "
        "v(tconst,titleType,primaryTitle,startYear,genres)",
        "title_basics",
    )
    copy(
        "SELECT * FROM (VALUES ('tt1',8.7,2000000),('tt2',5.0,300)) "
        "v(tconst,averageRating,numVotes)",
        "title_ratings",
    )
    copy("SELECT * FROM (VALUES ('tt1','nm9','\\N')) v(tconst,directors,writers)", "title_crew")
    copy(
        "SELECT * FROM (VALUES "
        "('nm1','Keanu Reeves'),('nm9','Lana Wachowski'),('nm5','Extra Person')) "
        "v(nconst,primaryName)",
        "name_basics",
    )
    copy(
        "SELECT * FROM (VALUES "
        "('tt1',1,'nm1','actor','[\"Neo\"]'),"
        "('tt1',99,'nm5','actor','[\"Bit Part\"]')) "
        "v(tconst,ordering,nconst,category,characters)",
        "title_principals",
    )
    return str(pq)


def _rows(parquet_dir, **kw):
    return {(t.subject, t.relation, t.obj) for t in structured_triples(parquet_dir, **kw)}


def test_emits_imdb_native_film_facts(tmp_path):
    rows = _rows(_fixture_parquet(tmp_path / "pq"), min_votes=1000, cast_cap=10)
    assert ("The Matrix", "HAS_GENRE", "Action") in rows
    assert ("The Matrix", "HAS_GENRE", "Sci-Fi") in rows
    assert ("The Matrix", "RELEASED_IN", "1999") in rows
    assert ("The Matrix", "DIRECTED_BY", "Lana Wachowski") in rows
    assert ("Keanu Reeves", "ACTED_IN", "The Matrix") in rows
    assert ("The Matrix", "FEATURES_CHARACTER", "Neo") in rows
    assert ("Neo", "PLAYED_BY", "Keanu Reeves") in rows


def test_min_votes_filters_unpopular_films(tmp_path):
    rows = _rows(_fixture_parquet(tmp_path / "pq"), min_votes=1000, cast_cap=10)
    assert not any(s == "Obscure" for s, _, _ in rows)


def test_cast_cap_drops_low_billed(tmp_path):
    rows = _rows(_fixture_parquet(tmp_path / "pq"), min_votes=1000, cast_cap=10)
    assert ("Extra Person", "ACTED_IN", "The Matrix") not in rows


def test_no_language_or_country_relations(tmp_path):
    rels = {t.relation for t in structured_triples(_fixture_parquet(tmp_path / "pq"))}
    assert "IN_LANGUAGE" not in rels and "FROM_COUNTRY" not in rels


def test_source_is_imdb_tconst(tmp_path):
    for t in structured_triples(_fixture_parquet(tmp_path / "pq")):
        assert t.source.startswith("imdb:")
