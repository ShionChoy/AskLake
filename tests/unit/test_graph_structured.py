from datasets.imdb_cmu.graph_structured import imdb_crew_triples, structured_triples


def _write_cmu(tmp_path):
    # movie.metadata: wiki_id, fb, name, release, box, runtime, langs, countries, genres (JSON)
    (tmp_path / "movie.metadata.tsv").write_text(
        "1\t/m/a\tThe Dark Knight\t2008-07-18\t\t152.0"
        '\t{"/m/02h40lc": "English Language"}\t{"/m/09c7w0": "United States of America"}'
        '\t{"/m/01jfsb": "Thriller", "/m/03npn": "Action"}\n'
        "9\t/m/z\tUnaligned Film\t1950\t\t80.0\t{}\t{}\t{}\n"
    )
    # character.metadata: wiki_id, fb, release, charName, dob, gender, height, ethnicity, actorName
    char_row = (
        "1\t/m/a\t2008-07-18\tBruce Wayne\t1974-06-09\tM\t1.8\t\t"
        "Christian Bale\t34\t/m/x\t/m/y\t/m/z\n"
    )
    (tmp_path / "character.metadata.tsv").write_text(char_row)
    return str(tmp_path)


def test_structured_triples_emit_aligned_film_facts(tmp_path):
    cmu = _write_cmu(tmp_path)
    aligned = {"1": ("The Dark Knight", "tt1", 2_700_000)}  # wiki 9 not aligned -> skipped
    rows = {(t.subject, t.relation, t.obj) for t in structured_triples(cmu, aligned)}
    assert ("The Dark Knight", "HAS_GENRE", "Thriller") in rows
    assert ("The Dark Knight", "HAS_GENRE", "Action") in rows
    assert ("The Dark Knight", "IN_LANGUAGE", "English Language") in rows
    assert ("The Dark Knight", "FROM_COUNTRY", "United States of America") in rows
    assert ("The Dark Knight", "RELEASED_IN", "2008") in rows
    assert ("The Dark Knight", "FEATURES_CHARACTER", "Bruce Wayne") in rows
    assert ("Bruce Wayne", "PLAYED_BY", "Christian Bale") in rows
    assert ("Christian Bale", "ACTED_IN", "The Dark Knight") in rows
    assert not any(s == "Unaligned Film" for s, _, _ in rows)  # wiki 9 skipped


def test_structured_triples_carry_cmu_source(tmp_path):
    cmu = _write_cmu(tmp_path)
    for t in structured_triples(cmu, {"1": ("The Dark Knight", "tt1", 1)}):
        assert t.source == "cmu:1"


def test_imdb_crew_triples_emit_authoritative_directors(tmp_path):
    import duckdb

    pq = tmp_path / "pq"
    pq.mkdir()
    con = duckdb.connect()
    con.execute(
        "COPY (SELECT * FROM (VALUES ('tt1','nm1,nm2'), ('tt2','nm1')) v(tconst, directors)) "
        f"TO '{pq}/title_crew.parquet' (FORMAT PARQUET)"
    )
    con.execute(
        "COPY (SELECT * FROM (VALUES ('nm1','Christopher Nolan'), ('nm2','Emma Thomas')) "
        f"v(nconst, primaryName)) TO '{pq}/name_basics.parquet' (FORMAT PARQUET)"
    )
    aligned = {"1": ("The Dark Knight", "tt1", 2_700_000), "2": ("Inception", "tt2", 2_400_000)}
    rows = {(t.subject, t.relation, t.obj, t.source) for t in imdb_crew_triples(str(pq), aligned)}
    assert ("The Dark Knight", "DIRECTED_BY", "Christopher Nolan", "imdb:tt1") in rows
    assert ("The Dark Knight", "DIRECTED_BY", "Emma Thomas", "imdb:tt1") in rows
    assert ("Inception", "DIRECTED_BY", "Christopher Nolan", "imdb:tt2") in rows


def test_structured_triples_dedup_repeated_character_rows(tmp_path):
    (tmp_path / "movie.metadata.tsv").write_text(
        "1\t/m/a\tThe Dark Knight\t2008-07-18\t\t152.0\t{}\t{}\t{}\n"
    )
    (tmp_path / "character.metadata.tsv").write_text(
        "1\t/m/a\t2008\tBruce Wayne\t\tM\t\t\tChristian Bale\t34\t/m/x\t/m/y\t/m/z\n"
        "1\t/m/a\t2008\tBatman\t\tM\t\t\tChristian Bale\t34\t/m/x\t/m/y\t/m/z\n"
    )
    aligned = {"1": ("The Dark Knight", "tt1", 1)}
    from datasets.imdb_cmu.graph_structured import structured_triples

    acted = [
        t
        for t in structured_triples(str(tmp_path), aligned)
        if t.relation == "ACTED_IN" and t.obj == "The Dark Knight"
    ]
    assert len(acted) == 1  # Christian Bale ACTED_IN once despite 2 character rows
