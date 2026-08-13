import gzip

import duckdb

from datasets.imdb.source import build_parquet


def _write_gz(path, header, rows):
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fixture(raw):
    _write_gz(
        raw / "title.basics.tsv.gz",
        [
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "isAdult",
            "startYear",
            "endYear",
            "runtimeMinutes",
            "genres",
        ],  # noqa: E501
        [
            ["tt1", "movie", "Pop Movie", "Pop Movie", "0", "2011", "\\N", "120", "Sci-Fi"],
            ["tt2", "movie", "Niche Movie", "Niche Movie", "0", "2009", "\\N", "90", "Drama"],
            ["tt3", "tvEpisode", "An Episode", "An Episode", "0", "2010", "\\N", "30", "Comedy"],
            ["tt4", "tvSeries", "A Show", "A Show", "0", "2012", "\\N", "\\N", "Drama"],
        ],
    )
    _write_gz(
        raw / "title.ratings.tsv.gz",
        ["tconst", "averageRating", "numVotes"],
        [
            ["tt1", "8.5", "5000"],
            ["tt2", "7.0", "50"],
            ["tt3", "6.0", "99999"],
            ["tt4", "8.0", "3000"],
        ],
    )
    _write_gz(
        raw / "title.crew.tsv.gz",
        ["tconst", "directors", "writers"],
        [["tt1", "nm1", "nm9"], ["tt2", "nm2", "\\N"]],
    )
    _write_gz(
        raw / "name.basics.tsv.gz",
        ["nconst", "primaryName", "birthYear", "deathYear", "primaryProfession", "knownForTitles"],
        [
            ["nm1", "Dir One", "1970", "\\N", "director", "tt1"],
            ["nm2", "Dir Two", "1980", "\\N", "director", "tt2"],
            ["nm3", "Actor One", "1990", "\\N", "actor", "tt1"],
        ],  # noqa: E501
    )
    _write_gz(
        raw / "title.principals.tsv.gz",
        ["tconst", "ordering", "nconst", "category", "job", "characters"],
        [
            ["tt1", "1", "nm3", "actor", "\\N", '["Neo"]'],
            ["tt1", "2", "nm1", "director", "\\N", "\\N"],
            ["tt2", "1", "nm2", "director", "\\N", "\\N"],
        ],
    )


def test_build_parquet_filters_movies_and_min_votes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fixture(raw)
    out = tmp_path / "out"
    written = build_parquet(str(raw), str(out), min_votes=1000)

    con = duckdb.connect()
    basics = con.execute(
        f"SELECT tconst FROM read_parquet('{out}/title_basics.parquet')"
    ).fetchall()  # noqa: E501
    assert {r[0] for r in basics} == {"tt1"}  # tt2 too few votes, tt3 not a movie

    names = con.execute(f"SELECT nconst FROM read_parquet('{out}/name_basics.parquet')").fetchall()
    assert {r[0] for r in names} == {"nm1", "nm3"}  # director + actor of the popular movie

    assert any(p.endswith("title_ratings.parquet") for p in written)
    assert any(p.endswith("title_crew.parquet") for p in written)
    assert any(p.endswith("title_principals.parquet") for p in written)


def test_build_parquet_includes_requested_title_types(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fixture(raw)
    out = tmp_path / "out"
    build_parquet(str(raw), str(out), min_votes=1000, title_types=("movie", "tvSeries"))
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT tconst, titleType FROM read_parquet('{out}/title_basics.parquet')"
    ).fetchall()
    got = {r[0] for r in rows}
    assert got == {"tt1", "tt4"}  # movie + tvSeries (tt2 too few votes, tt3 wrong type)


def test_build_parquet_default_is_movies_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fixture(raw)
    out = tmp_path / "out"
    build_parquet(str(raw), str(out), min_votes=1000)  # default title_types
    con = duckdb.connect()
    got = {
        r[0]
        for r in con.execute(
            f"SELECT tconst FROM read_parquet('{out}/title_basics.parquet')"
        ).fetchall()
    }
    assert got == {"tt1"}  # tt4 (tvSeries) excluded by the movies-only default
