import gzip

import duckdb

from datasets.imdb_cmu.source import build_parquet


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
        ],
    )
    _write_gz(
        raw / "title.ratings.tsv.gz",
        ["tconst", "averageRating", "numVotes"],
        [["tt1", "8.5", "5000"], ["tt2", "7.0", "50"], ["tt3", "6.0", "99999"]],
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
        ],  # noqa: E501
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
    assert {r[0] for r in names} == {"nm1"}  # only director of the popular movie

    assert any(p.endswith("title_ratings.parquet") for p in written)
    assert any(p.endswith("title_crew.parquet") for p in written)
