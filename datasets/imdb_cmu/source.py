# datasets/imdb_cmu/source.py
"""Connector: IMDb non-commercial TSVs -> filtered Parquet.

Filters to movies with at least `min_votes` votes to keep the dev working set
small (16 GB constraint). Raw TSVs are downloaded via scripts/download_data.sh and
never committed. This module is dataset-specific; the engine never imports it.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def build_parquet(raw_dir: str, out_dir: str, min_votes: int = 1000) -> list[str]:
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    def csv(name: str) -> str:
        path = (raw / name).as_posix()
        return (
            f"read_csv('{path}', delim='\t', header=true, quote='', "
            f"nullstr='\\N', compression='gzip', all_varchar=true)"
        )

    con.execute(
        f"""
        CREATE TABLE ratings AS
        SELECT tconst,
               TRY_CAST(averageRating AS DOUBLE) AS averageRating,
               TRY_CAST(numVotes AS BIGINT) AS numVotes
        FROM {csv("title.ratings.tsv.gz")}
        """
    )
    con.execute(
        f"""
        CREATE TABLE basics AS
        SELECT tconst, titleType, primaryTitle, originalTitle,
               TRY_CAST(startYear AS INTEGER) AS startYear,
               TRY_CAST(runtimeMinutes AS INTEGER) AS runtimeMinutes,
               genres
        FROM {csv("title.basics.tsv.gz")}
        WHERE titleType = 'movie'
        """
    )
    con.execute(
        f"""
        CREATE TABLE popular AS
        SELECT b.tconst
        FROM basics b JOIN ratings r ON b.tconst = r.tconst
        WHERE r.numVotes >= {int(min_votes)}
        """
    )
    con.execute(
        f"""
        CREATE TABLE crew AS
        SELECT tconst, directors, writers FROM {csv("title.crew.tsv.gz")}
        """
    )
    con.execute(
        """
        CREATE TABLE director_ids AS
        SELECT DISTINCT UNNEST(string_split(directors, ',')) AS nconst
        FROM crew
        WHERE tconst IN (SELECT tconst FROM popular) AND directors IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TABLE names AS
        SELECT nconst, primaryName,
               TRY_CAST(birthYear AS INTEGER) AS birthYear,
               TRY_CAST(deathYear AS INTEGER) AS deathYear,
               primaryProfession
        FROM {csv("name.basics.tsv.gz")}
        """
    )

    written: list[str] = []

    def export(select_sql: str, fname: str) -> None:
        path = (out / fname).as_posix()
        con.execute(f"COPY ({select_sql}) TO '{path}' (FORMAT PARQUET)")
        written.append(path)

    export(
        "SELECT * FROM basics WHERE tconst IN (SELECT tconst FROM popular)",
        "title_basics.parquet",
    )
    export(
        "SELECT * FROM ratings WHERE tconst IN (SELECT tconst FROM popular)",
        "title_ratings.parquet",
    )
    export(
        "SELECT * FROM crew WHERE tconst IN (SELECT tconst FROM popular)",
        "title_crew.parquet",
    )
    export(
        "SELECT * FROM names WHERE nconst IN (SELECT nconst FROM director_ids)",
        "name_basics.parquet",
    )
    return written
