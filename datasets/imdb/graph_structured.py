"""Deterministic structured triples built directly from IMDb parquet without an LLM.
Films are titled by IMDb primaryTitle so graph subjects match the SQL side. Covers the
numVotes>=min_votes movie set; cast capped at top-`cast_cap` billed principals. Dataset-specific;
the engine never imports this."""

from __future__ import annotations

import json
from collections.abc import Iterator

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.graph_store import Triple

_CAST_CATEGORIES = ("actor", "actress", "self")


def _characters(cell: str) -> list[str]:
    """IMDb `characters` is a JSON array string like '["Neo"]'; tolerate \\N / empty / bad JSON."""
    cell = (cell or "").strip()
    if not cell or cell == r"\N":
        return []
    try:
        return [c for c in json.loads(cell) if c]
    except (ValueError, TypeError):
        return []


def structured_triples(
    parquet_dir: str, *, min_votes: int = 1000, cast_cap: int = 10
) -> Iterator[Triple]:
    """Yield HAS_GENRE / RELEASED_IN / DIRECTED_BY / ACTED_IN / FEATURES_CHARACTER / PLAYED_BY
    for every movie with numVotes >= min_votes. source = 'imdb:<tconst>'."""
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    seen: set[tuple[str, str, str]] = set()

    def emit(s: str, rel: str, o: str, tconst: str) -> Iterator[Triple]:
        key = (s, rel, o)
        if s and o and key not in seen:
            seen.add(key)
            yield Triple(s, rel, o, f"imdb:{tconst}")

    # genres + year
    res = backend.run_sql(
        f"SELECT b.tconst, b.primaryTitle, b.startYear, b.genres "
        f"FROM title_basics b JOIN title_ratings r USING(tconst) "
        f"WHERE b.titleType='movie' AND r.numVotes >= {int(min_votes)}"
    )
    for tconst, title, year, genres in res.rows:
        for g in (genres or "").split(","):
            g = g.strip()
            if g and g != r"\N":
                yield from emit(title, "HAS_GENRE", g, tconst)
        if year not in (None, "", r"\N"):
            yield from emit(title, "RELEASED_IN", str(year), tconst)

    # directors (authoritative from IMDb)
    res = backend.run_sql(
        f"SELECT b.tconst, b.primaryTitle, n.primaryName FROM ("
        f"  SELECT tconst, UNNEST(string_split(directors, ',')) AS nconst FROM title_crew "
        f"  WHERE directors IS NOT NULL AND directors != '' AND directors != '\\N'"
        f") c JOIN title_basics b USING(tconst) JOIN title_ratings r USING(tconst) "
        f"JOIN name_basics n ON n.nconst = c.nconst "
        f"WHERE b.titleType='movie' AND r.numVotes >= {int(min_votes)}"
    )
    for tconst, title, director in res.rows:
        yield from emit(title, "DIRECTED_BY", director, tconst)

    # cast + characters (top-`cast_cap` billed)
    res = backend.run_sql(
        f"SELECT b.tconst, b.primaryTitle, n.primaryName, p.characters "
        f"FROM title_principals p "
        f"JOIN title_basics b USING(tconst) JOIN title_ratings r USING(tconst) "
        f"JOIN name_basics n ON n.nconst = p.nconst "
        f"WHERE b.titleType='movie' AND r.numVotes >= {int(min_votes)} "
        f"AND p.category IN {_CAST_CATEGORIES} AND p.ordering <= {int(cast_cap)}"
    )
    for tconst, title, actor, characters in res.rows:
        yield from emit(actor, "ACTED_IN", title, tconst)
        for ch in _characters(characters):
            yield from emit(title, "FEATURES_CHARACTER", ch, tconst)
            yield from emit(ch, "PLAYED_BY", actor, tconst)
