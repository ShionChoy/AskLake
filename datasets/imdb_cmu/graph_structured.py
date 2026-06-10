"""Deterministic structured triples from the CMU metadata TSVs (zero LLM) + authoritative
directors from the IMDb parquet. Films are titled by their aligned IMDb primaryTitle so graph
subjects match the SQL side. Dataset-specific; the engine never imports this."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.graph_store import Triple


def _names(cell: str) -> list[str]:
    """CMU genre/language/country columns are JSON dicts {freebase_id: name}; return the names."""
    cell = (cell or "").strip()
    if not cell or cell == "{}":
        return []
    try:
        return [v for v in json.loads(cell).values() if v]
    except (ValueError, AttributeError):
        return []


def _year(release_date: str) -> str | None:
    rd = (release_date or "").strip()
    return rd[:4] if len(rd) >= 4 and rd[:4].isdigit() else None


def structured_triples(cmu_dir: str, aligned: dict[str, tuple[str, str, int]]) -> Iterator[Triple]:
    """Yield film/genre/language/country/year + character/actor triples for the films in `aligned`
    ({wiki_id: (imdb_title, tconst, votes)}). Unaligned CMU films are skipped."""
    seen: set[tuple[str, str, str, str]] = set()
    cmu = Path(cmu_dir)
    with (cmu / "movie.metadata.tsv").open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0].strip() not in aligned:
                continue
            wiki_id = p[0].strip()
            title = aligned[wiki_id][0]
            src = f"cmu:{wiki_id}"
            for g in _names(p[8]):
                key = (title, "HAS_GENRE", g, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(title, "HAS_GENRE", g, src)
            for lang in _names(p[6]):
                key = (title, "IN_LANGUAGE", lang, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(title, "IN_LANGUAGE", lang, src)
            for country in _names(p[7]):
                key = (title, "FROM_COUNTRY", country, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(title, "FROM_COUNTRY", country, src)
            yr = _year(p[3])
            if yr:
                key = (title, "RELEASED_IN", yr, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(title, "RELEASED_IN", yr, src)
    char_path = cmu / "character.metadata.tsv"
    if not char_path.exists():
        return
    with char_path.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0].strip() not in aligned:
                continue
            wiki_id = p[0].strip()
            title = aligned[wiki_id][0]
            src = f"cmu:{wiki_id}"
            character, actor = p[3].strip(), p[8].strip()
            if character:
                key = (title, "FEATURES_CHARACTER", character, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(title, "FEATURES_CHARACTER", character, src)
            if character and actor:
                key = (character, "PLAYED_BY", actor, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(character, "PLAYED_BY", actor, src)
            if actor:
                key = (actor, "ACTED_IN", title, src)
                if key not in seen:
                    seen.add(key)
                    yield Triple(actor, "ACTED_IN", title, src)


def imdb_crew_triples(
    parquet_dir: str, aligned: dict[str, tuple[str, str, int]]
) -> Iterator[Triple]:
    """`film -DIRECTED_BY-> director`, sourced authoritatively from the IMDb parquet (CMU metadata
    has no director field). `aligned` = {wiki_id: (imdb_title, tconst, votes)}; one query over the
    whole title_crew⋈name_basics, filtered to the aligned tconsts. source='imdb:<tconst>'."""
    title_by_tconst = {tconst: title for title, tconst, _votes in aligned.values()}
    if not title_by_tconst:
        return
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    res = backend.run_sql(
        "SELECT t.tconst, n.primaryName FROM ("
        "SELECT tconst, UNNEST(string_split(directors, ',')) AS nconst FROM title_crew "
        "WHERE directors IS NOT NULL AND directors != ''"
        ") t JOIN name_basics n ON n.nconst = t.nconst"
    )
    seen: set[tuple[str, str, str, str]] = set()
    for tconst, director in res.rows:
        title = title_by_tconst.get(tconst)
        if title and director:
            key = (title, "DIRECTED_BY", director, f"imdb:{tconst}")
            if key not in seen:
                seen.add(key)
                yield Triple(title, "DIRECTED_BY", director, f"imdb:{tconst}")
