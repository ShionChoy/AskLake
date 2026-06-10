"""CMU plot corpus loader + IMDb title alignment for the GraphRAG build (dataset-specific, so
the engine stays dataset-agnostic). Emits PlotDocs for films present in BOTH the CMU corpus and
the top-N IMDb parquet, titled with the IMDb primaryTitle so graph subjects match the SQL side."""

from __future__ import annotations

import re
from pathlib import Path

from engine.graph.extraction import PlotDoc
from engine.lakehouse.duckdb_backend import DuckDBBackend

_NONALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(title: str) -> str:
    return _NONALNUM.sub(" ", title.lower()).strip()


def _year(release_date: str) -> int | None:
    rd = (release_date or "").strip()
    return int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None


def _read_metadata(cmu_dir: Path) -> dict[str, tuple[str, int | None]]:
    """wiki_id -> (name, year) from movie.metadata.tsv."""
    out: dict[str, tuple[str, int | None]] = {}
    with (cmu_dir / "movie.metadata.tsv").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            wiki_id, _freebase, name, release_date = parts[0].strip(), parts[1], parts[2], parts[3]
            out[wiki_id] = (name, _year(release_date))
    return out


def imdb_movie_index(parquet_dir: str) -> dict[str, tuple[str, int | None, int, str]]:
    """{normalized primaryTitle: (primaryTitle, startYear, numVotes, tconst)} for ALL movies (the
    full candidate pool); highest-votes wins on a normalized-title collision. tconst is threaded
    through so the structured loader can fetch IMDb-authoritative directors."""
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    res = backend.run_sql(
        "SELECT b.primaryTitle, b.startYear, r.numVotes, b.tconst "
        "FROM title_basics b JOIN title_ratings r USING (tconst) "
        "WHERE b.titleType = 'movie' ORDER BY r.numVotes DESC"
    )
    out: dict[str, tuple[str, int | None, int, str]] = {}
    for primary_title, start_year, num_votes, tconst in res.rows:
        key = _normalize(primary_title)
        if key and key not in out:  # rows are votes-desc, so first wins
            year = start_year if isinstance(start_year, int) else None
            out[key] = (primary_title, year, int(num_votes or 0), tconst)
    return out


def aligned_films(
    cmu_dir: str, imdb_index: dict[str, tuple[str, int | None, int, str]]
) -> dict[str, tuple[str, str, int]]:
    """{cmu wiki_id: (imdb primaryTitle, tconst, numVotes)} for CMU films that align to an IMDb
    movie by normalized title + year (±1, accepting a missing year on either side). Deduped by
    IMDb title (highest votes wins)."""
    meta = _read_metadata(Path(cmu_dir))  # {wiki_id: (name, year)}
    out: dict[str, tuple[str, str, int]] = {}
    best_votes: dict[str, int] = {}
    for wiki_id, (name, cmu_year) in meta.items():
        match = imdb_index.get(_normalize(name))
        if match is None:
            continue
        title, imdb_year, votes, tconst = match
        if imdb_year is not None and cmu_year is not None and abs(imdb_year - cmu_year) > 1:
            continue
        if title in best_votes and votes <= best_votes[title]:
            continue
        # drop any earlier wiki_id that mapped to this same title with fewer votes
        out = {w: v for w, v in out.items() if v[0] != title}
        out[wiki_id] = (title, tconst, votes)
        best_votes[title] = votes
    return out


def select_plot_docs(
    cmu_dir: str, imdb_index: dict[str, tuple[str, int | None, int, str]], max_films: int
) -> list[PlotDoc]:
    """Top-`max_films` aligned films by IMDb numVotes that also have a CMU plot, as PlotDocs
    (id='cmu:<wiki_id>', title=IMDb primaryTitle)."""
    aligned = aligned_films(cmu_dir, imdb_index)
    plots: dict[str, str] = {}
    with (Path(cmu_dir) / "plot_summaries.txt").open(encoding="utf-8") as f:
        for line in f:
            wiki_id, _, text = line.partition("\t")
            wiki_id, text = wiki_id.strip(), text.strip()
            if text and wiki_id in aligned:
                plots[wiki_id] = text
    ranked = sorted(plots.keys(), key=lambda w: aligned[w][2], reverse=True)  # votes at index 2
    return [PlotDoc(id=f"cmu:{w}", title=aligned[w][0], text=plots[w]) for w in ranked[:max_films]]
