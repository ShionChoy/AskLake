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


def top_imdb_titles(parquet_dir: str, n: int) -> dict[str, tuple[str, int | None]]:
    """{normalized primaryTitle: (primaryTitle, startYear)} for the top-n films by numVotes."""
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    # title_basics.parquet is pre-filtered to titleType='movie' by source.py —
    # no extra WHERE needed.
    res = backend.run_sql(
        "SELECT b.primaryTitle, b.startYear "
        "FROM title_basics b JOIN title_ratings r USING (tconst) "
        f"ORDER BY r.numVotes DESC LIMIT {int(n)}"
    )
    out: dict[str, tuple[str, int | None]] = {}
    for primary_title, start_year in res.rows:
        key = _normalize(primary_title)
        if key and key not in out:  # first (highest-votes) wins on collision
            out[key] = (primary_title, start_year if isinstance(start_year, int) else None)
    return out


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


def load_plot_docs(
    cmu_dir: str, imdb_titles: dict[str, tuple[str, int | None]], max_films: int
) -> list[PlotDoc]:
    """PlotDocs for CMU plots whose film aligns (normalized title + year within ±1, accepting a
    missing year on either side) to an IMDb film in `imdb_titles`. Title is the IMDb primaryTitle;
    deduped by film; capped to max_films."""
    cmu = Path(cmu_dir)
    meta = _read_metadata(cmu)
    docs: list[PlotDoc] = []
    seen: set[str] = set()
    with (cmu / "plot_summaries.txt").open(encoding="utf-8") as f:
        for line in f:
            wiki_id, _, text = line.partition("\t")
            wiki_id, text = wiki_id.strip(), text.strip()
            if not text or wiki_id not in meta:
                continue
            name, cmu_year = meta[wiki_id]
            match = imdb_titles.get(_normalize(name))
            if match is None:
                continue
            primary_title, imdb_year = match
            if imdb_year is not None and cmu_year is not None and abs(imdb_year - cmu_year) > 1:
                continue
            if primary_title in seen:
                continue
            seen.add(primary_title)
            docs.append(PlotDoc(id=f"cmu:{wiki_id}", title=primary_title, text=text))
            if len(docs) >= max_films:
                break
    return docs
