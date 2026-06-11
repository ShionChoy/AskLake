"""Select the top-N popular IMDb films and fetch their current Wikipedia plots as PlotDocs for the
GraphRAG theme-extraction build. Replaces the retired CMU corpus. Dataset-specific."""

from __future__ import annotations

from typing import Protocol

from engine.graph.extraction import PlotDoc
from engine.lakehouse.duckdb_backend import DuckDBBackend


class WikiSource(Protocol):
    def resolve(self, tconsts: list[str]) -> dict[str, str]: ...  # tconst -> enwiki title
    def plot(self, title: str) -> str | None: ...  # plot text or None


class _LiveWiki:
    """Default WikiSource backed by datasets.imdb_cmu.wiki_plots (live network at build time)."""

    def resolve(self, tconsts: list[str]) -> dict[str, str]:
        from datasets.imdb_cmu.wiki_plots import resolve_enwiki_titles

        return resolve_enwiki_titles(tconsts)

    def plot(self, title: str) -> str | None:
        from datasets.imdb_cmu.wiki_plots import fetch_plot

        return fetch_plot(title)


def select_plot_docs(
    parquet_dir: str, max_films: int, *, wiki: WikiSource | None = None
) -> list[PlotDoc]:
    """Top-`max_films` movies by numVotes that have a resolvable Wikipedia plot, as PlotDocs
    (id='wiki:<tconst>', title=IMDb primaryTitle, text=plot)."""
    wiki = wiki or _LiveWiki()
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    res = backend.run_sql(
        f"SELECT b.tconst, b.primaryTitle FROM title_basics b JOIN title_ratings r USING(tconst) "
        f"WHERE b.titleType='movie' ORDER BY r.numVotes DESC LIMIT {int(max_films)}"
    )
    ranked = [(tconst, title) for tconst, title in res.rows]
    titles = wiki.resolve([t for t, _ in ranked])
    docs: list[PlotDoc] = []
    for tconst, primary_title in ranked:
        art = titles.get(tconst)
        if not art:
            continue
        text = wiki.plot(art)
        if text:
            docs.append(PlotDoc(id=f"wiki:{tconst}", title=primary_title, text=text))
    return docs
