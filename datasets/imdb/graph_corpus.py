"""Select popular IMDb films and fetch their current Wikipedia plots for GraphRAG."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from engine.graph.extraction import PlotDoc
from engine.lakehouse.duckdb_backend import DuckDBBackend


class WikiSource(Protocol):
    def resolve(self, tconsts: list[str]) -> dict[str, str]: ...  # tconst -> enwiki title
    def plot(self, title: str) -> str | None: ...  # plot text or None


class _LiveWiki:
    """Default WikiSource backed by datasets.imdb.wiki_plots (live network at build time).
    Reuses one pooled httpx.Client across all calls."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx

            from datasets.imdb.wiki_plots import _UA

            self._client = httpx.Client(timeout=30.0, headers={"User-Agent": _UA})
        return self._client

    def resolve(self, tconsts: list[str]) -> dict[str, str]:
        from datasets.imdb.wiki_plots import resolve_enwiki_titles

        return resolve_enwiki_titles(tconsts, client=self._get_client())

    def plot(self, title: str) -> str | None:
        from datasets.imdb.wiki_plots import fetch_plot

        return fetch_plot(title, client=self._get_client())


def select_plot_docs(
    parquet_dir: str, max_films: int, *, wiki: WikiSource | None = None, workers: int = 8
) -> list[PlotDoc]:
    """Top-`max_films` movies by numVotes that have a resolvable Wikipedia plot, as PlotDocs
    (id='wiki:<tconst>', title=IMDb primaryTitle, text=plot). Plots are fetched concurrently with
    `workers` threads (httpx.Client is thread-safe); a film whose fetch fails after retries is
    skipped, not fatal."""
    wiki = wiki or _LiveWiki()
    backend = DuckDBBackend(parquet_dir=parquet_dir)
    res = backend.run_sql(
        f"SELECT b.tconst, b.primaryTitle FROM title_basics b JOIN title_ratings r USING(tconst) "
        f"WHERE b.titleType='movie' AND NOT COALESCE(b.isAdult, false) "
        f"ORDER BY r.numVotes DESC LIMIT {int(max_films)}"
    )
    ranked = [(tconst, title) for tconst, title in res.rows]
    titles = wiki.resolve([t for t, _ in ranked])
    targets = [(tc, title, titles[tc]) for tc, title in ranked if titles.get(tc)]
    print(
        f"[graph_corpus] resolved {len(targets)}/{len(ranked)} Wikipedia articles; "
        f"fetching plots with {workers} worker(s)",
        flush=True,
    )

    def _fetch(item: tuple[str, str, str]) -> PlotDoc | None:
        tconst, primary_title, art = item
        try:
            text = wiki.plot(art)
        except Exception:  # noqa: BLE001 - one film's network error must not abort the batch
            return None
        return PlotDoc(id=f"wiki:{tconst}", title=primary_title, text=text) if text else None

    docs: list[PlotDoc] = []
    skipped = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_fetch, it) for it in targets]):
            done += 1
            doc = fut.result()
            if doc is None:
                skipped += 1
            else:
                docs.append(doc)
            if done % 100 == 0:
                print(
                    f"  [graph_corpus] fetched {done}/{len(targets)} "
                    f"({len(docs)} ok, {skipped} skipped)",
                    flush=True,
                )
    if skipped:
        print(f"[graph_corpus] skipped {skipped} film(s) (no plot / fetch error)", flush=True)
    return docs
