"""Offline GraphRAG build: extract triples from IMDb-native structured data and Wikipedia plots
for the top-N popular films. One LLM call per plot for theme extraction.

Usage (from the repo root):
    DEEPSEEK_API_KEY=... uv run python -m scripts.build_graph
Env: ASKLAKE_PARQUET_DIR, ASKLAKE_GRAPH_PATH, GRAPH_FILMS (default 2000),
     GRAPH_MIN_VOTES (default 1000), GRAPH_CAST_CAP (default 10), GRAPH_WORKERS (default 12)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from datasets.imdb_cmu.graph_corpus import select_plot_docs
from datasets.imdb_cmu.graph_structured import structured_triples
from engine.graph.extraction import PlotDoc, extract_triples
from engine.graph.ontology import GraphOntology, load_ontology
from engine.graph.persistence import append_triples, save_triples
from engine.llm.factory import make_provider
from engine.ports.llm import LLMProvider

ONTOLOGY_YAML = "datasets/imdb_cmu/graph/ontology.yaml"

_THEME_RELATIONS = ("HAS_THEME", "SET_IN")
_THEME_HINT = (
    "Use the film's title as the subject. Emit HAS_THEME -> a short, reusable theme phrase "
    '(e.g. "memory", "identity", "time") and SET_IN -> a place or era. Prefer concise, reusable '
    "theme labels so films sharing a theme connect. Do NOT emit cast, directors, or genres."
)


def theme_ontology(ontology: GraphOntology) -> GraphOntology:
    """Restrict LLM extraction to themes + setting; IMDb supplies structured facts deterministically
    (so the plot-text pass must not re-extract director/cast/genre)."""
    return replace(ontology, relation_types=_THEME_RELATIONS, hint=_THEME_HINT)


def build_and_save(
    docs: list[PlotDoc], llm: LLMProvider, ontology: GraphOntology, out_path: str | Path
) -> int:
    """Extract triples from each doc and persist them as JSONL, written as they are produced so a
    mid-run failure still leaves the already-extracted films on disk. Returns the triple count.
    A provider exception aborts the build (the partial graph is kept)."""
    count = 0

    def _stream():
        nonlocal count
        for i, doc in enumerate(docs, start=1):
            extracted = extract_triples(llm, doc, ontology)
            count += len(extracted)
            print(f"  [{i}/{len(docs)}] {doc.title}: {len(extracted)} triple(s)", flush=True)
            yield from extracted

    save_triples(_stream(), out_path)
    return count


def _extract_with_retry(llm, doc, ontology, attempts=3, backoff=1.0):
    """Extract one doc's triples, retrying transient LLM errors; returns None if every attempt
    fails (so the build skips that film's themes instead of aborting the whole run)."""
    for i in range(attempts):
        try:
            return extract_triples(llm, doc, ontology)
        except Exception:  # noqa: BLE001 - one film's LLM error must not abort the batch
            if i == attempts - 1:
                return None
            time.sleep(backoff * (2**i))
    return None


def build_and_save_parallel(structured, docs, llm, ontology, out_path, workers=12, max_calls=None):
    """Write the deterministic `structured` triples first, then extract themes from `docs` with a
    bounded thread-pool fan-out (one LLM call per doc), appending as each completes. Returns the
    number of theme triples written. `max_calls` caps the number of LLM calls."""
    save_triples(structured, out_path)
    if max_calls is not None:
        docs = docs[:max_calls]

    def _stream():
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_extract_with_retry, llm, d, ontology): d for d in docs}
            for fut in as_completed(futures):
                done += 1
                doc = futures[fut]
                triples = fut.result()
                if triples is None:
                    failed += 1
                    print(
                        f"  [{done}/{len(docs)}] {doc.title}: FAILED after retries (skipped)",
                        flush=True,
                    )
                    continue
                print(f"  [{done}/{len(docs)}] {doc.title}: {len(triples)} triple(s)", flush=True)
                yield from triples
        if failed:
            print(f"[build_graph] {failed} film(s) skipped after extraction errors", flush=True)

    return append_triples(_stream(), out_path)


def main() -> int:
    parquet_dir = os.environ.get("ASKLAKE_PARQUET_DIR", "data/imdb/parquet")
    out_path = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
    n = int(os.environ.get("GRAPH_FILMS", "2000"))
    min_votes = int(os.environ.get("GRAPH_MIN_VOTES", "1000"))
    cast_cap = int(os.environ.get("GRAPH_CAST_CAP", "10"))
    workers = int(os.environ.get("GRAPH_WORKERS", "12"))

    if not Path(parquet_dir).exists():
        print(f"[build_graph] parquet dir not found: {parquet_dir} — run `make build-imdb` first.")
        return 1
    try:
        llm = make_provider()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[build_graph] no LLM provider ({exc}); set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY)."
        )
        return 1

    ontology = load_ontology(ONTOLOGY_YAML)
    theme_ont = theme_ontology(ontology)
    structured = list(structured_triples(parquet_dir, min_votes=min_votes, cast_cap=cast_cap))
    print(
        f"[build_graph] {len(structured)} structured triple(s) (numVotes>={min_votes}, "
        f"cast<={cast_cap}); fetching top-{n} Wikipedia plots for theme extraction -> {out_path}"
    )
    docs = select_plot_docs(parquet_dir, n)
    print(
        f"[build_graph] {len(docs)} plot(s) resolved from Wikipedia; extracting themes "
        f"with {workers} workers"
    )
    theme_count = build_and_save_parallel(
        structured, docs, llm, theme_ont, out_path, workers=workers
    )
    print(
        f"[build_graph] done: {len(structured)} structured + {theme_count} theme triple(s) "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
