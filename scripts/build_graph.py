"""Offline GraphRAG build: extract triples from CMU plots (aligned to the top-N IMDb films) and
persist them for the server to load. One LLM call per plot.

Usage (from the repo root):
    DEEPSEEK_API_KEY=... uv run python -m scripts.build_graph
Env: ASKLAKE_PARQUET_DIR, ASKLAKE_CMU_DIR, ASKLAKE_GRAPH_PATH, GRAPH_FILMS (default 2000),
     GRAPH_WORKERS (default 12)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets.imdb_cmu.graph_corpus import aligned_films, imdb_movie_index, select_plot_docs
from datasets.imdb_cmu.graph_structured import imdb_crew_triples, structured_triples
from engine.graph.extraction import PlotDoc, extract_triples
from engine.graph.ontology import GraphOntology, load_ontology
from engine.graph.persistence import append_triples, save_triples
from engine.llm.factory import make_provider
from engine.ports.llm import LLMProvider

ONTOLOGY_YAML = "datasets/imdb_cmu/graph/ontology.yaml"


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


def build_and_save_parallel(structured, docs, llm, ontology, out_path, workers=12, max_calls=None):
    """Write the deterministic `structured` triples first, then extract themes from `docs` with a
    bounded thread-pool fan-out (one LLM call per doc), appending as each completes. Returns the
    number of theme triples written. `max_calls` caps the number of LLM calls."""
    save_triples(structured, out_path)
    if max_calls is not None:
        docs = docs[:max_calls]

    def _stream():
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(extract_triples, llm, d, ontology): d for d in docs}
            for fut in as_completed(futures):
                done += 1
                doc = futures[fut]
                triples = fut.result()
                print(f"  [{done}/{len(docs)}] {doc.title}: {len(triples)} triple(s)", flush=True)
                yield from triples

    return append_triples(_stream(), out_path)


def main() -> int:
    parquet_dir = os.environ.get("ASKLAKE_PARQUET_DIR", "data/imdb/parquet")
    cmu_dir = os.environ.get("ASKLAKE_CMU_DIR", "data/cmu/raw/MovieSummaries")
    out_path = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
    n = int(os.environ.get("GRAPH_FILMS", "2000"))
    workers = int(os.environ.get("GRAPH_WORKERS", "12"))

    if not Path(parquet_dir).exists():
        print(f"[build_graph] parquet dir not found: {parquet_dir} — run `make build-imdb` first.")
        return 1
    if not (Path(cmu_dir) / "plot_summaries.txt").exists():
        print(f"[build_graph] CMU corpus not found in {cmu_dir} — run scripts/download_data.sh.")
        return 1

    try:
        llm = make_provider()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[build_graph] no LLM provider ({exc}); set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY)."
        )
        return 1

    index = imdb_movie_index(parquet_dir)
    aligned = aligned_films(cmu_dir, index)
    docs = select_plot_docs(cmu_dir, index, n)
    if not docs:
        print("[build_graph] no CMU plots aligned to IMDb films — nothing to build.")
        return 1
    ontology = load_ontology(ONTOLOGY_YAML)
    structured = list(structured_triples(cmu_dir, aligned)) + list(
        imdb_crew_triples(parquet_dir, aligned)
    )
    print(
        f"[build_graph] {len(structured)} structured triple(s); extracting themes from "
        f"{len(docs)} plot(s) with {workers} workers -> {out_path}"
    )
    theme_count = build_and_save_parallel(
        structured, docs, llm, ontology, out_path, workers=workers
    )
    print(
        f"[build_graph] done: {len(structured)} structured + {theme_count} theme triple(s) "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
