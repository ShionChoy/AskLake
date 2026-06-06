"""Offline GraphRAG build: extract triples from CMU plots (aligned to the top-N IMDb films) and
persist them for the server to load. One LLM call per plot.

Usage (from the repo root):
    DEEPSEEK_API_KEY=... uv run python -m scripts.build_graph
Env: ASKLAKE_PARQUET_DIR, ASKLAKE_CMU_DIR, ASKLAKE_GRAPH_PATH, GRAPH_FILMS (default 200)."""

from __future__ import annotations

import os
from pathlib import Path

from datasets.imdb_cmu.graph_corpus import load_plot_docs, top_imdb_titles
from engine.graph.extraction import PlotDoc, extract_triples
from engine.graph.ontology import GraphOntology, load_ontology
from engine.graph.persistence import save_triples
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


def main() -> int:
    parquet_dir = os.environ.get("ASKLAKE_PARQUET_DIR", "data/imdb/parquet")
    cmu_dir = os.environ.get("ASKLAKE_CMU_DIR", "data/cmu/raw/MovieSummaries")
    out_path = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
    n = int(os.environ.get("GRAPH_FILMS", "200"))

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

    docs = load_plot_docs(cmu_dir, top_imdb_titles(parquet_dir, n), n)
    if not docs:
        print("[build_graph] no CMU plots aligned to the top IMDb films — nothing to build.")
        return 1

    print(f"[build_graph] extracting from {len(docs)} film plot(s) -> {out_path}")
    count = build_and_save(docs, llm, load_ontology(ONTOLOGY_YAML), out_path)
    print(f"[build_graph] done: {count} triple(s) from {len(docs)} film(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
