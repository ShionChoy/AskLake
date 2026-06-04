from __future__ import annotations

from engine.ports.retrieval import RetrievalResult


class Synthesizer:
    """Fuses results from multiple RetrievalPaths into one cited answer (deterministic). The
    structured table from the first SQL-bearing path is kept as the primary `result`; every path's
    narrative is concatenated with a [path] tag so citations are preserved. An LLM-backed
    synthesizer can replace this behind the same fuse() signature later."""

    def fuse(self, question: str, results: list[RetrievalResult]) -> RetrievalResult:
        parts = [r for r in results if r is not None]
        if not parts:
            return RetrievalResult(
                path="none", sql=None, result=None, narrative="No answer produced.", chart_spec=None
            )
        primary = next((r for r in parts if r.result is not None and r.sql is not None), None)
        base = primary or parts[0]
        narrative = "\n\n".join(f"[{r.path}] {r.narrative}" for r in parts if r.narrative)
        return RetrievalResult(
            path="+".join(r.path for r in parts),
            sql=base.sql,
            result=base.result,
            narrative=narrative or "No answer produced.",
            chart_spec=base.chart_spec,
        )
