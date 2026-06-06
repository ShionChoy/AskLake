"""Persist a knowledge graph as JSON Lines (one triple per line) and reload it into an
InMemoryGraphStore. Additive — the GraphStore port is unchanged; the build script writes the
file and the server loads it."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple


def save_triples(triples: Iterable[Triple], path: str | Path) -> None:
    """Write triples as JSONL, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(
                json.dumps(
                    {
                        "subject": t.subject,
                        "relation": t.relation,
                        "obj": t.obj,
                        "source": t.source,
                    }
                )
                + "\n"
            )


def load_store(path: str | Path) -> InMemoryGraphStore:
    """Load a JSONL triple file into an InMemoryGraphStore.

    Raises FileNotFoundError if the file is absent, and json.JSONDecodeError on a malformed
    line (the file is written by save_triples, so corruption indicates a build problem).
    """
    store = InMemoryGraphStore()
    with Path(path).open(encoding="utf-8") as f:  # FileNotFoundError if missing
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            store.add(
                Triple(
                    subject=d["subject"],
                    relation=d["relation"],
                    obj=d["obj"],
                    source=d.get("source", ""),
                )
            )
    return store
