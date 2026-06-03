# AskLake

Governed multi-agent natural-language analytics over your own data
(Text-to-SQL + GraphRAG). See architecture docs (local).

## Quickstart (dev)
```bash
uv sync
make demo-p0   # in-process smoke demo (no Docker, no API key)
make dev       # docker compose core profile
```

## License
Apache-2.0. IMDb data is non-commercial (not redistributed); CMU corpus is CC BY-SA.
