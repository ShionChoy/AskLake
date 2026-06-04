.PHONY: dev lint test demo-p0 demo demo-p1 demo-p2 demo-p3 demo-p4 eval build-imdb

dev:
	docker compose --profile core up

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

demo: demo-p0

demo-p0:
	uv run python demos/demo_p0.py

demo-p1:
	uv run python demos/demo_p1.py

build-imdb:
	uv run python -c "import os; from datasets.imdb_cmu.source import build_parquet; build_parquet(os.environ.get('IMDB_RAW','data/imdb/raw'), os.environ.get('PARQUET_DIR','data/imdb/parquet'), int(os.environ.get('MIN_VOTES','1000')))"

demo-p2:
	uv run python demos/demo_p2.py

demo-p3:
	uv run python demos/demo_p3.py

demo-p4:
	uv run python demos/demo_p4.py

eval:
	uv run python -m eval.run
