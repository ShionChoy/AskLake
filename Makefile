.PHONY: dev lint test demo-p0 demo demo-p1 demo-p2 demo-p3 demo-p4 demo-p5 demo-p6 eval eval-real build-imdb build-graph build-crm eval-real-crm serve ui clean

dev:
	docker compose --profile core up

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

demo: demo-p0 demo-p1 demo-p2 demo-p3 demo-p4 demo-p5 demo-p6

demo-p0:
	uv run python demos/demo_p0.py

demo-p1:
	uv run python demos/demo_p1.py

build-imdb:
	uv run python -c "import os; from datasets.imdb_cmu.source import build_parquet; build_parquet(os.environ.get('IMDB_RAW','data/imdb/raw'), os.environ.get('PARQUET_DIR','data/imdb/parquet'), int(os.environ.get('MIN_VOTES','1000')))"

build-graph:
	uv run python -m scripts.build_graph

demo-p2:
	uv run python demos/demo_p2.py

demo-p3:
	uv run python demos/demo_p3.py

demo-p4:
	uv run python demos/demo_p4.py

demo-p5:
	uv run python demos/demo_p5.py

demo-p6:
	uv run python demos/demo_p6.py

eval:
	uv run python -m eval.run

eval-real:
	uv run python -m eval.real_run

build-crm:
	uv run python -c "from datasets.crm_demo.source import build_parquet; build_parquet('data/crm/parquet')"

eval-real-crm:
	uv run python -m eval.real_run --dataset crm

serve:
	uv run uvicorn api.serve:build_app --factory $(if $(wildcard .env),--env-file .env,) --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run ui/app.py --server.headless true --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .ruff_cache .mypy_cache
