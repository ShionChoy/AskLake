.PHONY: dev lint test demo-p0

dev:
	docker compose --profile core up

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

demo-p0:
	uv run python demos/demo_p0.py
