FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY engine ./engine
COPY api ./api
COPY ui ./ui
COPY datasets ./datasets
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.serve:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
