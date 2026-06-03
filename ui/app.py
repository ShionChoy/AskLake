from __future__ import annotations

import os

import streamlit as st

API_URL = os.environ.get("ASKLAKE_API_URL", "http://localhost:8000")


def _post(path: str, payload: dict) -> dict:
    import requests

    return requests.post(f"{API_URL}{path}", json=payload, timeout=120).json()


def _show_table(columns, rows) -> None:
    st.dataframe({c: [row[i] for row in rows] for i, c in enumerate(columns)})


def render() -> None:
    st.title("AskLake")

    st.header("Ask in natural language")
    question = st.text_input("Question", value="Highest-rated sci-fi films after 2010 (top 10)")
    if st.button("Ask"):
        resp = _post("/ask", {"question": question})
        if not resp.get("columns"):
            st.warning(resp.get("narrative", "No result."))
            if resp.get("sql"):
                st.code(resp["sql"], language="sql")
        else:
            st.code(resp["sql"], language="sql")
            _show_table(resp["columns"], resp["rows"])
            spec = resp.get("chart_spec")
            if spec and spec.get("type") == "bar":
                xi = resp["columns"].index(spec["x"])
                yi = resp["columns"].index(spec["y"])
                st.bar_chart({r[xi]: r[yi] for r in resp["rows"]})

    st.header("Raw SQL console")
    sql = st.text_area("SQL", value="SELECT 1 AS hello")
    if st.button("Run SQL"):
        resp = _post("/query", {"sql": sql})
        if resp.get("error"):
            st.error(resp["error"])
        else:
            _show_table(resp["columns"], resp["rows"])


if __name__ == "__main__":
    render()
