from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("ASKLAKE_API_URL", "http://localhost:8000")


def _post(path: str, payload: dict) -> dict:
    return requests.post(f"{API_URL}{path}", json=payload, timeout=120).json()


@st.cache_data(ttl=30)
def _info() -> dict:
    try:
        return requests.get(f"{API_URL}/info", timeout=5).json()
    except Exception:  # noqa: BLE001
        return {}


def _ask(question: str) -> dict:
    """Prefer the traced endpoint (rich steps); fall back to plain /ask."""
    try:
        r = requests.post(f"{API_URL}/ask_trace", json={"question": question}, timeout=180)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        pass
    return _post("/ask", {"question": question})


def _show_table(columns, rows) -> None:
    st.dataframe({c: [row[i] for row in rows] for i, c in enumerate(columns)})


def render() -> None:
    st.title("AskLake")

    info = _info()
    if info.get("model"):
        st.caption(
            f"🧠 model: **{info['model']}**  ·  {info.get('path', '')}  ·  "
            f"provider: {info.get('provider', '')}"
        )

    st.header("Ask in natural language")
    question = st.text_input("Question", value="Highest-rated sci-fi films after 2010 (top 10)")
    if st.button("Ask"):
        with st.spinner("Running the agent…"):
            resp = _ask(question)

        steps = resp.get("steps")
        if steps:
            st.subheader("Backend processing steps")
            for i, s in enumerate(steps, start=1):
                icon = "✅" if s.get("ok", True) else "❌"
                ms = s.get("ms")
                label = f"{icon} {i}. {s['step']}" + (f" — {ms:.0f} ms" if ms is not None else "")
                with st.expander(label, expanded=not s.get("ok", True)):
                    if s.get("detail"):
                        st.write(s["detail"])
                    if s.get("sql"):
                        st.code(s["sql"], language="sql")
            if resp.get("elapsed_ms") is not None:
                st.caption(f"total backend time: {resp['elapsed_ms']:.0f} ms")

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
