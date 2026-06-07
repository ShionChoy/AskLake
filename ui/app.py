from __future__ import annotations

import os

import requests
import streamlit as st

try:  # `streamlit run ui/app.py` puts ui/ on sys.path; pytest puts the repo root on sys.path
    from ui import credentials, graph_viz
except ImportError:  # pragma: no cover
    import credentials  # type: ignore[no-redef]
    import graph_viz  # type: ignore[no-redef]

API_URL = os.environ.get("ASKLAKE_API_URL", "http://localhost:8000")

PROVIDERS = ["deepseek", "anthropic"]
DEFAULT_MODELS = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"],
}
_CUSTOM = "(custom…)"
PATHS = ["auto", "sql", "graph", "fusion"]


def _creds_payload(state) -> dict:
    """Pick provider/model/api_key out of a dict-like state, omitting empty values."""
    out = {}
    for k in ("provider", "model", "api_key"):
        v = state.get(k, "")
        if v:
            out[k] = v
    return out


def _ask_body(question: str, state) -> dict:
    """Build the /ask_trace request body: question + retrieval path + (non-empty) credentials."""
    return {"question": question, "path": state.get("path", "auto"), **_creds_payload(state)}


def _post(path: str, payload: dict) -> dict:
    return requests.post(f"{API_URL}{path}", json=payload, timeout=120).json()


@st.cache_data(ttl=30)
def _info() -> dict:
    try:
        return requests.get(f"{API_URL}/info", timeout=5).json()
    except Exception:  # noqa: BLE001
        return {}


def _ask(question: str) -> dict:
    """Prefer the traced endpoint (rich steps + BYO credentials + routing); fall back to /ask."""
    body = _ask_body(question, st.session_state)
    try:
        r = requests.post(f"{API_URL}/ask_trace", json=body, timeout=180)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        pass
    # Fallback only fires if /ask_trace is unreachable; /ask ignores path/creds (server-key only).
    return _post("/ask", body)


def _show_table(columns, rows) -> None:
    st.dataframe({c: [row[i] for row in rows] for i, c in enumerate(columns)})


def _init_state() -> None:
    if st.session_state.get("creds_loaded"):
        return
    saved = credentials.load()
    st.session_state.provider = saved.get("provider", "deepseek")
    st.session_state.model = saved.get("model", DEFAULT_MODELS["deepseek"][0])
    st.session_state.api_key = saved.get("api_key", "")
    st.session_state.creds_loaded = True


def _sidebar() -> None:
    st.sidebar.header("⚙️ Model & API key")

    _saved_provider = st.session_state.get("provider", "deepseek")
    provider = st.sidebar.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(_saved_provider) if _saved_provider in PROVIDERS else 0,
    )
    st.session_state.provider = provider

    options = DEFAULT_MODELS[provider] + [_CUSTOM]
    current = st.session_state.get("model", options[0])
    index = options.index(current) if current in options else len(options) - 1
    choice = st.sidebar.selectbox("Model", options, index=index)
    if choice == _CUSTOM:
        model = st.sidebar.text_input("Custom model", value="" if current in options else current)
    else:
        model = choice
    st.session_state.model = model

    api_key = st.sidebar.text_input(
        "API key", type="password", value=st.session_state.get("api_key", "")
    )
    st.session_state.api_key = api_key

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Save locally"):
        credentials.save(provider, model, api_key)
        st.sidebar.success("Saved on this machine.")
    if col2.button("Delete saved key"):
        credentials.delete()
        st.session_state.api_key = ""
        st.sidebar.info("Saved key deleted.")

    if credentials.path().exists():
        st.sidebar.caption("🔐 A key is saved on this machine (plaintext, ~/.config/asklake/).")
    else:
        st.sidebar.caption("No saved key. A pasted key is sent per request and not stored.")

    st.sidebar.markdown("---")
    path_choice = st.sidebar.selectbox(
        "Retrieval path", PATHS, index=PATHS.index(st.session_state.get("path", "auto"))
    )
    st.session_state.path = path_choice
    st.sidebar.caption("Auto routes SQL vs. graph; Graph needs no API key.")


def render() -> None:
    _init_state()
    _sidebar()

    st.title("AskLake")

    selected_model = st.session_state.get("model")
    if selected_model:
        st.caption(
            f"🧠 model: **{selected_model}**  ·  provider: {st.session_state.get('provider', '')}"
        )
    else:
        info = _info()
        if info.get("model"):
            st.caption(f"🧠 model: **{info['model']}**  ·  {info.get('path', '')}")

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

        rpath = resp.get("path", "")
        if rpath:
            st.caption(f"path: {rpath}")

        if not resp.get("columns"):
            st.warning(resp.get("narrative", "No result."))
            if resp.get("sql"):
                st.code(resp["sql"], language="sql")
        else:
            if resp.get("narrative") and "graph" in (rpath or ""):
                st.info(resp["narrative"])  # cited graph / fusion narrative
            if resp.get("sql"):
                st.code(resp["sql"], language="sql")
            _show_table(resp["columns"], resp["rows"])
            spec = resp.get("chart_spec")
            if spec and spec.get("type") == "bar":
                xi = resp["columns"].index(spec["x"])
                yi = resp["columns"].index(spec["y"])
                st.bar_chart({r[xi]: r[yi] for r in resp["rows"]})
            triples = resp.get("graph_triples")
            if triples:
                with st.expander("🕸️ Network view", expanded=False):
                    graph_viz.render_network(triples)

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
