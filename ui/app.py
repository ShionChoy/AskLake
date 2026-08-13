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
    "anthropic": [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ],
}
_CUSTOM = "(custom…)"
PATHS = ["auto", "sql", "graph", "fusion"]
_RETIRED_MODELS = {"deepseek-chat", "deepseek-reasoner"}


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        code: str = "request_failed",
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id

    def __str__(self) -> str:
        suffix = f" · Request ID: {self.request_id}" if self.request_id else ""
        return f"{self.message}{suffix}"


def _creds_payload(state) -> dict:
    """Pick provider/model/api_key out of a dict-like state, omitting empty values."""
    out = {}
    for k in ("provider", "model", "api_key"):
        v = state.get(k, "")
        if v:
            out[k] = v
    return out


def _auth_headers(state) -> dict:
    """Build the Authorization header from a stored access token (separate from the LLM key)."""
    token = state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _ask_body(question: str, state) -> dict:
    """Build the /ask_trace request body: question + retrieval path + (non-empty) credentials."""
    return {"question": question, "path": state.get("path", "auto"), **_creds_payload(state)}


def _response_json(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(
            "The API returned an unreadable response.", status_code=response.status_code
        ) from exc
    if response.status_code >= 400:
        detail = payload.get("error") or payload.get("detail") or "The request was rejected."
        if not isinstance(detail, str):
            detail = "The request did not pass API validation."
        raise ApiError(
            detail,
            status_code=response.status_code,
            code=str(payload.get("code", "request_rejected")),
            request_id=str(payload.get("request_id", "")),
        )
    return payload


def _post(path: str, payload: dict) -> dict:
    response = requests.post(
        f"{API_URL}{path}", json=payload, headers=_auth_headers(st.session_state), timeout=120
    )
    return _response_json(response)


def _session() -> dict:
    response = requests.get(
        f"{API_URL}/session", headers=_auth_headers(st.session_state), timeout=10
    )
    return _response_json(response)


def _export_csv(sql: str) -> tuple[bytes, str]:
    response = requests.post(
        f"{API_URL}/export",
        json={"sql": sql},
        headers=_auth_headers(st.session_state),
        timeout=120,
    )
    if response.status_code >= 400:
        _response_json(response)
    disposition = response.headers.get("Content-Disposition", "")
    filename = "asklake-export.csv"
    if 'filename="' in disposition:
        filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
    return response.content, filename


@st.cache_data(ttl=30)
def _info() -> dict:
    try:
        return requests.get(f"{API_URL}/info", timeout=5).json()
    except Exception:  # noqa: BLE001
        return {}


def _ask(question: str) -> dict:
    """Prefer the traced endpoint (rich steps + BYO credentials + routing); fall back to /ask."""
    body = _ask_body(question, st.session_state)
    response = requests.post(
        f"{API_URL}/ask_trace",
        json=body,
        headers=_auth_headers(st.session_state),
        timeout=180,
    )
    # Compatibility with the small hermetic API; authentication/authorization failures must never
    # be hidden behind a second request.
    if response.status_code == 404:
        return _post("/ask", {"question": question})
    return _response_json(response)


def _show_table(columns, rows) -> None:
    st.dataframe({c: [row[i] for row in rows] for i, c in enumerate(columns)})


def _render_governance(governance: dict) -> None:
    if not governance:
        return
    st.caption(
        "🛡️ Enforced as "
        f"**{governance.get('role', 'unknown')}** · "
        f"action **{governance.get('action', 'unknown')}** · "
        f"maximum {governance.get('max_rows', '—')} rows · "
        f"policy v{governance.get('policy_version', '—')}"
    )
    notices = governance.get("notices") or []
    if notices:
        with st.expander("Data use and provenance", expanded=False):
            for notice in notices:
                st.caption(notice)


def _init_state() -> None:
    if not st.session_state.get("creds_loaded"):
        saved = credentials.load()
        provider = saved.get("provider", "deepseek")
        if provider not in PROVIDERS:
            provider = "deepseek"
        model = saved.get("model", DEFAULT_MODELS[provider][0])
        if model in _RETIRED_MODELS:
            model = DEFAULT_MODELS[provider][0]
        st.session_state.provider = provider
        st.session_state.model = model
        st.session_state.api_key = saved.get("api_key", "")
        st.session_state.access_token = saved.get("access_token", "")
        st.session_state.creds_loaded = True

    provider = st.session_state.get("provider", "deepseek")
    if provider not in PROVIDERS:
        provider = "deepseek"
        st.session_state.provider = provider
    model = st.session_state.get("model", DEFAULT_MODELS[provider][0])
    options = DEFAULT_MODELS[provider] + [_CUSTOM]
    st.session_state.setdefault("provider_choice", provider)
    if st.session_state.get("model_choice") not in options:
        st.session_state.model_choice = model if model in options else _CUSTOM
    if model not in DEFAULT_MODELS[provider]:
        st.session_state.custom_model = model
    else:
        st.session_state.setdefault("custom_model", "")


def _provider_changed() -> None:
    provider = st.session_state.provider_choice
    default = DEFAULT_MODELS[provider][0]
    st.session_state.provider = provider
    st.session_state.model_choice = default
    st.session_state.model = default
    st.session_state.custom_model = ""


def _sidebar() -> dict | None:
    st.sidebar.header("🛡️ Access & governance")
    st.sidebar.text_input(
        "Access token",
        type="password",
        key="access_token",
        help=(
            "The API resolves this credential to a server-side role. "
            "Leaving it empty uses the configured anonymous role."
        ),
    )
    st.sidebar.caption("Tokens are sent as Bearer credentials and are never saved by this UI.")

    active_session: dict | None = None
    try:
        active_session = _session()
    except ApiError as exc:
        if exc.status_code == 401:
            st.sidebar.error("Invalid, expired, or disabled access token.")
        else:
            st.sidebar.error(str(exc))
    except requests.RequestException:
        st.sidebar.error("Governance API is unavailable.")

    if active_session:
        principal = active_session.get("principal") or {}
        governance = active_session.get("governance") or {}
        role = principal.get("role", "unknown")
        user = principal.get("user", "unknown")
        method = principal.get("authentication_method", "unknown")
        if role == "public":
            st.sidebar.info(f"Effective role: **{role}** · {user}")
        else:
            st.sidebar.success(f"Effective role: **{role}** · {user}")
        actions = governance.get("actions") or []
        limits = governance.get("limits") or {}
        with st.sidebar.expander("Effective permissions", expanded=True):
            st.caption(f"Authentication: {method}")
            if principal.get("credential_id"):
                st.caption(f"Credential: {principal['credential_id']}")
            st.caption("Server-authorized actions")
            st.write(", ".join(actions) if actions else "None")
            st.caption(
                f"Query cap: {limits.get('query_rows', '—')} rows · "
                f"Graph cap: {limits.get('graph_triples', '—')} triples"
            )
            controls = governance.get("column_controls") or []
            filtered = governance.get("row_filtered_tables") or []
            if controls:
                st.caption(
                    "Column controls: "
                    + ", ".join(f"{item['column']} ({item['handling']})" for item in controls)
                )
            if filtered:
                st.caption("Row-filtered tables: " + ", ".join(filtered))

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Model & API key")

    provider = st.sidebar.selectbox(
        "Provider",
        PROVIDERS,
        key="provider_choice",
        on_change=_provider_changed,
    )
    st.session_state.provider = provider

    options = DEFAULT_MODELS[provider] + [_CUSTOM]
    choice = st.sidebar.selectbox("Model", options, key="model_choice")
    if choice == _CUSTOM:
        model = st.sidebar.text_input("Custom model", key="custom_model")
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
    return active_session


def _render_result(resp: dict) -> None:
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

    _render_governance(resp.get("governance") or {})


def render() -> None:
    _init_state()
    active_session = _sidebar()

    st.title("AskLake")

    if not active_session:
        st.error(
            "No valid governed session is available. Check the API and access token in the sidebar."
        )
        return

    principal = active_session.get("principal") or {}
    access = active_session.get("governance") or {}
    role = principal.get("role", "unknown")
    actions = set(access.get("actions") or [])
    limits = access.get("limits") or {}
    identity_col, policy_col, query_col, graph_col = st.columns(4)
    identity_col.metric("Effective role", role)
    policy_col.metric("Policy", f"v{access.get('policy_version', '—')}")
    query_col.metric("Query row cap", limits.get("query_rows", "—"))
    graph_col.metric("Graph triple cap", limits.get("graph_triples", "—"))

    with st.expander("Active governance controls", expanded=role == "public"):
        st.write(f"**Identity:** {principal.get('user', 'unknown')} · **Role:** {role}")
        st.write("**Authentication:** " + str(principal.get("authentication_method", "unknown")))
        if principal.get("credential_id"):
            st.write("**Credential ID:** " + str(principal["credential_id"]))
        st.write("**Authorized actions:** " + (", ".join(sorted(actions)) or "None"))
        filtered = access.get("row_filtered_tables") or []
        controls = access.get("column_controls") or []
        if filtered:
            st.write("**Row security:** " + ", ".join(filtered))
        else:
            st.write("**Row security:** no role-specific filters")
        if controls:
            st.write(
                "**Column security:** "
                + ", ".join(f"{item['column']} → {item['handling']}" for item in controls)
            )
        else:
            st.write("**Column security:** no role-specific masking or denied columns")

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
            try:
                st.session_state["last_resp"] = _ask(question)
                st.session_state.pop("last_error", None)
            except (ApiError, requests.RequestException) as exc:
                st.session_state.pop("last_resp", None)
                st.session_state["last_error"] = str(exc)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
    resp = st.session_state.get("last_resp")
    if resp:
        _render_result(resp)

    st.header("Raw SQL console")
    if "raw_sql" not in actions:
        st.info(
            f"The effective **{role}** role is not authorized for raw SQL. "
            "Use natural-language or graph queries, or authenticate with an analyst credential."
        )
    else:
        sql = st.text_area("SQL", value="SELECT 1 AS hello")
        if st.button("Run SQL"):
            try:
                sql_resp = _post("/query", {"sql": sql})
                _show_table(sql_resp["columns"], sql_resp["rows"])
                _render_governance(sql_resp.get("governance") or {})
            except (ApiError, requests.RequestException) as exc:
                st.error(str(exc))

    if "export" in actions:
        st.header("Governed CSV export")
        st.caption(
            "Exports use the same table, row, column, SQL, license, and result-size controls as "
            "interactive queries. Spreadsheet formulas are neutralized."
        )
        export_sql = st.text_area(
            "Export SQL", value="SELECT * FROM title_basics LIMIT 100", key="export_sql"
        )
        if st.button("Prepare governed export"):
            try:
                data, filename = _export_csv(export_sql)
                st.session_state["export_download"] = {"data": data, "filename": filename}
                st.session_state.pop("export_error", None)
            except (ApiError, requests.RequestException) as exc:
                st.session_state.pop("export_download", None)
                st.session_state["export_error"] = str(exc)
        if st.session_state.get("export_error"):
            st.error(st.session_state["export_error"])
        download = st.session_state.get("export_download")
        if download:
            st.download_button(
                "Download CSV",
                data=download["data"],
                file_name=download["filename"],
                mime="text/csv",
            )


if __name__ == "__main__":
    render()
