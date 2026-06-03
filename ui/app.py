# ui/app.py
from __future__ import annotations

import os

import streamlit as st

API_URL = os.environ.get("ASKLAKE_API_URL", "http://localhost:8000")


def render() -> None:
    st.title("AskLake — SQL Console (P0)")
    sql = st.text_area("SQL", value="SELECT 1 AS hello")
    if st.button("Run"):
        resp = st.session_state.get("_client", _default_post)(f"{API_URL}/query", {"sql": sql})
        if resp.get("error"):
            st.error(resp["error"])
        else:
            st.dataframe(
                {c: [row[i] for row in resp["rows"]] for i, c in enumerate(resp["columns"])}
            )


def _default_post(url: str, payload: dict) -> dict:
    import requests

    return requests.post(url, json=payload, timeout=30).json()


if __name__ == "__main__":
    render()
