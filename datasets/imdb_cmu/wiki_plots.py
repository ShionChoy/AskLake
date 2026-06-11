"""Resolve IMDb tconsts to English-Wikipedia articles (via the Wikidata P345 bridge) and fetch
each article's Plot section (CC BY-SA) through the MediaWiki API. Dataset-specific; offline build
step. HTTP client is injectable so tests run without network."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from urllib.parse import unquote

import httpx

_SPARQL = "https://query.wikidata.org/sparql"
_MW_API = "https://en.wikipedia.org/w/api.php"
_UA = "AskLake/1.0 (educational project; https://github.com/ShionChoy/AskLake)"
_PLOT_HEADINGS = {"plot", "plot summary", "synopsis", "plot synopsis", "story"}

_TRANSIENT = (httpx.TransportError,)
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_with_retry(client, url, params, *, attempts=4, backoff=1.0, sleep=time.sleep):
    """GET with retry on transient network errors and retryable HTTP status (exponential backoff).
    Re-raises the last error if every attempt fails. `sleep` is injectable for tests."""
    last_exc = None
    for i in range(attempts):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except _TRANSIENT as exc:
            last_exc = exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRY_STATUS:
                raise
            last_exc = exc
        if i < attempts - 1:
            sleep(backoff * (2**i))
    raise last_exc


def _client(client: httpx.Client | None) -> httpx.Client:
    return client or httpx.Client(timeout=30.0, headers={"User-Agent": _UA})


def resolve_enwiki_titles(
    tconsts: Iterable[str], *, client: httpx.Client | None = None, batch: int = 200
) -> dict[str, str]:
    """{tconst: enwiki article title} for tconsts that have a P345 statement + an enwiki sitelink.
    Missing ones are simply absent from the result."""
    c = _client(client)
    out: dict[str, str] = {}
    ids = [t for t in tconsts if t]
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        values = " ".join(f'"{t}"' for t in chunk)
        query = (
            "SELECT ?imdb ?article WHERE { "
            f"VALUES ?imdb {{ {values} }} "
            "?item wdt:P345 ?imdb . "
            "?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }"
        )
        try:
            resp = _get_with_retry(c, _SPARQL, {"query": query, "format": "json"})
        except httpx.HTTPError as exc:  # whole batch failed after retries — skip, keep the rest
            print(
                f"[wiki_plots] SPARQL batch failed after retries ({exc}); skipping {len(chunk)} ids"
            )
            continue
        for b in resp.json()["results"]["bindings"]:
            tconst = b["imdb"]["value"]
            url = b["article"]["value"]
            title = url.rsplit("/wiki/", 1)[-1].replace("_", " ")
            out[tconst] = unquote(title)
    return out


def _strip_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)  # shallow templates
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)  # [[a|b]] -> b, [[a]] -> a
    text = re.sub(r"'''?", "", text)  # bold/italic
    text = re.sub(r"<[^>]+>", "", text)  # stray html
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_plot_section(wikitext: str) -> str:
    """Return the cleaned text under the first Plot/Synopsis heading, or '' if none."""
    parts = re.split(r"^==+\s*(.*?)\s*==+\s*$", wikitext, flags=re.MULTILINE)
    # parts = [pre, heading1, body1, heading2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        if parts[i].strip().lower() in _PLOT_HEADINGS:
            return _strip_wikitext(parts[i + 1])
    return ""


def fetch_plot(title: str, *, client: httpx.Client | None = None) -> str | None:
    """Fetch the article's plot-section wikitext via the MediaWiki API and clean it. None if the
    article has no recognizable plot section."""
    c = _client(client)
    sec = _get_with_retry(
        c,
        _MW_API,
        {"action": "parse", "page": title, "prop": "sections", "format": "json", "redirects": "1"},
    )
    index = None
    for s in sec.json().get("parse", {}).get("sections", []):
        if s.get("line", "").strip().lower() in _PLOT_HEADINGS:
            index = s.get("index")
            break
    if not index:
        return None
    wt = _get_with_retry(
        c,
        _MW_API,
        {
            "action": "parse",
            "page": title,
            "section": index,
            "prop": "wikitext",
            "format": "json",
            "redirects": "1",
        },
    )
    raw = wt.json().get("parse", {}).get("wikitext", {}).get("*", "")
    cleaned = _strip_wikitext(re.sub(r"^==+.*?==+\s*", "", raw, count=1, flags=re.MULTILINE))
    return cleaned or None
