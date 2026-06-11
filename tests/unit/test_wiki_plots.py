from datasets.imdb_cmu.wiki_plots import _extract_plot_section, fetch_plot, resolve_enwiki_titles


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakeClient:
    """Returns queued payloads in order; records the URLs/params it was called with."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append((url, params))
        return _FakeResp(self._payloads.pop(0))


def test_extract_plot_section_pulls_plot_only():
    wikitext = (
        "Lead paragraph.\n\n== Plot ==\nNeo learns the world is a simulation.\n\n"
        "== Cast ==\n* Keanu Reeves as Neo\n"
    )
    assert "simulation" in _extract_plot_section(wikitext)
    assert "Keanu" not in _extract_plot_section(wikitext)


def test_extract_plot_section_handles_synopsis_heading():
    assert "rosebud" in _extract_plot_section("== Synopsis ==\nHe whispers rosebud.\n").lower()


def test_extract_plot_section_returns_empty_when_absent():
    assert _extract_plot_section("== Cast ==\nonly cast here\n") == ""


def test_resolve_enwiki_titles_maps_tconst_to_article():
    payload = {
        "results": {
            "bindings": [
                {
                    "imdb": {"value": "tt0133093"},
                    "article": {"value": "https://en.wikipedia.org/wiki/The_Matrix"},
                }
            ]
        }
    }
    client = _FakeClient([payload])
    out = resolve_enwiki_titles(["tt0133093"], client=client)
    assert out == {"tt0133093": "The Matrix"}


def test_fetch_plot_returns_cleaned_section():
    sections = {"parse": {"sections": [{"line": "Plot", "index": "1"}]}}
    secwt = {"parse": {"wikitext": {"*": "Neo takes the [[red pill]].<ref>x</ref>"}}}
    client = _FakeClient([sections, secwt])
    text = fetch_plot("The Matrix", client=client)
    assert "red pill" in text and "<ref>" not in text
