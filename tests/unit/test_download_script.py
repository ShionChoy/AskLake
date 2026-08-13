from pathlib import Path

SCRIPT = Path("scripts/download_data.sh").read_text()


def test_download_script_still_fetches_imdb():
    assert "datasets.imdbws.com" in SCRIPT
    assert "title.principals.tsv.gz" in SCRIPT
