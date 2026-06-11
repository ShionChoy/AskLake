from pathlib import Path

SCRIPT = Path("scripts/download_data.sh").read_text()


def test_download_script_has_no_cmu_reference():
    assert "cs.cmu.edu" not in SCRIPT
    assert "MovieSummaries" not in SCRIPT


def test_download_script_still_fetches_imdb():
    assert "datasets.imdbws.com" in SCRIPT
    assert "title.principals.tsv.gz" in SCRIPT
