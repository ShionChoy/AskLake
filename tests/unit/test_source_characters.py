import gzip

import duckdb

from datasets.imdb.source import build_parquet


def _raw(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)

    def w(name, text):
        with gzip.open(raw_dir / name, "wt", encoding="utf-8") as f:
            f.write(text)

    w(
        "title.basics.tsv.gz",
        "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\t"
        "runtimeMinutes\tgenres\n"
        "tt1\tmovie\tThe Matrix\tThe Matrix\t0\t1999\t\\N\t136\tAction,Sci-Fi\n",
    )
    w("title.ratings.tsv.gz", "tconst\taverageRating\tnumVotes\ntt1\t8.7\t2000000\n")
    w("title.crew.tsv.gz", "tconst\tdirectors\twriters\ntt1\tnm9\t\\N\n")
    w(
        "name.basics.tsv.gz",
        "nconst\tprimaryName\tbirthYear\tdeathYear\tprimaryProfession\tknownForTitles\n"
        "nm1\tKeanu Reeves\t1964\t\\N\tactor\ttt1\n"
        "nm9\tLana Wachowski\t1965\t\\N\tdirector\ttt1\n",
    )
    w(
        "title.principals.tsv.gz",
        'tconst\tordering\tnconst\tcategory\tjob\tcharacters\ntt1\t1\tnm1\tactor\t\\N\t["Neo"]\n',
    )
    return str(raw_dir)


def test_principals_parquet_includes_characters(tmp_path):
    raw = _raw(tmp_path / "raw")
    build_parquet(raw, str(tmp_path / "pq"), min_votes=1000)
    cols = (
        duckdb.connect()
        .execute(f"SELECT * FROM '{tmp_path}/pq/title_principals.parquet' LIMIT 0")
        .description
    )
    names = [c[0] for c in cols]
    assert "characters" in names
