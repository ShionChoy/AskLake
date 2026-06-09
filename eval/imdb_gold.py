"""Real IMDb gold set for the headline eval (run with a live LLM via eval.real_run).

Hand-authored NL question -> gold SQL over the REAL IMDb parquet (built by `make build-imdb`
into PARQUET_DIR). Unlike the hermetic MINI_CASES, these run against a shared persistent
backend, so `schema_sql` is unused (left empty) — eval.real_run loads the IMDb backend and
scores candidate vs gold result sets on it. Gold SQL is validated to execute + return rows
by tests/unit/test_imdb_gold.py when the parquet is present.

Tiers
-----
topn        SELECT … ORDER BY … LIMIT N — must be strictly tie-safe at the boundary.
            All topn cases route through _TOPN_PARAMS / _topn_case() so the tie-safety
            test can probe the boundary automatically.
aggregation GROUP BY / scalar aggregates, no LIMIT — unconditionally tie-safe.
multihop    Joins across title_principals / name_basics / title_crew.
"""

from __future__ import annotations

from eval.harness import EvalCase

PARQUET_DIR = "data/imdb/parquet"

# ---------------------------------------------------------------------------
# Top-N machinery
# ---------------------------------------------------------------------------
# Each entry is a 6-tuple:
#   (slug, question_template, where_sql, key_expr, projection, n)
# key_expr is the ORDER BY key; _topn_probe_sql fetches N+1 rows of that key
# so the tie-safety test can confirm rank[N-1] > rank[N].
#
# Safety rule: use tr.numVotes as key_expr wherever possible — it rarely ties.
# For averageRating or runtimeMinutes orderings, the vote threshold must be
# raised until the boundary is verified clean (done in Step 6).
# ---------------------------------------------------------------------------

_TOPN_PARAMS: list[tuple[str, str, str, str, str, int]] = [
    # ---- overall most-voted ------------------------------------------------
    (
        "most_voted_overall",
        "What are the {n} movies with the most votes?",
        "1=1",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    # ---- most-voted per genre (numVotes order — safe) ----------------------
    (
        "most_voted_horror",
        "What are the {n} most-voted horror movies?",
        "tb.genres LIKE '%Horror%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_animation",
        "What are the {n} most-voted animation movies with at least 100,000 votes?",
        "tb.genres LIKE '%Animation%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_scifi",
        "What are the {n} most-voted sci-fi movies with at least 100,000 votes?",
        "tb.genres LIKE '%Sci-Fi%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_comedy_romance",
        "What are the {n} most-voted comedy-romance movies with at least 50,000 votes?",
        "tb.genres LIKE '%Comedy%' AND tb.genres LIKE '%Romance%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_drama_1990s",
        "What are the {n} most-voted drama movies from the 1990s with at least 100,000 votes?",
        "tb.genres LIKE '%Drama%' AND tb.startYear BETWEEN 1990 AND 1999 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
    (
        "most_voted_recent",
        "What are the {n} most-voted movies released in 2020 or later with at least 50,000 votes?",
        "tb.startYear >= 2020 AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
    (
        "most_voted_action",
        "What are the {n} most-voted action movies with at least 100,000 votes?",
        "tb.genres LIKE '%Action%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_crime",
        "What are the {n} most-voted crime movies with at least 100,000 votes?",
        "tb.genres LIKE '%Crime%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        10,
    ),
    (
        "most_voted_thriller",
        "What are the {n} most-voted thriller movies with at least 50,000 votes?",
        "tb.genres LIKE '%Thriller%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        10,
    ),
    (
        "most_voted_adventure",
        "What are the {n} most-voted adventure movies with at least 100,000 votes?",
        "tb.genres LIKE '%Adventure%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        10,
    ),
    (
        "most_voted_mystery",
        "What are the {n} most-voted mystery movies with at least 50,000 votes?",
        "tb.genres LIKE '%Mystery%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        10,
    ),
    (
        "most_voted_biography",
        "What are the {n} most-voted biography movies with at least 50,000 votes?",
        "tb.genres LIKE '%Biography%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        10,
    ),
    (
        "most_voted_war",
        "What are the {n} most-voted war movies with at least 50,000 votes?",
        "tb.genres LIKE '%War%' AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_western",
        "What are the {n} most-voted western movies with at least 25,000 votes?",
        "tb.genres LIKE '%Western%' AND tr.numVotes >= 25000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_musical",
        "What are the {n} most-voted musical movies with at least 25,000 votes?",
        "tb.genres LIKE '%Musical%' AND tr.numVotes >= 25000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_family",
        "What are the {n} most-voted family movies with at least 100,000 votes?",
        "tb.genres LIKE '%Family%' AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    # ---- most-voted per decade (numVotes order — safe) ---------------------
    (
        "most_voted_1970s",
        "What are the {n} most-voted movies from the 1970s with at least 50,000 votes?",
        "(tb.startYear // 10) * 10 = 1970 AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
    (
        "most_voted_1980s",
        "What are the {n} most-voted movies from the 1980s with at least 50,000 votes?",
        "(tb.startYear // 10) * 10 = 1980 AND tr.numVotes >= 50000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
    (
        "most_voted_1990s",
        "What are the {n} most-voted movies from the 1990s with at least 100,000 votes?",
        "(tb.startYear // 10) * 10 = 1990 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        10,
    ),
    (
        "most_voted_2000s",
        "What are the {n} most-voted movies from the 2000s with at least 100,000 votes?",
        "(tb.startYear // 10) * 10 = 2000 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        10,
    ),
    (
        "most_voted_2010s",
        "What are the {n} most-voted movies from the 2010s with at least 200,000 votes?",
        "(tb.startYear // 10) * 10 = 2010 AND tr.numVotes >= 200000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        10,
    ),
    # ---- most-voted for specific years (numVotes order — safe) -------------
    (
        "most_voted_year_1994",
        "What are the {n} most-voted movies released in 1994 with at least 100,000 votes?",
        "tb.startYear = 1994 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_year_1999",
        "What are the {n} most-voted movies released in 1999 with at least 100,000 votes?",
        "tb.startYear = 1999 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_year_2019",
        "What are the {n} most-voted movies released in 2019 with at least 200,000 votes?",
        "tb.startYear = 2019 AND tr.numVotes >= 200000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_year_2022",
        "What are the {n} most-voted movies released in 2022 with at least 100,000 votes?",
        "tb.startYear = 2022 AND tr.numVotes >= 100000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    # ---- highest-rated (averageRating key — thresholds chosen for tie safety) ---
    # top1 is always safe if rank[0] != rank[1]; verified on the 243K dataset
    (
        "highest_rated_popular",
        "What is the title and average rating of the highest-rated movie"
        " that has at least 50,000 votes?",
        "tr.numVotes >= 50000",
        "tr.averageRating",
        "tb.primaryTitle, tr.averageRating",
        1,
    ),
    # Horror top-5 by rating >=50k — boundary verified clean (8.4 vs 8.2)
    (
        "top_rated_horror",
        "What are the {n} highest-rated horror movies with at least 50,000 votes?",
        "tb.genres LIKE '%Horror%' AND tr.numVotes >= 50000",
        "tr.averageRating",
        "tb.primaryTitle, tr.averageRating, tr.numVotes",
        5,
    ),
    # Longest runtime — boundary verified clean (321 vs 242 minutes)
    (
        "longest_runtime",
        "What are the {n} longest movies (by runtime) with more than 180 minutes"
        " and at least 50,000 votes?",
        "tb.runtimeMinutes > 180 AND tr.numVotes >= 50000",
        "tb.runtimeMinutes",
        "tb.primaryTitle, tb.runtimeMinutes, tr.averageRating",
        5,
    ),
    (
        "most_voted_history",
        "What are the {n} most-voted history movies with at least 25,000 votes?",
        "tb.genres LIKE '%History%' AND tr.numVotes >= 25000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_sport",
        "What are the {n} most-voted sport movies with at least 25,000 votes?",
        "tb.genres LIKE '%Sport%' AND tr.numVotes >= 25000",
        "tr.numVotes",
        "tb.primaryTitle, tr.numVotes",
        5,
    ),
    (
        "most_voted_1960s",
        "What are the {n} most-voted movies from the 1960s with at least 25,000 votes?",
        "(tb.startYear // 10) * 10 = 1960 AND tr.numVotes >= 25000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
    (
        "most_voted_1950s",
        "What are the {n} most-voted movies from the 1950s with at least 10,000 votes?",
        "(tb.startYear // 10) * 10 = 1950 AND tr.numVotes >= 10000",
        "tr.numVotes",
        "tb.primaryTitle, tb.startYear, tr.numVotes",
        5,
    ),
]


def _topn_case(p: tuple[str, str, str, str, str, int]) -> EvalCase:
    slug, question, where_sql, key_expr, projection, n = p
    gold = (
        f"SELECT {projection}\n"
        "FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
        f"WHERE {where_sql}\n"
        f"ORDER BY {key_expr} DESC LIMIT {n}"
    )
    return EvalCase(
        name=f"topn_{slug}",
        schema_sql="",
        question=question.format(n=n),
        gold_sql=gold,
        tier="topn",
    )


def _topn_probe_sql(p: tuple[str, str, str, str, str, int]) -> str:
    _slug, _q, where_sql, key_expr, _proj, n = p
    return (
        f"SELECT {key_expr} AS k\n"
        "FROM title_basics tb JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
        f"WHERE {where_sql}\n"
        f"ORDER BY {key_expr} DESC LIMIT {n + 1}"
    )


# ---------------------------------------------------------------------------
# Aggregation-tier cases (GROUP BY / scalar aggregates, no LIMIT)
# ---------------------------------------------------------------------------

_YEAR_RANGES: list[tuple[int, int]] = [
    (2000, 2009),
    (2010, 2019),
    (2015, 2024),
    (1990, 1999),
    (1980, 1989),
]

_RATING_THRESHOLDS: list[float] = [7.0, 7.5, 8.0, 8.5]

_GENRES_AGG: list[str] = [
    "Drama",
    "Comedy",
    "Action",
    "Horror",
    "Thriller",
    "Romance",
    "Sci-Fi",
    "Biography",
]


def _count_per_year(start: int, end: int) -> EvalCase:
    return EvalCase(
        name=f"count_per_year_{start}_{end}",
        schema_sql="",
        question=(
            f"How many movies were released each year from {start} to {end}?"
            " List each year and its count in chronological order."
        ),
        gold_sql=(
            "SELECT startYear, COUNT(*) AS num_movies\n"
            "FROM title_basics\n"
            f"WHERE startYear BETWEEN {start} AND {end}\n"
            "GROUP BY startYear\n"
            "ORDER BY startYear"
        ),
        tier="aggregation",
    )


def _avg_rating_per_year_since(since: int) -> EvalCase:
    return EvalCase(
        name=f"avg_rating_per_year_since_{since}",
        schema_sql="",
        question=(
            f"What is the average movie rating for each year from {since} onwards, ordered by year?"
            " Include the number of titles per year."
        ),
        gold_sql=(
            "SELECT tb.startYear, AVG(tr.averageRating) AS avg_rating,"
            " COUNT(*) AS num_titles\n"
            "FROM title_basics tb\n"
            "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
            f"WHERE tb.startYear >= {since}\n"
            "GROUP BY tb.startYear\n"
            "ORDER BY tb.startYear"
        ),
        tier="aggregation",
    )


def _count_per_genre(genre: str, min_votes: int) -> EvalCase:
    slug = genre.lower().replace("-", "_")
    return EvalCase(
        name=f"count_{slug}_movies",
        schema_sql="",
        question=(f"How many {genre} movies have at least {min_votes:,} votes?"),
        gold_sql=(
            "SELECT COUNT(*) AS num_movies\n"
            "FROM title_basics tb\n"
            "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
            f"WHERE tb.genres LIKE '%{genre}%' AND tr.numVotes >= {min_votes}"
        ),
        tier="aggregation",
    )


def _count_above_rating(threshold: float, min_votes: int) -> EvalCase:
    slug = str(threshold).replace(".", "_")
    return EvalCase(
        name=f"count_movies_rating_ge_{slug}",
        schema_sql="",
        question=(
            f"How many movies have an average rating of at least {threshold}"
            f" with at least {min_votes:,} votes?"
        ),
        gold_sql=(
            "SELECT COUNT(*) AS num_movies\n"
            "FROM title_basics tb\n"
            "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
            f"WHERE tr.averageRating >= {threshold} AND tr.numVotes >= {min_votes}"
        ),
        tier="aggregation",
    )


def _movies_per_decade() -> EvalCase:
    return EvalCase(
        name="count_per_decade_1920_2020",
        schema_sql="",
        question=(
            "How many movies were released in each decade from the 1920s to the 2020s?"
            " Group by decade and order chronologically."
        ),
        gold_sql=(
            "SELECT (startYear // 10) * 10 AS decade,"
            " COUNT(*) AS num_movies\n"
            "FROM title_basics\n"
            "WHERE startYear BETWEEN 1920 AND 2029\n"
            "GROUP BY decade\n"
            "ORDER BY decade"
        ),
        tier="aggregation",
    )


def _avg_rating_per_decade_since(since: int) -> EvalCase:
    return EvalCase(
        name=f"avg_rating_per_decade_since_{since}",
        schema_sql="",
        question=(
            f"What is the average movie rating per decade from {since} onwards,"
            " along with the number of titles per decade?"
        ),
        gold_sql=(
            "SELECT (tb.startYear // 10) * 10 AS decade,\n"
            "       AVG(tr.averageRating) AS avg_rating,\n"
            "       COUNT(*) AS num_titles\n"
            "FROM title_basics tb\n"
            "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
            f"WHERE tb.startYear >= {since}\n"
            "GROUP BY decade\n"
            "ORDER BY decade"
        ),
        tier="aggregation",
    )


def _years_having_more_than(k: int) -> EvalCase:
    return EvalCase(
        name=f"years_having_gt_{k}_movies",
        schema_sql="",
        question=(f"Which years have more than {k:,} movies? List each year and its count."),
        gold_sql=(
            "SELECT startYear, COUNT(*) AS num_movies\n"
            "FROM title_basics\n"
            "WHERE startYear IS NOT NULL\n"
            "GROUP BY startYear\n"
            f"HAVING COUNT(*) > {k}\n"
            "ORDER BY startYear"
        ),
        tier="aggregation",
    )


_AGGREGATION_CASES: list[EvalCase] = (
    # count per year over various ranges
    [_count_per_year(s, e) for s, e in _YEAR_RANGES]
    # avg rating per year since various baselines
    + [_avg_rating_per_year_since(y) for y in [2000, 2010, 2015]]
    # count per genre (scalar)
    + [_count_per_genre(g, 10000) for g in _GENRES_AGG]
    # count movies above rating threshold
    + [_count_above_rating(t, 10000) for t in _RATING_THRESHOLDS]
    # decade-level aggregations
    + [
        _movies_per_decade(),
        _avg_rating_per_decade_since(2000),
    ]
    # years HAVING > K movies
    + [_years_having_more_than(k) for k in [2000, 5000]]
    # direct additional hand-authored aggregation cases
    + [
        EvalCase(
            name="count_drama_action_movies_per_year_2010_2020",
            schema_sql="",
            question=(
                "How many drama-action movies were released each year from 2010 to 2020?"
                " List year and count in order."
            ),
            gold_sql=(
                "SELECT startYear, COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Drama%' AND genres LIKE '%Action%'\n"
                "AND startYear BETWEEN 2010 AND 2020\n"
                "GROUP BY startYear\n"
                "ORDER BY startYear"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="avg_runtime_horror_films",
            schema_sql="",
            question=(
                "What is the average runtime in minutes of horror movies with a known runtime?"
            ),
            gold_sql=(
                "SELECT AVG(runtimeMinutes) AS avg_runtime_minutes\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Horror%' AND runtimeMinutes IS NOT NULL"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="avg_runtime_per_genre_action_drama_comedy",
            schema_sql="",
            question=(
                "What is the average runtime for each of the following genres:"
                " Action, Drama, and Comedy? Include only movies with a known runtime."
            ),
            gold_sql=(
                "SELECT 'Action' AS genre,"
                " AVG(runtimeMinutes) AS avg_runtime_minutes\n"
                "FROM title_basics WHERE genres LIKE '%Action%'"
                " AND runtimeMinutes IS NOT NULL\n"
                "UNION ALL\n"
                "SELECT 'Drama', AVG(runtimeMinutes)\n"
                "FROM title_basics WHERE genres LIKE '%Drama%'"
                " AND runtimeMinutes IS NOT NULL\n"
                "UNION ALL\n"
                "SELECT 'Comedy', AVG(runtimeMinutes)\n"
                "FROM title_basics WHERE genres LIKE '%Comedy%'"
                " AND runtimeMinutes IS NOT NULL"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="count_horror_movies_per_decade",
            schema_sql="",
            question=(
                "How many horror movies were released in each decade?"
                " Group by decade and order chronologically."
            ),
            gold_sql=(
                "SELECT (startYear // 10) * 10 AS decade,\n"
                "       COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Horror%' AND startYear IS NOT NULL\n"
                "GROUP BY decade\n"
                "ORDER BY decade"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="count_scifi_movies_per_year_2000_2020",
            schema_sql="",
            question=(
                "How many sci-fi movies were released each year from 2000 to 2020?"
                " List year and count in chronological order."
            ),
            gold_sql=(
                "SELECT startYear, COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Sci-Fi%' AND startYear BETWEEN 2000 AND 2020\n"
                "GROUP BY startYear\n"
                "ORDER BY startYear"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="count_movies_runtime_gt_120",
            schema_sql="",
            question=("How many movies in the dataset have a runtime greater than 120 minutes?"),
            gold_sql=(
                "SELECT COUNT(*) AS num_movies\nFROM title_basics\nWHERE runtimeMinutes > 120"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="avg_rating_per_year_2015_2020",
            schema_sql="",
            question=(
                "What is the average rating and number of movies for each year from 2015 to 2020?"
            ),
            gold_sql=(
                "SELECT tb.startYear, AVG(tr.averageRating) AS avg_rating,"
                " COUNT(*) AS num_titles\n"
                "FROM title_basics tb\n"
                "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
                "WHERE tb.startYear BETWEEN 2015 AND 2020\n"
                "GROUP BY tb.startYear\n"
                "ORDER BY tb.startYear"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="count_total_movies_in_dataset",
            schema_sql="",
            question="How many movies are in the dataset in total?",
            gold_sql=("SELECT COUNT(*) AS total_movies\nFROM title_basics"),
            tier="aggregation",
        ),
        EvalCase(
            name="count_animation_movies_per_year_2000_2020",
            schema_sql="",
            question=(
                "How many animation movies were released each year from 2000 to 2020?"
                " List year and count in chronological order."
            ),
            gold_sql=(
                "SELECT startYear, COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Animation%' AND startYear BETWEEN 2000 AND 2020\n"
                "GROUP BY startYear\n"
                "ORDER BY startYear"
            ),
            tier="aggregation",
        ),
        # Count long movies (runtime > 120 min) per decade
        EvalCase(
            name="count_long_movies_per_decade",
            schema_sql="",
            question=(
                "How many movies with a runtime greater than 120 minutes were released"
                " in each decade? Group by decade and order chronologically."
            ),
            gold_sql=(
                "SELECT (startYear // 10) * 10 AS decade,\n"
                "       COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE runtimeMinutes > 120 AND startYear IS NOT NULL\n"
                "GROUP BY decade\n"
                "ORDER BY decade"
            ),
            tier="aggregation",
        ),
        # Avg rating per genre scalar (Drama, Horror, Sci-Fi, Comedy)
        EvalCase(
            name="avg_rating_drama_movies",
            schema_sql="",
            question=(
                "What is the average rating of drama movies that have at least 10,000 votes?"
            ),
            gold_sql=(
                "SELECT AVG(tr.averageRating) AS avg_rating\n"
                "FROM title_basics tb\n"
                "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
                "WHERE tb.genres LIKE '%Drama%' AND tr.numVotes >= 10000"
            ),
            tier="aggregation",
        ),
        EvalCase(
            name="avg_rating_horror_movies",
            schema_sql="",
            question=(
                "What is the average rating of horror movies that have at least 10,000 votes?"
            ),
            gold_sql=(
                "SELECT AVG(tr.averageRating) AS avg_rating\n"
                "FROM title_basics tb\n"
                "JOIN title_ratings tr ON tb.tconst = tr.tconst\n"
                "WHERE tb.genres LIKE '%Horror%' AND tr.numVotes >= 10000"
            ),
            tier="aggregation",
        ),
        # Count drama movies per decade
        EvalCase(
            name="count_drama_per_decade",
            schema_sql="",
            question=(
                "How many drama movies were released in each decade?"
                " Group by decade and order chronologically."
            ),
            gold_sql=(
                "SELECT (startYear // 10) * 10 AS decade,\n"
                "       COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Drama%' AND startYear IS NOT NULL\n"
                "GROUP BY decade\n"
                "ORDER BY decade"
            ),
            tier="aggregation",
        ),
        # Count crime movies per year 2000-2020
        EvalCase(
            name="count_crime_per_year_2000_2020",
            schema_sql="",
            question=(
                "How many crime movies were released each year from 2000 to 2020?"
                " List year and count in chronological order."
            ),
            gold_sql=(
                "SELECT startYear, COUNT(*) AS num_movies\n"
                "FROM title_basics\n"
                "WHERE genres LIKE '%Crime%' AND startYear BETWEEN 2000 AND 2020\n"
                "GROUP BY startYear\n"
                "ORDER BY startYear"
            ),
            tier="aggregation",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Multi-hop tier (joins across title_principals / name_basics / title_crew)
# ---------------------------------------------------------------------------

_ACTOR_FILMOGRAPHIES: list[tuple[str, int, int]] = [
    # (actor_name, min_votes, n)  — ORDER BY numVotes DESC (tie-safe)
    ("Tom Hanks", 50000, 5),
    ("Leonardo DiCaprio", 50000, 5),
    ("Brad Pitt", 50000, 5),
    ("Morgan Freeman", 50000, 5),
    ("Robert De Niro", 50000, 5),
    ("Matt Damon", 50000, 5),
    ("Denzel Washington", 25000, 5),
    ("Cate Blanchett", 50000, 5),
    ("Tom Cruise", 50000, 5),
    ("Will Smith", 50000, 5),
    ("Christian Bale", 50000, 5),
    ("Natalie Portman", 50000, 5),
    ("Samuel L. Jackson", 50000, 5),
    ("Anne Hathaway", 50000, 5),
    ("Joaquin Phoenix", 50000, 5),
]

_DIRECTOR_BEST_FILMS: list[tuple[str, int]] = [
    # (director_name, min_votes) — ORDER BY averageRating DESC LIMIT 1 (safe if rank0 > rank1)
    ("Christopher Nolan", 100000),
    ("Steven Spielberg", 100000),
    ("Quentin Tarantino", 100000),
    ("Martin Scorsese", 100000),
    ("Peter Jackson", 100000),
    ("David Fincher", 100000),
]

_DIRECTOR_TOP_N_BY_VOTES: list[tuple[str, int, int]] = [
    # (director_name, min_votes, n) — ORDER BY numVotes DESC (tie-safe)
    ("Quentin Tarantino", 0, 5),
    ("David Fincher", 100000, 5),
    ("Peter Jackson", 100000, 5),
]


def _actor_top_n_probe(actor: str, min_votes: int, n: int) -> str:
    return (
        f"SELECT tr.numVotes AS k\n"
        "FROM title_principals tp\n"
        "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
        "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
        "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
        f"WHERE nb.primaryName = '{actor}' AND tp.category IN ('actor', 'actress')\n"
        f"AND tr.numVotes >= {min_votes}\n"
        f"ORDER BY tr.numVotes DESC LIMIT {n + 1}"
    )


def _actor_top_n(actor: str, min_votes: int, n: int) -> EvalCase:
    slug = actor.lower().replace(" ", "_")
    return EvalCase(
        name=f"actor_top{n}_{slug}",
        schema_sql="",
        question=(
            f"What are the {n} most-voted movies that {actor} starred in"
            f" with at least {min_votes:,} votes?"
        ),
        gold_sql=(
            f"SELECT tb.primaryTitle, tr.numVotes\n"
            "FROM title_principals tp\n"
            "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
            "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
            "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
            f"WHERE nb.primaryName = '{actor}'"
            " AND tp.category IN ('actor', 'actress')\n"
            f"AND tr.numVotes >= {min_votes}\n"
            f"ORDER BY tr.numVotes DESC LIMIT {n}"
        ),
        tier="multihop",
    )


def _director_best_rated(director: str, min_votes: int) -> EvalCase:
    slug = director.lower().replace(" ", "_")
    return EvalCase(
        name=f"director_best_rated_{slug}",
        schema_sql="",
        question=(
            f"What is the highest-rated movie directed by {director}"
            f" with at least {min_votes:,} votes?"
        ),
        gold_sql=(
            "SELECT tb.primaryTitle, tr.averageRating\n"
            "FROM title_principals tp\n"
            "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
            "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
            "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
            f"WHERE nb.primaryName = '{director}' AND tp.category = 'director'\n"
            f"AND tr.numVotes >= {min_votes}\n"
            "ORDER BY tr.averageRating DESC LIMIT 1"
        ),
        tier="multihop",
    )


def _director_best_rated_probe(director: str, min_votes: int) -> str:
    return (
        "SELECT tr.averageRating AS k\n"
        "FROM title_principals tp\n"
        "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
        "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
        "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
        f"WHERE nb.primaryName = '{director}' AND tp.category = 'director'\n"
        f"AND tr.numVotes >= {min_votes}\n"
        "ORDER BY tr.averageRating DESC LIMIT 2"
    )


def _director_top_n_votes(director: str, min_votes: int, n: int) -> EvalCase:
    slug = director.lower().replace(" ", "_")
    votes_clause = f"AND tr.numVotes >= {min_votes}\n" if min_votes > 0 else ""
    return EvalCase(
        name=f"director_top{n}_votes_{slug}",
        schema_sql="",
        question=(
            f"What are the {n} most-voted movies directed by {director}"
            + (f" with at least {min_votes:,} votes?" if min_votes > 0 else "?")
        ),
        gold_sql=(
            "SELECT tb.primaryTitle, tr.numVotes\n"
            "FROM title_principals tp\n"
            "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
            "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
            "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
            f"WHERE nb.primaryName = '{director}' AND tp.category = 'director'\n"
            f"{votes_clause}"
            f"ORDER BY tr.numVotes DESC LIMIT {n}"
        ),
        tier="multihop",
    )


def _director_top_n_votes_probe(director: str, min_votes: int, n: int) -> str:
    votes_clause = f"AND tr.numVotes >= {min_votes}\n" if min_votes > 0 else ""
    return (
        "SELECT tr.numVotes AS k\n"
        "FROM title_principals tp\n"
        "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
        "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
        "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
        f"WHERE nb.primaryName = '{director}' AND tp.category = 'director'\n"
        f"{votes_clause}"
        f"ORDER BY tr.numVotes DESC LIMIT {n + 1}"
    )


# ---------------------------------------------------------------------------
# Multihop tie-safety probes
# Each entry: (name, probe_sql, n)
# probe_sql selects the ordering key AS k for n+1 rows; the test asserts
# rows[n-1][0] > rows[n][0] when len(rows) > n.
# ---------------------------------------------------------------------------

_MULTIHOP_PROBES: list[tuple[str, str, int]] = (
    # Actor top-N by numVotes
    [
        (
            f"actor_top{n}_{actor.lower().replace(' ', '_')}",
            _actor_top_n_probe(actor, mv, n),
            n,
        )
        for actor, mv, n in _ACTOR_FILMOGRAPHIES
    ]
    # Director best-rated LIMIT 1 (rating key)
    + [
        (
            f"director_best_rated_{d.lower().replace(' ', '_')}",
            _director_best_rated_probe(d, mv),
            1,
        )
        for d, mv in _DIRECTOR_BEST_FILMS
    ]
    # Director top-N by numVotes
    + [
        (
            f"director_top{n}_votes_{d.lower().replace(' ', '_')}",
            _director_top_n_votes_probe(d, mv, n),
            n,
        )
        for d, mv, n in _DIRECTOR_TOP_N_BY_VOTES
    ]
    # Hand-authored director top-5 cases
    + [
        (
            "multihop_spielberg_top5_by_votes",
            (
                "SELECT tr.numVotes AS k\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Steven Spielberg' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 6"
            ),
            5,
        ),
        (
            "multihop_nolan_top5_by_votes",
            (
                "SELECT tr.numVotes AS k\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Christopher Nolan' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 6"
            ),
            5,
        ),
        (
            "multihop_scorsese_top5_by_votes",
            (
                "SELECT tr.numVotes AS k\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Martin Scorsese' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 6"
            ),
            5,
        ),
        # Director leaderboard: ORDER BY COUNT(*) DESC LIMIT 3
        # Boundary verified: rank[2]=8 (Tarantino) vs rank[3]=7 (Fincher) — clean
        (
            "directors_most_popular_films",
            (
                "SELECT COUNT(*) AS k\n"
                "FROM (\n"
                "    SELECT tconst, UNNEST(string_split(directors, ',')) AS nconst\n"
                "    FROM title_crew\n"
                "    WHERE directors IS NOT NULL AND directors != ''\n"
                ") t\n"
                "JOIN title_basics tb ON t.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON t.tconst = tr.tconst\n"
                "JOIN name_basics nb ON nb.nconst = t.nconst\n"
                "WHERE tr.numVotes >= 500000\n"
                "GROUP BY nb.primaryName\n"
                "ORDER BY k DESC\n"
                "LIMIT 4"
            ),
            3,
        ),
    ]
)


_MULTIHOP_CASES: list[EvalCase] = (
    [_actor_top_n(actor, mv, n) for actor, mv, n in _ACTOR_FILMOGRAPHIES]
    + [_director_best_rated(d, mv) for d, mv in _DIRECTOR_BEST_FILMS]
    + [_director_top_n_votes(d, mv, n) for d, mv, n in _DIRECTOR_TOP_N_BY_VOTES]
    + [
        # Director leaderboard — top 3 directors by # films with >=500k votes
        # Boundary: rank[2]=8 (Tarantino) vs rank[3]=7 (Fincher) — clean
        EvalCase(
            name="directors_most_popular_films",
            schema_sql="",
            question=(
                "Which 3 directors have directed the most movies with at least 500,000 votes?"
                " List each director and their movie count."
            ),
            gold_sql=(
                "SELECT nb.primaryName AS director, COUNT(*) AS num_movies\n"
                "FROM (\n"
                "    SELECT tconst, UNNEST(string_split(directors, ',')) AS nconst\n"
                "    FROM title_crew\n"
                "    WHERE directors IS NOT NULL AND directors != ''\n"
                ") t\n"
                "JOIN title_basics tb ON t.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON t.tconst = tr.tconst\n"
                "JOIN name_basics nb ON nb.nconst = t.nconst\n"
                "WHERE tr.numVotes >= 500000\n"
                "GROUP BY nb.primaryName\n"
                "ORDER BY num_movies DESC\n"
                "LIMIT 3"
            ),
            tier="multihop",
        ),
        # Top 5 most-voted Spielberg films via title_principals
        EvalCase(
            name="multihop_spielberg_top5_by_votes",
            schema_sql="",
            question=("What are the 5 most-voted movies directed by Steven Spielberg?"),
            gold_sql=(
                "SELECT tb.primaryTitle, tr.numVotes\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Steven Spielberg' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 5"
            ),
            tier="multihop",
        ),
        # Top 5 most-voted Nolan films
        EvalCase(
            name="multihop_nolan_top5_by_votes",
            schema_sql="",
            question=("What are the 5 most-voted movies directed by Christopher Nolan?"),
            gold_sql=(
                "SELECT tb.primaryTitle, tr.numVotes\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Christopher Nolan' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 5"
            ),
            tier="multihop",
        ),
        # Count films per actor (scalar aggregation with join)
        EvalCase(
            name="multihop_count_films_tom_hanks",
            schema_sql="",
            question=(
                "How many movies has Tom Hanks appeared in as actor or actress"
                " with at least 10,000 votes?"
            ),
            gold_sql=(
                "SELECT COUNT(*) AS num_movies\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Tom Hanks' AND tp.category IN ('actor', 'actress')\n"
                "AND tr.numVotes >= 10000"
            ),
            tier="multihop",
        ),
        # Writer credits
        EvalCase(
            name="multihop_count_films_nolan_writer",
            schema_sql="",
            question=("How many movies has Christopher Nolan received writing credit for?"),
            gold_sql=(
                "SELECT COUNT(*) AS num_movies\n"
                "FROM (\n"
                "    SELECT tconst, UNNEST(string_split(writers, ',')) AS nconst\n"
                "    FROM title_crew\n"
                "    WHERE writers IS NOT NULL AND writers != ''\n"
                ") t\n"
                "JOIN name_basics nb ON nb.nconst = t.nconst\n"
                "WHERE nb.primaryName = 'Christopher Nolan'"
            ),
            tier="multihop",
        ),
        # Co-starring query
        EvalCase(
            name="multihop_costarring_dicaprio_pitt",
            schema_sql="",
            question=("Which movies feature both Leonardo DiCaprio and Brad Pitt as actors?"),
            gold_sql=(
                "SELECT tb.primaryTitle, tr.averageRating, tr.numVotes\n"
                "FROM title_principals tp1\n"
                "JOIN name_basics nb1 ON tp1.nconst = nb1.nconst\n"
                "JOIN title_principals tp2 ON tp1.tconst = tp2.tconst\n"
                "JOIN name_basics nb2 ON tp2.nconst = nb2.nconst\n"
                "JOIN title_basics tb ON tp1.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp1.tconst = tr.tconst\n"
                "WHERE nb1.primaryName = 'Leonardo DiCaprio'"
                " AND nb2.primaryName = 'Brad Pitt'\n"
                "AND tp1.category IN ('actor', 'actress')"
                " AND tp2.category IN ('actor', 'actress')"
            ),
            tier="multihop",
        ),
        # Most-voted Scorsese films via principals
        EvalCase(
            name="multihop_scorsese_top5_by_votes",
            schema_sql="",
            question=("What are the 5 most-voted movies directed by Martin Scorsese?"),
            gold_sql=(
                "SELECT tb.primaryTitle, tr.numVotes\n"
                "FROM title_principals tp\n"
                "JOIN title_basics tb ON tp.tconst = tb.tconst\n"
                "JOIN title_ratings tr ON tp.tconst = tr.tconst\n"
                "JOIN name_basics nb ON tp.nconst = nb.nconst\n"
                "WHERE nb.primaryName = 'Martin Scorsese' AND tp.category = 'director'\n"
                "ORDER BY tr.numVotes DESC LIMIT 5"
            ),
            tier="multihop",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Assemble IMDB_GOLD
# ---------------------------------------------------------------------------

IMDB_GOLD: list[EvalCase] = (
    [_topn_case(p) for p in _TOPN_PARAMS] + _AGGREGATION_CASES + _MULTIHOP_CASES
)
