"""Real IMDb gold set for the headline eval (run with a live LLM via eval.real_run).

Hand-authored NL question -> gold SQL over the REAL IMDb parquet (built by `make build-imdb`
into PARQUET_DIR). Unlike the hermetic MINI_CASES, these run against a shared persistent
backend, so `schema_sql` is unused (left empty) — eval.real_run loads the IMDb backend and
scores candidate vs gold result sets on it. Gold SQL is validated to execute + return rows
by tests/unit/test_imdb_gold.py when the parquet is present."""
from __future__ import annotations

from eval.harness import EvalCase

PARQUET_DIR = "data/imdb/parquet"

IMDB_GOLD: list[EvalCase] = [
    EvalCase(
        name="highest_rated_popular_movie",
        schema_sql="",
        question=(
            "What is the title and average rating of the highest-rated movie"
            " that has at least 50,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tr.numVotes >= 50000
ORDER BY tr.averageRating DESC
LIMIT 1
""".strip(),
    ),
    EvalCase(
        name="most_voted_movies_top5",
        schema_sql="",
        question=(
            "What are the 5 movies with the most votes, ordered from most to fewest votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.numVotes
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
ORDER BY tr.numVotes DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="count_movies_per_year_2015_2024",
        schema_sql="",
        question=(
            "How many movies were released each year from 2015 to 2024?"
            " List each year and its count in chronological order."
        ),
        gold_sql="""
SELECT startYear, COUNT(*) AS num_movies
FROM title_basics
WHERE startYear BETWEEN 2015 AND 2024
GROUP BY startYear
ORDER BY startYear
""".strip(),
    ),
    EvalCase(
        name="avg_rating_by_year_since_2000",
        schema_sql="",
        question=(
            "What is the average movie rating for each year from 2000 onwards, ordered by year?"
            " Include the number of titles per year."
        ),
        gold_sql="""
SELECT tb.startYear, ROUND(AVG(tr.averageRating), 2) AS avg_rating, COUNT(*) AS num_titles
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.startYear >= 2000
GROUP BY tb.startYear
ORDER BY tb.startYear
""".strip(),
    ),
    EvalCase(
        name="top5_horror_movies_by_rating",
        schema_sql="",
        question=(
            "What are the top 5 highest-rated horror movies with at least 50,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.averageRating, tr.numVotes
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.genres LIKE '%Horror%' AND tr.numVotes >= 50000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="top5_animation_movies_by_rating",
        schema_sql="",
        question=(
            "What are the top 5 highest-rated animation movies with at least 100,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.genres LIKE '%Animation%' AND tr.numVotes >= 100000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="top5_scifi_movies_by_rating",
        schema_sql="",
        question=(
            "What are the 5 best sci-fi movies by rating among those with at least 100,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.genres LIKE '%Sci-Fi%' AND tr.numVotes >= 100000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="top5_drama_1990_2000",
        schema_sql="",
        question=(
            "What are the 5 highest-rated drama movies released between 1990 and 2000"
            " with at least 100,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tb.startYear, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.genres LIKE '%Drama%' AND tb.startYear BETWEEN 1990 AND 2000 AND tr.numVotes >= 100000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="top5_recent_movies_since_2020",
        schema_sql="",
        question=(
            "What are the 5 highest-rated movies released in 2020 or later"
            " with at least 50,000 votes?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tb.startYear, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.startYear >= 2020 AND tr.numVotes >= 50000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="top5_comedy_romance_by_rating",
        schema_sql="",
        question=(
            "What are the 5 best comedy-romance movies with at least 50,000 votes,"
            " ordered by rating?"
        ),
        gold_sql="""
SELECT tb.primaryTitle, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.genres LIKE '%Comedy%' AND tb.genres LIKE '%Romance%' AND tr.numVotes >= 50000
ORDER BY tr.averageRating DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="longest_movies_top5",
        schema_sql="",
        question=(
            "What are the 5 longest movies (by runtime in minutes) with at least 50,000 votes?"
            " Include their runtime and rating."
        ),
        gold_sql="""
SELECT tb.primaryTitle, tb.runtimeMinutes, tr.averageRating
FROM title_basics tb
JOIN title_ratings tr ON tb.tconst = tr.tconst
WHERE tb.runtimeMinutes > 180 AND tr.numVotes >= 50000
ORDER BY tb.runtimeMinutes DESC
LIMIT 5
""".strip(),
    ),
    EvalCase(
        name="directors_most_popular_films",
        schema_sql="",
        question=(
            "Which 5 directors have directed the most movies with at least 50,000 votes?"
            " List each director and their movie count."
        ),
        gold_sql="""
SELECT nb.primaryName AS director, COUNT(*) AS num_movies
FROM (
    SELECT tconst, UNNEST(string_split(directors, ',')) AS nconst
    FROM title_crew
    WHERE directors IS NOT NULL AND directors != ''
) t
JOIN title_basics tb ON t.tconst = tb.tconst
JOIN title_ratings tr ON t.tconst = tr.tconst
JOIN name_basics nb ON nb.nconst = t.nconst
WHERE tr.numVotes >= 50000
GROUP BY nb.primaryName
ORDER BY num_movies DESC
LIMIT 5
""".strip(),
    ),
]
