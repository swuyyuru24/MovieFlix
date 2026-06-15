"""Unit tests for the MovieFlix recommendation engine.

These run against a small synthetic dataset so they're fast and deterministic
and don't require the MovieLens download.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movieflix.data import MovieData  # noqa: E402
from movieflix.engine import RecommenderEngine  # noqa: E402


def _toy_data() -> MovieData:
    """Two clear genre clusters: sci-fi/action vs. romance/drama."""
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": ["Spacewar", "Robot Wars", "Galaxy Quest", "Love Letters", "Sweet Romance", "Heartbreak"],
            "year": [2000, 2001, 2002, 2003, 2004, 2005],
            "genres": [
                ["Sci-Fi", "Action"],
                ["Sci-Fi", "Action"],
                ["Sci-Fi", "Adventure"],
                ["Romance", "Drama"],
                ["Romance", "Comedy"],
                ["Romance", "Drama"],
            ],
        }
    )
    # Users 1-3 love sci-fi; users 4-6 love romance.
    rows = []
    for u in (1, 2, 3):
        for m in (1, 2, 3):
            rows.append((u, m, 5.0))
        rows.append((u, 4, 1.0))
    for u in (4, 5, 6):
        for m in (4, 5, 6):
            rows.append((u, m, 5.0))
        rows.append((u, 1, 1.0))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    ratings["timestamp"] = 0

    movie_index = {int(m): i for i, m in enumerate(movies["movieId"])}
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).rename_axis("movieId")
    return MovieData(
        movies=movies,
        ratings=ratings,
        genres=sorted({g for row in movies["genres"] for g in row}),
        movie_index=movie_index,
        rating_stats=stats,
    )


@pytest.fixture(scope="module")
def engine():
    return RecommenderEngine(data=_toy_data(), n_factors=4)


def test_content_profile_recommends_same_genre(engine):
    # A user who loves a sci-fi film should get the other sci-fi films first.
    recs = engine.recommend({1: 5.0}, top_n=3, alpha=0.0, min_ratings=0)
    rec_ids = [r["movieId"] for r in recs]
    assert 1 not in rec_ids  # already rated, excluded
    # Top picks should be the remaining sci-fi titles (2 and 3), not romance.
    assert set(rec_ids[:2]) == {2, 3}


def test_collaborative_signal(engine):
    # Pure collaborative: liking sci-fi movie 1 should surface 2 and 3.
    recs = engine.recommend({1: 5.0, 2: 5.0}, top_n=3, alpha=1.0, min_ratings=0)
    assert recs[0]["movieId"] == 3


def test_genre_filter(engine):
    recs = engine.recommend({1: 5.0}, top_n=5, alpha=0.5, min_ratings=0, genres=["Romance"])
    assert recs, "expected some romance results"
    assert all("Romance" in r["genres"] for r in recs)


def test_excludes_already_rated(engine):
    rated = {1: 5.0, 2: 4.0, 3: 3.0}
    recs = engine.recommend(rated, top_n=10, alpha=0.5, min_ratings=0)
    assert not (set(rated) & {r["movieId"] for r in recs})


def test_unknown_movie_falls_back_to_popularity(engine):
    recs = engine.recommend({99999: 5.0}, top_n=3, min_ratings=0)
    assert len(recs) == 3  # falls back, still returns picks


def test_scores_present_and_ranked(engine):
    recs = engine.recommend({1: 5.0, 2: 5.0}, top_n=4, alpha=0.5, min_ratings=0)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_minmax_constant_array():
    from movieflix.hybrid import minmax

    out = minmax(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(out, 0.0)
