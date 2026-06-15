"""Dataset loading for MovieLens (ml-latest-small).

Loads the ``movies.csv`` and ``ratings.csv`` files into pandas frames and
derives a few convenient columns (release year, genre lists, popularity). If
the dataset is missing it is downloaded automatically on first use.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
import pandas as pd

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Repository-relative default location for the extracted dataset.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "ml-latest-small")

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def ensure_dataset(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Return *data_dir*, downloading and extracting MovieLens if needed."""
    movies = os.path.join(data_dir, "movies.csv")
    ratings = os.path.join(data_dir, "ratings.csv")
    if os.path.exists(movies) and os.path.exists(ratings):
        return data_dir

    target_root = os.path.dirname(data_dir)
    os.makedirs(target_root, exist_ok=True)
    with urlopen(ML_URL) as resp:  # noqa: S310 - trusted GroupLens URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(target_root)
    if not os.path.exists(movies):
        raise FileNotFoundError(
            f"MovieLens download did not produce expected files in {data_dir}"
        )
    return data_dir


def _split_title_year(raw_title: str) -> tuple[str, int | None]:
    match = _YEAR_RE.search(raw_title)
    if not match:
        return raw_title.strip(), None
    year = int(match.group(1))
    title = _YEAR_RE.sub("", raw_title).strip()
    return title, year


@dataclass
class MovieData:
    """Container for the loaded and pre-processed MovieLens dataset.

    Attributes
    ----------
    movies:
        One row per movie with columns ``movieId, title, year, genres``
        (``genres`` is a list of strings; ``(no genres listed)`` becomes ``[]``).
    ratings:
        Raw ratings with columns ``userId, movieId, rating, timestamp``.
    genres:
        Sorted list of every distinct genre in the catalogue.
    movie_index:
        Maps ``movieId`` -> row position (0..n_movies-1) used by the models.
    rating_stats:
        Per-movie ``count`` and ``mean`` rating, indexed by ``movieId``.
    """

    movies: pd.DataFrame
    ratings: pd.DataFrame
    genres: list[str]
    movie_index: dict[int, int]
    rating_stats: pd.DataFrame

    @property
    def movie_ids(self) -> np.ndarray:
        return self.movies["movieId"].to_numpy()

    def title(self, movie_id: int) -> str:
        return self._by_id.loc[movie_id, "title"]

    def record(self, movie_id: int) -> dict:
        row = self._by_id.loc[movie_id]
        return {
            "movieId": int(movie_id),
            "title": row["title"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": list(row["genres"]),
            "rating_count": int(self.rating_stats.loc[movie_id, "count"])
            if movie_id in self.rating_stats.index
            else 0,
            "avg_rating": round(float(self.rating_stats.loc[movie_id, "mean"]), 2)
            if movie_id in self.rating_stats.index
            else None,
        }

    def __post_init__(self) -> None:
        # Indexed view for O(1) per-id lookups.
        self._by_id = self.movies.set_index("movieId")


def load_movie_data(data_dir: str = DEFAULT_DATA_DIR) -> MovieData:
    """Load and pre-process the MovieLens dataset into a :class:`MovieData`."""
    data_dir = ensure_dataset(data_dir)
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))

    titles_years = movies["title"].map(_split_title_year)
    movies["title"] = [t for t, _ in titles_years]
    movies["year"] = [y for _, y in titles_years]
    movies["genres"] = movies["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    all_genres = sorted({g for row in movies["genres"] for g in row})
    movie_index = {int(mid): i for i, mid in enumerate(movies["movieId"])}

    stats = (
        ratings.groupby("movieId")["rating"].agg(["count", "mean"]).rename_axis("movieId")
    )

    return MovieData(
        movies=movies.reset_index(drop=True),
        ratings=ratings,
        genres=all_genres,
        movie_index=movie_index,
        rating_stats=stats,
    )
