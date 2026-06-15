"""Content-based component.

Each movie is represented as a TF-IDF weighted vector over its genres. A user
profile is the rating-weighted average of the genres of the movies they have
rated, so a movie scores highly when its genres match the user's tastes.

Implemented with numpy only (no scikit-learn dependency).
"""

from __future__ import annotations

import numpy as np

from .data import MovieData


class ContentRecommender:
    def __init__(self, data: MovieData):
        self.data = data
        self.genres = data.genres
        self._genre_pos = {g: i for i, g in enumerate(self.genres)}
        self.movie_vectors = self._build_genre_matrix()

    def _build_genre_matrix(self) -> np.ndarray:
        """Return an L2-normalized, TF-IDF weighted (n_movies, n_genres) matrix."""
        n_movies = len(self.data.movies)
        n_genres = len(self.genres)
        multihot = np.zeros((n_movies, n_genres), dtype=np.float64)
        for row_i, genres in enumerate(self.data.movies["genres"]):
            for g in genres:
                multihot[row_i, self._genre_pos[g]] = 1.0

        # Inverse document frequency over genres (smoothed).
        doc_freq = np.count_nonzero(multihot, axis=0)
        idf = np.log((1 + n_movies) / (1 + doc_freq)) + 1.0
        weighted = multihot * idf

        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return weighted / norms

    def build_profile(self, ratings: dict[int, float]) -> np.ndarray | None:
        """Build a genre-space profile vector from ``{movieId: rating}``.

        Ratings are centered at 3.0 so that disliked movies (rating < 3) push
        the profile *away* from their genres while liked movies pull toward
        them. Returns ``None`` when none of the rated movies are known.
        """
        idx = self.data.movie_index
        profile = np.zeros(len(self.genres), dtype=np.float64)
        used = 0
        for movie_id, rating in ratings.items():
            pos = idx.get(int(movie_id))
            if pos is None:
                continue
            weight = float(rating) - 3.0
            profile += weight * self.movie_vectors[pos]
            used += 1
        if used == 0:
            return None
        norm = np.linalg.norm(profile)
        if norm == 0:
            return None
        return profile / norm

    def score(self, profile: np.ndarray) -> np.ndarray:
        """Cosine similarity of every movie to *profile* (movie vectors are unit)."""
        return self.movie_vectors @ profile

    def top_matching_genres(self, profile: np.ndarray, movie_id: int, k: int = 2):
        """Genres most responsible for a movie matching the profile (for 'why')."""
        pos = self.data.movie_index.get(int(movie_id))
        if pos is None:
            return []
        contrib = self.movie_vectors[pos] * profile
        order = np.argsort(contrib)[::-1]
        return [self.genres[i] for i in order if contrib[i] > 0][:k]
