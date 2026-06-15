"""Collaborative-filtering component (latent-factor / SVD model).

We build a user x movie ratings matrix, mean-center each movie's column, and
factor it with a truncated SVD. The right singular vectors give every movie a
low-dimensional "taste" vector.

A brand-new user (someone who just rated a handful of movies in the web UI) is
handled by *fold-in*: we project their centered ratings onto the movie factor
space to obtain a user vector, then score every movie by the dot product. This
captures patterns like "people who liked these also liked..." without retraining.
"""

from __future__ import annotations

import numpy as np

from .data import MovieData


class CollaborativeRecommender:
    def __init__(self, data: MovieData, n_factors: int = 50):
        self.data = data
        self.n_movies = len(data.movies)
        self.n_factors = n_factors
        self._fit()

    def _fit(self) -> None:
        ratings = self.data.ratings
        idx = self.data.movie_index

        user_ids = ratings["userId"].to_numpy()
        unique_users = np.unique(user_ids)
        user_pos = {int(u): i for i, u in enumerate(unique_users)}
        n_users = len(unique_users)

        # Dense user x movie matrix (small dataset: 610 x ~9700).
        matrix = np.zeros((n_users, self.n_movies), dtype=np.float64)
        rated_mask = np.zeros((n_users, self.n_movies), dtype=bool)
        for u, m, r in zip(user_ids, ratings["movieId"].to_numpy(), ratings["rating"].to_numpy()):
            row = user_pos[int(u)]
            col = idx[int(m)]
            matrix[row, col] = r
            rated_mask[row, col] = True

        # Per-movie mean over observed ratings only; unrated entries -> 0 after centering.
        counts = rated_mask.sum(axis=0)
        sums = matrix.sum(axis=0)
        safe_counts = np.where(counts == 0, 1, counts)
        self.movie_mean = np.where(counts == 0, self.data.ratings["rating"].mean(), sums / safe_counts)

        centered = np.where(rated_mask, matrix - self.movie_mean, 0.0)

        k = min(self.n_factors, n_users, self.n_movies)
        # Vt rows are orthonormal directions in movie space: shape (k, n_movies).
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.item_factors = vt[:k]  # (k, n_movies)
        self.popularity = counts  # number of ratings per movie (for cold-start/filtering)

    def build_profile(self, ratings: dict[int, float]) -> np.ndarray | None:
        """Fold a new user's ``{movieId: rating}`` into the latent space."""
        idx = self.data.movie_index
        centered = np.zeros(self.n_movies, dtype=np.float64)
        used = 0
        for movie_id, rating in ratings.items():
            pos = idx.get(int(movie_id))
            if pos is None:
                continue
            centered[pos] = float(rating) - self.movie_mean[pos]
            used += 1
        if used == 0:
            return None
        # Project onto factor space: user_vec (k,) = centered (n,) @ Vt^T (n,k).
        return self.item_factors @ centered

    def score(self, profile: np.ndarray) -> np.ndarray:
        """Predicted mean-centered affinity for every movie."""
        return profile @ self.item_factors  # (n_movies,)
