"""High-level recommendation engine tying the components together."""

from __future__ import annotations

import numpy as np

from . import hybrid
from .collaborative import CollaborativeRecommender
from .content_based import ContentRecommender
from .data import MovieData, load_movie_data


class RecommenderEngine:
    """Loads the dataset and serves hybrid recommendations.

    Construction trains the collaborative model (a one-off SVD), so build the
    engine once at application start-up and reuse it across requests.
    """

    def __init__(self, data: MovieData | None = None, n_factors: int = 50):
        self.data = data if data is not None else load_movie_data()
        self.content = ContentRecommender(self.data)
        self.collab = CollaborativeRecommender(self.data, n_factors=n_factors)

    # -- catalogue helpers -------------------------------------------------

    def popular_movies(self, n: int = 40, min_ratings: int = 50) -> list[dict]:
        """Most-rated movies, for seeding the "rate these" UI."""
        stats = self.data.rating_stats
        eligible = stats[stats["count"] >= min_ratings]
        top_ids = eligible.sort_values("count", ascending=False).head(n).index
        return [self.data.record(int(mid)) for mid in top_ids]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Case-insensitive substring search over titles."""
        query = query.strip().lower()
        if not query:
            return []
        movies = self.data.movies
        mask = movies["title"].str.lower().str.contains(query, regex=False)
        hits = movies[mask].head(limit)
        return [self.data.record(int(mid)) for mid in hits["movieId"]]

    # -- recommendations ---------------------------------------------------

    def recommend(
        self,
        ratings: dict[int, float],
        top_n: int = 12,
        alpha: float = 0.5,
        min_ratings: int = 20,
        genres: list[str] | None = None,
    ) -> list[dict]:
        """Return the top-N hybrid recommendations for a user's ratings.

        Parameters
        ----------
        ratings:
            ``{movieId: rating}`` collected from the user (1.0-5.0).
        top_n:
            Number of movies to return.
        alpha:
            Collaborative weight in the hybrid blend (0 = pure content,
            1 = pure collaborative).
        min_ratings:
            Only recommend movies with at least this many ratings, to avoid
            surfacing obscure titles with unstable scores.
        genres:
            Optional genre filter; a movie qualifies if it has any of them.
        """
        ratings = {int(k): float(v) for k, v in ratings.items()}

        content_profile = self.content.build_profile(ratings)
        if content_profile is None:
            # No known movies rated -> fall back to popularity.
            return self.popular_movies(n=top_n, min_ratings=min_ratings)

        content_scores = self.content.score(content_profile)
        collab_profile = self.collab.build_profile(ratings)
        collab_scores = self.collab.score(collab_profile) if collab_profile is not None else None

        blended = hybrid.combine(content_scores, collab_scores, alpha)

        # Build candidate filter mask.
        movie_ids = self.data.movie_ids
        rated_set = set(ratings)
        popularity = self.collab.popularity
        wanted_genres = set(genres) if genres else None

        candidates = []
        content_norm = hybrid.minmax(content_scores)
        collab_norm = hybrid.minmax(collab_scores) if collab_scores is not None else None
        for pos, mid in enumerate(movie_ids):
            mid = int(mid)
            if mid in rated_set:
                continue
            if popularity[pos] < min_ratings:
                continue
            record = self.data.record(mid)
            if wanted_genres and not (wanted_genres & set(record["genres"])):
                continue
            candidates.append((pos, mid, record))

        candidates.sort(key=lambda c: blended[c[0]], reverse=True)

        results = []
        for pos, mid, record in candidates[:top_n]:
            record["score"] = round(float(blended[pos]), 4)
            record["content_score"] = round(float(content_norm[pos]), 4)
            record["collab_score"] = (
                round(float(collab_norm[pos]), 4) if collab_norm is not None else None
            )
            record["why"] = self._explain(content_profile, mid, record, collab_norm, pos)
            results.append(record)
        return results

    def _explain(self, content_profile, movie_id, record, collab_norm, pos) -> str:
        matched = self.content.top_matching_genres(content_profile, movie_id)
        parts = []
        if matched:
            parts.append("matches your taste for " + " & ".join(matched))
        if collab_norm is not None and collab_norm[pos] > 0.6:
            parts.append("loved by users with similar ratings")
        if not parts:
            parts.append("a well-rated pick for you")
        return "; ".join(parts).capitalize()
