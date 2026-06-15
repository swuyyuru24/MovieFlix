"""Score-combination helpers for the hybrid recommender."""

from __future__ import annotations

import numpy as np


def minmax(scores: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1]; returns zeros for a constant array."""
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def combine(
    content_scores: np.ndarray,
    collab_scores: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Blend normalized content and collaborative scores.

    ``alpha`` is the collaborative weight in ``[0, 1]``. When the collaborative
    profile is unavailable (e.g. the user rated only movies nobody else has
    rated) we fall back to pure content scores.
    """
    content_n = minmax(content_scores)
    if collab_scores is None:
        return content_n
    collab_n = minmax(collab_scores)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * collab_n + (1.0 - alpha) * content_n
