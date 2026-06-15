"""MovieFlix — a personalized hybrid movie recommendation engine.

The package exposes a single high-level entry point, :class:`RecommenderEngine`,
which loads the MovieLens dataset and serves hybrid (content-based +
collaborative-filtering) recommendations for an on-the-fly user profile.
"""

from .engine import RecommenderEngine

__all__ = ["RecommenderEngine"]
__version__ = "1.0.0"
