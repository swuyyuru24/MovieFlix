"""MovieFlix Flask web application.

Run with::

    python app.py

then open http://127.0.0.1:5000 in a browser. Rate a few movies, tune the
"personalization mix", and get hybrid recommendations.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from movieflix import RecommenderEngine

app = Flask(__name__)

# Build the engine once at start-up (trains the collaborative SVD).
print("Loading MovieFlix engine (downloading MovieLens on first run)...")
engine = RecommenderEngine()
print(f"Engine ready: {len(engine.data.movies)} movies, {len(engine.data.ratings)} ratings.")


@app.route("/")
def index():
    return render_template(
        "index.html",
        seed_movies=engine.popular_movies(n=36, min_ratings=50),
        genres=engine.data.genres,
    )


@app.get("/api/search")
def api_search():
    query = request.args.get("q", "")
    return jsonify(engine.search(query, limit=15))


@app.post("/api/recommend")
def api_recommend():
    payload = request.get_json(force=True) or {}
    raw = payload.get("ratings", {})
    try:
        ratings = {int(k): float(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        return jsonify({"error": "ratings must be a {movieId: rating} map"}), 400

    if not ratings:
        return jsonify({"error": "Rate at least one movie to get recommendations."}), 400

    alpha = float(payload.get("alpha", 0.5))
    top_n = int(payload.get("top_n", 12))
    genres = payload.get("genres") or None

    recs = engine.recommend(ratings, top_n=top_n, alpha=alpha, genres=genres)
    return jsonify({"recommendations": recs})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
