// MovieFlix front-end: collect ratings, call the API, render recommendations.

const state = {
  ratings: {}, // movieId -> {rating, title}
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// ---- Ratings model -------------------------------------------------------

function setRating(id, title, rating) {
  state.ratings[id] = { rating, title };
  renderRatedStrip();
  syncStarWidgets();
}

function removeRating(id) {
  delete state.ratings[id];
  renderRatedStrip();
  syncStarWidgets();
}

function renderRatedStrip() {
  const ids = Object.keys(state.ratings);
  const strip = $("#rated-strip");
  $("#rated-count").textContent = ids.length;
  strip.classList.toggle("hidden", ids.length === 0);
  $("#rated-list").innerHTML = ids
    .map((id) => {
      const { title, rating } = state.ratings[id];
      return `<span class="chip">${escapeHtml(title)} <b>${"★".repeat(
        Math.round(rating)
      )}</b> <span class="x" data-remove="${id}">✕</span></span>`;
    })
    .join("");
  $("#recommend-btn").disabled = ids.length === 0;
}

// Reflect current ratings on any star widget present in the DOM.
function syncStarWidgets() {
  $$(".stars").forEach((widget) => {
    const id = widget.dataset.id;
    const current = state.ratings[id]?.rating || 0;
    $$(".star", widget).forEach((star) => {
      star.classList.toggle("active", Number(star.dataset.val) <= current);
    });
  });
}

// ---- Star widget wiring (delegated so it works for dynamic cards too) ----

function wireStars(scope = document) {
  $$(".stars", scope).forEach((widget) => {
    if (widget.dataset.wired) return;
    widget.dataset.wired = "1";
    const id = widget.dataset.id;
    const card = widget.closest(".card") || widget.closest("[data-title]");
    const title = card?.dataset.title || `Movie ${id}`;
    $$(".star", widget).forEach((star) => {
      const val = Number(star.dataset.val);
      star.addEventListener("mouseenter", () =>
        $$(".star", widget).forEach((s) =>
          s.classList.toggle("hot", Number(s.dataset.val) <= val)
        )
      );
      star.addEventListener("mouseleave", () =>
        $$(".star", widget).forEach((s) => s.classList.remove("hot"))
      );
      star.addEventListener("click", () => setRating(id, title, val));
    });
  });
}

// ---- Search --------------------------------------------------------------

let searchTimer = null;
function onSearchInput(e) {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  const box = $("#search-results");
  if (q.length < 2) {
    box.classList.add("hidden");
    return;
  }
  searchTimer = setTimeout(async () => {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const hits = await res.json();
    if (!hits.length) {
      box.innerHTML = `<div class="row"><span class="muted">No matches</span></div>`;
    } else {
      box.innerHTML = hits
        .map((m) => {
          const label = `${m.title}${m.year ? ` (${m.year})` : ""}`;
          return `<div class="row" data-id="${m.movieId}" data-title="${escapeHtml(
            label
          )}"><span>${escapeHtml(label)}</span><small>⭐ ${
            m.avg_rating ?? "–"
          }</small></div>`;
        })
        .join("");
    }
    box.classList.remove("hidden");
  }, 180);
}

function onSearchPick(e) {
  const row = e.target.closest(".row[data-id]");
  if (!row) return;
  // Default a search pick to 5 stars; user can re-click chip to remove.
  setRating(row.dataset.id, row.dataset.title, 5);
  $("#search").value = "";
  $("#search-results").classList.add("hidden");
}

// ---- Recommendations -----------------------------------------------------

async function getRecommendations() {
  const btn = $("#recommend-btn");
  btn.disabled = true;
  btn.textContent = "Thinking…";

  const ratings = {};
  for (const [id, v] of Object.entries(state.ratings)) ratings[id] = v.rating;

  const body = {
    ratings,
    alpha: Number($("#alpha").value) / 100,
    genres: $("#genre").value ? [$("#genre").value] : null,
    top_n: 12,
  };

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResults(data.recommendations);
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Get my recommendations →";
  }
}

function renderResults(recs) {
  const section = $("#results-section");
  const grid = $("#results");
  if (!recs.length) {
    grid.innerHTML = `<p class="muted">No matches with these filters — try widening the genre.</p>`;
  } else {
    grid.innerHTML = recs
      .map((m, i) => {
        const genres = m.genres.join(" · ");
        const collab =
          m.collab_score != null
            ? `<span class="scoredot"></span>${Math.round(m.collab_score * 100)}% peer match`
            : "";
        return `
        <div class="card">
          <span class="rank">#${i + 1}</span>
          <div class="poster" style="--h:${m.movieId % 360}"><span>${escapeHtml(
          m.title.slice(0, 2).toUpperCase()
        )}</span></div>
          <div class="card-body">
            <div class="card-title">${escapeHtml(m.title)}</div>
            <div class="card-meta">${m.year ?? ""} · ${escapeHtml(genres)}</div>
            <div class="why">${escapeHtml(m.why)}</div>
            <div class="scorebar">${collab}</div>
          </div>
        </div>`;
      })
      .join("");
  }
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- Utilities + init ----------------------------------------------------

function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

document.addEventListener("DOMContentLoaded", () => {
  wireStars();
  renderRatedStrip();
  $("#search").addEventListener("input", onSearchInput);
  $("#search-results").addEventListener("click", onSearchPick);
  $("#recommend-btn").addEventListener("click", getRecommendations);
  document.addEventListener("click", (e) => {
    const rm = e.target.closest("[data-remove]");
    if (rm) removeRating(rm.dataset.remove);
    if (!e.target.closest(".search-wrap")) $("#search-results").classList.add("hidden");
  });
});
