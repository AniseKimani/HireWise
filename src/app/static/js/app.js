// HireWise frontend (Day 10): request browsing -> recommendations ->
// explanation -> worker details -> optional graph view. All data comes
// from the /api routes via api.js; nothing here scores or explains.

import { api } from "./api.js";
import { clear, definitionList, el, setStatus } from "./dom.js";
import {
  COMPONENT_LABELS, COMPONENT_ORDER, PATH_TYPE_LABELS, formatContribution, formatExperience, formatPercent,
  formatPercentile, formatReviews, formatScore, friendlyError, shareOfScore,
} from "./format.js";

const PER_PAGE = 10;
const MAX_SKILLS_SHOWN = 8;

const state = { page: 1, totalPages: 0, filters: {}, selected: null, graphAvailable: null, workerId: null };

const $ = (id) => document.getElementById(id);
const ui = {
  health: $("health-status"),
  filters: $("request-filters"),
  q: $("filter-q"),
  category: $("filter-category"),
  skill: $("filter-skill"),
  reset: $("filter-reset"),
  listStatus: $("request-list-status"),
  list: $("request-list"),
  prev: $("page-prev"),
  next: $("page-next"),
  pageInfo: $("page-info"),
  selected: $("selected-request"),
  recommendForm: $("recommend-form"),
  kSelect: $("k-select"),
  recommendButton: $("recommend-button"),
  resultsStatus: $("results-status"),
  results: $("recommendation-list"),
  dialog: $("worker-dialog"),
  dialogTitle: $("worker-dialog-title"),
  dialogClose: $("worker-dialog-close"),
  workerStatus: $("worker-status"),
  workerProfile: $("worker-profile"),
  graphLoad: $("graph-load"),
  graphStatus: $("graph-status"),
  graph: $("worker-graph"),
};

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
async function loadHealth() {
  try {
    const health = await api.health();
    state.graphAvailable = health.neo4j && health.neo4j.available !== false;
    if (!health.recommender || !health.recommender.ready) {
      ui.health.textContent = "Recommendations are currently unavailable.";
      ui.health.className = "health health--down";
    } else if (health.postgresql && (health.postgresql.available === false || health.postgresql.data_loaded === false)) {
      ui.health.textContent = "Application data is temporarily unavailable.";
      ui.health.className = "health health--down";
    } else if (health.status === "degraded") {
      ui.health.textContent = "Recommender ready. Graph details temporarily unavailable.";
      ui.health.className = "health health--degraded";
    } else {
      ui.health.textContent = "Recommender ready. Knowledge graph connected.";
      ui.health.className = "health health--ok";
    }
  } catch (error) {
    ui.health.textContent = "System status unknown: the server could not be reached.";
    ui.health.className = "health health--down";
  }
}

// ---------------------------------------------------------------------------
// Request browsing
// ---------------------------------------------------------------------------
async function loadRequests() {
  setStatus(ui.listStatus, "Loading requests…");
  ui.list.setAttribute("aria-busy", "true");
  try {
    const result = await api.listRequests({ page: state.page, perPage: PER_PAGE, ...state.filters });
    populateCategories(result.available_categories);
    renderRequestList(result);
  } catch (error) {
    clear(ui.list);
    setStatus(ui.listStatus, friendlyError(error), "error");
    state.totalPages = 0;
    updatePagination();
  } finally {
    ui.list.removeAttribute("aria-busy");
  }
}

function populateCategories(categories) {
  if (ui.category.options.length > 1 || !Array.isArray(categories)) return;
  for (const category of categories) ui.category.append(el("option", { text: category, attrs: { value: category } }));
}

function renderRequestList(result) {
  clear(ui.list);
  state.totalPages = result.total_pages;
  if (result.total === 0) {
    setStatus(ui.listStatus, "No requests match these filters.");
  } else {
    setStatus(ui.listStatus, `${result.total} request${result.total === 1 ? "" : "s"} found.`);
  }
  for (const item of result.items) {
    const button = el("button", {
      className: "request-item",
      attrs: { type: "button", "aria-pressed": state.selected && state.selected.link === item.link ? "true" : "false" },
    }, [
      el("span", { className: "request-item__title", text: item.title || "(untitled request)" }),
      el("span", { className: "request-item__meta", text: `${item.category} · ${item.required_skills.slice(0, 3).join(", ")}` }),
    ]);
    button.dataset.link = item.link; // exact request id, kept as data (never parsed as HTML)
    button.addEventListener("click", () => selectRequest(item));
    ui.list.append(el("li", {}, [button]));
  }
  updatePagination();
}

function updatePagination() {
  ui.prev.disabled = state.page <= 1;
  ui.next.disabled = state.page >= state.totalPages;
  ui.pageInfo.textContent = state.totalPages ? `Page ${state.page} of ${state.totalPages}` : "";
}

function selectRequest(item) {
  state.selected = item;
  for (const button of ui.list.querySelectorAll(".request-item")) {
    button.setAttribute("aria-pressed", button.dataset.link === item.link ? "true" : "false");
  }

  clear(ui.selected);
  ui.selected.append(
    el("h3", { className: "selected-request__title", text: item.title || "(untitled request)" }),
    definitionList([
      ["Category", item.category],
      ["Required skills", item.required_skills.join(", ") || "none listed"],
      ["Pay type", item.pay_type || "not stated"],
      ["Budget level", formatPercentile(item.price_tier)],
      ["Location requirement", item.location_requirement || "none"],
      ["Eligible workers", String(item.n_eligible_workers)],
    ]),
  );
  ui.recommendButton.disabled = false;
  clear(ui.results);
  setStatus(ui.resultsStatus, "Ready. Choose how many workers to show, then select “Get recommendations”.");
}

// ---------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------
async function loadRecommendations() {
  if (!state.selected) return;
  const k = Number(ui.kSelect.value);
  ui.recommendButton.disabled = true;
  clear(ui.results);
  setStatus(ui.resultsStatus, "Computing recommendations…");
  ui.results.setAttribute("aria-busy", "true");
  try {
    const result = await api.recommendations(state.selected.link, k);
    if (result.n_returned === 0) {
      setStatus(ui.resultsStatus, "No eligible workers were found for this request.");
    } else {
      setStatus(ui.resultsStatus,
        `Top ${result.n_returned} of ${result.n_candidates} eligible workers (model: ${result.model.name}).`);
    }
    for (const recommendation of result.recommendations) ui.results.append(renderRecommendation(recommendation));
  } catch (error) {
    setStatus(ui.resultsStatus, friendlyError(error), "error");
  } finally {
    ui.results.removeAttribute("aria-busy");
    ui.recommendButton.disabled = false;
  }
}

function renderRecommendation(rec) {
  const worker = rec.worker;
  const skills = worker.declared_skills;
  const shownSkills = skills.slice(0, MAX_SKILLS_SHOWN);
  const detailsButton = el("button", {
    className: "button button--link", text: `View worker ${worker.worker_id}`, attrs: { type: "button" },
  });
  detailsButton.addEventListener("click", () => openWorker(worker.worker_id));

  return el("li", { className: "recommendation" }, [
    el("div", { className: "recommendation__header" }, [
      el("span", { className: "recommendation__rank", text: `#${rec.rank}`, attrs: { "aria-label": `Rank ${rec.rank}` } }),
      el("h3", { className: "recommendation__worker", text: worker.worker_id }),
      el("span", { className: "recommendation__score", text: `Hybrid score ${formatScore(rec.score)}` }),
    ]),
    el("p", {
      className: "recommendation__meta",
      text: [worker.primary_category, worker.country, formatExperience(worker.experience_level, worker.experience_years),
        formatReviews(worker.training_review_count, worker.training_mean_rating)].filter(Boolean).join(" · "),
    }),
    el("ul", { className: "chips", attrs: { "aria-label": "Declared skills" } }, [
      ...shownSkills.map((s) => el("li", { className: "chip", text: s })),
      skills.length > shownSkills.length ? el("li", { className: "chip chip--more", text: `+${skills.length - shownSkills.length} more` }) : null,
    ]),
    renderContributions(rec),
    el("p", { className: "recommendation__explanation", text: rec.explanation }),
    renderEvidence(rec),
    detailsButton,
  ]);
}

function renderContributions(rec) {
  const bar = el("div", { className: "contribution-bar", attrs: { role: "img" } });
  const legend = el("ul", { className: "contribution-legend" });
  const spoken = [];
  for (const component of COMPONENT_ORDER) {
    const value = rec.contributions[component];
    const segment = el("span", { className: `contribution-bar__segment contribution--${component}` });
    segment.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`; // CSSOM, permitted by the CSP
    bar.append(segment);
    const share = formatPercent(shareOfScore(value, rec.score));
    legend.append(el("li", {}, [
      el("span", { className: `swatch contribution--${component}`, attrs: { "aria-hidden": "true" } }),
      `${COMPONENT_LABELS[component]}: ${formatContribution(value)} (${share} of score)`,
    ]));
    spoken.push(`${COMPONENT_LABELS[component]} ${formatContribution(value)}`);
  }
  bar.setAttribute("aria-label", `Score contributions: ${spoken.join(", ")}. Total ${formatScore(rec.score)}.`);
  const dominant = rec.dominant_component ? COMPONENT_LABELS[rec.dominant_component] : "none";
  return el("div", { className: "contributions" }, [
    el("p", { className: "contributions__title", text: `Score contributions (largest: ${dominant})` }),
    bar,
    legend,
  ]);
}

function renderEvidence(rec) {
  const details = el("details", { className: "evidence" }, [el("summary", { text: "Why this recommendation?" })]);
  if (!rec.kg_summary && (!rec.kg_evidence || rec.kg_evidence.length === 0)) {
    details.append(el("p", { text: "No knowledge-graph evidence is available for this worker." }));
    return details;
  }
  if (rec.kg_summary) details.append(el("p", { className: "evidence__summary", text: rec.kg_summary }));
  if (rec.kg_evidence && rec.kg_evidence.length) {
    details.append(
      el("p", { className: "evidence__heading", text: "Strongest knowledge-graph paths" }),
      el("ul", { className: "evidence__list" }, rec.kg_evidence.map((evidence) =>
        el("li", {}, [el("strong", { text: `${PATH_TYPE_LABELS[evidence.path_type] || evidence.path_type}: ` }), evidence.text]))),
    );
  }
  return details;
}

// ---------------------------------------------------------------------------
// Worker details + optional graph view
// ---------------------------------------------------------------------------
async function openWorker(workerId) {
  state.workerId = workerId;
  ui.dialogTitle.textContent = `Worker ${workerId}`;
  clear(ui.workerProfile);
  clear(ui.graph);
  setStatus(ui.graphStatus, state.graphAvailable === false
    ? "Graph details are temporarily unavailable (the knowledge-graph database is not reachable)." : "");
  ui.graphLoad.disabled = false;
  setStatus(ui.workerStatus, "Loading worker profile…");
  if (!ui.dialog.open) ui.dialog.showModal();
  try {
    const worker = await api.worker(workerId);
    setStatus(ui.workerStatus, "");
    ui.workerProfile.append(definitionList([
      ["Primary category", worker.primary_category],
      ["Secondary categories", worker.secondary_categories.join(", ") || "none"],
      ["Location", worker.country || "not stated"],
      ["Experience", formatExperience(worker.experience_level, worker.experience_years)],
      ["Rate (synthetic profile value)", worker.rate === null ? "not stated" : worker.rate.toFixed(2)],
      ["Price level", formatPercentile(worker.price_tier)],
      ["Reviews", formatReviews(worker.training_review_count, worker.training_mean_rating)],
      ["Joined (simulated day)", worker.join_day === null ? "not stated" : worker.join_day.toFixed(0)],
      ["Declared skills", worker.declared_skills.join(", ") || "none"],
    ]));
  } catch (error) {
    setStatus(ui.workerStatus, friendlyError(error), "error");
  }
}

async function loadGraph() {
  const workerId = state.workerId;
  if (!workerId) return;
  ui.graphLoad.disabled = true;
  clear(ui.graph);
  setStatus(ui.graphStatus, "Loading graph details…");
  try {
    const graph = await api.workerGraph(workerId);
    if (workerId !== state.workerId) return;
    setStatus(ui.graphStatus, "");
    ui.graph.append(definitionList([
      ["Skills (HAS_SKILL)", graph.skills.join(", ") || "none"],
      ["Specialisations (SPECIALIZES_IN)", graph.categories.map((c) => `${c.category} (${c.role})`).join(", ") || "none"],
      ["Location (LOCATED_IN)", graph.location || "none"],
    ]));
  } catch (error) {
    setStatus(ui.graphStatus, friendlyError(error), error.code === "neo4j_unavailable" ? "warning" : "error");
  } finally {
    ui.graphLoad.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
ui.filters.addEventListener("submit", (event) => {
  event.preventDefault();
  state.filters = { q: ui.q.value.trim(), category: ui.category.value, skill: ui.skill.value.trim() };
  state.page = 1;
  loadRequests();
});
ui.reset.addEventListener("click", () => {
  ui.filters.reset();
  state.filters = {};
  state.page = 1;
  loadRequests();
});
ui.prev.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadRequests(); } });
ui.next.addEventListener("click", () => { if (state.page < state.totalPages) { state.page += 1; loadRequests(); } });
ui.recommendForm.addEventListener("submit", (event) => { event.preventDefault(); loadRecommendations(); });
ui.dialogClose.addEventListener("click", () => ui.dialog.close());
ui.graphLoad.addEventListener("click", loadGraph);

loadHealth();
loadRequests();
