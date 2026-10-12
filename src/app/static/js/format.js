// Pure presentation helpers (Day 10). No DOM access, no network, and no
// recommendation logic: these only format values the API already returned.

export const COMPONENT_ORDER = ["cbf", "cf", "kg"];

export const COMPONENT_LABELS = {
  cbf: "Profile/content",
  cf: "Collaborative",
  kg: "Knowledge graph",
};

// Display names for the KG path types the API returns (labels only; the
// evidence text itself always comes from the API).
export const PATH_TYPE_LABELS = {
  skill: "Required skill",
  related_skill: "Related skill",
  category: "Specialisation",
  repeat_hire: "Your hiring history",
  hired_similar_skill: "Similar to a past hire",
  co_hire: "Shared clients",
};

export function formatScore(value) {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

// Exact contribution values: tiny but non-zero values are never shown as 0.
export function formatContribution(value) {
  if (typeof value !== "number") return "n/a";
  if (value === 0) return "0";
  if (value < 0.001) return "<0.001";
  return value.toFixed(3);
}

export function shareOfScore(value, score) {
  return score > 0 ? value / score : 0;
}

export function formatPercent(fraction) {
  if (fraction === 0) return "0%";
  if (fraction < 0.001) return "<0.1%";
  return `${(fraction * 100).toFixed(1)}%`;
}

export function formatPercentile(priceTier) {
  if (typeof priceTier !== "number") return "not stated";
  return `Percentile ${Math.round(priceTier * 100)} of 100 within its category`;
}

export function formatExperience(level, years) {
  const parts = [];
  if (level) parts.push(level.charAt(0).toUpperCase() + level.slice(1));
  if (typeof years === "number") parts.push(`${years.toFixed(1)} years`);
  return parts.length ? parts.join(", ") : "not stated";
}

export function formatReviews(count, mean) {
  if (!count) return "No training-period reviews yet";
  return `${mean.toFixed(2)}/5 average from ${count} training-period review${count === 1 ? "" : "s"}`;
}

const FRIENDLY_ERRORS = {
  network_error: "The HireWise server could not be reached. Check that it is running and try again.",
  unknown_request: "That request could not be found. It may have been removed; please choose another.",
  request_not_recommendable: "Recommendations are only available for requests from the evaluation period.",
  recommender_unavailable: "The recommender is not available right now. Please try again later.",
  neo4j_unavailable: "Graph details are temporarily unavailable (the knowledge-graph database is not reachable). Recommendations still work.",
  unknown_worker: "That worker could not be found.",
  invalid_k: "Please choose a valid number of workers.",
  invalid_page: "That page does not exist.",
  invalid_per_page: "Invalid page size.",
  invalid_q: "The search text is too long.",
  invalid_category: "The category filter is too long.",
  invalid_skill: "The skill filter is too long.",
  invalid_response: "The server returned an unexpected response. Please try again.",
};

export function friendlyError(error) {
  const code = error && error.code;
  return FRIENDLY_ERRORS[code] || "Something went wrong on the server. Please try again.";
}
