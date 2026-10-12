// HireWise API client (Day 10). The ONLY module that calls fetch().
// Every call returns parsed JSON or throws ApiError with a stable `code`.

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

function buildUrl(path, params) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value)); // URLSearchParams encodes request links safely
    }
  }
  return url;
}

async function getJson(path, params, { acceptErrorBody = false } = {}) {
  let response;
  try {
    response = await fetch(buildUrl(path, params), { headers: { Accept: "application/json" } });
  } catch (networkError) {
    throw new ApiError(0, "network_error", "The HireWise server could not be reached.");
  }
  let body;
  try {
    body = await response.json();
  } catch (parseError) {
    throw new ApiError(response.status, "invalid_response", "The server returned an unreadable response.");
  }
  if (!response.ok && !acceptErrorBody) {
    const error = body && body.error ? body.error : {};
    throw new ApiError(response.status, error.code || "http_error", error.message || "Request failed.");
  }
  return body;
}

export const api = {
  // /api/health answers 503 with a normal status body when the recommender is down.
  health: () => getJson("/api/health", null, { acceptErrorBody: true }),
  listRequests: ({ page, perPage, q, category, skill }) =>
    getJson("/api/requests", { page, per_page: perPage, q, category, skill }),
  getRequest: (link) => getJson("/api/requests", { link }),
  recommendations: (link, k) => getJson("/api/recommendations", { link, k }),
  worker: (workerId) => getJson(`/api/workers/${encodeURIComponent(workerId)}`),
  workerGraph: (workerId) => getJson(`/api/workers/${encodeURIComponent(workerId)}/graph`),
};
