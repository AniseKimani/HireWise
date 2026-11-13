// Notification bell (pre-deployment Milestone 3). Progressive enhancement only: the server renders the unread
// count on every page; this module refreshes it every data-poll-seconds (configuration, never a constant here) while
// the tab is visible, and stops polling while it is hidden or once the session has ended (401). The request goes
// through the shared API client (api.js, the only module that calls fetch). No inline code; nothing outside the bell
// is touched.
import { api } from "./api.js";

const MIN_SECONDS = 5; // guard against a misconfigured value hammering the server

function init() {
  const bell = document.querySelector("[data-notification-bell]");
  if (!bell) return;
  const badge = bell.querySelector("[data-notification-count]");
  const url = bell.getAttribute("data-count-url");
  const seconds = parseInt(bell.getAttribute("data-poll-seconds"), 10);
  if (!url || !badge || !(seconds >= MIN_SECONDS)) return;

  let timer = null;
  let stopped = false;

  function render(count) {
    badge.textContent = count > 99 ? "99+" : count > 0 ? String(count) : "";
    badge.hidden = !(count > 0);
    bell.setAttribute("aria-label", count > 0 ? `Notifications, ${count} unread` : "Notifications");
  }

  function stop() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  async function refresh() {
    if (stopped || document.visibilityState === "hidden") return;
    try {
      const data = await api.unreadNotifications(url);
      if (data && typeof data.unread === "number") render(data.unread);
    } catch (error) {
      if (error && error.status === 401) {
        stopped = true; // signed out: stop for good
        stop();
      }
      // offline or server unavailable: keep the last count
    }
  }

  function start() {
    if (stopped || timer !== null) return;
    timer = window.setInterval(refresh, seconds * 1000);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      stop();
    } else {
      refresh();
      start();
    }
  });

  if (document.visibilityState !== "hidden") start();
}

init(); // modules run after the document has been parsed
