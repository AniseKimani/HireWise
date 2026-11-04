// Shared site behaviour (mobile responsiveness, Milestone 1). Progressive
// enhancement only: without this script the full navigation stays visible
// at every width. The phone menu collapses only after initialisation below
// succeeds and adds the "nav-enhanced" class to <html>. No inline code, no
// network calls, nothing outside the site header is touched.
(function () {
  "use strict";

  var PHONE_QUERY = "(max-width: 767px)"; // keep in step with the phone breakpoint in style.css
  var ENHANCED_CLASS = "nav-enhanced";
  var OPEN_CLASS = "is-open";

  function closeDetails(details, restoreFocus) {
    if (!details || !details.open) return;
    details.open = false;
    if (restoreFocus) {
      var summary = details.querySelector("summary");
      if (summary) summary.focus();
    }
  }

  function init() {
    var toggle = document.querySelector(".nav-toggle");
    var menu = toggle ? document.getElementById(toggle.getAttribute("aria-controls")) : null;
    var accountMenus = Array.prototype.slice.call(document.querySelectorAll(".account-menu"));
    var phone = window.matchMedia ? window.matchMedia(PHONE_QUERY) : null;

    function isOpen() {
      return menu.classList.contains(OPEN_CLASS);
    }

    function setOpen(open, restoreFocus) {
      menu.classList.toggle(OPEN_CLASS, open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (!open) {
        accountMenus.forEach(function (details) { closeDetails(details, false); });
        if (restoreFocus) toggle.focus();
      }
    }

    // ---- Account menu (<details>): Escape and outside clicks close it, at every width.
    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape" && event.key !== "Esc") return;
      var handled = false;
      accountMenus.forEach(function (details) {
        if (details.open && details.contains(document.activeElement)) {
          closeDetails(details, true);
          handled = true;
        }
      });
      if (handled) return;
      if (menu && toggle && isOpen() && phone && phone.matches) {
        var focusInside = menu.contains(document.activeElement) || document.activeElement === toggle;
        setOpen(false, focusInside || document.activeElement === document.body);
      }
    });

    document.addEventListener("click", function (event) {
      accountMenus.forEach(function (details) {
        if (details.open && !details.contains(event.target)) closeDetails(details, false);
      });
      if (menu && toggle && isOpen() && !menu.contains(event.target) && !toggle.contains(event.target)) {
        setOpen(false, false);
      }
    });

    // ---- Phone menu toggle. Without both elements, the navigation simply stays expanded.
    if (!toggle || !menu) return;

    toggle.addEventListener("click", function () {
      setOpen(!isOpen(), false);
    });

    // Choosing a navigation link closes the menu (also covers same-page #anchor links).
    menu.addEventListener("click", function (event) {
      var link = event.target.closest ? event.target.closest("a[href]") : null;
      if (link && menu.contains(link)) setOpen(false, false);
    });

    // Leaving the phone layout resets the menu so it never reappears open unexpectedly.
    if (phone) {
      var onChange = function () { if (!phone.matches) setOpen(false, false); };
      if (phone.addEventListener) { phone.addEventListener("change", onChange); }
      else if (phone.addListener) { phone.addListener(onChange); }
    }

    setOpen(false, false);
    // Last step: only now may the stylesheet collapse the navigation.
    document.documentElement.classList.add(ENHANCED_CLASS);
  }

  function safeInit() {
    try {
      init();
    } catch (error) {
      // Leave the navigation fully expanded if anything goes wrong.
      document.documentElement.classList.remove(ENHANCED_CLASS);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", safeInit);
  } else {
    safeInit();
  }
})();
