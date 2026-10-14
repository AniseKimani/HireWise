// Registration steps (Day 12 redesign). Progressive enhancement only: the
// server validates everything and rejects the other role's fields, so this
// script is about presentation -- one step at a time, and only the selected
// role's fields are shown, required and submitted.
(function () {
  "use strict";
  var form = document.getElementById("register-form");
  if (!form) return;
  document.documentElement.classList.add("has-js");

  var STEPS = ["role", "account", "details"];
  var sections = {};
  STEPS.forEach(function (name) { sections[name] = form.querySelector('[data-step="' + name + '"]'); });
  var stepper = form.querySelector(".stepper");

  function selectedRole() {
    var checked = form.querySelector('input[name="role"]:checked');
    return checked ? checked.value : null;
  }

  // Show only the selected role's fields; disable (so they are neither
  // validated nor submitted) and clear the other role's fields.
  function applyRole(role, clearOther) {
    form.querySelectorAll("[data-role]").forEach(function (el) {
      var active = el.getAttribute("data-role") === role;
      el.hidden = !active;
      if (el.tagName === "FIELDSET") {
        el.disabled = !active;
        el.querySelectorAll("input, select, textarea").forEach(function (input) {
          if (!active && clearOther) {
            if (input.tagName === "SELECT") { input.selectedIndex = 0; } else { input.value = ""; }
          }
          if (input.hasAttribute("data-required")) { input.required = active; }
        });
      }
    });
    var label = role === "worker" ? "worker" : role === "client" ? "client" : "HireWise";
    form.querySelectorAll("[data-role-label]").forEach(function (el) { el.textContent = label; });
    form.querySelectorAll("[data-role-text]").forEach(function (el) {
      el.textContent = role === "worker" ? "Worker" : role === "client" ? "Client" : "Account";
    });
  }

  function showStep(name, moveFocus) {
    STEPS.forEach(function (step) { sections[step].hidden = step !== name; });
    if (stepper) {
      stepper.hidden = name === "role";
      var index = { account: 0, details: 1 }[name];
      stepper.querySelectorAll("li").forEach(function (li, i) {
        li.classList.toggle("is-done", i < index);
        if (i === index) { li.setAttribute("aria-current", "step"); } else { li.removeAttribute("aria-current"); }
      });
    }
    if (moveFocus) {
      var heading = sections[name].querySelector("h1, h2");
      if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus(); }
    }
  }

  function stepIsValid(name) {
    var fields = sections[name].querySelectorAll("input, select, textarea");
    for (var i = 0; i < fields.length; i++) {
      if (!fields[i].disabled && !fields[i].checkValidity()) { fields[i].reportValidity(); return false; }
    }
    if (name === "account") {
      var first = form.querySelector("[data-secret-entry]"), confirm = form.querySelector("[data-secret-confirm]");
      confirm.setCustomValidity(first.value === confirm.value ? "" : "The two entries do not match.");
      if (!confirm.checkValidity()) { confirm.reportValidity(); return false; }
    }
    return true;
  }

  form.addEventListener("change", function (event) {
    if (event.target.name === "role") {
      applyRole(event.target.value, true);
      form.querySelector("#role-missing").hidden = true;
    }
  });

  form.querySelectorAll("[data-next]").forEach(function (button) {
    button.addEventListener("click", function () {
      var current = button.closest("[data-step]").getAttribute("data-step");
      if (current === "role" && !selectedRole()) { form.querySelector("#role-missing").hidden = false; return; }
      if (current !== "role" && !stepIsValid(current)) { return; }
      showStep(button.getAttribute("data-next"), true);
    });
  });
  form.querySelectorAll("[data-back]").forEach(function (button) {
    button.addEventListener("click", function () { showStep(button.getAttribute("data-back"), true); });
  });

  // Initial state: keep server-side values; open the first step with an error.
  applyRole(selectedRole(), false);
  var firstError = form.querySelector(".field-error:not(.js-only)");
  var errorStep = firstError ? firstError.closest("[data-step]") : null;
  if (!selectedRole()) { showStep("role", false); }
  else if (errorStep) { showStep(errorStep.getAttribute("data-step"), false); }
  else { showStep("role", false); }
})();
