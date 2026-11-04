// Continue-with-Google completion form (Milestone 2). Progressive enhancement only: the server validates
// everything and rejects the other account type's fields, so this script just shows, requires and submits
// the selected type's fields (the same behaviour as the registration form's register.js).
(function () {
  "use strict";
  var form = document.getElementById("google-complete-form");
  if (!form) return;
  document.documentElement.classList.add("has-js");

  function selectedRole() {
    var checked = form.querySelector('input[name="role"]:checked');
    return checked ? checked.value : null;
  }

  function applyRole(role, clearOther) {
    form.querySelectorAll("fieldset[data-role]").forEach(function (fieldset) {
      var active = fieldset.getAttribute("data-role") === role;
      fieldset.hidden = !active;
      fieldset.disabled = !active;
      fieldset.querySelectorAll("input, select, textarea").forEach(function (input) {
        if (!active && clearOther) {
          if (input.tagName === "SELECT") { input.selectedIndex = 0; } else { input.value = ""; }
        }
        if (input.hasAttribute("data-required")) { input.required = active; }
      });
    });
  }

  form.addEventListener("change", function (event) {
    if (event.target.name === "role") { applyRole(event.target.value, true); }
  });
  applyRole(selectedRole(), false);
})();
