// Create-request form (R4): show only the amount fields that apply to the chosen pay type.
// Progressive enhancement only. Without this script every group stays visible and the form works
// unchanged. The server reads only the chosen pay type's amounts (fixed: budget; hourly: at least one
// rate; unspecified: none), so hidden groups are simply hidden: never disabled, cleared or made
// required here, and the submitted field names and values are exactly what the user entered.
(function () {
  "use strict";
  var select = document.getElementById("pay_type");
  if (!select) return;
  var groups = Array.prototype.slice.call(document.querySelectorAll("[data-pay]"));
  if (!groups.length) return;

  function update() {
    var type = select.value;
    groups.forEach(function (group) {
      group.hidden = group.getAttribute("data-pay") !== type;
    });
  }

  select.addEventListener("change", update);
  update();
})();
