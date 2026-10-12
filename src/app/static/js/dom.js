// Text-safe DOM construction (Day 10). API values are always inserted with
// textContent, never parsed as HTML.

export function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
  for (const [name, value] of Object.entries(options.attrs || {})) {
    if (value !== undefined && value !== null && value !== false) node.setAttribute(name, String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function setStatus(node, message, kind = "info") {
  node.textContent = message || "";
  node.dataset.kind = kind;
}

export function definitionList(rows) {
  const dl = el("dl", { className: "facts" });
  for (const [term, value] of rows) {
    dl.append(el("dt", { text: term }), el("dd", { text: value }));
  }
  return dl;
}
