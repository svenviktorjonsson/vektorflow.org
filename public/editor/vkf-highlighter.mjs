const KEYWORDS = new Set(["true", "false"]);
const BUILTINS = new Set(["any", "bit", "chr", "dig", "int", "num", "str", "type"]);

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function token(kind, value) {
  return `<span class="vf-token ${kind}">${escapeHtml(value)}</span>`;
}

function dimensionToken(value) {
  return token(`dimension dimension-${value}`, value);
}

// A structural suffix is a run of one-character axis names (or compile-time
// digits). Multi-character suffixes remain ordinary identifiers: x_min and
// particle_count must never be split by a lexical highlighter.
function structuralIdentifier(value) {
  const separator = value.lastIndexOf("_");
  if (separator <= 0 || separator === value.length - 1) return null;
  const base = value.slice(0, separator);
  const suffix = value.slice(separator + 1);
  // Common ordinary underscore identifiers take precedence in the lexical
  // fallback. A semantic/LSP provider can override this with resolved names.
  if (new Set(["min", "max", "minima", "maxima", "pot", "tot", "count"]).has(suffix)) {
    return null;
  }
  if (!/^(?:[A-Za-z]|\d)+$/u.test(suffix)) return null;
  if (/[A-Za-z]/u.test(suffix) && ![...suffix].every((axis) => /[a-z]/u.test(axis))) {
    return null;
  }
  return { base, suffix };
}

function identifierKind(source, start, value, end) {
  const lineStart = source.lastIndexOf("\n", start - 1) + 1;
  const before = source.slice(lineStart, start);
  const after = source.slice(end);
  if (/^\s*$/u.test(before) && /^\s*:(?!:)/u.test(after)) return "binding";
  if (KEYWORDS.has(value)) return "keyword";
  if (BUILTINS.has(value)) return "builtin";
  if (/^\s*(?:\[[^\]\n]*\]\s*)?\(/u.test(after)) return "function";
  if (/^[A-Z]/u.test(value)) return "type";
  return null;
}

export function highlightVkf(source) {
  let html = "";
  let cursor = 0;
  while (cursor < source.length) {
    const rest = source.slice(cursor);
    const comment = /^#[^\n]*/u.exec(rest);
    if (comment) {
      html += token("comment", comment[0]);
      cursor += comment[0].length;
      continue;
    }
    const string = /^(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*")/u.exec(rest);
    if (string) {
      html += token("string", string[0]);
      cursor += string[0].length;
      continue;
    }
    const number = /^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/u.exec(rest);
    if (number) {
      html += token("number", number[0]);
      cursor += number[0].length;
      continue;
    }
    const identifier = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(rest);
    if (identifier) {
      const value = identifier[0];
      const structural = structuralIdentifier(value);
      if (structural) {
        const identifierEnd = cursor + value.length;
        const kind = identifierKind(source, cursor, structural.base, identifierEnd);
        html += kind ? token(kind, structural.base) : escapeHtml(structural.base);
        html += token("operator", "_");
        for (const axis of structural.suffix) html += dimensionToken(axis);
      } else {
        const kind = identifierKind(source, cursor, value, cursor + value.length);
        html += kind ? token(kind, value) : escapeHtml(value);
      }
      cursor += value.length;
      continue;
    }
    const anonymousDimensions = /^\.\.\./u.exec(rest);
    if (anonymousDimensions) {
      html += token("anonymous-dimension", anonymousDimensions[0]);
      cursor += anonymousDimensions[0].length;
      continue;
    }
    const operator = /^(?:::|>>|==|~=|!=|<=|>=|=>|->|\/\/|\.\.|><|\/\\|\\\/|@::|@:|@>|@\||@!|[=<>+\-*/^%&~:$?.|])/u.exec(rest);
    if (operator) {
      html += token("operator", operator[0]);
      cursor += operator[0].length;
      continue;
    }
    const punctuation = /^[;,()[\]{}]/u.exec(rest);
    if (punctuation) {
      html += token("punctuation", punctuation[0]);
      cursor += punctuation[0].length;
      continue;
    }
    html += escapeHtml(source[cursor]);
    cursor += 1;
  }
  return html;
}
