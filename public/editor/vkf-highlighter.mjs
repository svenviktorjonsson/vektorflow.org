const BOOLEANS = new Set(["true", "false"]);
const BUILTINS = new Set(["any", "bit", "chr", "dig", "int", "num", "str", "type"]);
const EMPTY_IDENTIFIERS = new Set();
const STRUCTURAL_MEMBER_FUNCTIONS = new Set([
  "correlation", "count", "covariance", "deriv", "differentiate", "diff",
  "depends_on", "integ", "integrate", "iqr", "limit", "max", "mean", "median",
  "min", "mode", "normalize", "percentile", "polynomial", "range", "roots", "std",
  "substitute", "sum", "variance", "zscore",
]);

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

function dimensionColor(value) {
  if (/^[a-z]$/u.test(value)) {
    const index = value.codePointAt(0) - "a".codePointAt(0);
    return `hsl(${Math.round((200 + index * 137.508) % 360)} 72% 72%)`;
  }
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return `hsl(${hash % 360} 72% 72%)`;
}

function dimensionToken(value) {
  return `<span class="vf-token dimension" style="--vf-dimension-color:${dimensionColor(value)}">${escapeHtml(value)}</span>`;
}

function codeOnly(source) {
  let output = "";
  let cursor = 0;
  while (cursor < source.length) {
    const rest = source.slice(cursor);
    const hidden = /^(?:##(?:[\s\S]*?##|[\s\S]*)|#(?!#)[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*")/u.exec(rest);
    if (!hidden) {
      output += source[cursor];
      cursor += 1;
      continue;
    }
    output += hidden[0].replace(/[^\n]/gu, " ");
    cursor += hidden[0].length;
  }
  return output;
}

function functionDeclaration(line) {
  const head = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[([^\]\n]*)\]\s*)?\(/u.exec(line);
  if (!head) return null;
  const open = head[0].lastIndexOf("(");
  let depth = 0;
  let close = -1;
  for (let index = open; index < line.length; index++) {
    if (line[index] === "(") depth += 1;
    if (line[index] !== ")") continue;
    depth -= 1;
    if (depth === 0) {
      close = index;
      break;
    }
  }
  if (close < 0) return null;
  const tail = line.slice(close + 1).trimStart();
  if (!tail.startsWith(":") && !(tail.startsWith("->") && tail.includes(":"))) return null;
  return { name: head[1], compileParameters: head[2], parameters: line.slice(open + 1, close) };
}

function declaredIdentifiers(source) {
  const declared = new Set();
  for (const line of codeOnly(source).split(/[\n;]/u)) {
    const functionDefinition = functionDeclaration(line);
    if (functionDefinition) {
      declared.add(functionDefinition.name);
      for (const parameters of [functionDefinition.compileParameters, functionDefinition.parameters]) {
        if (!parameters) continue;
        for (const match of parameters.matchAll(/(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:/gu)) {
          declared.add(match[1]);
        }
      }
      continue;
    }
    const binding = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:(?!:)/u.exec(line);
    if (binding) declared.add(binding[1]);
  }
  return declared;
}

// Resolve a complete declared name first. Otherwise a trailing run of
// lowercase index labels (or compile-time digits) is one structural suffix;
// its letters never select different highlighting categories.
function structuralIdentifier(value, { memberCall = false, declared = EMPTY_IDENTIFIERS } = {}) {
  const separator = value.lastIndexOf("_");
  if (separator <= 0 || separator === value.length - 1) return null;
  if (declared.has(value)) return null;
  const base = value.slice(0, separator);
  const suffix = value.slice(separator + 1);
  // A dotted call normally has an ordinary snake_case member name. Only the
  // standard axis-polymorphic operations specialize a member call by suffix.
  if (memberCall && !STRUCTURAL_MEMBER_FUNCTIONS.has(base)) return null;
  if (!/^(?:[A-Za-z]|\d)+$/u.test(suffix)) return null;
  if (/[A-Za-z]/u.test(suffix) && ![...suffix].every((axis) => /[a-z]/u.test(axis))) {
    return null;
  }
  return { base, suffix };
}

function identifierKind(source, start, value, end) {
  const lineStart = Math.max(
    source.lastIndexOf("\n", start - 1),
    source.lastIndexOf(";", start - 1),
  ) + 1;
  const before = source.slice(lineStart, start);
  const after = source.slice(end);
  if (/^\s*$/u.test(before) && /^\s*:(?!:)/u.test(after)) return "binding";
  if (BOOLEANS.has(value)) return "boolean";
  if (BUILTINS.has(value)) return "builtin";
  if (/^\s*(?:\[[^\]\n]*\]\s*)?\(/u.test(after)) return "function";
  if (/^[A-Z]/u.test(value)) return "type";
  return null;
}

export function highlightVkf(source) {
  const declared = declaredIdentifiers(source);
  let html = "";
  let cursor = 0;
  while (cursor < source.length) {
    const rest = source.slice(cursor);
    const comment = /^(?:##(?:[\s\S]*?##|[\s\S]*)|#(?!#)[^\n]*)/u.exec(rest);
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
      const identifierEnd = cursor + value.length;
      const attached = value.startsWith("_")
        && /[0-9'"\])}]/u.test(source[cursor - 1] ?? "")
        ? structuralIdentifier(`value${value}`)
        : null;
      if (attached) {
        html += token("operator", "_");
        html += dimensionToken(attached.suffix);
        cursor += value.length;
        continue;
      }
      const memberCall = /\.\s*$/u.test(source.slice(0, cursor))
        && /^\s*\(/u.test(source.slice(identifierEnd));
      const structural = structuralIdentifier(value, { memberCall, declared });
      if (structural) {
        const kind = identifierKind(source, cursor, structural.base, identifierEnd);
        html += kind ? token(kind, structural.base) : escapeHtml(structural.base);
        html += token("operator", "_");
        html += dimensionToken(structural.suffix);
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
    const operator = /^(?:::|>>|==|~=|!=|!\?|<=|>=|=>|->|\/\/|\.\.|><|\/\\|\\\/|@::|@:|@>|@\||@!|[=<>+\-*/^%&~:$?.|!])/u.exec(rest);
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
