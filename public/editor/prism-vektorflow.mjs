export const TEXTMATE_GRAMMAR_SHA256 = 'a81f226ba1183be77f99ce14ce48e76d19958b16ae0c2acf53a4f08455b2ca64';

export function registerVektorFlowPrism(Prism) {
  if (!Prism?.languages) throw new TypeError('Prism.languages is required.');

  const interpolation = {
    pattern: /(^|[^\\])\$\([\s\S]*?\)/,
    lookbehind: true,
    inside: {
      punctuation: /^\$\(|\)$/,
      number: /\b\d+(?:\.\d+)?\b/,
    operator: /\.\.\.|:::|::|>>|==|~=|!=|<=|>=|=>|->|\/\/|\.\.|><|\/\\|\\\/|@::|@:|@>|@\||@!|[=<>+\-*/^%&~:$?.|]/,
      function: /\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/,
      variable: /\b[a-zA-Z_][a-zA-Z0-9_]*\b/,
    },
  };

  Prism.languages.vektorflow = {
    comment: /#.*/,
    'triple-quoted-string': [
      { pattern: /"""[\s\S]*?"""/, greedy: true, alias: 'string' },
      { pattern: /'''[\s\S]*?'''/, greedy: true, alias: 'string' },
    ],
    string: {
      pattern: /"(?:\\.|[^"\\])*"/,
      greedy: true,
      inside: {
        interpolation,
        escape: /\\(?:["\\nrt$]|.)/,
      },
    },
    'line-print-sugar': { pattern: /:::/, alias: 'keyword' },
    module: {
      pattern: /(:)(\.)([a-zA-Z_][a-zA-Z0-9_]*)/,
      inside: {
        punctuation: /^:\./,
        namespace: /[a-zA-Z_][a-zA-Z0-9_]*$/,
      },
    },
    'structural-axis-call': {
      pattern: /\b(?:(?:stat\.(?:sum|mean|variance|std|range|count|min|max|percentile|median|iqr|mode|zscore|normalize|covariance|correlation))|(?:symbolic\.(?:integ|integrate|deriv|differentiate|diff|limit|roots|polynomial|depends_on|substitute)))_([a-z]+)(?=\s*\()/,
      inside: {
        namespace: /^(?:stat|symbolic)/,
        punctuation: /\./,
        function: /(?:sum|mean|variance|std|range|count|min|max|percentile|median|iqr|mode|zscore|normalize|covariance|correlation|integ|integrate|deriv|differentiate|diff|limit|roots|polynomial|depends_on|substitute)(?=_)/,
        dimension: /_[a-z]+$/,
      },
    },
    'stdlib-call': {
      pattern: /\b(?:math|stat|random|time|io|collections|errors|system|process|regex|linalg|physics|symbolic)\b\s*\.\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?=\()/,
      inside: {
        namespace: /^[a-zA-Z_][a-zA-Z0-9_]*/,
        punctuation: /\./,
        function: /[a-zA-Z_][a-zA-Z0-9_]*$/,
      },
    },
    'function-definition': {
      pattern: /^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:\[[^\]\n]*\]\s*)?\([^\n)]*\)\s*(?:->\s*[^:\n]+)?\s*:(?:\s*(?:#.*)?)?$/m,
      inside: {
        function: /^[\s]*[a-zA-Z_][a-zA-Z0-9_]*/,
        parameter: /\([^)]*\)/,
        operator: /->/,
        punctuation: /[():]/,
      },
    },
    binding: { pattern: /^\s*[a-zA-Z_][a-zA-Z0-9_]*(?=\s*:(?!::))/m, alias: 'variable' },
    boolean: /\b(?:true|false)\??\b/,
    wildcard: { pattern: /\b_\?/, alias: 'keyword' },
    builtin: /\b(?:bit|chr|dig|int|num|str|type|any)\b/,
    number: /\b(?:\d+\.\d+|\d+)\b/,
    'anonymous-dimension': /\.\.\./,
    dimension: {
      pattern: /\b[A-Za-z][A-Za-z0-9]*_(?!min\b|max\b|pot\b|tot\b|count\b)(?:[a-z]+|[0-9]+)\b(?!\s*\()/,
      alias: 'dimension',
    },
    operator: /:::|::|>>|==|~=|!=|<=|>=|=>|->|\/\/|\.\.|><|\/\\|\\\/|@::|@:|@>|@\||@!|\(\s*[+\-*/]\s*\)|\{\s*[+\-*/]\s*\}|[=<>+\-*/^%&~:$?.|]/,
    function: /\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/,
    'class-name': /\b[A-Z][a-zA-Z0-9_]*\b/,
    variable: /\b[a-zA-Z_][a-zA-Z0-9_]*\b/,
    punctuation: /[;,()[\]{}]/,
  };
  Prism.languages.vkf = Prism.languages.vektorflow;
  return Prism.languages.vektorflow;
}
