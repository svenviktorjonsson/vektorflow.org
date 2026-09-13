import { highlightVkf } from "../editor/vkf-highlighter.mjs";
import { mountRetainedSceneResult } from "../inline-retained-scene-view.mjs";
import { loadSharedCompiler } from "./vkf-shared-compiler.mjs";
import { LIVE_EXAMPLE_GROUPS, LIVE_EXAMPLES_BY_ID } from "./examples.mjs";

const source = document.querySelector("#source");
const highlight = document.querySelector("#highlight");
const compileButton = document.querySelector("#compile");
const playButton = document.querySelector("#play");
const example = document.querySelector("#example");
const output = document.querySelector("#output");
const outputHeading = document.querySelector("#output-heading");
const visualization = document.querySelector("#visualization");
const status = document.querySelector("#status");

const EXAMPLES = Object.freeze({
  console: Object.freeze({
    id: "console",
    title: "Dependency chain",
    source: "base: 40\nfirst: base + 1\nsecond: first + 1\nsecond + 1",
    kind: "console",
  }),
  "console-arithmetic": Object.freeze({
    id: "console-arithmetic",
    title: "Grouped arithmetic",
    source: "value: 100\n:: (value - (20 + 4) * 2) // (3 + 1)",
    kind: "console",
  }),
  ...LIVE_EXAMPLES_BY_ID,
});

for (const group of LIVE_EXAMPLE_GROUPS) {
  const options = document.createElement("optgroup");
  options.label = group.label;
  for (const item of group.examples) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.title;
    options.append(option);
  }
  example.append(options);
}

function renderHighlight() {
  highlight.innerHTML = highlightVkf(`${source.value}\n`);
}

function synchronizeScroll() {
  const pre = highlight.parentElement;
  pre.scrollTop = source.scrollTop;
  pre.scrollLeft = source.scrollLeft;
}

renderHighlight();
source.addEventListener("input", renderHighlight);
source.addEventListener("scroll", synchronizeScroll);

const compiler = await loadSharedCompiler();
let stopVisualization;
let catalogExample;
compileButton.disabled = false;
status.value = "Ready";

function selectedExample() {
  if (example.value === "catalog" && catalogExample) return catalogExample;
  return EXAMPLES[example.value] ?? EXAMPLES.console;
}

function selectedExampleIsTimed() {
  return selectedExample().kind.endsWith("-time");
}

function selectedExampleIsSurface() {
  return selectedExample().kind.startsWith("surface");
}

function stopAnimation() {
  stopVisualization?.();
  stopVisualization = undefined;
  visualization.replaceChildren();
  playButton.textContent = "Play";
}

function showConsole() {
  output.hidden = false;
  visualization.hidden = true;
  outputHeading.textContent = selectedExample().kind === "console" ? "Console" : "Result";
}

function showVisualization() {
  output.hidden = true;
  visualization.hidden = false;
  outputHeading.textContent = "Result";
}

async function compileSource() {
  stopAnimation();
  const started = performance.now();
  try {
    const result = compiler.run(source.value);
    if (result.kind === "visual") {
      stopVisualization = mountRetainedSceneResult(
        visualization, result.retained_scene_arenas);
      showVisualization();
    } else {
      showConsole();
      output.textContent = result.stdout;
    }
    status.value = `Compiled and ran in ${(performance.now() - started).toFixed(1)} ms`;
  } catch (error) {
    output.textContent = error.cause?.message ?? error.message;
    showConsole();
    status.value = "Compile or runtime error";
  }
}

playButton.addEventListener("click", async () => {
  await compileSource();
});

example.addEventListener("change", () => {
  stopAnimation();
  const chosen = selectedExample();
  source.value = chosen.source;
  compileButton.disabled = false;
  playButton.hidden = !selectedExampleIsTimed();
  history.replaceState(null, "", `?example=${encodeURIComponent(example.value)}`);
  renderHighlight();
  compileSource();
});

compileButton.addEventListener("click", compileSource);
source.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    compileSource();
  }
});

function catalogueSourceUrl(path) {
  if (!/^(?:examples|benchmarks)\/[a-zA-Z0-9_./-]+\.vkf$/u.test(path) || path.split("/").includes("..")) {
    throw new TypeError("Invalid catalogue source path");
  }
  return `./generated/sources/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function readmeSourceUrl(path) {
  if (!/^snippets\/readme-\d+\.vkf$/u.test(path) || path.split("/").includes("..")) {
    throw new TypeError("Invalid README source path");
  }
  return `../generated/${path.split("/").map(encodeURIComponent).join("/")}`;
}

async function loadInitialExample() {
  const parameters = new URLSearchParams(location.search);
  const requestedSource = parameters.get("source");
  const requestedReadme = parameters.get("readme");
  if (requestedSource || requestedReadme) {
    const response = await fetch(requestedReadme
      ? readmeSourceUrl(requestedReadme)
      : catalogueSourceUrl(requestedSource));
    if (!response.ok) throw new Error(`Source request failed (${response.status})`);
    const title = parameters.get("title") || (requestedSource || requestedReadme).split("/").at(-1);
    const browserRunnable = parameters.get("browserRunnable") === "true";
    catalogExample = Object.freeze({
      source: await response.text(),
      kind: parameters.get("kind") || "console",
      browserRunnable,
    });
    const option = document.createElement("option");
    option.value = "catalog";
    option.textContent = `README · ${title}`;
    example.append(option);
    example.value = "catalog";
    source.value = catalogExample.source;
    playButton.hidden = true;
    compileButton.disabled = false;
    renderHighlight();
    await compileSource();
    return;
  }

  const requestedExample = parameters.get("example");
  if (requestedExample && EXAMPLES[requestedExample]) {
    example.value = requestedExample;
    source.value = EXAMPLES[requestedExample].source;
  }
  playButton.hidden = !selectedExampleIsTimed();
  renderHighlight();
  await compileSource();
}

try {
  await loadInitialExample();
} catch (error) {
  output.textContent = error.cause?.message ?? error.message;
  showConsole();
  status.value = "Source load error";
}
