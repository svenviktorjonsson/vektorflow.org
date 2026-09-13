export function resolveReleaseVersion({ requested, versions, latest }) {
  if (versions.includes(requested)) return requested;
  return latest;
}

export function withReleaseVersion(value, href, latest) {
  const url = new URL(href);
  if (value === latest) url.searchParams.delete("version");
  else url.searchParams.set("version", value);
  return url;
}

export function applyReleaseVersion(root, value) {
  for (const panel of root.querySelectorAll("[data-release-version]")) {
    panel.hidden = panel.dataset.releaseVersion !== value;
  }
}

export function releaseRuntimeFor(selector, value) {
  const option = [...selector.options].find((candidate) => candidate.value === value);
  return Object.freeze({ version: value, wasmUrl: option?.dataset.browserWasm || null });
}

export function browserUnavailableMessage(version) {
  return `VKF ${version} has no published browser compiler.`;
}

export function applyReleaseRuntime(root, runtime) {
  for (const label of root.querySelectorAll("[data-vkf-run-version]")) {
    label.textContent = runtime.version;
  }
  for (const button of root.querySelectorAll(".readme-example-play")) {
    button.disabled = !runtime.wasmUrl;
    button.textContent = runtime.wasmUrl ? "Run" : "Unavailable";
  }
}

const selector = globalThis.document?.querySelector("#vkf-release-version");
if (selector) {
  const versions = [...selector.options].map(({ value }) => value);
  const latest = selector.dataset.latest;
  const current = new URL(globalThis.location.href);
  const activate = (value) => {
    const runtime = releaseRuntimeFor(selector, value);
    selector.value = value;
    applyReleaseVersion(globalThis.document, value);
    applyReleaseRuntime(globalThis.document, runtime);
    globalThis.history.replaceState(
      null,
      "",
      withReleaseVersion(value, globalThis.location.href, latest),
    );
    for (const link of globalThis.document.querySelectorAll('a[href$=".html"], a[href*=".html#"], a[href*=".html?"]')) {
      const url = new URL(link.href, globalThis.location.href);
      if (url.origin === globalThis.location.origin) link.href = withReleaseVersion(value, url, latest);
    }
    globalThis.dispatchEvent(new CustomEvent("vf-release-version-change", {
      detail: runtime,
    }));
  };
  activate(resolveReleaseVersion({
    requested: current.searchParams.get("version"), versions, latest,
  }));
  selector.addEventListener("change", () => activate(selector.value));
}
