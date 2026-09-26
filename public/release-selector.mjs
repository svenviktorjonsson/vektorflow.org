export function releaseRuntimeFor(marker) {
  if (!marker?.dataset.releaseVersion) throw new Error("missing stable release metadata");
  return Object.freeze({
    version: marker.dataset.releaseVersion,
    wasmUrl: marker.dataset.browserWasm || null,
  });
}

export function browserUnavailableMessage(version) {
  return `VKF ${version} has no published browser compiler.`;
}
