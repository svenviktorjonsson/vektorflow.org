import { createSharedCompiler } from "./playground/vkf-shared-compiler.mjs";

const NO_HOST_IMPORTS = Object.freeze({});

function compilerFor(module) {
  if (!(module instanceof WebAssembly.Module)) {
    throw new Error("browser compiler module is unavailable");
  }
  if (WebAssembly.Module.imports(module).length !== 0) {
    throw new Error("browser compiler requested forbidden host imports");
  }
  return createSharedCompiler({ instance: new WebAssembly.Instance(module, NO_HOST_IMPORTS) });
}

export function createInlineWorkerRequestHandler() {
  let applicationCompiler = null;
  let applicationSourceSha256 = null;
  return function handleInlineWorkerRequest(data) {
    try {
      if (data?.type === "compile-application") {
        if (typeof data.source !== "string" || typeof data.sourceId !== "string" ||
            !data.assets || typeof data.assets !== "object" || Array.isArray(data.assets)) {
          throw new TypeError("invalid inline application compile request");
        }
        applicationCompiler = compilerFor(data.module);
        const bundle = applicationCompiler.compileApplication({
          source: data.source, sourceId: data.sourceId, assets: data.assets,
        });
        applicationSourceSha256 = bundle.sourceSha256;
        return { id: data.id, status: "ok", bundle };
      }
      if (data?.type === "dispatch-application") {
        if (!applicationCompiler || typeof data.sourceSha256 !== "string" ||
            !data.event || typeof data.event !== "object" || Array.isArray(data.event)) {
          throw new TypeError("invalid inline application dispatch request");
        }
        if (data.sourceSha256 !== applicationSourceSha256) {
          throw new Error("stale application event");
        }
        return { id: data.id, status: "ok", dispatch: applicationCompiler.dispatchApplication({
          sourceSha256: data.sourceSha256, event: data.event,
        }) };
      }
      if (data?.type !== "run" || typeof data.source !== "string") {
        throw new TypeError("invalid inline worker request");
      }
      if (data.measurePerformance === true) {
        const instantiateStarted = globalThis.performance?.now?.() ?? Date.now();
        const compiler = compilerFor(data.module);
        const compilerInstantiateMs = (globalThis.performance?.now?.() ?? Date.now()) - instantiateStarted;
        const measuredOutput = compiler.run(data.source, { measurePerformance: true });
        const { timing, ...output } = measuredOutput;
        return { id: data.id, status: "ok", output,
          timing: { compilerInstantiateMs, ...timing } };
      }
      const compiler = compilerFor(data.module);
      return { id: data.id, status: "ok", output: compiler.run(data.source) };
    } catch (error) {
      return {
        id: data?.id,
        status: "error",
        message: error instanceof Error ? error.message : "VKF execution failed",
      };
    }
  };
}

const handleInlineWorkerRequest = createInlineWorkerRequestHandler();

export function runInlineWorkerRequest(data) {
  return createInlineWorkerRequestHandler()(data);
}

globalThis.onmessage = ({ data }) => {
  const response = handleInlineWorkerRequest(data);
  const transfers = response.output?.retained_scene_arenas
    ? response.output.retained_scene_arenas.map((packet) => packet.arena.buffer)
    : response.bundle?.staticAssets
      ? response.bundle.staticAssets.map((asset) => asset.bytes.buffer)
      : [];
  globalThis.postMessage(response, transfers);
};
