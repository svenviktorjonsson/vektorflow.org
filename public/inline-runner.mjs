const DEFAULT_WASM_URL = new URL("./playground/artifacts/vkf-shared-compiler.wasm", import.meta.url);
const WORKER_URL = new URL("./inline-runner-worker.mjs", import.meta.url);

function resultPackets(output) {
  if (Array.isArray(output?.retained_scene_arenas) && output.retained_scene_arenas.length > 0) {
    if (!output.retained_scene_arenas.every((packet) =>
      packet?.schema === "vektor-flow/retained-scene-arena"
      && packet.version === 1
      && packet.arena instanceof Uint8Array)) {
      throw new TypeError("browser compiler returned an invalid retained scene arena");
    }
    return output.retained_scene_arenas;
  }
  return null;
}

export function createInlineRunner({
  wasmUrl = DEFAULT_WASM_URL,
  compileModule = globalThis.WebAssembly?.compile,
  fetchImpl = globalThis.fetch,
  WorkerClass = globalThis.Worker,
  timeoutMs = 2_000,
} = {}) {
  if (typeof compileModule !== "function"
      || typeof fetchImpl !== "function"
      || typeof WorkerClass !== "function") {
    throw new Error("inline browser execution is unavailable in this environment");
  }
  const now = () => globalThis.performance?.now?.() ?? Date.now();
  const downloadStarted = now();
  let downloadMs = 0;
  let moduleCompileMs = 0;
  const compilerModule = fetchImpl(wasmUrl).then((response) => {
    if (!response.ok) throw new Error("browser compiler WASM is unavailable");
    return response.arrayBuffer();
  }).then((bytes) => {
    downloadMs = now() - downloadStarted;
    const compileStarted = now();
    return compileModule(bytes).then((module) => {
      moduleCompileMs = now() - compileStarted;
      return module;
    });
  });
  let sequence = 0;
  let activeApplication = null;

  function request(worker, message, { terminate = false } = {}) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (callback) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        worker.onerror = null;
        worker.onmessage = null;
        if (terminate) worker.terminate();
        callback();
      };
      const timer = setTimeout(() => finish(() => {
        worker.terminate();
        reject(new Error("VKF execution timed out; worker terminated"));
      }), timeoutMs);
      worker.onerror = () => finish(() => {
        worker.terminate();
        reject(new Error("VKF execution worker failed"));
      });
      worker.onmessage = ({ data }) => {
        if (data?.id !== message.id) return;
        if (data.status === "error") {
          finish(() => reject(new Error(data.message)));
          return;
        }
        finish(() => resolve(data));
      };
      worker.postMessage(message);
    });
  }

  return Object.freeze({
    async prewarm() {
      await compilerModule;
      return Object.freeze({ downloadMs, moduleCompileMs });
    },
    async run(source) {
      if (typeof source !== "string") throw new TypeError("inline VKF source must be a string");
      const module = await compilerModule;
      const worker = new WorkerClass(WORKER_URL, { type: "module", name: "vkf-inline-runner" });
      const id = ++sequence;
      const data = await request(worker, {
        type: "run", id, source, module, measurePerformance: true,
      }, { terminate: true });
      const execution = { output: data.output, packets: resultPackets(data.output) };
      if (data.timing) execution.timing = data.timing;
      return execution;
    },
    async compileApplication({ source, sourceId, assets } = {}) {
      if (typeof source !== "string" || typeof sourceId !== "string" ||
          !assets || typeof assets !== "object" || Array.isArray(assets)) {
        throw new TypeError("inline VKF application request is malformed");
      }
      activeApplication?.reset();
      const module = await compilerModule;
      const worker = new WorkerClass(WORKER_URL, {
        type: "module", name: "vkf-inline-application-runner",
      });
      const id = ++sequence;
      let response;
      try {
        response = await request(worker, {
          type: "compile-application", id, source, sourceId, assets, module,
        });
      } catch (error) {
        worker.terminate();
        throw error;
      }
      const bundle = response.bundle;
      if (bundle?.schema !== "vektor-flow/application-bundle" || bundle.version !== 1 ||
          typeof bundle.sourceSha256 !== "string" || !bundle.initialVisual ||
          !Array.isArray(bundle.staticAssets) ||
          bundle.retainedEventProgram?.schema !== "vektor-flow/retained-event-program") {
        worker.terminate();
        throw new TypeError("browser worker returned an invalid VKF application bundle");
      }
      let active = true;
      let pending = Promise.resolve();
      const execution = Object.freeze({
        bundle,
        dispatch(event) {
          const operation = pending.then(async () => {
            if (!active) throw new Error("stale application event");
            const dispatchId = ++sequence;
            const data = await request(worker, {
              type: "dispatch-application", id: dispatchId,
              sourceSha256: bundle.sourceSha256, event,
            });
            return data.dispatch;
          });
          pending = operation.catch(() => {});
          return operation;
        },
        reset() {
          if (!active) return;
          active = false;
          worker.terminate();
          if (activeApplication === execution) activeApplication = null;
        },
      });
      activeApplication = execution;
      return execution;
    },
  });
}
