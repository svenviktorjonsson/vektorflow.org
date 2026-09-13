// Transport only. Parsing, typing, lowering and execution are owned by WASM.
export function createSharedCompiler({instance}) {
  const api = instance.exports;
  let clockSequence = 0;
  let documentSequence = 0;
  let documentState = null;
  api._initialize?.();
  function response() {
    return JSON.parse(new TextDecoder().decode(new Uint8Array(api.memory.buffer,
      api.vkf_result_pointer(), api.vkf_result_length())));
  }
  function withSource(source, callback) {
    if (typeof source !== 'string') throw new TypeError('browser compiler source must be a string');
    const bytes = new TextEncoder().encode(source);
    const pointer = api.malloc(bytes.length + 1);
    if (!pointer) throw new Error('browser compiler source allocation failed');
    try {
      new Uint8Array(api.memory.buffer, pointer, bytes.length).set(bytes);
      return callback(pointer, bytes.length);
    } finally { api.free(pointer); }
  }
  function checkedResponse(status, phase = 'discovery') {
      const result = response();
      if (status !== 0) {
        const error = new Error(result.message);
        error.phase = phase;
        throw error;
      }
      return result;
  }
  function compile(source) {
    return withSource(source, (pointer, length) => checkedResponse(api.vkf_compile_source(pointer, length), 'frontend'));
  }
  function normalizeClockSnapshot(value) {
    if (!value || typeof value !== 'object') throw new TypeError('clock arena snapshot must be an object');
    if (value.schema !== 'vektor-flow/clock-arena') throw new TypeError('clock arena schema is invalid');
    if (value.version !== 1) throw new RangeError(`clock arena version ${value.version} is unsupported`);
    if (!Number.isSafeInteger(value.sequence) || value.sequence <= 0 || value.sequence > 0xffffffff) {
      throw new RangeError('clock arena snapshot sequence is stale or invalid');
    }
    if (!Number.isFinite(value.wallSeconds) || !Number.isFinite(value.monotonicSeconds)) {
      throw new RangeError('clock arena samples must be finite');
    }
    return Object.freeze({...value});
  }
  function captureClockSnapshot() {
    clockSequence += 1;
    const monotonicMilliseconds = globalThis.performance?.now?.();
    return Object.freeze({
      schema: 'vektor-flow/clock-arena', version: 1, sequence: clockSequence,
      wallSeconds: Date.now() / 1000,
      monotonicSeconds: Number.isFinite(monotonicMilliseconds)
        ? monotonicMilliseconds / 1000 : 0,
    });
  }
  function installClockSnapshot(programApi, value) {
    const snapshot = normalizeClockSnapshot(value);
    const pointer = programApi.vkf_vm_clock_arena_ptr?.();
    const length = programApi.vkf_vm_clock_arena_len?.();
    if (!Number.isInteger(pointer) || length !== 32) {
      throw new TypeError('compiled program clock arena ABI is unavailable');
    }
    const view = new DataView(programApi.memory.buffer, pointer, length);
    view.setUint32(0, snapshot.version, true);
    view.setUint32(4, length, true);
    view.setUint32(8, snapshot.sequence, true);
    view.setUint32(12, 0, true);
    view.setFloat64(16, snapshot.wallSeconds, true);
    view.setFloat64(24, snapshot.monotonicSeconds, true);
    return snapshot;
  }
  function freshDocumentBytes(length) {
    const bytes = new Uint8Array(length);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 1, true);
    view.setUint32(4, length, true);
    return bytes;
  }
  function normalizeDocumentSnapshot(value, length) {
    if (!value || typeof value !== 'object') throw new TypeError('document arena snapshot must be an object');
    if (value.schema !== 'vektor-flow/document-arena') throw new TypeError('document arena schema is invalid');
    if (value.version !== 1) throw new RangeError(`document arena version ${value.version} is unsupported`);
    if (!Number.isSafeInteger(value.sequence) || value.sequence <= 0) {
      throw new RangeError('document arena snapshot sequence is stale or invalid');
    }
    const bytes = value.bytes instanceof Uint8Array ? value.bytes
      : value.bytes instanceof ArrayBuffer ? new Uint8Array(value.bytes) : null;
    if (!bytes || bytes.byteLength !== length) throw new RangeError('document arena snapshot length is invalid');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== 1 || view.getUint32(4, true) !== length) {
      throw new TypeError('document arena snapshot header is invalid');
    }
    return Uint8Array.from(bytes);
  }
  function installDocumentSnapshot(programApi, supplied) {
    const pointer = programApi.vkf_vm_document_arena_ptr?.();
    const length = programApi.vkf_vm_document_arena_len?.();
    if (!Number.isInteger(pointer) || !Number.isInteger(length) || length <= 32) {
      throw new TypeError('compiled program document arena ABI is unavailable');
    }
    const bytes = supplied ? normalizeDocumentSnapshot(supplied, length)
      : documentState?.byteLength === length ? Uint8Array.from(documentState)
      : freshDocumentBytes(length);
    documentSequence += 1;
    new DataView(bytes.buffer).setUint32(8, documentSequence, true);
    new Uint8Array(programApi.memory.buffer, pointer, length).set(bytes);
    return {pointer, length};
  }
  function documentRuntimeError(programApi, pointer) {
    const code = new DataView(programApi.memory.buffer, pointer, 32).getUint32(20, true);
    const messages = {
      1: 'document path must be a normalized relative path',
      2: 'document does not exist in this session',
      3: 'document arena document-count quota exceeded (64)',
      4: 'document arena per-document byte quota exceeded (8192)',
      5: 'document arena total byte quota exceeded (49152)',
    };
    return messages[code] ? new Error(messages[code]) : null;
  }
  function compileApplication({source, sourceId, assets} = {}) {
    if (typeof source !== 'string') throw new TypeError('application source must be a string');
    if (typeof sourceId !== 'string') throw new TypeError('application sourceId must be a string');
    if (!assets || typeof assets !== 'object' || Array.isArray(assets)) {
      throw new TypeError('application assets must be a path-to-bytes object');
    }
    const encoder = new TextEncoder();
    const encodedAssets = Object.keys(assets).sort().map((path) => {
      const value = assets[path];
      const bytes = typeof value === 'string' ? encoder.encode(value)
        : value instanceof Uint8Array ? value
        : value instanceof ArrayBuffer ? new Uint8Array(value)
        : null;
      if (!bytes) throw new TypeError(`application asset "${path}" must be a string or bytes`);
      return {path, bytes: Array.from(bytes)};
    });
    return withSource(JSON.stringify({source, sourceId, assets: encodedAssets}), (pointer, length) => {
      if (typeof api.vkf_compile_application !== 'function') {
        throw new Error('browser compiler does not support application bundles');
      }
      const response = checkedResponse(api.vkf_compile_application(pointer, length), 'application');
      const bundle = response.bundle;
      if (bundle?.schema !== 'vektor-flow/application-bundle' || bundle.version !== 1 ||
          !Array.isArray(bundle.staticAssets)) {
        throw new TypeError('invalid VKF application bundle');
      }
      return Object.freeze({
        ...bundle,
        staticAssets: bundle.staticAssets.map((asset) => Object.freeze({
          ...asset, bytes: Uint8Array.from(asset.bytes),
        })),
      });
    });
  }
  function dispatchApplication({sourceSha256, event} = {}) {
    if (typeof sourceSha256 !== 'string') {
      throw new TypeError('application dispatch sourceSha256 must be a string');
    }
    if (!event || typeof event !== 'object' || Array.isArray(event)) {
      throw new TypeError('application dispatch event must be an object');
    }
    return withSource(JSON.stringify({sourceSha256, event}), (pointer, length) => {
      if (typeof api.vkf_dispatch_application_event !== 'function') {
        throw new Error('browser compiler does not support application event dispatch');
      }
      const response = checkedResponse(
        api.vkf_dispatch_application_event(pointer, length), 'application-event');
      const dispatch = response.dispatch;
      if (dispatch?.schema !== 'vektor-flow/application-dispatch' ||
          dispatch.version !== 1 || dispatch.sourceSha256 !== sourceSha256 ||
          dispatch.packet?.kind !== 'display.replace') {
        throw new TypeError('invalid VKF application dispatch packet');
      }
      return Object.freeze(dispatch);
    });
  }
  return Object.freeze({
    compile,
    compileApplication,
    dispatchApplication,
    captureClockSnapshot,
    captureDocumentSnapshot() {
      const bytes = documentState ? Uint8Array.from(documentState)
        : freshDocumentBytes(32 + 64 * 8464);
      documentSequence += 1;
      new DataView(bytes.buffer).setUint32(8, documentSequence, true);
      return Object.freeze({
        schema: 'vektor-flow/document-arena', version: 1,
        sequence: documentSequence, bytes,
      });
    },
    resetDocuments() {
      documentState = null;
      documentSequence += 1;
    },
    describeTests(source, identity = '<browser>') {
      return withSource(source, (pointer, length) => withSource(identity, (identityPointer, identityLength) =>
        checkedResponse(api.vkf_describe_tests(pointer, length, identityPointer, identityLength))));
    },
    selectTestFiles(paths) {
      return withSource(JSON.stringify(paths), (pointer, length) =>
        checkedResponse(api.vkf_select_test_files(pointer, length))).files;
    },
    run(source, {clockSnapshot, documentSnapshot, measurePerformance = false} = {}) {
      const now = () => globalThis.performance?.now?.() ?? Date.now();
      const compileStarted = now();
      compile(source);
      const status = api.vkf_emit_program();
      const result = checkedResponse(status, 'lowering');
      const module = new WebAssembly.Module(new Uint8Array(api.memory.buffer,
        api.vkf_program_pointer(), api.vkf_program_length()));
      if (WebAssembly.Module.imports(module).length !== 0) {
        throw new Error('compiled browser program must not import host capabilities');
      }
      const programInstance = new WebAssembly.Instance(module);
      const programApi = programInstance.exports;
      const compileMs = now() - compileStarted;
      installClockSnapshot(programApi, clockSnapshot ?? captureClockSnapshot());
      const installedDocuments = installDocumentSnapshot(programApi, documentSnapshot);
      if (result.manifest?.schema !== 'vektor-flow.symbolic-kernel'
          || !result.manifest.functions || typeof result.manifest.functions !== 'object') {
        throw new TypeError('invalid VKF symbolic kernel manifest');
      }
      const entry = result.manifest.functions.$vkf_main;
      if (!entry) throw new RangeError('unknown VKF function "$vkf_main"');
      if (entry.parameters !== 0) {
        throw new RangeError(`$vkf_main expects ${entry.parameters} arguments, got 0`);
      }
      // Entry metadata and opaque addresses are transport. Never decode a VKF value.
      let invocationStatus;
      const executeStarted = now();
      try {
        invocationStatus = programApi.vkf_vm_invoke(entry.index, 0);
      } catch (error) {
        throw documentRuntimeError(programApi, installedDocuments.pointer) ?? error;
      }
      if (invocationStatus !== 0) {
        throw new Error(`VKF invocation "$vkf_main" failed with status ${invocationStatus}`);
      }
      const executeMs = now() - executeStarted;
      documentState = Uint8Array.from(new Uint8Array(programApi.memory.buffer,
        installedDocuments.pointer, installedDocuments.length));
      // Copy raw slots, preserving relative pointers and every numeric bit.
      // Native display formatting executes in the compiler WASM, not JavaScript.
      const used = Math.max(programApi.vkf_vm_heap_ptr(),
        programApi.vkf_vm_results_ptr() + programApi.vkf_vm_value_slot_size());
      const formatStarted = now();
      const memory = new Uint8Array(programApi.memory.buffer, 0, used);
      const pointer = api.malloc(memory.length);
      if (!pointer) throw new Error('browser compiler stdout allocation failed');
      try {
        new Uint8Array(api.memory.buffer, pointer, memory.length).set(memory);
        const retained = checkedResponse(api.vkf_format_retained_ui_packets(
          pointer, memory.length, programApi.vkf_vm_results_ptr()), 'output');
        const formatted = checkedResponse(api.vkf_format_stdout(pointer, memory.length,
          programApi.vkf_vm_results_ptr()), 'output');
        if (retained.retained_scene_arenas?.length) {
          const output = {
            kind: 'visual',
            stdout: formatted.stdout,
            stderr: '',
            retained_scene_arenas: retained.retained_scene_arenas.map((packet) => ({
              ...packet,
              arena: Uint8Array.from(packet.arena),
            })),
          };
          if (measurePerformance) output.timing = {
            compileMs, executeMs, formatMs: now() - formatStarted,
          };
          return output;
        }
        const output = {kind: 'console', stdout: formatted.stdout, stderr: ''};
        if (measurePerformance) output.timing = {
          compileMs, executeMs, formatMs: now() - formatStarted,
        };
        return output;
      } finally { api.free(pointer); }
    },
  });
}

export const PACKAGED_SHARED_COMPILER_URL = new URL(
  "./artifacts/vkf-shared-compiler.wasm",
  import.meta.url,
);

export async function loadSharedCompiler({
  wasm = PACKAGED_SHARED_COMPILER_URL,
  fetchImpl = globalThis.fetch,
  compileModule = globalThis.WebAssembly?.compile,
} = {}) {
  if (typeof fetchImpl !== "function" || typeof compileModule !== "function") {
    throw new Error("WebAssembly compiler loading is unavailable");
  }
  const response = await fetchImpl(wasm);
  if (!response.ok) throw new Error("VKF compiler WASM is unavailable");
  const module = await compileModule(await response.arrayBuffer());
  if (WebAssembly.Module.imports(module).length !== 0) {
    throw new Error("VKF compiler WASM must not import host capabilities");
  }
  return createSharedCompiler({ instance: new WebAssembly.Instance(module) });
}
