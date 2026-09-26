const WORKER_URL = globalThis.__vkfConcurrentWorkerScope
  ? new URL('process-worker.mjs', globalThis.__vkfConcurrentWorkerScope)
  : new URL('./vkf-concurrent-worker.mjs', import.meta.url);
const RESULT_CAPACITY = 4 * 1024 * 1024;
const RESULT_OFFSET = 24;

export function startValueProcess(module, functionIndex, argumentGraph, {
  WorkerType = globalThis.Worker, sharedSnapshot = null, brokerPort = null,
  sharedBackings = [], workerArity = 1, readResult = null, onClose = () => {},
} = {}) {
  if (!(module instanceof WebAssembly.Module) ||
      !Number.isSafeInteger(functionIndex) || functionIndex < 0 ||
      argumentGraph?.schema !== 'vektor-flow/concurrent-value-v1' ||
      !Number.isSafeInteger(workerArity) || workerArity < 0 || workerArity > 255 ||
      typeof WorkerType !== 'function' || typeof SharedArrayBuffer !== 'function') {
    throw new TypeError('browser process requires a worker, shared memory and one VKF value');
  }
  const completion = new SharedArrayBuffer(RESULT_OFFSET + RESULT_CAPACITY);
  const state = new Int32Array(completion, 0, 2);
  let worker = new WorkerType(WORKER_URL, {type: 'module'});
  try {
    worker.postMessage({module, functionIndex,
      argumentGraph: sharedSnapshot ? null : argumentGraph, workerArity,
      sharedSnapshot, sharedBackings, brokerPort, completion},
      brokerPort ? [brokerPort] : []);
  } catch (error) {
    worker.terminate();
    throw error;
  }
  let joined = false;
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    worker?.terminate();
    worker = null;
    onClose();
  };
  return Object.freeze({
    join() {
      if (joined) throw new Error('process result has already been joined');
      if (typeof window !== 'undefined' && globalThis === window) {
        throw new Error('blocking process join must run in a Web Worker');
      }
      while (Atomics.load(state, 0) === 0) Atomics.wait(state, 0, 0);
      joined = true;
      close();
      const status = Atomics.load(state, 0);
      if (status !== 1 && status !== -3 && status !== -4) {
        throw new Error('process was killed or failed');
      }
      const length = Atomics.load(state, 1);
      let bytes;
      if (length === -1) {
        if (typeof readResult !== 'function') {
          throw new Error('process result requires broker-backed transport');
        }
        bytes = readResult(new DataView(completion).getUint32(8, true));
      } else {
        if (length < 0 || length > RESULT_CAPACITY) {
          throw new Error('invalid process result length');
        }
        bytes = Uint8Array.from(new Uint8Array(completion, RESULT_OFFSET, length));
      }
      if (status === -4) {
        throw new Error(new TextDecoder().decode(bytes));
      }
      return {bytes, errorMask: status === -3
        ? new DataView(completion).getUint32(16, true) : 0};
    },
    kill() {
      if (Atomics.compareExchange(state, 0, 0, -2) !== 0) return false;
      close();
      Atomics.notify(state, 0);
      return true;
    },
  });
}

export function startNumericProcess(module, functionIndex, argument, WorkerType = globalThis.Worker) {
  if (!(module instanceof WebAssembly.Module) ||
      !Number.isSafeInteger(functionIndex) || functionIndex < 0 ||
      typeof argument !== 'number') {
    throw new TypeError('numeric process requires a compiled worker and one number');
  }
  if (typeof WorkerType !== 'function' || typeof SharedArrayBuffer !== 'function') {
    throw new Error('browser process requires Web Workers and shared memory');
  }
  const completion = new SharedArrayBuffer(16);
  const state = new Int32Array(completion, 0, 2);
  let worker = new WorkerType(WORKER_URL, {type: 'module'});
  try {
    worker.postMessage({module, functionIndex, argument, completion});
  } catch (error) {
    worker.terminate();
    throw error;
  }
  let joined = false;
  return Object.freeze({
    join() {
      if (joined) throw new Error('process result has already been joined');
      if (typeof window !== 'undefined' && globalThis === window) {
        throw new Error('blocking process join must run in a Web Worker');
      }
      while (Atomics.load(state, 0) === 0) Atomics.wait(state, 0, 0);
      joined = true;
      worker?.terminate();
      worker = null;
      if (Atomics.load(state, 0) !== 1) throw new Error('process was killed or failed');
      return new DataView(completion).getFloat64(8, true);
    },
    kill() {
      if (Atomics.compareExchange(state, 0, 0, -2) !== 0) return false;
      worker?.terminate();
      worker = null;
      Atomics.notify(state, 0);
      return true;
    },
  });
}
