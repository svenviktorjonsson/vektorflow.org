// One isolated numeric VKF worker. Completion is published into shared memory
// so a parent VKF Web Worker can implement synchronous join without relying on
// a message callback that cannot run while that parent is blocked.
import {createBrowserConcurrentImport} from './vkf-concurrent-host.mjs';
import {captureValue, materializeValue, encodeValueGraph, decodeValueGraph}
  from './vkf-concurrent-values.mjs';
import {readSharedSnapshot} from './vkf-concurrent-shared.mjs';
const nodeThreads = typeof process !== 'undefined' && process.versions?.node
  ? await import('node:worker_threads') : null;
const WorkerType = nodeThreads?.Worker ?? globalThis.Worker;
const RESULT_OFFSET = 24;

globalThis.onmessage = ({data}) => {
  const state = new Int32Array(data.completion, 0, 2);
  let host;
  let api;
  try {
    if (!(data.module instanceof WebAssembly.Module) ||
        !Number.isInteger(data.functionIndex) || data.functionIndex < 0 ||
        (typeof data.argument !== 'number' &&
         data.argumentGraph?.schema !== 'vektor-flow/concurrent-value-v1' &&
         !data.sharedSnapshot)) {
      throw new TypeError('invalid process launch');
    }
    const imports = WebAssembly.Module.imports(data.module);
    host = imports.length === 1 && imports[0].module === 'vkf.concurrent' &&
      imports[0].name === 'invoke' && imports[0].kind === 'function'
      ? createBrowserConcurrentImport(data.module, () => api,
          WorkerType, data.brokerPort ?? null, data.sharedBackings ?? []) : {};
    api = new WebAssembly.Instance(data.module, host).exports;
    const hasGraph = Boolean(data.argumentGraph || data.sharedSnapshot);
    const workerArity = hasGraph ? data.workerArity : 1;
    const slotSize = api.vkf_vm_value_slot_size();
    if (slotSize < 24) throw new RangeError('process VM lacks complex-capable external slots');
    if (!Number.isInteger(workerArity) || workerArity < 0 ||
        api.vkf_vm_arguments_capacity() < workerArity) {
      throw new RangeError('numeric process argument slot is unavailable');
    }
    const argument = new DataView(api.memory.buffer, api.vkf_vm_arguments_ptr(), slotSize);
    if (hasGraph) {
      const graph = data.sharedSnapshot
        ? decodeValueGraph(readSharedSnapshot(data.sharedSnapshot).bytes)
        : data.argumentGraph;
      const pointer = materializeValue(api, graph);
      if (workerArity === 1) {
        const width = new DataView(api.memory.buffer).getUint32(pointer, true) === 10 ? 24 : 16;
        new Uint8Array(api.memory.buffer, api.vkf_vm_arguments_ptr(), width)
          .set(new Uint8Array(api.memory.buffer, pointer, width));
      } else {
        const tuple = new DataView(api.memory.buffer, pointer, 16);
        if (tuple.getUint32(0, true) !== 6 ||
            tuple.getUint32(4, true) !== workerArity) {
          throw new TypeError('process arguments do not match worker arity');
        }
        const members = tuple.getUint32(8, true);
        for (let index = 0; index < workerArity; ++index) {
          const member = new DataView(api.memory.buffer).getUint32(members + index * 4, true);
          const width = new DataView(api.memory.buffer).getUint32(member, true) === 10 ? 24 : 16;
          new Uint8Array(api.memory.buffer, api.vkf_vm_arguments_ptr() + index * slotSize, width)
            .set(new Uint8Array(api.memory.buffer, member, width));
        }
      }
    } else {
      argument.setUint32(0, 2, true);
      argument.setUint32(4, 0, true);
      argument.setFloat64(8, data.argument, true);
    }
    if (api.vkf_vm_invoke(data.functionIndex, workerArity) !== 0) {
      throw new Error('numeric process worker failed');
    }
    // Shared descriptors are broker-owned and remain valid after this Worker
    // exits. Returning one must not expand the complete backing in the child.
    const result = new DataView(api.memory.buffer, api.vkf_vm_results_ptr(), slotSize);
    if (hasGraph) {
      const bytes = encodeValueGraph(captureValue(
        api.memory, api.vkf_vm_results_ptr(), api.vkf_vm_heap_ptr()));
      if (bytes.length > data.completion.byteLength - RESULT_OFFSET) {
        new DataView(data.completion).setUint32(8, host.publishResult(bytes), true);
        Atomics.store(state, 1, -1);
      } else {
        new Uint8Array(data.completion, RESULT_OFFSET, bytes.length).set(bytes);
        Atomics.store(state, 1, bytes.length);
      }
    } else {
      if (result.getUint32(0, true) !== 2) {
        throw new TypeError('numeric process worker returned a non-numeric value');
      }
      new DataView(data.completion).setFloat64(8, result.getFloat64(8, true), true);
    }
    Atomics.compareExchange(state, 0, 0, 1);
  } catch (failure) {
    try {
      const control = api?.vkf_vm_error_control_ptr?.value;
      if (!Number.isInteger(control)) throw new Error('missing typed error control');
      const error = new DataView(api.memory.buffer, control, 28);
      const mask = error.getUint32(24, true);
      const pointer = error.getUint32(20, true);
      if (!mask || !pointer) throw new Error('worker failed without a VKF error');
      const bytes = encodeValueGraph(captureValue(
        api.memory, pointer, api.vkf_vm_heap_ptr()));
      if (bytes.length > data.completion.byteLength - RESULT_OFFSET) {
        new DataView(data.completion).setUint32(8, host.publishResult(bytes), true);
        Atomics.store(state, 1, -1);
      } else {
        new Uint8Array(data.completion, RESULT_OFFSET, bytes.length).set(bytes);
        Atomics.store(state, 1, bytes.length);
      }
      new DataView(data.completion).setUint32(16, mask, true);
      Atomics.compareExchange(state, 0, 0, -3);
    } catch {
      const message = new TextEncoder().encode(
        failure instanceof Error ? failure.message : 'browser process failed');
      const capacity = data.completion.byteLength - RESULT_OFFSET;
      const length = Math.min(message.length, capacity);
      new Uint8Array(data.completion, RESULT_OFFSET, length)
        .set(message.subarray(0, length));
      Atomics.store(state, 1, length);
      Atomics.compareExchange(state, 0, 0, -4);
    }
  } finally {
    host?.close?.();
    Atomics.notify(state, 0);
    globalThis.close?.();
  }
};

// Node's worker_threads provides the release test host for the same worker
// module. The browser branch has no Node import or process dependency.
nodeThreads?.parentPort?.on('message', (data) => globalThis.onmessage({data}));
