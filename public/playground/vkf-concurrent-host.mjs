import {createBrokerClient} from './vkf-concurrent-broker-client.mjs';
import {startValueProcess} from './vkf-concurrent-process.mjs';
import {captureValue, materializeValue, encodeValueGraph, decodeValueGraph}
  from './vkf-concurrent-values.mjs';
import {publishSharedGraph, readSharedNode} from './vkf-concurrent-shared-graph.mjs';

// The compiler emits this import only for programs containing a concurrent
// intrinsic. The host owns worker and channel resources, while value layout
// remains the compiler's tagged-slot ABI.
export function createBrowserConcurrentImport(
  module, exportsOf, WorkerType = globalThis.Worker, inheritedPort = null,
  inheritedSharedBackings = []
) {
  if (!(module instanceof WebAssembly.Module) || typeof exportsOf !== 'function') {
    throw new TypeError('concurrent import requires a compiled module and instance');
  }
  const resources = new Map();
  // Structured cloning a SharedArrayBuffer passes the same backing to a
  // child Worker; the large numeric payload need not cross the broker reply
  // cell or be reconstructed inside the child's WebAssembly heap.
  const sharedValues = new Map(inheritedSharedBackings);
  const sharedNodeCache = new Map();
  const numericValues = (handle) => {
    const buffer = sharedValues.get(handle);
    return buffer && buffer.kind === 'numeric' ? new Float64Array(buffer.buffer) : null;
  };
  const broker = createBrokerClient({WorkerType, port: inheritedPort});
  let nextProcessHandle = -1;
  const api = () => {
    const value = exportsOf();
    if (!value) throw new Error('concurrent program instance is unavailable');
    return value;
  };
  const handleTags = {process: 15, queue: 16, broadcast: 17};
  const numberAt = (pointer, kind) => {
    const view = new DataView(api().memory.buffer, pointer, 16);
    if (view.getUint32(0, true) !== handleTags[kind]) {
      throw new TypeError(`concurrent ${kind} handle has the wrong value type`);
    }
    return view.getFloat64(8, true);
  };
  const resourceAt = (pointer, kind) => {
    const handle = numberAt(pointer, kind);
    if (!Number.isSafeInteger(handle) || !resources.has(handle) ||
        resources.get(handle).kind !== kind) {
      throw new TypeError(`invalid concurrent ${kind} handle`);
    }
    return resources.get(handle).value;
  };
  const allocate = (tag, value = 0) => {
    const pointer = api().vkf_vm_alloc(16);
    if (!pointer) throw new RangeError('concurrent result allocation failed');
    const view = new DataView(api().memory.buffer, pointer, 16);
    view.setUint32(0, tag, true);
    view.setUint32(4, 0, true);
    if (tag === 1) view.setUint32(8, value ? 1 : 0, true);
    else view.setFloat64(8, value, true);
    return pointer;
  };
  const handleFor = (kind, value) => {
    const handle = nextProcessHandle--;
    resources.set(handle, {kind, value});
    return allocate(handleTags[kind], handle);
  };
  const graphNode = (handle, index) => {
    const backing = sharedValues.get(handle);
    if (backing?.kind === 'graph') return readSharedNode(backing.buffer, index);
    const key = `${handle}:${index}`;
    if (!sharedNodeCache.has(key)) {
      sharedNodeCache.set(key, broker.sharedNode(handle, index));
    }
    return sharedNodeCache.get(key);
  };
  const graphReference = (handle, index) => {
    const node = graphNode(handle, index);
    const pointer = api().vkf_vm_alloc(16);
    if (!pointer) throw new RangeError('shared graph reference allocation failed');
    const slot = new DataView(api().memory.buffer, pointer, 16);
    slot.setUint32(0, 14, true);
    slot.setUint32(4, node.count, true);
    slot.setUint32(8, handle, true);
    slot.setUint32(12, index, true);
    return pointer;
  };
  const materializeGraphNode = (handle, index, seen = new Map()) => {
    if (seen.has(index)) return seen.get(index);
    const node = graphNode(handle, index);
    const width = node.tag === 10 ? 24 : 16;
    const pointer = api().vkf_vm_alloc(width);
    if (!pointer) throw new RangeError('shared graph node allocation failed');
    seen.set(index, pointer);
    new Uint8Array(api().memory.buffer, pointer, width).set(node.slot.subarray(0, width));
    if (node.tag === 3) {
      const bytes = api().vkf_vm_alloc(Math.max(1, node.bytes.length));
      if (!bytes) throw new RangeError('shared string allocation failed');
      new Uint8Array(api().memory.buffer, bytes, node.bytes.length).set(node.bytes);
      new DataView(api().memory.buffer).setUint32(pointer + 8, bytes, true);
    } else if ([4, 5, 6, 9, 11].includes(node.tag)) {
      const bytes = api().vkf_vm_alloc(Math.max(4, node.children.length * 4));
      if (!bytes) throw new RangeError('shared collection allocation failed');
      for (let child = 0; child < node.children.length; ++child) {
        new DataView(api().memory.buffer).setUint32(bytes + child * 4,
          materializeGraphNode(handle, node.children[child], seen), true);
      }
      new DataView(api().memory.buffer).setUint32(pointer + 8, bytes, true);
    } else if (node.tag === 8 || node.tag === 12) {
      const view = new DataView(api().memory.buffer);
      view.setUint32(pointer + 4,
        materializeGraphNode(handle, node.children[0], seen), true);
      view.setUint32(pointer + 8,
        materializeGraphNode(handle, node.children[1], seen), true);
    }
    return pointer;
  };
  const shallowGraphNode = (handle, index) => {
    const node = graphNode(handle, index);
    return [4, 5, 6, 9, 11].includes(node.tag)
      ? graphReference(handle, index)
      : materializeGraphNode(handle, index);
  };
  const materializeShared = (pointer) => {
    const slot = new DataView(api().memory.buffer, pointer, 16);
    if (slot.getUint32(0, true) === 14) {
      return materializeGraphNode(slot.getUint32(8, true), slot.getUint32(12, true));
    }
    if (slot.getUint32(0, true) !== 13) return pointer;
    const values = numericValues(slot.getUint32(8, true));
    const count = slot.getUint32(4, true);
    if (values && values.length !== count) throw new TypeError('shared vector shape mismatch');
    const result = api().vkf_vm_alloc(16);
    const members = api().vkf_vm_alloc(Math.max(4, count * 4));
    if (!result || !members) throw new RangeError('shared vector materialization failed');
    for (let index = 0; index < count; ++index) {
      const member = allocate(2, values
        ? values[index] : broker.sharedNumber(slot.getUint32(8, true), index, count));
      new DataView(api().memory.buffer).setUint32(members + index * 4, member, true);
    }
    const record = new DataView(api().memory.buffer, result, 16);
    record.setUint32(0, 4, true);
    record.setUint32(4, count, true);
    record.setUint32(8, members, true);
    return result;
  };
  const invoke = (operation, argumentsPointer, argumentCount, workerIndex) => {
    if (operation === 11) return materializeShared(argumentsPointer);
    if (operation === 10) {
      const view = new DataView(api().memory.buffer, argumentsPointer, 16);
      if (workerIndex >= view.getUint32(4, true)) {
        throw new RangeError('invalid shared vector access');
      }
      if (view.getUint32(0, true) === 14) {
        const node = graphNode(view.getUint32(8, true), view.getUint32(12, true));
        if (![4, 5, 6, 9, 11].includes(node.tag)) {
          throw new TypeError('shared graph node is not indexable');
        }
        const child = (node.tag === 5 || node.tag === 9)
          ? node.children[workerIndex * 2 + 1] : node.children[workerIndex];
        return shallowGraphNode(view.getUint32(8, true), child);
      }
      if (view.getUint32(0, true) !== 13) {
        throw new TypeError('value is not a shared vector');
      }
      const values = numericValues(view.getUint32(8, true));
      if (values && values.length !== view.getUint32(4, true)) {
        throw new TypeError('shared vector shape mismatch');
      }
      return allocate(2, values
        ? values[workerIndex] : broker.sharedNumber(view.getUint32(8, true),
          workerIndex, view.getUint32(4, true)));
    }
    if (operation === 12) {
      const view = new DataView(api().memory.buffer, argumentsPointer, 16);
      if (view.getUint32(0, true) !== 14) throw new TypeError('value is not a shared record');
      const handle = view.getUint32(8, true);
      const node = graphNode(handle, view.getUint32(12, true));
      if (node.tag !== 5) throw new TypeError('shared graph node is not a record');
      const key = new DataView(api().memory.buffer, workerIndex, 16);
      const keyBytes = new Uint8Array(api().memory.buffer,
        key.getUint32(8, true), key.getUint32(4, true));
      for (let field = 0; field < node.children.length; field += 2) {
        const name = graphNode(handle, node.children[field]);
        if (name.bytes.length === keyBytes.length &&
            name.bytes.every((byte, index) => byte === keyBytes[index])) {
          return shallowGraphNode(handle, node.children[field + 1]);
        }
      }
      return allocate(0);
    }
    const operands = new Uint32Array(api().memory.buffer, argumentsPointer, argumentCount);
    const pointers = Array.from(operands);
    switch (operation) {
      case 0: return allocate(16, broker.createQueue());
      case 1: return allocate(17, broker.createBroadcast());
      case 2:
        broker.putQueue(numberAt(pointers[0], 'queue'), encodeValueGraph(
          captureValue(api().memory, pointers[1], api().vkf_vm_heap_ptr())));
        return allocate(0);
      case 3: return materializeValue(api(), decodeValueGraph(
        broker.getQueue(numberAt(pointers[0], 'queue'))));
      case 4:
        broker.putBroadcast(numberAt(pointers[0], 'broadcast'), encodeValueGraph(
          captureValue(api().memory, pointers[1], api().vkf_vm_heap_ptr())));
        return allocate(0);
      case 5:
        return allocate(16, broker.subscribe(numberAt(pointers[0], 'broadcast')));
      case 6: {
        const workerArity = workerIndex & 255;
        const functionIndex = workerIndex >>> 8;
        const child = broker.createChildPort();
        try {
          const argumentGraph = captureValue(api().memory, pointers[0],
            api().vkf_vm_heap_ptr());
          const sharedBackings = [];
          const seenBackings = new Set();
          const includeBacking = (handle) => {
            if (seenBackings.has(handle)) return;
            seenBackings.add(handle);
            const backing = sharedValues.get(handle);
            if (!backing) return; // Broker demand-read remains valid for remote handles.
            sharedBackings.push([handle, backing]);
            for (const nested of backing.references ?? []) includeBacking(nested);
          };
          for (const node of argumentGraph.nodes) {
            if (node.tag !== 13 && node.tag !== 14) continue;
            const slot = Uint8Array.from(node.slot);
            const handle = new DataView(slot.buffer).getUint32(8, true);
            includeBacking(handle);
          }
          return handleFor('process', startValueProcess(
            module, functionIndex, argumentGraph,
            {WorkerType,
             brokerPort: child.port, sharedBackings, workerArity,
             readResult: broker.readTransport,
             onClose: () => broker.detachChild(child.id)}));
        } catch (error) {
          child.port.close?.();
          broker.detachChild(child.id);
          throw error;
        }
      }
      case 7: {
        const handle = numberAt(pointers[0], 'process');
        const process = resourceAt(pointers[0], 'process');
        const result = process.join();
        resources.delete(handle);
        const value = materializeValue(api(), decodeValueGraph(result.bytes));
        if (result.errorMask) {
          const control = api().vkf_vm_error_control_ptr?.value;
          if (!Number.isInteger(control)) throw new Error('typed VM error channel is unavailable');
          const view = new DataView(api().memory.buffer);
          view.setUint32(control + 20, value, true);
          view.setUint32(control + 24, result.errorMask, true);
          return allocate(0);
        }
        return value;
      }
      case 8: {
        const stopped = resourceAt(pointers[0], 'process').kill();
        // Keep the local handle tombstone: aliases observe a second kill as
        // false, while join still reports that the process was killed.
        return allocate(1, stopped ? 1 : 0);
      }
      case 9: {
        const pointer = pointers[0];
        const view = new DataView(api().memory.buffer, pointer, 16);
        if ([0, 1, 2, 10, 13, 14, 16, 17].includes(view.getUint32(0, true))) {
          return pointer; // Scalars and existing shared descriptors need no republishing.
        }
        if (view.getUint32(0, true) === 4) {
          const count = view.getUint32(4, true);
          const members = view.getUint32(8, true);
          const numbers = new Float64Array(new SharedArrayBuffer(count * 8));
          let numeric = true;
          for (let index = 0; index < count; ++index) {
            const item = new DataView(api().memory.buffer)
              .getUint32(members + index * 4, true);
            const scalar = new DataView(api().memory.buffer, item, 16);
            if (scalar.getUint32(0, true) !== 2) { numeric = false; break; }
            numbers[index] = scalar.getFloat64(8, true);
          }
          if (numeric) {
            const handle = broker.publishShared('numeric', numbers.buffer);
            sharedValues.set(handle, {kind: 'numeric', buffer: numbers.buffer});
            const result = allocate(13);
            const slot = new DataView(api().memory.buffer, result, 16);
            slot.setUint32(4, count, true);
            slot.setUint32(8, handle, true);
            return result;
          }
        }
        const graph = captureValue(api().memory, pointer, api().vkf_vm_heap_ptr());
        const published = publishSharedGraph(graph);
        const handle = broker.publishShared('graph', published.buffer);
        const references = graph.nodes.filter((node) => node.tag === 13 || node.tag === 14)
          .map((node) => new DataView(Uint8Array.from(node.slot).buffer)
            .getUint32(8, true));
        sharedValues.set(handle, {kind: 'graph', buffer: published.buffer, references});
        return graphReference(handle, published.root);
      }
      default: throw new RangeError(`unknown browser concurrent operation ${operation}`);
    }
  };
  return Object.freeze({
    'vkf.concurrent': Object.freeze({invoke}),
    publishResult(bytes) {
      if (!(bytes instanceof Uint8Array)) throw new TypeError('process result must be bytes');
      const backing = new SharedArrayBuffer(bytes.length);
      new Uint8Array(backing).set(bytes);
      return broker.publishShared('transport', backing);
    },
    materializeOutput() {
      const seen = new Set();
      const visit = (address) => {
        if (seen.has(address)) return;
        seen.add(address);
        let view = new DataView(api().memory.buffer, address, 16);
        let tag = view.getUint32(0, true);
        if (tag === 13 || tag === 14) {
          const ordinary = materializeShared(address);
          new Uint8Array(api().memory.buffer, address, 16)
            .set(new Uint8Array(api().memory.buffer, ordinary, 16));
          view = new DataView(api().memory.buffer, address, 16);
          tag = new DataView(api().memory.buffer, address, 4).getUint32(0, true);
        }
        const count = view.getUint32(4, true);
        const payload = view.getUint32(8, true);
        if (tag === 4 || tag === 6 || tag === 11) {
          for (let index = 0; index < count; ++index) {
            visit(new DataView(api().memory.buffer)
              .getUint32(payload + index * 4, true));
          }
        } else if (tag === 5 || tag === 9) {
          for (let index = 0; index < count; ++index) {
            const entry = payload + index * 8;
            visit(new DataView(api().memory.buffer).getUint32(entry, true));
            visit(new DataView(api().memory.buffer).getUint32(entry + 4, true));
          }
        } else if (tag === 8 || tag === 12) {
          visit(count);
          visit(payload);
        }
      };
      visit(api().vkf_vm_results_ptr());
    },
    close() {
      for (const resource of resources.values()) resource.value.kill();
      resources.clear();
      broker.close();
    },
  });
}
