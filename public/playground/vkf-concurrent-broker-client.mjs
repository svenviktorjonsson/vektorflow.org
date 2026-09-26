const BROKER_URL = globalThis.__vkfConcurrentWorkerScope
  ? new URL('broker-worker.mjs', globalThis.__vkfConcurrentWorkerScope)
  : new URL('./vkf-concurrent-broker.mjs', import.meta.url);
const MAX_REPLY_BYTES = 256 * 1024 * 1024;
const decoder = new TextDecoder();

export function createBrokerClient({WorkerType = globalThis.Worker,
  ChannelType = globalThis.MessageChannel, port = null} = {}) {
  if (typeof SharedArrayBuffer !== 'function' || typeof ChannelType !== 'function') {
    throw new Error('concurrent broker requires shared memory and MessageChannel');
  }
  let broker = null;
  let clientPort = port;
  if (!clientPort) {
    if (typeof WorkerType !== 'function') throw new Error('concurrent broker requires Web Workers');
    broker = new WorkerType(BROKER_URL, {type: 'module'});
    const channel = new ChannelType();
    clientPort = channel.port2;
    broker.postMessage({op: 'attach', port: channel.port1}, [channel.port1]);
  }
  const request = (op, data = {}, transfers = [], capacity = 4096) => {
    if (typeof window !== 'undefined' && globalThis === window) {
      throw new Error('blocking concurrent operation must run in a Web Worker');
    }
    if (!Number.isSafeInteger(capacity) || capacity < 0 || capacity > MAX_REPLY_BYTES) {
      throw new RangeError('concurrent broker reply exceeds value graph limit');
    }
    const cell = new SharedArrayBuffer(16 + capacity);
    const state = new Int32Array(cell, 0, 4);
    clientPort.postMessage({op, cell, ...data}, transfers);
    while (Atomics.load(state, 0) === 0) Atomics.wait(state, 0, 0);
    const length = Atomics.load(state, 1);
    if (length < 0 || length > capacity) throw new RangeError('invalid broker reply length');
    const bytes = Uint8Array.from(new Uint8Array(cell, 16, length));
    if (Atomics.load(state, 0) !== 1) throw new Error(decoder.decode(bytes));
    return {kind: Atomics.load(state, 2), bytes};
  };
  const handle = (reply) => {
    if (reply.kind !== 1 || reply.bytes.length !== 8) {
      throw new TypeError('invalid concurrent handle reply');
    }
    return new DataView(reply.bytes.buffer).getFloat64(0, true);
  };
  const numberWindows = new Map();
  const sharedRange = (id, index, count) => {
    if (!Number.isSafeInteger(count) || count < 0 || count * 8 > MAX_REPLY_BYTES) {
      throw new RangeError('invalid shared range length');
    }
    const reply = request('shared.range', {handle: id, index, count}, [], count * 8);
    if (reply.kind !== 2 || reply.bytes.length !== count * 8) {
      throw new TypeError('invalid shared numeric range reply');
    }
    return new Float64Array(reply.bytes.buffer,
      reply.bytes.byteOffset, count);
  };
  return Object.freeze({
    createQueue: () => handle(request('queue.create')),
    createBroadcast: () => handle(request('broadcast.create')),
    subscribe: (id) => handle(request('broadcast.subscribe', {handle: id})),
    putQueue: (id, bytes) => request('queue.put', {handle: id, bytes}),
    reserveQueue: (id) => handle(request('queue.get', {handle: id})),
    takeQueue: (id, size) => {
      const reply = request('queue.take', {handle: id}, [], size);
      if (reply.kind !== 2) throw new TypeError('invalid concurrent queue value reply');
      return reply.bytes;
    },
    getQueue: (id) => {
      const size = handle(request('queue.get', {handle: id}));
      const reply = request('queue.take', {handle: id}, [], size);
      if (reply.kind !== 2) throw new TypeError('invalid concurrent queue value reply');
      return reply.bytes;
    },
    putBroadcast: (id, bytes) => request('broadcast.put', {handle: id, bytes}),
    publishShared: (kind, buffer) => handle(request('shared.publish', {kind, buffer})),
    readTransport: (id) => {
      const size = handle(request('shared.transport-size', {handle: id}));
      if (!Number.isSafeInteger(size) || size < 0 || size > 0x7fffffff) {
        throw new RangeError('process result exceeds transport value limit');
      }
      const result = new Uint8Array(size);
      for (let index = 0; index < size; index += 1024 * 1024) {
        const count = Math.min(1024 * 1024, size - index);
        const reply = request('shared.transport-chunk',
          {handle: id, index, count}, [], count);
        if (reply.kind !== 2 || reply.bytes.length !== count) {
          throw new TypeError('invalid process result chunk');
        }
        result.set(reply.bytes, index);
      }
      return result;
    },
    sharedRange,
    sharedNumber: (id, index, length) => {
      if (!Number.isSafeInteger(index) || !Number.isSafeInteger(length) ||
          index < 0 || index >= length) {
        throw new RangeError('shared numeric index out of bounds');
      }
      const initialWindow = 256;
      const maximumWindow = 8192;
      let window = numberWindows.get(id);
      if (!window || index < window.start ||
          index >= window.start + window.values.length) {
        // Child-origin shared buffers may be known only to the broker. Grow
        // demand reads for sequential scans, while random access keeps a
        // small initial fetch instead of copying the whole backing.
        const sequential = window && index === window.start + window.values.length;
        const size = sequential
          ? Math.min(window.size * 2, maximumWindow) : initialWindow;
        const start = sequential ? index : Math.floor(index / initialWindow) * initialWindow;
        window = {start, size,
          values: sharedRange(id, start, Math.min(size, length - start))};
        numberWindows.set(id, window);
      }
      return window.values[index - window.start];
    },
    sharedNode: (id, index) => {
      const size = handle(request('shared.node-size', {handle: id, index}));
      const reply = request('shared.node', {handle: id, index}, [], size);
      if (reply.kind !== 2 || reply.bytes.length !== size) {
        throw new TypeError('invalid shared graph node reply');
      }
      const view = new DataView(reply.bytes.buffer);
      const children = view.getUint32(24, true);
      const byteLength = view.getUint32(28, true);
      if (32 + children * 4 + byteLength !== size) {
        throw new TypeError('shared graph node reply is malformed');
      }
      return {
        tag: view.getUint32(0, true), count: view.getUint32(4, true),
        slot: reply.bytes.subarray(0, 24),
        children: Array.from({length: children}, (_, child) =>
          view.getUint32(32 + child * 4, true)),
        bytes: reply.bytes.subarray(32 + children * 4),
      };
    },
    createChildPort() {
      const channel = new ChannelType();
      const id = handle(request('attach-child', {port: channel.port1}, [channel.port1]));
      return {port: channel.port2, id};
    },
    detachChild: (id) => request('detach-client', {target: id}),
    close() {
      try { request('detach-self'); } catch { /* broker may have exited */ }
      clientPort.close?.();
      broker?.terminate();
    },
  });
}
