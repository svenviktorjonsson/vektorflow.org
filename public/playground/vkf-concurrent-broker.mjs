// Event-driven channel owner. Clients may block in Atomics.wait, but this
// dedicated Worker keeps receiving MessagePort requests and never polls.
import {readSharedNode} from './vkf-concurrent-shared-graph.mjs';
const encoder = new TextEncoder();
const queues = new Map();
const broadcasts = new Map();
const clients = new Map();
const sharedBackings = new Map();
let nextId = 1;
let nextClientId = 1;

function reply(cell, kind, value) {
  const state = new Int32Array(cell, 0, 4);
  try {
    let bytes = new Uint8Array();
    if (kind === 1) {
      bytes = new Uint8Array(8);
      new DataView(bytes.buffer).setFloat64(0, value, true);
    } else if (kind === 2) {
      bytes = value;
    }
    if (bytes.length > cell.byteLength - 16) {
      throw new RangeError('concurrent message exceeds broker reply capacity');
    }
    new Uint8Array(cell, 16, bytes.length).set(bytes);
    Atomics.store(state, 1, bytes.length);
    Atomics.store(state, 2, kind);
    Atomics.store(state, 0, 1);
  } catch (error) {
    const bytes = encoder.encode(error.message);
    const length = Math.min(bytes.length, cell.byteLength - 16);
    new Uint8Array(cell, 16, length).set(bytes.subarray(0, length));
    Atomics.store(state, 1, length);
    Atomics.store(state, 0, -1);
  }
  Atomics.notify(state, 0);
}

function fail(cell, message) {
  const state = new Int32Array(cell, 0, 4);
  const bytes = encoder.encode(message);
  const length = Math.min(bytes.length, cell.byteLength - 16);
  new Uint8Array(cell, 16, length).set(bytes.subarray(0, length));
  Atomics.store(state, 1, length);
  Atomics.store(state, 0, -1);
  Atomics.notify(state, 0);
}

function queueFor(id) {
  const queue = queues.get(id);
  if (!queue) throw new TypeError('invalid concurrent queue handle');
  return queue;
}

function sharedFor(id, kind) {
  const backing = sharedBackings.get(id);
  if (!backing || backing.kind !== kind) {
    throw new TypeError('invalid immutable shared backing handle');
  }
  return backing.buffer;
}

function encodeSharedNode(node) {
  const bytes = new Uint8Array(32 + node.children.length * 4 + node.bytes.length);
  const view = new DataView(bytes.buffer);
  bytes.set(node.slot, 0);
  view.setUint32(24, node.children.length, true);
  view.setUint32(28, node.bytes.length, true);
  node.children.forEach((child, index) =>
    view.setUint32(32 + index * 4, child, true));
  bytes.set(node.bytes, 32 + node.children.length * 4);
  return bytes;
}

function put(queue, bytes) {
  const item = {sequence: queue.nextSequence++, bytes};
  if (queue.waiters.length) {
    const waiter = queue.waiters.shift();
    queue.reserved.set(waiter.clientId, item);
    reply(waiter.cell, 1, bytes.length);
  } else queue.items.push(item);
}

function detach(clientId) {
  const client = clients.get(clientId);
  if (!client) return;
  clients.delete(clientId);
  for (const queue of queues.values()) {
    const reserved = queue.reserved.get(clientId);
    if (reserved) {
      queue.reserved.delete(clientId);
      const insertion = queue.items.findIndex((item) =>
        item.sequence > reserved.sequence);
      queue.items.splice(insertion < 0 ? queue.items.length : insertion, 0, reserved);
    }
    queue.waiters = queue.waiters.filter((waiter) => {
      if (waiter.clientId !== clientId) return true;
      fail(waiter.cell, 'concurrent queue receiver exited');
      return false;
    });
  }
  // A subscriber queue is a transferable VKF value. It may have been sent to
  // a parent through join/queue before this client's exit; deleting it here
  // would invalidate that handle. All channel backing lives until the broker
  // execution context closes (no public close/generation API exists yet).
  client.port.close?.();
}

function dispatch(request, clientId) {
  const {op, cell, handle} = request;
  try {
    if (!(cell instanceof SharedArrayBuffer)) throw new TypeError('missing broker reply cell');
    if (!clients.has(clientId)) throw new Error('concurrent client has exited');
    switch (op) {
      case 'queue.create': {
        const id = nextId++;
        queues.set(id, {items: [], waiters: [], reserved: new Map(), nextSequence: 0});
        reply(cell, 1, id);
        break;
      }
      case 'broadcast.create': {
        const id = nextId++;
        broadcasts.set(id, new Set());
        reply(cell, 1, id);
        break;
      }
      case 'queue.put':
        put(queueFor(handle), request.bytes);
        reply(cell, 0);
        break;
      case 'queue.get': {
        const queue = queueFor(handle);
        if (queue.reserved.has(clientId)) {
          throw new Error('concurrent client already has a pending queue value');
        }
        if (queue.items.length) {
          const item = queue.items.shift();
          queue.reserved.set(clientId, item);
          reply(cell, 1, item.bytes.length);
        }
        else queue.waiters.push({cell, clientId});
        break;
      }
      case 'queue.take': {
        const queue = queueFor(handle);
        const item = queue.reserved.get(clientId);
        if (!item) throw new Error('concurrent client has no reserved queue value');
        reply(cell, 2, item.bytes);
        if (Atomics.load(new Int32Array(cell, 0, 4), 0) === 1) {
          queue.reserved.delete(clientId);
        }
        break;
      }
      case 'broadcast.subscribe': {
        const subscribers = broadcasts.get(handle);
        if (!subscribers) throw new TypeError('invalid concurrent broadcast handle');
        const id = nextId++;
        queues.set(id, {items: [], waiters: [], reserved: new Map(),
          nextSequence: 0, subscriberOf: handle});
        subscribers.add(id);
        reply(cell, 1, id);
        break;
      }
      case 'broadcast.put': {
        const subscribers = broadcasts.get(handle);
        if (!subscribers) throw new TypeError('invalid concurrent broadcast handle');
        for (const id of subscribers) {
          const queue = queues.get(id);
          if (queue) put(queue, request.bytes);
          else subscribers.delete(id);
        }
        reply(cell, 0);
        break;
      }
      case 'shared.publish': {
        if (!(request.buffer instanceof SharedArrayBuffer) ||
            !['numeric', 'graph', 'transport'].includes(request.kind)) {
          throw new TypeError('shared publish requires immutable SAB backing');
        }
        const id = nextId++;
        sharedBackings.set(id, {kind: request.kind, buffer: request.buffer});
        reply(cell, 1, id);
        break;
      }
      case 'shared.number': {
        const values = new Float64Array(sharedFor(handle, 'numeric'));
        if (!Number.isInteger(request.index) || request.index < 0 ||
            request.index >= values.length) throw new RangeError('shared index out of bounds');
        reply(cell, 1, values[request.index]);
        break;
      }
      case 'shared.range': {
        const values = new Float64Array(sharedFor(handle, 'numeric'));
        const {index, count} = request;
        if (!Number.isSafeInteger(index) || !Number.isSafeInteger(count) ||
            index < 0 || count < 0 || index + count > values.length) {
          throw new RangeError('shared range out of bounds');
        }
        reply(cell, 2, new Uint8Array(values.buffer,
          values.byteOffset + index * 8, count * 8));
        break;
      }
      case 'shared.transport-size': {
        reply(cell, 1, sharedFor(handle, 'transport').byteLength);
        break;
      }
      case 'shared.transport-chunk': {
        const bytes = new Uint8Array(sharedFor(handle, 'transport'));
        const {index, count} = request;
        if (!Number.isSafeInteger(index) || !Number.isSafeInteger(count) ||
            index < 0 || count < 0 || index + count > bytes.length) {
          throw new RangeError('shared transport range out of bounds');
        }
        reply(cell, 2, bytes.subarray(index, index + count));
        break;
      }
      case 'shared.node-size': {
        const node = readSharedNode(sharedFor(handle, 'graph'), request.index);
        reply(cell, 1, 32 + node.children.length * 4 + node.bytes.length);
        break;
      }
      case 'shared.node': {
        const node = readSharedNode(sharedFor(handle, 'graph'), request.index);
        reply(cell, 2, encodeSharedNode(node));
        break;
      }
      case 'attach-child':
        reply(cell, 1, attach(request.port, clientId));
        break;
      case 'detach-client':
        if (clients.has(request.target) && request.target !== clientId &&
            clients.get(request.target)?.parentId !== clientId) {
          throw new TypeError('concurrent client cannot detach unrelated client');
        }
        reply(cell, 0);
        detach(request.target);
        break;
      case 'detach-self':
        reply(cell, 0);
        detach(clientId);
        break;
      default: throw new RangeError(`unknown concurrent broker request ${op}`);
    }
  } catch (error) {
    if (cell instanceof SharedArrayBuffer) fail(cell, error.message);
  }
}

function attach(port, parentId = null) {
  if (!port || typeof port.postMessage !== 'function') {
    throw new TypeError('invalid concurrent broker port');
  }
  const clientId = nextClientId++;
  clients.set(clientId, {port, parentId});
  if (typeof port.addEventListener === 'function') {
    port.addEventListener('message', ({data}) => dispatch(data, clientId));
    port.start?.();
  } else {
    port.on('message', (data) => dispatch(data, clientId));
    port.on?.('close', () => detach(clientId));
  }
  return clientId;
}

if (typeof process !== 'undefined' && process.versions?.node) {
  const {parentPort} = await import('node:worker_threads');
  parentPort?.on('message', ({op, port}) => {
    if (op === 'attach') attach(port);
  });
} else {
  globalThis.onmessage = ({data}) => {
    if (data?.op === 'attach') attach(data.port);
  };
}
