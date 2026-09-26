// Browser concurrency transport. These buffers are shared between actual Web
// Workers; no callback on the blocked worker's event loop is needed to wake a
// receiver. The numeric lane matches the current native concurrent ABI.
const HEADER_WORDS = 6;
const LOCK = 0;
const HEAD = 1;
const TAIL = 2;
const COUNT = 3;
const EPOCH = 4;
const CLOSED = 5;
const DEFAULT_CAPACITY = 4096;
const VALUE_SLOTS = 64;
const VALUE_SLOT_BYTES = 64 * 1024;

function capacityOf(buffer) {
  if (!(buffer instanceof SharedArrayBuffer) || buffer.byteLength < HEADER_WORDS * 4 + 8 ||
      (buffer.byteLength - HEADER_WORDS * 4) % 8 !== 0) {
    throw new TypeError('concurrent queue requires a valid shared buffer');
  }
  return (buffer.byteLength - HEADER_WORDS * 4) / 8;
}

export function createNumericQueue(capacity = DEFAULT_CAPACITY) {
  if (!Number.isSafeInteger(capacity) || capacity < 1) {
    throw new RangeError('concurrent queue capacity must be positive');
  }
  return new SharedArrayBuffer(HEADER_WORDS * 4 + capacity * 8);
}

function views(buffer) {
  const capacity = capacityOf(buffer);
  return {
    state: new Int32Array(buffer, 0, HEADER_WORDS),
    values: new Float64Array(buffer, HEADER_WORDS * 4, capacity),
    capacity,
  };
}

function waitForChange(state, previous) {
  if (typeof window !== 'undefined' && globalThis === window) {
    throw new Error('blocking concurrent operation must run in a Web Worker');
  }
  Atomics.wait(state, EPOCH, previous);
}

function lock(state) {
  while (Atomics.compareExchange(state, LOCK, 0, 1) !== 0) {
    const epoch = Atomics.load(state, EPOCH);
    if (Atomics.load(state, LOCK) !== 0) waitForChange(state, epoch);
  }
}

function unlock(state) {
  Atomics.store(state, LOCK, 0);
  Atomics.add(state, EPOCH, 1);
  Atomics.notify(state, EPOCH);
}

export function putNumeric(buffer, value) {
  if (typeof value !== 'number') throw new TypeError('numeric queue accepts one number');
  const {state, values, capacity} = views(buffer);
  for (;;) {
    lock(state);
    if (Atomics.load(state, CLOSED)) {
      unlock(state);
      throw new Error('concurrent queue is closed');
    }
    if (Atomics.load(state, COUNT) < capacity) {
      const tail = Atomics.load(state, TAIL);
      values[tail] = value;
      Atomics.store(state, TAIL, (tail + 1) % capacity);
      Atomics.add(state, COUNT, 1);
      unlock(state);
      return;
    }
    const epoch = Atomics.load(state, EPOCH);
    unlock(state);
    waitForChange(state, epoch + 1);
  }
}

export function getNumeric(buffer) {
  const {state, values, capacity} = views(buffer);
  for (;;) {
    lock(state);
    if (Atomics.load(state, COUNT) !== 0) {
      const head = Atomics.load(state, HEAD);
      const value = values[head];
      Atomics.store(state, HEAD, (head + 1) % capacity);
      Atomics.sub(state, COUNT, 1);
      unlock(state);
      return value;
    }
    if (Atomics.load(state, CLOSED)) {
      unlock(state);
      throw new Error('concurrent queue is closed');
    }
    const epoch = Atomics.load(state, EPOCH);
    unlock(state);
    waitForChange(state, epoch + 1);
  }
}

export function closeNumericQueue(buffer) {
  const {state} = views(buffer);
  lock(state);
  Atomics.store(state, CLOSED, 1);
  unlock(state);
}

export function createNumericBroadcast() {
  const subscribers = new Set();
  return Object.freeze({
    subscribe(capacity) {
      const queue = createNumericQueue(capacity);
      subscribers.add(queue);
      return queue;
    },
    put(value) {
      for (const queue of subscribers) putNumeric(queue, value);
    },
    unsubscribe(queue) {
      if (!subscribers.delete(queue)) return false;
      closeNumericQueue(queue);
      return true;
    },
    close() {
      for (const queue of subscribers) closeNumericQueue(queue);
      subscribers.clear();
    },
  });
}

function valueViews(buffer) {
  const header = HEADER_WORDS * 4;
  const lengthsSize = VALUE_SLOTS * 4;
  if (!(buffer instanceof SharedArrayBuffer) ||
      buffer.byteLength !== header + lengthsSize + VALUE_SLOTS * VALUE_SLOT_BYTES) {
    throw new TypeError('concurrent value queue requires a valid shared buffer');
  }
  return {
    state: new Int32Array(buffer, 0, HEADER_WORDS),
    lengths: new Uint32Array(buffer, header, VALUE_SLOTS),
    data: new Uint8Array(buffer, header + lengthsSize),
  };
}

export function createValueQueue() {
  if (typeof SharedArrayBuffer !== 'function') {
    throw new Error('browser concurrent values require cross-origin-isolated shared memory');
  }
  return new SharedArrayBuffer(
    HEADER_WORDS * 4 + VALUE_SLOTS * 4 + VALUE_SLOTS * VALUE_SLOT_BYTES);
}

export function putValue(buffer, bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength > VALUE_SLOT_BYTES) {
    throw new RangeError('concurrent value exceeds the shared-message limit');
  }
  const {state, lengths, data} = valueViews(buffer);
  for (;;) {
    lock(state);
    if (Atomics.load(state, CLOSED)) {
      unlock(state);
      throw new Error('concurrent value queue is closed');
    }
    if (Atomics.load(state, COUNT) < VALUE_SLOTS) {
      const tail = Atomics.load(state, TAIL);
      data.set(bytes, tail * VALUE_SLOT_BYTES);
      lengths[tail] = bytes.byteLength;
      Atomics.store(state, TAIL, (tail + 1) % VALUE_SLOTS);
      Atomics.add(state, COUNT, 1);
      unlock(state);
      return;
    }
    const epoch = Atomics.load(state, EPOCH);
    unlock(state);
    waitForChange(state, epoch + 1);
  }
}

export function getValue(buffer) {
  const {state, lengths, data} = valueViews(buffer);
  for (;;) {
    lock(state);
    if (Atomics.load(state, COUNT) !== 0) {
      const head = Atomics.load(state, HEAD);
      const length = lengths[head];
      const value = Uint8Array.from(data.subarray(
        head * VALUE_SLOT_BYTES, head * VALUE_SLOT_BYTES + length));
      Atomics.store(state, HEAD, (head + 1) % VALUE_SLOTS);
      Atomics.sub(state, COUNT, 1);
      unlock(state);
      return value;
    }
    if (Atomics.load(state, CLOSED)) {
      unlock(state);
      throw new Error('concurrent value queue is closed');
    }
    const epoch = Atomics.load(state, EPOCH);
    unlock(state);
    waitForChange(state, epoch + 1);
  }
}

export function createValueBroadcast() {
  const subscribers = new Set();
  return Object.freeze({
    subscribe() {
      const queue = createValueQueue();
      subscribers.add(queue);
      return queue;
    },
    put(bytes) {
      for (const queue of subscribers) putValue(queue, bytes);
    },
    unsubscribe(queue) {
      if (!subscribers.delete(queue)) return false;
      const {state} = valueViews(queue);
      lock(state); Atomics.store(state, CLOSED, 1); unlock(state);
      return true;
    },
  });
}
