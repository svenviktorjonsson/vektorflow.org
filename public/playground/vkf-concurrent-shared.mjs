// Immutable, publish-once shared generation. Public VKF rebinding/update
// syntax is intentionally not inferred here.
const HEADER_BYTES = 16;

export function createSharedSnapshot(bytes) {
  if (!(bytes instanceof Uint8Array) || typeof SharedArrayBuffer !== 'function') {
    throw new TypeError('shared snapshot requires bytes and SharedArrayBuffer');
  }
  const buffer = new SharedArrayBuffer(HEADER_BYTES + bytes.byteLength);
  new Uint8Array(buffer, HEADER_BYTES).set(bytes);
  const header = new Int32Array(buffer, 0, 4);
  Atomics.store(header, 1, 1); // generation
  Atomics.store(header, 2, bytes.byteLength);
  Atomics.store(header, 0, 1); // published last
  Atomics.notify(header, 0);
  return buffer;
}

export function readSharedSnapshot(buffer) {
  if (!(buffer instanceof SharedArrayBuffer) || buffer.byteLength < HEADER_BYTES) {
    throw new TypeError('invalid shared snapshot');
  }
  const header = new Int32Array(buffer, 0, 4);
  if (Atomics.load(header, 0) !== 1 || Atomics.load(header, 1) !== 1) {
    throw new Error('shared snapshot is not published');
  }
  const length = Atomics.load(header, 2);
  if (length !== buffer.byteLength - HEADER_BYTES) {
    throw new RangeError('shared snapshot length is invalid');
  }
  return Object.freeze({generation: 1,
    bytes: Uint8Array.from(new Uint8Array(buffer, HEADER_BYTES, length))});
}
