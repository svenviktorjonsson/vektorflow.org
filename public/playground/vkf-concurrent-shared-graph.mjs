// Immutable, random-access VKF graph backing. Publish once into SAB; workers
// receive only the SAB descriptor and materialize demanded nodes lazily.
const MAGIC = 0x564b4631;
const HEADER = 16;
const NODE_BYTES = 40;
const MAX_NODES = 1_000_000;

export function publishSharedGraph(graph) {
  if (graph?.schema !== 'vektor-flow/concurrent-value-v1' ||
      !Array.isArray(graph.nodes) || graph.nodes.length > MAX_NODES ||
      !Number.isInteger(graph.root) || graph.root < 0 ||
      graph.root >= graph.nodes.length) {
    throw new TypeError('invalid immutable shared value graph');
  }
  const childCount = graph.nodes.reduce((sum, node) => sum + node.children.length, 0);
  const byteCount = graph.nodes.reduce((sum, node) => sum + node.bytes.length, 0);
  const childrenStart = HEADER + graph.nodes.length * NODE_BYTES;
  const bytesStart = childrenStart + childCount * 4;
  const allocation = bytesStart + byteCount;
  if (!Number.isSafeInteger(allocation) || allocation > 256 * 1024 * 1024) {
    throw new RangeError('shared value graph exceeds 256 MiB');
  }
  const buffer = new SharedArrayBuffer(allocation);
  const view = new DataView(buffer);
  const raw = new Uint8Array(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, graph.nodes.length, true);
  view.setUint32(8, graph.root, true);
  view.setUint32(12, allocation, true);
  let childCursor = childrenStart;
  let byteCursor = bytesStart;
  graph.nodes.forEach((node, index) => {
    const address = HEADER + index * NODE_BYTES;
    if (!Array.isArray(node.slot) || node.slot.length > 24 ||
        !Array.isArray(node.children) || !Array.isArray(node.bytes)) {
      throw new TypeError('invalid immutable shared value node');
    }
    raw.set(node.slot, address);
    view.setUint32(address + 24, childCursor, true);
    view.setUint32(address + 28, node.children.length, true);
    view.setUint32(address + 32, byteCursor, true);
    view.setUint32(address + 36, node.bytes.length, true);
    for (const child of node.children) {
      if (!Number.isInteger(child) || child < 0 || child >= graph.nodes.length) {
        throw new RangeError('shared graph child index is invalid');
      }
      view.setUint32(childCursor, child, true);
      childCursor += 4;
    }
    raw.set(node.bytes, byteCursor);
    byteCursor += node.bytes.length;
  });
  return {buffer, root: graph.root};
}

export function readSharedNode(buffer, index) {
  if (!(buffer instanceof SharedArrayBuffer) || buffer.byteLength < HEADER) {
    throw new TypeError('invalid shared graph storage');
  }
  const view = new DataView(buffer);
  const count = view.getUint32(4, true);
  if (view.getUint32(0, true) !== MAGIC ||
      view.getUint32(12, true) !== buffer.byteLength ||
      !Number.isInteger(index) || index < 0 || index >= count ||
      HEADER + count * NODE_BYTES > buffer.byteLength) {
    throw new RangeError('invalid shared graph node');
  }
  const address = HEADER + index * NODE_BYTES;
  const childStart = view.getUint32(address + 24, true);
  const childCount = view.getUint32(address + 28, true);
  const byteStart = view.getUint32(address + 32, true);
  const byteCount = view.getUint32(address + 36, true);
  if (childStart + childCount * 4 > buffer.byteLength ||
      byteStart + byteCount > buffer.byteLength) {
    throw new RangeError('shared graph node exceeds backing storage');
  }
  return {
    tag: view.getUint32(address, true),
    count: view.getUint32(address + 4, true),
    slot: new Uint8Array(buffer, address, 24),
    children: new Uint32Array(buffer, childStart, childCount),
    bytes: new Uint8Array(buffer, byteStart, byteCount),
  };
}
