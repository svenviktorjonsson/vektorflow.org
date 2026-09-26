// Lossless transport for immutable VKF tagged-value graphs. The host copies
// layout bytes and relocates pointers; it never evaluates a VKF expression.
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', {fatal: true});
const MAX_NODES = 1_000_000;
const MAX_PAYLOAD = 64 * 1024 * 1024;

function sourceView(memory, used) {
  if (!(memory instanceof WebAssembly.Memory) ||
      !Number.isSafeInteger(used) || used < 0 || used > memory.buffer.byteLength) {
    throw new TypeError('invalid concurrent value source memory');
  }
  return new DataView(memory.buffer, 0, used);
}

function bounded(view, pointer, length) {
  if (!Number.isSafeInteger(pointer) || !Number.isSafeInteger(length) ||
      pointer < 0 || length < 0 || pointer + length > view.byteLength ||
      length > MAX_PAYLOAD) {
    throw new RangeError('concurrent value graph exceeds available memory');
  }
}

export function captureValue(memory, pointer, used) {
  const view = sourceView(memory, used);
  const indexes = new Map();
  const nodes = [];
  function capture(address) {
    bounded(view, address, 16);
    if (indexes.has(address)) return indexes.get(address);
    if (nodes.length >= MAX_NODES) throw new RangeError('concurrent value graph is too large');
    const tag = view.getUint32(address, true);
    const count = view.getUint32(address + 4, true);
    const payload = view.getUint32(address + 8, true);
    const slotSize = tag === 10 ? 24 : 16;
    bounded(view, address, slotSize);
    const node = {tag, sourcePointer: address, slot: Array.from(new Uint8Array(
      view.buffer, address, slotSize)), children: [], bytes: []};
    const index = nodes.length;
    indexes.set(address, index);
    nodes.push(node);
    const children = (base, entries, stride) => {
      bounded(view, base, entries * stride);
      for (let item = 0; item < entries; ++item) {
        for (let offset = 0; offset < stride; offset += 4) {
          node.children.push(capture(view.getUint32(base + item * stride + offset, true)));
        }
      }
    };
    switch (tag) {
      case 0: case 1: case 2: case 10: case 13: case 14:
      case 16: case 17: break;
      case 15:
        throw new TypeError('process handle cannot cross Worker ownership boundary');
      case 3:
        bounded(view, payload, count);
        node.bytes = Array.from(new Uint8Array(view.buffer, payload, count));
        break;
      case 4: case 6: case 11:
        children(payload, count, 4);
        break;
      case 5: case 9:
        children(payload, count, 8);
        break;
      case 8: case 12:
        node.children.push(capture(count), capture(payload));
        break;
      default:
        throw new TypeError(`concurrent transport cannot copy VKF tag ${tag}`);
    }
    return index;
  }
  return {schema: 'vektor-flow/concurrent-value-v1', root: capture(pointer), nodes};
}

export function materializeValue(api, graph) {
  if (!api?.memory || typeof api.vkf_vm_alloc !== 'function' ||
      graph?.schema !== 'vektor-flow/concurrent-value-v1' ||
      !Array.isArray(graph.nodes) || graph.nodes.length > MAX_NODES ||
      !Number.isInteger(graph.root) || graph.root < 0 || graph.root >= graph.nodes.length) {
    throw new TypeError('invalid concurrent value graph');
  }
  const slots = graph.nodes.map((node) => {
    if (!Number.isInteger(node?.tag) || !Array.isArray(node.slot) ||
        node.slot.length !== (node.tag === 10 ? 24 : 16) ||
        !Array.isArray(node.children) || !Array.isArray(node.bytes)) {
      throw new TypeError('invalid concurrent value node');
    }
    if (node.tag === 15) {
      throw new TypeError('process handle cannot cross Worker ownership boundary');
    }
    if (new DataView(Uint8Array.from(node.slot).buffer).getUint32(0, true) !== node.tag) {
      throw new TypeError('concurrent value node tag does not match its slot');
    }
    const pointer = api.vkf_vm_alloc(node.slot.length);
    if (!pointer) throw new RangeError('concurrent value allocation failed');
    new Uint8Array(api.memory.buffer, pointer, node.slot.length).set(node.slot);
    return pointer;
  });
  const childPointer = (index) => {
    if (!Number.isInteger(index) || index < 0 || index >= slots.length) {
      throw new RangeError('concurrent value graph has invalid edge');
    }
    return slots[index];
  };
  graph.nodes.forEach((node, index) => {
    const address = slots[index];
    const view = new DataView(api.memory.buffer);
    if (node.tag === 3) {
      if (node.bytes.length > MAX_PAYLOAD) throw new RangeError('concurrent string is too large');
      decoder.decode(Uint8Array.from(node.bytes));
      const payload = api.vkf_vm_alloc(Math.max(1, node.bytes.length));
      if (!payload) throw new RangeError('concurrent string allocation failed');
      new Uint8Array(api.memory.buffer, payload, node.bytes.length).set(node.bytes);
      new DataView(api.memory.buffer).setUint32(address + 8, payload, true);
    } else if (node.tag === 4 || node.tag === 5 || node.tag === 6 ||
               node.tag === 9 || node.tag === 11) {
      const width = node.tag === 5 || node.tag === 9 ? 8 : 4;
      const count = view.getUint32(address + 4, true);
      if (node.children.length !== count * (width / 4) ||
          node.children.length * 4 > MAX_PAYLOAD) {
        throw new TypeError('concurrent collection length is invalid');
      }
      const payload = api.vkf_vm_alloc(Math.max(4, node.children.length * 4));
      if (!payload) throw new RangeError('concurrent collection allocation failed');
      node.children.forEach((child, offset) =>
        new DataView(api.memory.buffer).setUint32(payload + offset * 4, childPointer(child), true));
      new DataView(api.memory.buffer).setUint32(address + 8, payload, true);
    } else if (node.tag === 8 || node.tag === 12) {
      if (node.children.length !== 2) throw new TypeError('concurrent display wrapper is invalid');
      view.setUint32(address + 4, childPointer(node.children[0]), true);
      view.setUint32(address + 8, childPointer(node.children[1]), true);
    }
  });
  return slots[graph.root];
}

export function encodeValueGraph(graph) {
  return encoder.encode(JSON.stringify(graph));
}

export function decodeValueGraph(bytes) {
  const value = JSON.parse(decoder.decode(bytes));
  if (value?.schema !== 'vektor-flow/concurrent-value-v1') {
    throw new TypeError('invalid concurrent value graph schema');
  }
  return value;
}
