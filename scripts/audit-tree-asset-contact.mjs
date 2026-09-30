import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {createTriangleSurfaceAdmissionReference} from '../public/previews/0.6.0/live/tree/runtime/vf-stone-triangle-contact.mjs';

const variant = process.argv[2] ?? 'original';
if (!['original', 'uniform', 'normal', 'triangular'].includes(variant)) throw new Error('Unknown tree variant');
const name = variant === 'original' ? 'tree-mesh-feedback-8.bin.gz'
  : `tree-mesh-${variant}-feedback-8.bin.gz`;
const url = new URL(`../public/previews/0.6.0/live/tree/assets/${name}`, import.meta.url);
const raw = gunzipSync(await readFile(url));
const data = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
assert.equal(raw.toString('ascii', 0, 8), 'VFTREE02');
let offset = 20;
const meshes = [];
for (let part = 0; part < data.getUint32(8, true); part += 1) {
  const [metaSize, vertexCount, indexCount, uvCount, roughnessCount] =
    Array.from({length: 5}, (_, i) => data.getUint32(offset + i * 4, true));
  offset += 20;
  const meta = JSON.parse(raw.toString('utf8', offset, offset + metaSize));
  offset += metaSize + (4 - metaSize % 4) % 4;
  const vertices = new Float32Array(raw.buffer, raw.byteOffset + offset, vertexCount);
  offset += vertexCount * 4;
  const indices = new Uint32Array(raw.buffer, raw.byteOffset + offset, indexCount);
  offset += indexCount * 4;
  const uvs = new Float32Array(raw.buffer, raw.byteOffset + offset, uvCount);
  offset += (uvCount + roughnessCount) * 4;
  meshes.push({...meta, vertices, indices, uvs});
}
assert.equal(offset, raw.byteLength);
const [wood, foliage] = meshes;
const surface = createTriangleSurfaceAdmissionReference(wood);
const leafStride = foliage.leaf_vertex_count;
assert.ok(leafStride > 0 && foliage.vertices.length % (leafStride * 10) === 0);
const total = foliage.vertices.length / (leafStride * 10);
const blades = Array.from({length: total}, () => []);
for (let offset = 0; offset < foliage.indices.length; offset += 3) {
  const triangle = foliage.indices.subarray(offset, offset + 3);
  const leaf = Math.floor(triangle[0] / leafStride);
  assert.ok(triangle.every((index) => Math.floor(index / leafStride) === leaf));
  blades[leaf].push(...Array.from(triangle, (index) => index - leaf * leafStride));
}
let penetrations = 0;
for (let leaf = 0; leaf < total; leaf += 1) {
  const first = leaf * leafStride * 10;
  const packet = {
    vertices: foliage.vertices.subarray(first, first + leafStride * 10),
    indices: new Uint32Array(blades[leaf]),
  };
  if (surface.intersects(packet)) penetrations += 1;
}
console.log(JSON.stringify({asset: url.pathname, leaves: total, penetrations}));
assert.equal(penetrations, 0, 'Live tree asset contains leaves or stalks penetrating wood');
