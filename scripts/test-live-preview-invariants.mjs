import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { treeProGenPresets, treeProGenAsset } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-pro-gen-presets.mjs';
import { createVfLiveWorldStackReference } from '../public/previews/0.6.0/live/runtime/vf-live-world-stack.mjs';
import { createGranularParticleWorldGpuSeedReference } from '../public/previews/0.6.0/live/sand/runtime/vf-granular-particle-world-gpu.mjs';
import { createGrassMaterialFieldReference, createGrassRendererGpuBatchPacketsReference } from '../public/previews/0.6.0/live/tree/ui/vf-grass-material-field.mjs';
import { reconstructGrassBladeGpuInstancesReference } from '../public/previews/0.6.0/live/tree/ui/vf-grass-blade-gpu.mjs';

const stack = createVfLiveWorldStackReference();
const data = { x_i: [0, 1, 2], y_i: [0, 1, 4] };
const falling = stack.append({ id: 'falling' }).add(data).add({ embedding: 'points' })
  .push({ id: 'gravity', collisions: false, advance({ world, dt }) {
    const y = world.read('y_i');
    for (let index = 0; index < y.length; index += 1) y[index] -= 9.82 * dt;
  } });
stack.append({ id: 'rigid' }).add({ x_i: [0], y_i: [0] })
  .push({ id: 'contacts', collisions: true, advance() {} });
assert.equal(stack.flip('falling'), falling);
stack.active.advance({ dt: 0.5 });
assert.deepEqual(data.y_i, [-4.91, -3.91, -0.9100000000000001]);
assert.equal(stack.snapshots()[0].physics[0].collisions, 'none');
assert.equal(stack.snapshots()[1].physics[0].collisions, 'rigid');
assert.equal(stack.snapshots()[0].layerCount, 2);
assert.throws(() => stack.append({ id: 'falling' }), /already exists/);

const sand = createGranularParticleWorldGpuSeedReference();
assert.ok(sand.count >= 920 * 4, 'sand should retain at least four times the original grains');
assert.equal(sand.initialOverlapPairCount, 0);
assert.equal(sand.policy.grainRadius, 0.004);

const field = createGrassMaterialFieldReference({ generator: 'vkf.conditioned', version: 1,
  seed: [0x01234567, 0x89abcdef], domain: 'material',
  hierarchy: ['world:boreal', 'grass-field:test'], lod: 0, channel: 'surface' });
const cells = Array.from({ length: 96 }, (_, index) => [index % 12, Math.floor(index / 12)]);
const grass = createGrassRendererGpuBatchPacketsReference(field,
  { cells, detailLevel: 4, footprint: 0.02, bladeBudget: 49152 });
const packet = grass.packets[0];
assert.equal(packet.instance_count, 49152);
assert.equal(packet.grass_gpu.cell_records.length / 12, 96, 'all demanded cells must receive grass');
assert.equal(packet.grass_gpu.blades_per_cell, 512);
assert.ok(packet.cell_instance_ranges.every(({ count }) => count === 512));
const instances = reconstructGrassBladeGpuInstancesReference(packet.grass_gpu, 512);
const rootsX = Array.from({ length: 512 }, (_, index) => instances[index * 16]);
assert.ok(Math.min(...rootsX) < 0.08 && Math.max(...rootsX) > 0.92,
  'grass roots must cover cell edges without a grid margin');
console.log(JSON.stringify({ worlds: stack.snapshots(), sandGrains: sand.count,
  initialSandOverlaps: sand.initialOverlapPairCount, grassBlades: packet.instance_count,
  grassCells: 96, bladesPerCell: 512 }));

const variants = new Set();
for (const name of Object.keys(treeProGenPresets)) {
  const bytes = gunzipSync(readFileSync(new URL(`../public/previews/0.6.0/live/tree/assets/${treeProGenAsset(name)}`, import.meta.url)));
  assert.equal(bytes.subarray(0, 8).toString(), 'VFTREE02');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assert.equal(view.getUint32(8, true), 2);
  assert.ok(view.getUint32(12, true) <= 393216);
  assert.ok(view.getUint32(16, true) <= 2359296);
  let offset = 20; let vertices = 0; let indices = 0;
  for (let mesh = 0; mesh < 2; mesh++) {
    const metaLength = view.getUint32(offset, true), words = view.getUint32(offset + 4, true);
    const indexCount = view.getUint32(offset + 8, true), uvCount = view.getUint32(offset + 12, true);
    const roughnessCount = view.getUint32(offset + 16, true); offset += 20;
    JSON.parse(bytes.subarray(offset, offset + metaLength).toString());
    offset += metaLength + ((4 - metaLength % 4) % 4);
    assert.equal(words % 10, 0); const count = words / 10;
    for (let i = 0; i < words; i++) assert.ok(Number.isFinite(view.getFloat32(offset + i * 4, true)), `${name}: finite vertex`);
    offset += words * 4;
    for (let i = 0; i < indexCount; i++) assert.ok(view.getUint32(offset + i * 4, true) < count, `${name}: valid index`);
    offset += (indexCount + uvCount + roughnessCount) * 4;
    vertices += count; indices += indexCount;
  }
  assert.equal(offset, bytes.length); assert.equal(vertices, view.getUint32(12, true));
  assert.equal(indices, view.getUint32(16, true));
  variants.add(createHash('sha256').update(bytes).digest('hex'));
}
assert.equal(variants.size, 4, 'all distribution variants must generate different geometry');
assert.throws(() => treeProGenAsset('../bad'), /Unknown/);
console.log('Four distinct, finite, bounded pro-gen tree assets verified');
