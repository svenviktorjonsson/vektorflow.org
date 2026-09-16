import assert from 'node:assert/strict';
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
