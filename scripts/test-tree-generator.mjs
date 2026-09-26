import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {generateTreeMeshes,TREE_SPECIES} from '../public/previews/0.6.0/live/tree/runtime/vf-tree-live-generation.mjs';
import {prepareMechanicalInitialState} from '../public/previews/0.6.0/compiled/runtime-tree-12/vf-world-mechanical-runtime.mjs';

const require=createRequire(import.meta.url);
const bridge=require('../public/previews/0.6.0/compiled/runtime-tree-12/vf-compiled-runtime-bridge.js');
const base=new URL('../public/previews/0.6.0/compiled/tree-air-12/',import.meta.url);
const runtime=bridge.instantiateWasmRuntime({
  bytes:await readFile(new URL('main.wasm',base)),
  manifest:JSON.parse(await readFile(new URL('manifest.json',base))),
});
runtime.init();
const program=runtime.worldProgram(),world=program.gpu_worlds[0],arenas=runtime.worldLayerViews();
const controls=program.views[program.active_view].controls.tree_generation;
assert.deepEqual(controls.species,['oak','birch','beech']);
assert.deepEqual(controls.distribution,['uniform','normal','triangular']);
assert.deepEqual(controls.height,{max:8,min:3,step:.25,value:8});
assert.deepEqual(controls.split_factor,{max:1,min:0,step:.05,value:.65});
assert.deepEqual(controls.turning_factor,{max:1,min:0,step:.05,value:.5});

const height=mesh=>{
  let low=Infinity,high=-Infinity;
  for(const part of mesh.meshes)for(let i=2;i<part.vertices.length;i+=10){low=Math.min(low,part.vertices[i]);high=Math.max(high,part.vertices[i]);}
  return high-low;
};
const start=performance.now(),noSplit=generateTreeMeshes({splitFactor:0,turnFactor:0}),equalSplit=generateTreeMeshes({splitFactor:1,turnFactor:1});
assert.ok(noSplit.primitiveCount<equalSplit.primitiveCount/10,'Split 0 must suppress lateral branches');
assert.ok(equalSplit.primitiveCount>500,'Split 1 must produce a developed crown');
assert.equal(noSplit.meshes[1].leaf_vertex_count,30);
assert.equal(equalSplit.meshes[1].leaf_vertex_count,30);
const short=generateTreeMeshes({height:3}),tall=generateTreeMeshes({height:8});
assert.ok(height(tall)>height(short)*2,'Height control must affect generated geometry');
for(const [species,profile] of Object.entries(TREE_SPECIES)){
  const result=generateTreeMeshes({species});
  assert.equal(result.controls.species,species);
  assert.equal(result.meshes[1].leaf_vertex_count,30);
  assert.ok(result.meshes[0].use_vertex_bark);
  assert.ok(result.meshes.every(mesh=>mesh.vertices.every(Number.isFinite)));
  const prepared=prepareMechanicalInitialState(structuredClone(world),arenas,result.meshes);
  assert.equal(prepared.initial.nodes.length,prepared.initial.nodeCount*16);
  assert.ok(prepared.initial.geometry.every(Number.isFinite));
  assert.ok(prepared.meshes.every(mesh=>mesh.vertices.every(Number.isFinite)));
  assert.ok(profile.leafOutline===species);
}
console.log(JSON.stringify({passed:true,split0:noSplit.primitiveCount,split1:equalSplit.primitiveCount,
  shortHeight:height(short),tallHeight:height(tall),species:Object.keys(TREE_SPECIES),
  generationMs:performance.now()-start}));
