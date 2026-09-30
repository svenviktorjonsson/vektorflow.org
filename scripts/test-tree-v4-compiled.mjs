import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import test from 'node:test';

const root=new URL('../public/',import.meta.url);
const bundle=JSON.parse(await readFile(new URL('previews/0.6.0/compiled/bundle.json',root),'utf8'));
const record=bundle.applications.tree;
const runtimeBase=new URL(`previews/0.6.0/compiled/${record.runtime_directory}/`,root);
const applicationBase=new URL(`previews/0.6.0/compiled/${record.directory}/`,root);
const require=createRequire(import.meta.url);
const bridge=require(fileURLToPath(new URL('vf-compiled-runtime-bridge.js',runtimeBase)));

test('Coming Soon tree runs authored contact Law over its physical V4 geometry',async()=>{
  const bytes=await readFile(new URL('main.wasm',applicationBase));
  const manifest=JSON.parse(await readFile(new URL('manifest.json',applicationBase),'utf8'));
  assert.ok(WebAssembly.validate(bytes));
  const application=bridge.instantiateWasmRuntime({bytes,manifest});
  application.init();
  const world=application.worldProgram().gpu_worlds.find(world=>world.kind==='wind');
  assert.equal(world.surface_contact.kind,'surface_contact');
  const {parseMechanicalAssetBytes}=await import(new URL('vf-world-mechanical-asset.mjs',runtimeBase));
  const {surfaceContactLawParameters,prepareMechanicalInitialState}=await import(
    new URL('vf-world-mechanical-runtime.mjs',runtimeBase));
  assert.ok(surfaceContactLawParameters(world).pairCapacity>0);
  const assetUrl=world.solid_properties.asset;
  const assetPath=new URL(assetUrl,'https://vektorflow.org').pathname;
  assert.match(assetPath,/tree-mesh-contact-v4\.bin\.gz$/);
  const compressed=await readFile(new URL(assetPath.slice(1),root));
  const raw=gunzipSync(compressed);
  assert.equal(raw.toString('ascii',0,8),'VFWORLD4');
  const asset=parseMechanicalAssetBytes(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  assert.equal(asset.meshes.length,2);
  assert.ok(asset.contact.surfaceTopology.length>0);
  assert.ok(asset.contact.islandCount>asset.meshes[1].leaf_vertex_count);
  const {initial,meshes}=prepareMechanicalInitialState(world,application.worldLayerViews(),asset.meshes);
  assert.ok(initial.nodes.every(Number.isFinite));
  assert.ok(meshes.every(mesh=>mesh.vertices.every(Number.isFinite)));
});
