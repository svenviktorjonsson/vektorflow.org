import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { gunzipSync } from 'node:zlib';
import { applications } from '../public/coming-soon.mjs';
const root=new URL('../public/',import.meta.url),bundle=JSON.parse(await readFile(new URL('previews/0.6.0/compiled/bundle.json',root),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const bridge=createRequire(import.meta.url)('../public/previews/0.6.0/compiled/runtime/vf-compiled-runtime-bridge.js');
for(const [name,digest] of Object.entries(bundle.assets)){const bytes=await readFile(new URL(name,root));assert.equal(hash(bytes),digest,name);const decoded=gunzipSync(bytes);assert.equal(decoded.subarray(0,8).toString(),'VFTREE02');}
for(const {id,files} of applications){
  const record=bundle.applications[id],base=new URL(`previews/0.6.0/compiled/${id}/`,root),bytes=await readFile(new URL('main.wasm',base)),manifestBytes=await readFile(new URL('manifest.json',base));
  assert.equal(hash(bytes),record.wasm);assert.equal(hash(manifestBytes),record.manifest);assert.ok(WebAssembly.validate(bytes));assert.deepEqual(Object.keys(record.sources).sort(),[...files].sort());
  for(const name of files)assert.equal(hash(await readFile(new URL(`sources/coming-soon/${id}/${name}`,root))),record.sources[name]);
  const runtime=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(manifestBytes)});runtime.init();const program=runtime.worldProgram();assert.ok(program.gpu_worlds.length);assert.ok(program.views.every(view=>view.axis===false));
  const arenas=runtime.worldLayerViews();assert.ok(arenas.every(arena=>arena.state.every(Number.isFinite)));
  for(const world of program.gpu_worlds){assert.ok(runtime.readBinding(`$world$gpu$${world.world_id}$physics`).includes('@compute'));assert.throws(()=>runtime.stepWorld(world.world_id),WebAssembly.RuntimeError);}
  if(id==='wheel'){assert.deepEqual(program.gpu_worlds.map(world=>world.kind),['liquid','granular']);assert.equal(program.gpu_worlds[0].geometry.radius,.5);assert.ok(arenas[2].layer.count>=1808);assert.ok(arenas[5].layer.count>=3680);}
  else {assert.equal(program.gpu_worlds[0].kind,id==='stones'?'rigid':'wind');assert.equal(arenas[id==='stones'?0:3].layer.count,id==='stones'?5:8192);}
  if(id==='stones'){const asset=gunzipSync(await readFile(new URL(program.gpu_worlds[0].properties.asset.slice(1),root)));assert.equal(asset.readUInt32LE(8),5);let offset=20;for(let i=0;i<5;i++){const sizes=Array.from({length:5},(_,j)=>asset.readUInt32LE(offset+j*4));offset+=20;const metadata=JSON.parse(asset.subarray(offset,offset+sizes[0]));for(let axis=0;axis<3;axis++)assert.ok(Math.abs(metadata.collision.center[axis]-arenas[0].state[i*9+axis*3])<1e-8,'asset and add placement must agree');offset+=sizes[0]+(4-sizes[0]%4)%4+(sizes[1]+sizes[2]+sizes[3]+sizes[4])*4;}assert.equal(offset,asset.length);}
  console.log(`${id}: executable source, finite initial data and artifact hashes verified`);
}
for(const [name,digest] of Object.entries(bundle.runtime)){const source=await readFile(new URL(`previews/0.6.0/compiled/runtime/${name}`,root));assert.equal(hash(source),digest);for(const match of source.toString().matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))assert.ok(bundle.runtime[match[1]],`Missing transitive module ${match[1]}`);}
const page=await readFile(new URL('coming-soon.html',root),'utf8');assert.equal((page.match(/<iframe /g)||[]).length,3);assert.doesNotMatch(page,/not yet the compiled source|Intended VKF|simulation-tabs/);
console.log('Three independent sections and complete module closure verified');
