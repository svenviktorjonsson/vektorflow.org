import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { gunzipSync } from 'node:zlib';
import { applications } from '../public/coming-soon.mjs';
const root=new URL('../public/',import.meta.url),bundle=JSON.parse(await readFile(new URL('previews/0.6.0/compiled/bundle.json',root),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const runtimeDirectory=bundle.runtime_directory??'runtime';
const {prepareMechanicalInitialState}=await import(`../public/previews/0.6.0/compiled/${runtimeDirectory}/vf-world-mechanical-runtime.mjs`);
function decodeMeshes(bytes){const b=gunzipSync(bytes);let offset=20;const meshes=[];for(let i=0;i<b.readUInt32LE(8);i++){const sizes=Array.from({length:5},(_,j)=>b.readUInt32LE(offset+j*4));offset+=20;const meta=JSON.parse(b.subarray(offset,offset+sizes[0]));offset+=sizes[0]+(4-sizes[0]%4)%4;const vertices=new Float32Array(b.buffer.slice(b.byteOffset+offset,b.byteOffset+offset+sizes[1]*4));offset+=sizes[1]*4;const indices=new Uint32Array(b.buffer.slice(b.byteOffset+offset,b.byteOffset+offset+sizes[2]*4));offset+=(sizes[2]+sizes[3]+sizes[4])*4;meshes.push({...meta,vertices,indices});}assert.equal(offset,b.length);return meshes;}
function verifyInitialExclusion(world,arena){const radius=world.kind==='granular'?world.properties.radius:world.properties.spacing*.46,diameter=radius*2,points=Array.from({length:arena.layer.count},(_,i)=>[arena.state[i*6],arena.state[i*6+3]]),g=world.geometry,c=Math.cos(g.rotation),s=Math.sin(g.rotation),rotate=([x,y])=>[g.center[0]+c*x-s*y,g.center[1]+s*x+c*y];let minimum=Infinity;
  for(let i=0;i<points.length;i++){const p=points[i];assert.ok(Math.hypot(p[0]-g.center[0],p[1]-g.center[1])<=g.radius-radius-g.half_width+1e-7);for(const segment of g.segments){const a=rotate(segment.slice(0,2)),b=rotate(segment.slice(2)),v=b.map((x,k)=>x-a[k]),t=Math.max(0,Math.min(1,((p[0]-a[0])*v[0]+(p[1]-a[1])*v[1])/(v[0]**2+v[1]**2)));assert.ok(Math.hypot(p[0]-a[0]-t*v[0],p[1]-a[1]-t*v[1])>=radius+g.half_width-1e-7);}for(let j=i+1;j<points.length;j++)minimum=Math.min(minimum,(p[0]-points[j][0])**2+(p[1]-points[j][1])**2);}
  assert.ok(minimum>=diameter**2,`${world.kind}: initial hard entities overlap`);console.log(`${world.kind}: ${points.length} initial entities have zero overlaps and clear all boundaries`);
}
const bridge=createRequire(import.meta.url)(`../public/previews/0.6.0/compiled/${runtimeDirectory}/vf-compiled-runtime-bridge.js`);
for(const [name,digest] of Object.entries(bundle.assets)){const bytes=await readFile(new URL(name,root));assert.equal(hash(bytes),digest,name);const decoded=gunzipSync(bytes);assert.equal(decoded.subarray(0,8).toString(),'VFTREE02');}
for(const {id,files} of applications){
  const record=bundle.applications[id],base=new URL(`previews/0.6.0/compiled/${record.directory??id}/`,root),bytes=await readFile(new URL('main.wasm',base)),manifestBytes=await readFile(new URL('manifest.json',base));
  assert.equal(hash(bytes),record.wasm);assert.equal(hash(manifestBytes),record.manifest);assert.ok(WebAssembly.validate(bytes));assert.deepEqual(Object.keys(record.sources).sort(),[...files].sort());
  for(const name of files)assert.equal(hash(await readFile(new URL(`sources/coming-soon/${id}/${name}`,root))),record.sources[name]);
  const runtime=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(manifestBytes)});runtime.init();const program=runtime.worldProgram();assert.ok(program.gpu_worlds.length);assert.ok(program.views.every(view=>view.axis===false));
  const arenas=runtime.worldLayerViews();assert.ok(arenas.every(arena=>arena.state.every(Number.isFinite)));
  if(id==='wheel')for(const world of program.gpu_worlds){verifyInitialExclusion(world,arenas.find(a=>a.layer.id===world.layer_id));const kernel=runtime.readBinding(`$world$gpu$${world.world_id}$physics`);for(const name of ['motion_find_toi','motion_bar_toi','preventive_predict','motion_update_ledger','motion_guard_end','motion_rotate'])assert.ok(kernel.includes(`fn ${name}`),`Compiled ${world.kind} missing ${name}`);assert.match(kernel,/bitcast<u32>\(fraction\)&0x7fffffffu/);}
  if(id==='tree'){
    const world=program.gpu_worlds[0];for(const url of [world.solid_properties.asset,...Object.values(world.solid_properties.variants)]){
      const asset=decodeMeshes(await readFile(new URL(url.slice(1),root))),{initial,meshes}=prepareMechanicalInitialState(structuredClone(world),arenas,asset);assert.equal(initial.nodes.length,initial.nodeCount*16);assert.ok(initial.geometry.every(Number.isFinite));assert.ok(meshes.every(m=>m.vertices.every(Number.isFinite)&&m.vertices.length%24===0));
      const leaves=meshes[asset.findIndex(m=>m.id.includes('foliage'))].vertices;for(let base=0;base<leaves.length;base+=12*24){const anchor=[0,1,2].map(a=>(leaves[base+a]+leaves[base+24+a])*.5);for(let j=0;j<12;j++){assert.equal(leaves[base+j*24+19],1);for(let a=0;a<3;a++)assert.ok(Math.abs(leaves[base+j*24+16+a]-anchor[a])<1e-6);assert.ok(leaves[base+j*24+23]>0);}}
    }console.log('Four tree variants: finite 96-byte geometry, shared leaf anchors and elastic leaf coefficients verified');
  }
  for(const world of program.gpu_worlds){assert.ok(runtime.readBinding(`$world$gpu$${world.world_id}$physics`).includes('@compute'));assert.throws(()=>runtime.stepWorld(world.world_id),WebAssembly.RuntimeError);}
  if(id==='wheel'){assert.deepEqual(program.gpu_worlds.map(world=>world.kind),['liquid','granular']);assert.equal(program.gpu_worlds[0].geometry.radius,.5);assert.ok(arenas[2].layer.count>=1808);assert.ok(arenas[5].layer.count>=3680);}
  else {assert.equal(program.gpu_worlds[0].kind,id==='stones'?'rigid':'wind');assert.equal(arenas.find(a=>a.layer.id===program.gpu_worlds[0].layer_id).layer.count,id==='stones'?5:8192);}
  if(id==='stones'){const asset=gunzipSync(await readFile(new URL(program.gpu_worlds[0].properties.asset.slice(1),root)));assert.equal(asset.readUInt32LE(8),5);let offset=20;for(let i=0;i<5;i++){const sizes=Array.from({length:5},(_,j)=>asset.readUInt32LE(offset+j*4));offset+=20;const metadata=JSON.parse(asset.subarray(offset,offset+sizes[0]));for(let axis=0;axis<3;axis++)assert.ok(Math.abs(metadata.collision.center[axis]-arenas[0].state[i*9+axis*3])<1e-8,'asset and add placement must agree');offset+=sizes[0]+(4-sizes[0]%4)%4+(sizes[1]+sizes[2]+sizes[3]+sizes[4])*4;}assert.equal(offset,asset.length);}
  console.log(`${id}: executable source, finite initial data and artifact hashes verified`);
}
for(const [name,digest] of Object.entries(bundle.runtime)){const source=await readFile(new URL(`previews/0.6.0/compiled/${runtimeDirectory}/${name}`,root));assert.equal(hash(source),digest);for(const match of source.toString().matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))assert.ok(bundle.runtime[match[1]],`Missing transitive module ${match[1]}`);}
const page=await readFile(new URL('coming-soon.html',root),'utf8');assert.equal((page.match(/<iframe /g)||[]).length,3);assert.doesNotMatch(page,/not yet the compiled source|Intended VKF|simulation-tabs/);
console.log('Three independent sections and complete module closure verified');
