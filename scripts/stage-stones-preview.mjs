// Publish one source-compiled application; preserve all other applications.
import {readFile,copyFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const compiler=path.resolve(process.argv[2]??'../vektor-flow/build/branches/pre-gen');
const root=path.join(site,'public'),compiled=path.join(root,'previews/0.6.0/compiled');
const id=process.argv[3]??'stones';if(!['stones','tree','wheel'].includes(id))throw Error('Unknown application');
const committedRuntime=process.argv.includes('--committed-runtime');
const preserveRuntime=process.argv.includes('--preserve-runtime');
if(preserveRuntime&&id!=='wheel')throw Error('Preserved runtime is only staged for the wheel');
const runtime_directory=id==='wheel'?'runtime-wheel-39':id==='stones'?'runtime-stones-16':'runtime-tree-11',directory=id==='wheel'?'wheel-water-only-39':id==='stones'?'stones-mixed-16':'tree-air-11';
const bundlePath=path.join(compiled,'bundle.json'),bundle=JSON.parse(await readFile(bundlePath));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),runtime={};
const input=path.join(compiler,'examples',id==='wheel'?'world-wheel':id==='stones'?'world-stones':'world-tree');
const entry=id==='wheel'?'water.vkf':'main.vkf';
const names=id==='wheel'?[entry,'geometry.vkf','materials.vkf','particles.vkf']:['main.vkf','geometry.vkf','materials.vkf'],snapshots=new Map();
for(const name of names)snapshots.set(name,await readFile(path.join(input,name)));
const buildDir=path.join(input,'.vkfbuild',path.parse(entry).name);
const compilerExe=process.env.VKF_PREVIEW_COMPILER??path.join(compiler,'.work/world-model-ninja/bin/vkf.exe');
const emitterExe=process.env.VKF_PREVIEW_WASM_EMITTER??path.join(compiler,'.work/world-model-ninja/bin/vkf_wasm_artifact_smoke.exe');
// Never pair current source with a pre-existing executable. Compile those
// exact inputs, then fail closed if any imported file changed during build.
for(const [exe,args] of [[compilerExe,['--source',path.join(input,entry),'--aot','--emit-wasm']],[emitterExe,['--source',path.join(input,entry),'--typed-ir',path.join(buildDir,'typed-ir.json')]]]){
  const result=spawnSync(exe,args,{cwd:compiler,encoding:'utf8',windowsHide:true});
  if(result.status!==0)throw Error(`Preview compilation failed: ${result.error??result.stderr??result.stdout}`);
}
for(const [name,bytes] of snapshots)if(hash(await readFile(path.join(input,name)))!==hash(bytes))throw Error(`Source changed during compilation: ${name}`);
const build={entry:'main.vkf',compiler:hash(await readFile(compilerExe)),emitter:hash(await readFile(emitterExe)),typed_ir:hash(await readFile(path.join(buildDir,'typed-ir.json')))};
function committedBytes(args){const result=spawnSync('git',args,{cwd:compiler,windowsHide:true,maxBuffer:16*1024*1024});if(result.status!==0)throw Error('Cannot read committed runtime: '+result.stderr);return result.stdout;}
if(committedRuntime)build.runtime_revision=committedBytes(['rev-parse','HEAD']).toString().trim();
await mkdir(path.join(compiled,runtime_directory),{recursive:true});
async function copyRuntime(name){
  if(runtime[name])return;
  if(!/^[A-Za-z0-9_.-]+\.(js|mjs)$/.test(name))throw Error(`Runtime module outside flat adapter closure: ${name}`);
  const original=preserveRuntime?await readFile(path.join(compiled,runtime_directory,name)):
    committedRuntime?committedBytes(['show',`${build.runtime_revision}:web/vf-ui/${name}`]):await readFile(path.join(compiler,'web/vf-ui',name));
  const source=Buffer.from(original.toString('utf8').replaceAll('\r\n','\n'));
  if(!preserveRuntime)await writeFile(path.join(compiled,runtime_directory,name),source);
  runtime[name]=hash(source);
  for(const match of source.toString().matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))await copyRuntime(match[1].split('?')[0]);
}
for(const name of ['vf-world-layer-runtime.js','vf-compiled-runtime-bridge.js',id==='wheel'?'vf-world-material-runtime.mjs':'vf-world-mechanical-runtime.mjs'])await copyRuntime(name);
const output=path.join(compiled,directory),sources=path.join(root,'sources/coming-soon',id);
await mkdir(output,{recursive:true});await mkdir(sources,{recursive:true});
const bytes=await readFile(path.join(buildDir,`${path.parse(entry).name}.wasm`));
if(!WebAssembly.validate(bytes))throw Error('Invalid application WASM');
await writeFile(path.join(output,'main.wasm'),bytes);
await copyFile(path.join(buildDir,'wasm-manifest.json'),path.join(output,'manifest.json'));
const hashes={};for(const [name,bytes] of snapshots){const source=Buffer.from(bytes.toString('utf8').replaceAll('\r\n','\n'));const publicName=name===entry?'main.vkf':name;await writeFile(path.join(sources,publicName),source);hashes[publicName]=hash(source);}
build.sources=hashes;build.wasm=hash(bytes);
bundle.applications[id]={directory,runtime_directory,runtime,wasm:hash(bytes),manifest:hash(await readFile(path.join(output,'manifest.json'))),sources:hashes,build};
const bridge=createRequire(import.meta.url)(path.join(compiler,'web/vf-ui/vf-compiled-runtime-bridge.js'));
const app=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(await readFile(path.join(output,'manifest.json')))});app.init();
const world=app.worldProgram().gpu_worlds[0],p=world.kind==='wind'?world.solid_properties:world.properties;
if(id==='wheel'){
  const {LIQUID_PARTICLE_WORLD_GPU_WGSL}=await import(pathToFileURL(path.join(compiler,'web/vf-ui/vf-liquid-contained-world-gpu.mjs')).href);
  const {SWEPT_WHEEL_CONTACT_WGSL}=await import(pathToFileURL(path.join(compiler,'web/vf-ui/vf-swept-wheel-contact-gpu.mjs')).href);
  const water=app.worldProgram().gpu_worlds.find(world=>world.kind==='liquid');
  if(app.worldProgram().gpu_worlds.length!==1||!water)throw Error('Coming Soon wheel must contain only water');
  const compiledWaterLaw=app.readBinding(`${water.binding_prefix??`$world$gpu$${water.world_id}`}$physics`);
  const liquidMarkers=['fn sweep_wheel','telemetry_base()+9u','let boundary_layer = params.fluid.y * 0.5'];
  if(liquidMarkers.some(marker=>!LIQUID_PARTICLE_WORLD_GPU_WGSL.includes(marker)||!compiledWaterLaw.includes(marker)))
    throw Error('Wheel WASM contains a stale liquid GPU law; rebuild vkf_wasm_artifact_smoke');
  const sweptMarker='let angular_velocity=delta/max(elapsed,1.0e-6);';
  if(!SWEPT_WHEEL_CONTACT_WGSL.includes(sweptMarker)||!compiledWaterLaw.includes(sweptMarker))
    throw Error('Wheel WASM contains a stale swept-contact GPU law; rebuild vkf_wasm_artifact_smoke');
}
for(const url of [p.asset,...Object.values(p.variants??{})].filter(Boolean)){const asset=new URL(url,'https://vektorflow.org').pathname.slice(1);if(!asset.startsWith('previews/0.6.0/live/'))throw Error('Asset out of scope');bundle.assets[asset]=hash(await readFile(path.join(root,asset)));}
await writeFile(bundlePath,JSON.stringify(bundle,null,2)+'\n');
console.log(JSON.stringify({application:id,directory,runtime_directory,modules:Object.keys(runtime).length}));
