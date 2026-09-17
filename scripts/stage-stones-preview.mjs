// Publish one source-compiled application; preserve all other applications.
import {readFile,copyFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const compiler=path.resolve(process.argv[2]??'../vektor-flow/build/branches/pre-gen');
const root=path.join(site,'public'),compiled=path.join(root,'previews/0.6.0/compiled');
const id=process.argv[3]??'stones';if(!['stones','tree','wheel'].includes(id))throw Error('Unknown application');
const committedRuntime=process.argv.includes('--committed-runtime');
const runtime_directory=id==='wheel'?'runtime-wheel-12':id==='stones'?'runtime-stones-8':'runtime-tree-9',directory=id==='wheel'?'wheel-granular-12':id==='stones'?'stones-granite-8':'tree-wind-9';
const bundlePath=path.join(compiled,'bundle.json'),bundle=JSON.parse(await readFile(bundlePath));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),runtime={};
const input=path.join(compiler,'examples',id==='wheel'?'world-wheel':id==='stones'?'world-stones':'world-tree');
const names=id==='wheel'?['main.vkf','geometry.vkf','materials.vkf','particles.vkf']:['main.vkf','geometry.vkf','materials.vkf'],snapshots=new Map();
for(const name of names)snapshots.set(name,await readFile(path.join(input,name)));
const compilerExe=process.env.VKF_PREVIEW_COMPILER??path.join(compiler,'.work/world-model-ninja/bin/vkf.exe');
const emitterExe=process.env.VKF_PREVIEW_WASM_EMITTER??path.join(compiler,'.work/world-model-ninja/bin/vkf_wasm_artifact_smoke.exe');
// Never pair current source with a pre-existing executable. Compile those
// exact inputs, then fail closed if any imported file changed during build.
for(const [exe,args] of [[compilerExe,['--source',path.join(input,'main.vkf'),'--aot','--emit-wasm']],[emitterExe,['--source',path.join(input,'main.vkf'),'--typed-ir',path.join(input,'.vkfbuild/main/typed-ir.json')]]]){
  const result=spawnSync(exe,args,{cwd:compiler,encoding:'utf8',windowsHide:true});
  if(result.status!==0)throw Error(`Preview compilation failed: ${result.error??result.stderr??result.stdout}`);
}
for(const [name,bytes] of snapshots)if(hash(await readFile(path.join(input,name)))!==hash(bytes))throw Error(`Source changed during compilation: ${name}`);
const build={entry:'main.vkf',compiler:hash(await readFile(compilerExe)),emitter:hash(await readFile(emitterExe)),typed_ir:hash(await readFile(path.join(input,'.vkfbuild/main/typed-ir.json')))};
function committedBytes(args){const result=spawnSync('git',args,{cwd:compiler,windowsHide:true,maxBuffer:16*1024*1024});if(result.status!==0)throw Error('Cannot read committed runtime: '+result.stderr);return result.stdout;}
if(committedRuntime)build.runtime_revision=committedBytes(['rev-parse','HEAD']).toString().trim();
await mkdir(path.join(compiled,runtime_directory),{recursive:true});
async function copyRuntime(name){
  if(runtime[name])return;
  if(!/^[A-Za-z0-9_.-]+\.(js|mjs)$/.test(name))throw Error('Runtime module outside flat adapter closure');
  const source=committedRuntime?committedBytes(['show',`${build.runtime_revision}:web/vf-ui/${name}`]):await readFile(path.join(compiler,'web/vf-ui',name));
  await writeFile(path.join(compiled,runtime_directory,name),source);runtime[name]=hash(source);
  for(const match of source.toString().matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))await copyRuntime(match[1]);
}
for(const name of ['vf-world-layer-runtime.js','vf-compiled-runtime-bridge.js',id==='wheel'?'vf-world-material-runtime.mjs':'vf-world-mechanical-runtime.mjs'])await copyRuntime(name);
const output=path.join(compiled,directory),sources=path.join(root,'sources/coming-soon',id);
await mkdir(output,{recursive:true});await mkdir(sources,{recursive:true});
const bytes=await readFile(path.join(input,'.vkfbuild/main/main.wasm'));
if(!WebAssembly.validate(bytes))throw Error('Invalid application WASM');
await writeFile(path.join(output,'main.wasm'),bytes);
await copyFile(path.join(input,'.vkfbuild/main/wasm-manifest.json'),path.join(output,'manifest.json'));
const hashes={};for(const [name,source] of snapshots){await writeFile(path.join(sources,name),source);hashes[name]=hash(source);}
build.sources=hashes;build.wasm=hash(bytes);
bundle.applications[id]={directory,runtime_directory,runtime,wasm:hash(bytes),manifest:hash(await readFile(path.join(output,'manifest.json'))),sources:hashes,build};
const bridge=createRequire(import.meta.url)(path.join(compiler,'web/vf-ui/vf-compiled-runtime-bridge.js'));
const app=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(await readFile(path.join(output,'manifest.json')))});app.init();
const world=app.worldProgram().gpu_worlds[0],p=world.kind==='wind'?world.solid_properties:world.properties;
for(const url of [p.asset,...Object.values(p.variants??{})].filter(Boolean)){const asset=new URL(url,'https://vektorflow.org').pathname.slice(1);if(!asset.startsWith('previews/0.6.0/live/'))throw Error('Asset out of scope');bundle.assets[asset]=hash(await readFile(path.join(root,asset)));}
await writeFile(bundlePath,JSON.stringify(bundle,null,2)+'\n');
console.log(JSON.stringify({application:id,directory,runtime_directory,modules:Object.keys(runtime).length}));
