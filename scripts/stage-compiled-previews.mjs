import { readFile,copyFile,mkdir,readdir,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const compiler=path.resolve(process.argv[2]??'../vektor-flow/build/branches/pre-gen');
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtimeDirectory='runtime-feedback-4';
const destination=path.join(site,'public/previews/0.6.0/compiled',runtimeDirectory);await mkdir(destination,{recursive:true});
const copied=new Set(),bundle={version:1,runtime_directory:runtimeDirectory,applications:{},runtime:{},assets:{}};
const bridge=createRequire(import.meta.url)(path.join(compiler,'web/vf-ui/vf-compiled-runtime-bridge.js'));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
async function runtime(name){if(copied.has(name))return;copied.add(name);const file=path.join(compiler,'web/vf-ui',name),source=await readFile(file,'utf8');await copyFile(file,path.join(destination,name));for(const match of source.matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))await runtime(match[1]);}
for(const name of ['vf-world-layer-runtime.js','vf-compiled-runtime-bridge.js','vf-world-material-runtime.mjs','vf-world-mechanical-runtime.mjs'])await runtime(name);
for(const [id,folder] of [['wheel','world-wheel'],['stones','world-stones'],['tree','world-tree']]){
  const directory=`${id}-feedback-4`;
  const input=path.join(compiler,'examples',folder),output=path.join(site,'public/previews/0.6.0/compiled',directory),sources=path.join(site,'public/sources/coming-soon',id);await mkdir(output,{recursive:true});await mkdir(sources,{recursive:true});
  await copyFile(path.join(input,'.vkfbuild/main/main.wasm'),path.join(output,'main.wasm'));await copyFile(path.join(input,'.vkfbuild/main/wasm-manifest.json'),path.join(output,'manifest.json'));
  const bytes=await readFile(path.join(output,'main.wasm'));if(!WebAssembly.validate(bytes))throw new Error(`${id}: invalid compiled artifact`);
  const hashes={};for(const name of await readdir(input))if(name.endsWith('.vkf')){await copyFile(path.join(input,name),path.join(sources,name));hashes[name]=digest(await readFile(path.join(input,name)));}
  bundle.applications[id]={directory,wasm:digest(bytes),manifest:digest(await readFile(path.join(output,'manifest.json'))),sources:hashes};
  const app=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(await readFile(path.join(output,'manifest.json'),'utf8'))});app.init();
  for(const world of app.worldProgram().gpu_worlds){const properties=world.kind==='wind'?world.solid_properties:world.properties;for(const url of [properties.asset,...Object.values(properties.variants??{})].filter(Boolean)){const pathname=new URL(url,'https://vektorflow.org').pathname;if(!pathname.startsWith('/previews/0.6.0/live/'))throw new Error('Geometry asset outside live preview scope');bundle.assets[pathname.slice(1)]=digest(await readFile(path.join(site,'public',...pathname.slice(1).split('/'))));}}
  console.log(`${id}: valid WASM, executable source staged`);
}
console.log(`${copied.size} runtime modules staged`);
for(const name of copied)bundle.runtime[name]=digest(await readFile(path.join(destination,name)));
await writeFile(path.join(destination,'../bundle.json'),JSON.stringify(bundle,null,2)+'\n');
