// Publish only the accepted stone application; do not replace the wheel/tree.
import {readFile,copyFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const compiler=path.resolve(process.argv[2]??'../vektor-flow/build/branches/pre-gen');
const root=path.join(site,'public'),compiled=path.join(root,'previews/0.6.0/compiled');
const id=process.argv[3]??'stones';if(!['stones','tree'].includes(id))throw Error('Unknown mechanical application');
const runtime_directory=id==='stones'?'runtime-stones-7':'runtime-tree-8',directory=id==='stones'?'stones-granite-7':'tree-leaves-8';
const bundlePath=path.join(compiled,'bundle.json'),bundle=JSON.parse(await readFile(bundlePath));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),runtime={};
await mkdir(path.join(compiled,runtime_directory),{recursive:true});
async function copyRuntime(name){
  if(runtime[name])return;const source=await readFile(path.join(compiler,'web/vf-ui',name));
  await writeFile(path.join(compiled,runtime_directory,name),source);runtime[name]=hash(source);
  for(const match of source.toString().matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))await copyRuntime(match[1]);
}
for(const name of ['vf-world-layer-runtime.js','vf-compiled-runtime-bridge.js','vf-world-mechanical-runtime.mjs'])await copyRuntime(name);
const input=path.join(compiler,'examples',id==='stones'?'world-stones':'world-tree'),output=path.join(compiled,directory),sources=path.join(root,'sources/coming-soon',id);
await mkdir(output,{recursive:true});await mkdir(sources,{recursive:true});
const bytes=await readFile(path.join(input,'.vkfbuild/main/main.wasm'));
if(!WebAssembly.validate(bytes))throw Error('Invalid stone WASM');
await writeFile(path.join(output,'main.wasm'),bytes);
await copyFile(path.join(input,'.vkfbuild/main/wasm-manifest.json'),path.join(output,'manifest.json'));
const hashes={};for(const name of ['main.vkf','geometry.vkf','materials.vkf']){const source=await readFile(path.join(input,name));await writeFile(path.join(sources,name),source);hashes[name]=hash(source);}
bundle.applications[id]={directory,runtime_directory,runtime,wasm:hash(bytes),manifest:hash(await readFile(path.join(output,'manifest.json'))),sources:hashes};
const bridge=createRequire(import.meta.url)(path.join(compiler,'web/vf-ui/vf-compiled-runtime-bridge.js'));
const app=bridge.instantiateWasmRuntime({bytes,manifest:JSON.parse(await readFile(path.join(output,'manifest.json')))});app.init();
const world=app.worldProgram().gpu_worlds[0],p=world.kind==='wind'?world.solid_properties:world.properties;
for(const url of [p.asset,...Object.values(p.variants??{})]){const asset=new URL(url,'https://vektorflow.org').pathname.slice(1);if(!asset.startsWith('previews/0.6.0/live/'))throw Error('Asset out of scope');bundle.assets[asset]=hash(await readFile(path.join(root,asset)));}
await writeFile(bundlePath,JSON.stringify(bundle,null,2)+'\n');
console.log(JSON.stringify({application:id,directory,runtime_directory,modules:Object.keys(runtime).length}));
