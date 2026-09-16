import { readFile,copyFile,mkdir,readdir,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const compiler=path.resolve(process.argv[2]??'../vektor-flow/build/branches/pre-gen');
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const destination=path.join(site,'public/previews/0.6.0/compiled/runtime');await mkdir(destination,{recursive:true});
const copied=new Set(),bundle={version:1,applications:{},runtime:{}};
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
async function runtime(name){if(copied.has(name))return;copied.add(name);const file=path.join(compiler,'web/vf-ui',name),source=await readFile(file,'utf8');await copyFile(file,path.join(destination,name));for(const match of source.matchAll(/(?:from\s*|import\s*)['"]\.\/([^'"]+)['"]/g))await runtime(match[1]);}
for(const name of ['vf-world-layer-runtime.js','vf-compiled-runtime-bridge.js','vf-world-material-runtime.mjs','vf-world-mechanical-runtime.mjs'])await runtime(name);
for(const [id,folder] of [['wheel','world-wheel'],['stones','world-stones'],['tree','world-tree']]){
  const input=path.join(compiler,'examples',folder),output=path.join(site,'public/previews/0.6.0/compiled',id),sources=path.join(site,'public/sources/coming-soon',id);await mkdir(output,{recursive:true});await mkdir(sources,{recursive:true});
  await copyFile(path.join(input,'.vkfbuild/main/main.wasm'),path.join(output,'main.wasm'));await copyFile(path.join(input,'.vkfbuild/main/wasm-manifest.json'),path.join(output,'manifest.json'));
  const bytes=await readFile(path.join(output,'main.wasm'));if(!WebAssembly.validate(bytes))throw new Error(`${id}: invalid compiled artifact`);
  const hashes={};for(const name of await readdir(input))if(name.endsWith('.vkf')){await copyFile(path.join(input,name),path.join(sources,name));hashes[name]=digest(await readFile(path.join(input,name)));}
  bundle.applications[id]={wasm:digest(bytes),manifest:digest(await readFile(path.join(output,'manifest.json'))),sources:hashes};
  console.log(`${id}: valid WASM, executable source staged`);
}
console.log(`${copied.size} runtime modules staged`);
for(const name of copied)bundle.runtime[name]=digest(await readFile(path.join(destination,name)));
await writeFile(path.join(destination,'../bundle.json'),JSON.stringify(bundle,null,2)+'\n');
