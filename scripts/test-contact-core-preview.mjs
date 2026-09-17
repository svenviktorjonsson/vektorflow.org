import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const base=new URL('../public/previews/0.6.0/contact-core-1/',import.meta.url);
const html=await readFile(new URL('index.html',base),'utf8'),run=await readFile(new URL('run.mjs',base),'utf8');
assert.ok(html.includes('This is not the wheel simulation.'));
assert.ok(html.includes('value="10000"')&&html.includes('value="1000000"'));
assert.ok(run.includes('.filter(n=>n<=maxParticles)'));
assert.ok(!run.includes('fetch('),'results must stay on the device');
for(const name of ['vf-contact-prefix-scan-gpu.mjs','vf-contact-graph-gpu.mjs','vf-contact-operator-gpu.mjs','vf-contact-dual-solve-gpu.mjs','vf-contact-velocity-audit-gpu.mjs']){
 const source=await readFile(new URL('runtime/'+name,base),'utf8');
 for(const match of source.matchAll(/from ['"]([^'"]+)['"]/g))await readFile(new URL(match[1],new URL('runtime/'+name,base)));
}
// Exercise the real asynchronous browser bootstrap with an unsupported GPU.
// This catches cyclic module-startup deadlocks without claiming GPU validation.
const nodes=Object.fromEntries(['status','progress','result','start','particles'].map(id=>[id,{textContent:'',disabled:false,value:''}]));
let finished;const done=new Promise(resolve=>finished=resolve);
Object.defineProperty(nodes.status,'textContent',{set(value){if(value.startsWith('FAIL'))finished(value);}});
globalThis.document={getElementById:id=>nodes[id]};globalThis.window={};globalThis.location={search:'?run=1&particles=10000'};
Object.defineProperty(globalThis,'navigator',{value:{},configurable:true});
let timer;
try{
 await import(new URL('report.mjs',base));
 await Promise.race([done,new Promise((_,reject)=>timer=setTimeout(()=>reject(Error('Browser bootstrap did not finish')),3000))]);
 assert.equal(window.__contactCoreResult.passed,false);
 assert.equal(window.__contactCoreResult.requestedMaxParticles,10000);
 assert.equal(nodes.start.disabled,false);
 assert.ok(nodes.result.textContent.includes('a712cf70'));
}finally{clearTimeout(timer);}
console.log('Contact test module closure, workload selection and failure bootstrap passed');
