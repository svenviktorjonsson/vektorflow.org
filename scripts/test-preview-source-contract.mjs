import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {applications,previewSourceUrl,verifyPreviewBytes} from '../public/coming-soon.mjs';
const root=new URL('../public/',import.meta.url),bundle=JSON.parse(await readFile(new URL('previews/0.6.0/compiled/bundle.json',root)));
const hash=b=>createHash('sha256').update(b).digest('hex');
const pages={wheel:'material-wheel',stones:'rocks',tree:'tree'};
const comingSoon=await readFile(new URL('coming-soon.html',root),'utf8');
const linkedPreviews=[...comingSoon.matchAll(/(?:href|src)="\.\/previews\/0\.6\.0\/live\/(material-wheel|rocks|tree)\/\?v=([^"]+)"/g)];
for(const {id,files} of applications){
 const record=bundle.applications[id],index=await readFile(new URL(`previews/0.6.0/live/${pages[id]}/index.html`,root),'utf8');
 const links=linkedPreviews.filter(([,page])=>page===pages[id]);
 assert.equal(links.length,2,`${id}: Coming Soon must link and embed exactly one preview`);
 assert.ok(links.every(([,page,version])=>version===record.directory),`${id}: stale Coming Soon preview URL`);
 assert.ok(index.includes(`../../compiled/${record.directory}/main.wasm`),'iframe must run the artifact associated with shown source');
 assert.ok(index.includes(`../../compiled/${record.runtime_directory??bundle.runtime_directory}/vf-world-layer-runtime.js`));
 for(const name of files){const bytes=await readFile(new URL(`sources/coming-soon/${id}/${name}`,root));await verifyPreviewBytes(bytes,record.sources[name]);
  assert.equal(previewSourceUrl(id,name,record.sources[name]),`./sources/coming-soon/${id}/${name}?v=${record.sources[name]}`);
  const changed=Buffer.concat([bytes,Buffer.from('\n# changed import')]);await assert.rejects(verifyPreviewBytes(changed,record.sources[name]),/Source\/build mismatch/);
  const text=bytes.toString(),highlight=globalThis.Prism.highlight(text,globalThis.Prism.languages.vkf,'vkf');assert.ok(highlight.includes('class="token'));assert.ok(!highlight.includes('<script'));
 }
 assert.equal(record.build.wasm,record.wasm);assert.deepEqual(record.build.sources,record.sources);assert.equal(record.build.entry,'main.vkf');assert.match(record.build.compiler,/^[a-f0-9]{64}$/);
}
const attack=globalThis.Prism.highlight('# <script>alert(1)</script>',globalThis.Prism.languages.vkf,'vkf');assert.ok(attack.includes('&lt;script>'));assert.ok(!attack.includes('<script>'));
assert.equal(hash(await readFile(new URL('vendor/prism/prism-core-1.30.0.min.js',root))).length,64);
console.log('Prism read-only code, modified-import rejection, build receipts and iframe artifact identity verified');
