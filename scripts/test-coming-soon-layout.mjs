import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';

const root=new URL('../public/',import.meta.url);
const html=await readFile(new URL('coming-soon.html',root),'utf8');
assert.equal((html.match(/<h1\b/g)??[]).length,1);
assert.equal((html.match(/<h2\b/g)??[]).length,4);
assert.equal((html.match(/<h[3-6]\b/g)??[]).length,0);
assert.match(html,/0\.6\.0 work is in progress/);
assert.match(html,/href="\.\/index\.html">Home<\/a>/);
assert.match(html,/href="\.\/origins\.html">Origins<\/a>/);

const sections=['contact-tests','wheel','stones','tree'];
for(const id of sections){
  const section=html.match(new RegExp(`<section id="${id}"[\\s\\S]*?<\\/section>`))?.[0];
  assert.ok(section,`Missing ${id} preview`);
  assert.match(section,/Purpose:/);
  assert.match(section,/State:/);
  assert.equal((section.match(/<a href="\.\/previews\/0\.6\.0\//g)??[]).length,1,
    `${id} should have one direct preview link`);
  if(id!=='contact-tests')assert.match(section,/<iframe[^>]+loading="lazy"/);
}
for(const [,address] of html.matchAll(/\b(?:href|src)="(\.\/[^"#]+)"/g)){
  const pathname=address.split(/[?#]/)[0];
  let target=new URL(pathname,root);
  if(target.pathname.endsWith('/'))target=new URL('index.html',target);
  assert.ok((await stat(target)).isFile(),`Broken page link or preview: ${address}`);
}
assert.match(html,/\.session-shell iframe \{ display: block; width: 100%; height: min\(76vh, 760px\);/);
assert.match(html,/@media \(max-width: 680px\)/);
console.log('Coming Soon: flat headings, four distinct previews, and every local link/iframe resolved');
