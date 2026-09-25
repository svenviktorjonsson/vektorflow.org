import assert from 'node:assert/strict';
import {readFile, readdir} from 'node:fs/promises';

const publicUrl = new URL('../public/', import.meta.url);
const pages = (await readdir(publicUrl)).filter(name => name.endsWith('.html'));
for (const name of pages) {
  const html = await readFile(new URL(name, publicUrl), 'utf8');
  assert.match(html, /data-latest="0\.4\.7"/u, `${name}: current selection`);
  assert.match(html, /<option value="0\.4\.7" data-browser-wasm="" selected>/u,
    `${name}: visible current option`);
}
for (const name of ['index.html', 'performance.html']) {
  const html = await readFile(new URL(name, publicUrl), 'utf8');
  assert.match(html, /<section data-release-version="0\.4\.7">/u, `${name}: current panel`);
  assert.match(html, /<section data-release-version="0\.4\.5" hidden>/u,
    `${name}: historical panel hidden by default`);
}
const install = await readFile(new URL('install.html', publicUrl), 'utf8');
assert.match(install, /<section class="release-panel" data-release-version="0\.4\.7">/u);
assert.match(install, /<section class="release-panel" data-release-version="0\.4\.5" hidden>/u);
console.log(`${pages.length} current pages select the current release; historical panels remain separate`);
