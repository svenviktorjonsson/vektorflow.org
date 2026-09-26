import assert from 'node:assert/strict';
import {readFile, readdir} from 'node:fs/promises';

const publicUrl = new URL('../public/', import.meta.url);
const pages = (await readdir(publicUrl)).filter(name => name.endsWith('.html'));
for (const name of pages) {
  const html = await readFile(new URL(name, publicUrl), 'utf8');
  assert.match(html, /data-latest="0\.4\.7"/u, `${name}: current selection`);
  assert.match(html,
    /<option value="0\.4\.7" data-browser-wasm="https:\/\/vektorflow\.org\/runtimes\/v0\.4\.7\/vkf-shared-compiler\.wasm\?sha256=[a-f0-9]{64}" selected>0\.4\.7<\/option>/u,
    `${name}: supported browser compiler`);
  assert.equal((html.match(/<option /gu) ?? []).length, 1, `${name}: one supported compiler`);
  assert.doesNotMatch(html, /value="0\.4\.[0-6]"/u, `${name}: no superseded compiler`);
}
const install = await readFile(new URL('install.html', publicUrl), 'utf8');
assert.match(install, /<section class="release-panel" data-release-version="0\.4\.7">/u);
assert.match(install, /Package unavailable/u);
assert.doesNotMatch(install, /data-release-version="0\.4\.[0-6]"/u);
console.log(`${pages.length} pages expose only the supported stable compiler`);
