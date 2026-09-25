import {execFileSync} from 'node:child_process';
import {readFile, readdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const publicRoot = fileURLToPath(new URL('../public/', import.meta.url));
const selectorPattern = /<label class="release-selector"[\s\S]*?<\/label>/gu;
const index = await readFile(new URL('../public/index.html', import.meta.url), 'utf8');
const canonical = [...index.matchAll(selectorPattern)];
if (canonical.length !== 1) throw new Error('index must have exactly one release selector');
const currentOption = /<option value="0\.4\.7"[^>]*>[^<]*<\/option>/u.exec(canonical[0][0])?.[0];
if (!currentOption) throw new Error('index must have the current release option');

function updatedSelector(original) {
  const matches = [...original.matchAll(selectorPattern)];
  if (matches.length !== 1) throw new Error('page must have exactly one release selector');
  return matches[0][0]
    .replace(/<option value="0\.4\.7"[^>]*>[^<]*<\/option>/gu, '')
    .replace(/(<option\b[^>]*?) selected(?=>)/gu, '$1')
    .replace(/data-latest="[^"]+"/u, 'data-latest="0.4.7"')
    .replace(/(<select\b[^>]*>)/u, `$1${currentOption}`);
}

const pages = (await readdir(publicRoot)).filter(name => name.endsWith('.html'));
const changes = [];
for (const name of pages) {
  const url = new URL(`../public/${name}`, import.meta.url);
  const original = await readFile(url, 'utf8');
  const committed = execFileSync('git', ['show', `HEAD:public/${name}`], {
    cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8',
  });
  const updated = original.replace(selectorPattern, updatedSelector(committed));
  if (updated !== original) changes.push({url, updated, name});
}
for (const {url, updated} of changes) await writeFile(url, updated, 'utf8');
console.log(`Synced ${changes.length} current-page release selectors`);
