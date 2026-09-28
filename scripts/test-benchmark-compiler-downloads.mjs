import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';

const release = new URL('../public/downloads/v0.4.7/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('benchmark-compilers.json', release)));
const page = await readFile(new URL('../public/performance.html', import.meta.url), 'utf8');
assert.equal(manifest.version, '0.4.7');
assert.equal(Object.keys(manifest.platforms).length, 2);
for (const [platform, entry] of Object.entries(manifest.platforms)) {
  const compiler = await readFile(new URL(entry.compiler, release));
  const receiptBytes = await readFile(new URL(entry.receipt, release));
  const receipt = JSON.parse(receiptBytes);
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  assert.equal(digest(compiler), entry.sha256, `${platform}: compiler bytes`);
  assert.equal(digest(receiptBytes), entry.receiptSha256, `${platform}: receipt bytes`);
  assert.equal(receipt.version, manifest.version);
  assert.equal(receipt.options.runs, 10);
  assert.equal(receipt.results.length, 20);
  assert.equal(receipt.correctness.passed, true);
  assert.ok(page.includes(`./downloads/v0.4.7/${entry.compiler}`), `${platform}: download link`);
  assert.ok(page.includes(entry.sha256), `${platform}: visible compiler hash`);
  assert.ok(page.includes(entry.receipt.split('/').at(-1)), `${platform}: receipt link`);
}
console.log('Benchmark compiler downloads and 10-run receipts verified');
