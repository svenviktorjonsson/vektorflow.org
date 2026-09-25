import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const wheelTest = fileURLToPath(new URL('./test-wheel-page-gpu.mjs', import.meta.url));

function sample(mode) {
  const run = spawnSync(process.execPath, [wheelTest, mode], {
    encoding: 'utf8',
    maxBuffer: 8 * 1024 * 1024,
    timeout: 120_000,
  });
  const reportLine = run.stdout?.trim().split(/\r?\n/).at(-1);
  let report;
  try { report = JSON.parse(reportLine); }
  catch { throw new Error(`${mode} produced no GPU report: ${run.stderr || run.stdout}`); }
  if (run.status !== 0 || !report.passed || !report.shape) {
    throw new Error(`${mode} failed: ${report.error || run.stderr || 'GPU regression failed'}`);
  }
  return report.shape;
}

const dry = sample('--sand-dropcastle-dry');
const wet = sample('--sand-dropcastle-wet');
const width = shape => shape.bounds[2] - shape.bounds[0];
const dryWidth = width(dry), wetWidth = width(wet);
const passed = wetWidth <= dryWidth - 0.08 && wet.meanY >= dry.meanY + 0.025;
const result = {
  passed,
  dry: {width: dryWidth, meanY: dry.meanY, coverage: dry.diskAreaFraction},
  wet: {width: wetWidth, meanY: wet.meanY, coverage: wet.diskAreaFraction},
};
console.log(JSON.stringify(result));
if (!passed) process.exitCode = 1;
