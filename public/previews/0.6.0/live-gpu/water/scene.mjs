import {
  createLiquidParticleWorldGpuRuntime,
} from './runtime/vf-liquid-particle-world-gpu.mjs';
import {
  createLiquidParticleEmbeddingGpu,
} from './runtime/vf-liquid-particle-embedding-gpu.mjs';
import {
  createFixedStepRealtimeClock,
} from './runtime/fixed-step-realtime-clock.mjs';

const canvas = document.getElementById('water');
const playButton = document.getElementById('play');
const resetButton = document.getElementById('reset');
const fluidButton = document.getElementById('fluid');
const particlesButton = document.getElementById('particles');
const status = document.getElementById('status');
const errorBox = document.getElementById('error');

let playing = true;
let mode = 'fluid';
let runtime = null;
let embedding = null;
let realtimeClock = null;
let previousTimestamp = null;
let simulatedTime = 0;
let renderedFrames = 0;
let capturePending = false;
const frameIntervals = [];

const fail = (error) => {
  playing = false;
  const message = String(error?.stack || error);
  errorBox.hidden = false;
  errorBox.textContent = message;
  status.textContent = 'WebGPU unavailable';
  window.__water2dGpuResult = { outcome: 'fail', error: message };
};

const setMode = (nextMode) => {
  mode = nextMode;
  fluidButton.setAttribute('aria-pressed', String(mode === 'fluid'));
  particlesButton.setAttribute('aria-pressed', String(mode === 'particles'));
};

const measuredFps = () => {
  if (frameIntervals.length < 2) return 0;
  const sorted = [...frameIntervals].sort((left, right) => left - right);
  const median = sorted[Math.floor(sorted.length / 2)];
  return median > 0 ? 1000 / median : 0;
};

const updateStatus = () => {
  const fps = measuredFps();
  const timing = realtimeClock.snapshot();
  status.textContent = `${fps ? fps.toFixed(0) : '—'} Hz display · ${runtime.primaryCount}`
    + ` liquid + ${runtime.diffuseCapacity} diffuse slots · GPU compute`
    + ` · ${(timing.debtSeconds * 1000).toFixed(1)} ms debt`
    + ` · ${(timing.droppedSimulationSeconds * 1000).toFixed(1)} ms dropped`;
};

const timingTelemetry = () => Object.freeze({
  ...realtimeClock.snapshot(),
  sceneSimulatedSeconds: simulatedTime,
  playing,
});

const draw = (timestamp, advance = true) => {
  if (!runtime || !embedding) return;
  const encoder = runtime.device.createCommandEncoder({
    label: 'VKF 2D GPU liquid frame',
  });
  if (advance && playing && previousTimestamp != null) {
    const elapsed = Math.max(0, (timestamp - previousTimestamp) / 1000);
    const timing = realtimeClock.advance(elapsed);
    runtime.stepMany(encoder, timing.steps, runtime.policy.timeStep);
    simulatedTime += timing.simulatedDeltaSeconds;
  }
  embedding.render(encoder, { time: simulatedTime, mode });
  runtime.device.queue.submit([encoder.finish()]);
};

const frame = (timestamp) => {
  try {
    if (capturePending) {
      previousTimestamp = timestamp;
      requestAnimationFrame(frame);
      return;
    }
    if (previousTimestamp != null) {
      const interval = timestamp - previousTimestamp;
      if (interval > 0 && interval < 100) {
        frameIntervals.push(interval);
        if (frameIntervals.length > 120) frameIntervals.shift();
      }
    }
    draw(timestamp);
    previousTimestamp = timestamp;
    renderedFrames += 1;
    if (renderedFrames % 20 === 0) updateStatus();
    requestAnimationFrame(frame);
  } catch (error) {
    fail(error);
  }
};

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(playing));
  previousTimestamp = null;
});
resetButton.addEventListener('click', () => {
  runtime.reset();
  simulatedTime = 0;
  realtimeClock.reset();
  previousTimestamp = null;
  frameIntervals.length = 0;
  draw(performance.now(), false);
  updateStatus();
});
fluidButton.addEventListener('click', () => setMode('fluid'));
particlesButton.addEventListener('click', () => setMode('particles'));

try {
  if (!navigator.gpu) throw new Error('This proof requires WebGPU; no CPU particle fallback is used.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  device.lost.then((information) => fail(new Error(`WebGPU device lost: ${information.message}`)));
  device.addEventListener('uncapturederror', (event) => fail(event.error));
  runtime = await createLiquidParticleWorldGpuRuntime(device);
  realtimeClock = createFixedStepRealtimeClock({
    timeStep: runtime.policy.timeStep,
  });
  embedding = await createLiquidParticleEmbeddingGpu(device, canvas, runtime, {
    maximumPixelRatio: 2,
  });
  window.__water2dGpuRuntime = runtime;
  window.__water2dGpuEmbedding = embedding;
  window.__water2dGpuTiming = timingTelemetry;
  window.__water2dGpuResult = null;
  window.__water2dGpuCapture = async ({ steps = 1, captureMode = mode } = {}) => {
    capturePending = true;
    try {
      const count = Math.max(0, Math.floor(steps));
      const encoder = device.createCommandEncoder();
      runtime.stepMany(encoder, count, runtime.policy.timeStep);
      simulatedTime += count * runtime.policy.timeStep;
      embedding.render(encoder, { time: simulatedTime, mode: captureMode });
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const telemetry = await runtime.readTelemetry();
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
      return { step: runtime.frameIndex, simulatedTime, mode: captureMode,
        timing: timingTelemetry(), telemetry,
        dataUrl: canvas.toDataURL('image/png') };
    } finally {
      capturePending = false;
    }
  };
  requestAnimationFrame(frame);
} catch (error) {
  fail(error);
}
