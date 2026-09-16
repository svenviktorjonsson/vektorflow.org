import { createLiquidParticleWorldGpuRuntime } from './runtime/vf-liquid-particle-world-gpu.mjs?v=wheel-pressure-9';
import { createLiquidParticleEmbeddingGpu } from './runtime/vf-liquid-particle-embedding-gpu.mjs?v=raw-particles-11';
import { createFixedStepRealtimeClock } from './runtime/fixed-step-realtime-clock.mjs';
import { createGranularParticleWorldGpuRuntime } from '../sand/runtime/vf-granular-particle-world-gpu.mjs?v=sand-repose-11';
import { createGranularParticleEmbeddingGpu } from '../sand/runtime/vf-granular-particle-embedding-gpu.mjs?v=sand-density-11';
import { createWheelEmbeddingGpu } from './wheel-embedding-gpu.mjs';
import { createVfLiveWorldStackReference } from '../runtime/vf-live-world-stack.mjs?v=world-stack-1';

const canvas = document.getElementById('stage');
const waterButton = document.getElementById('water');
const sandButton = document.getElementById('sand');
const particleButton = document.getElementById('particles');
const playButton = document.getElementById('play');
const resetButton = document.getElementById('reset');
const status = document.getElementById('status');
const errorBox = document.getElementById('error');

let material = 'water';
let showParticles = false;
let playing = true;
let previousTimestamp = null;
let wheelAngle = 0.22;
let wheelAngularVelocity = 0;
let wheelDrag = null;
let frameCount = 0;
let waterRuntime;
let sandRuntime;
let waterEmbedding;
let sandEmbedding;
let wheelEmbedding;
let waterClock;
let sandClock;
let device;
let context;
let worldStack;
const MAX_DRUM_ANGULAR_SPEED = 2.4;
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

const fail = (error) => {
  if (!errorBox.hidden) return;
  playing = false;
  errorBox.hidden = false;
  errorBox.textContent = String(error?.message || error?.stack || error);
  status.textContent = 'WebGPU unavailable';
  window.__materialWheelResult = { outcome: 'fail', error: errorBox.textContent };
};

const selected = () => {
  const world = worldStack?.active;
  if (!world) return material === 'water'
    ? { runtime: waterRuntime, embedding: waterEmbedding, clock: waterClock,
      mode: showParticles ? 'particles' : 'fluid' }
    : { runtime: sandRuntime, embedding: sandEmbedding, clock: sandClock,
      mode: showParticles ? 'particles' : 'sand' };
  return { world, runtime: world.read('runtime'), embedding: world.read('embedding'),
    clock: world.read('clock'), mode: showParticles ? 'particles' : world.read('embeddedMode') };
};

const setMaterial = (next) => {
  material = next;
  worldStack?.flip(next);
  waterButton.setAttribute('aria-pressed', String(next === 'water'));
  sandButton.setAttribute('aria-pressed', String(next === 'sand'));
  previousTimestamp = null;
};

waterButton.addEventListener('click', () => setMaterial('water'));
sandButton.addEventListener('click', () => setMaterial('sand'));
particleButton.addEventListener('click', () => {
  showParticles = !showParticles;
  particleButton.setAttribute('aria-pressed', String(showParticles));
});
playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(playing));
  previousTimestamp = null;
});
resetButton.addEventListener('click', () => {
  const active = selected();
  if (active.world) active.world.reset();
  else { active.runtime.reset(); active.clock.reset(); }
  wheelAngle = 0.22;
  wheelAngularVelocity = 0;
  previousTimestamp = null;
});

const normalizeAngle = (value) => Math.atan2(Math.sin(value), Math.cos(value));

canvas.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  if (!wheelEmbedding) return;
  const active = selected();
  const point = wheelEmbedding.screenToWorld(event, active.runtime.policy);
  const wheel = active.runtime.wheel;
  if (Math.hypot(point[0] - wheel.center[0], point[1] - wheel.center[1]) > wheel.radius * 1.24) return;
  const pointerAngle = Math.atan2(point[1] - wheel.center[1], point[0] - wheel.center[0]);
  wheelDrag = { pointerId: event.pointerId, offset: normalizeAngle(pointerAngle - wheelAngle),
    lastAngle: wheelAngle, lastTime: performance.now() };
  wheelAngularVelocity = 0;
  canvas.setPointerCapture(event.pointerId);
  canvas.dataset.dragging = 'true';
});

canvas.addEventListener('pointermove', (event) => {
  event.preventDefault();
  if (!wheelDrag || event.pointerId !== wheelDrag.pointerId) return;
  const active = selected();
  const point = wheelEmbedding.screenToWorld(event, active.runtime.policy);
  const wheel = active.runtime.wheel;
  const pointerAngle = Math.atan2(point[1] - wheel.center[1], point[0] - wheel.center[0]);
  const next = normalizeAngle(pointerAngle - wheelDrag.offset);
  const now = performance.now();
  const dt = Math.max(1 / 240, Math.min(0.05, (now - wheelDrag.lastTime) / 1000));
  const delta = normalizeAngle(next - wheelDrag.lastAngle);
  wheelAngularVelocity = clamp(wheelAngularVelocity * 0.35 + delta / dt * 0.65,
    -MAX_DRUM_ANGULAR_SPEED, MAX_DRUM_ANGULAR_SPEED);
  wheelAngle = next;
  wheelDrag.lastAngle = next;
  wheelDrag.lastTime = now;
});

const releaseWheel = (event) => {
  event.preventDefault();
  if (!wheelDrag || event.pointerId !== wheelDrag.pointerId) return;
  wheelDrag = null;
  wheelAngularVelocity = 0;
  canvas.dataset.dragging = 'false';
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
};
canvas.addEventListener('pointerup', releaseWheel);
canvas.addEventListener('pointercancel', releaseWheel);

// iOS can hand a gesture from an iframe to the parent scroller before pointer
// capture settles. A non-passive touch guard keeps the whole simulation surface
// owned by the drum interaction from the first sample onward.
for (const type of ['touchstart', 'touchmove', 'touchend', 'touchcancel']) {
  canvas.addEventListener(type, (event) => event.preventDefault(), { passive: false });
}

const renderFrame = (timestamp) => {
  try {
    const active = selected();
    if (!active.runtime || !active.embedding) return;
    const elapsed = previousTimestamp == null ? 0
      : Math.max(0, Math.min(0.08, (timestamp - previousTimestamp) / 1000));
    previousTimestamp = timestamp;
    if (!wheelDrag) {
      wheelAngle = normalizeAngle(wheelAngle + wheelAngularVelocity * elapsed);
      wheelAngularVelocity *= Math.exp(-1.05 * elapsed);
      if (Math.abs(wheelAngularVelocity) < 0.004) wheelAngularVelocity = 0;
    }
    active.runtime.setWheel({ angle: wheelAngle, angularVelocity: wheelAngularVelocity });
    const encoder = device.createCommandEncoder({ label: 'VKF material wheel frame' });
    if (playing && elapsed > 0) {
      if (active.world) active.world.advance({ encoder, elapsed });
      else {
        const timing = active.clock.advance(elapsed);
        if (timing.steps > 0) active.runtime.stepMany(encoder, timing.steps,
          active.runtime.policy.timeStep);
      }
    }
    active.embedding.render(encoder, { time: timestamp / 1000, mode: active.mode });
    wheelEmbedding.render(encoder, context.getCurrentTexture().createView(),
      active.runtime.policy, active.runtime.wheel, wheelAngle);
    device.queue.submit([encoder.finish()]);
    frameCount += 1;
    if (frameCount % 18 === 0) {
      const count = active.runtime.particleCount ?? active.runtime.primaryCount;
      status.textContent = `${material === 'water' ? 'Water' : 'Sand'} · ${count} particles · GPU compute`
        + ` · Ø 1.0 m · drum ${wheelAngularVelocity.toFixed(2)} rad/s`;
    }
    requestAnimationFrame(renderFrame);
  } catch (error) { fail(error); }
};

try {
  if (!navigator.gpu) throw new Error('This live session requires WebGPU.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  device = await adapter.requestDevice();
  device.lost.then((information) => fail(new Error(`WebGPU device lost: ${information.message}`)));
  device.addEventListener('uncapturederror', (event) => {
    console.error(event.error?.message || event.error);
    fail(event.error);
  });
  waterRuntime = await createLiquidParticleWorldGpuRuntime(device);
  sandRuntime = await createGranularParticleWorldGpuRuntime(device);
  waterEmbedding = await createLiquidParticleEmbeddingGpu(device, canvas, waterRuntime, { maximumPixelRatio: 2 });
  sandEmbedding = await createGranularParticleEmbeddingGpu(device, canvas, sandRuntime, { maximumPixelRatio: 2 });
  const format = navigator.gpu.getPreferredCanvasFormat();
  context = canvas.getContext('webgpu');
  wheelEmbedding = await createWheelEmbeddingGpu(device, canvas, format);
  waterClock = createFixedStepRealtimeClock({ timeStep: waterRuntime.policy.timeStep });
  sandClock = createFixedStepRealtimeClock({ timeStep: sandRuntime.policy.timeStep });
  worldStack = createVfLiveWorldStackReference();
  const waterWorld = worldStack.append({ id: 'water', label: 'Water' })
    .add({ runtime: waterRuntime, particleCount: waterRuntime.particleCount })
    .add({ embedding: waterEmbedding, clock: waterClock, embeddedMode: 'fluid' })
    .push({ id: 'gravity-sph-local-rules', collisions: 'fluid',
      advance({ encoder, elapsed }) {
        const timing = waterClock.advance(elapsed);
        if (timing.steps > 0) waterRuntime.stepMany(encoder, timing.steps,
          waterRuntime.policy.timeStep);
      }, reset() { waterRuntime.reset(); waterClock.reset(); } });
  const sandWorld = worldStack.append({ id: 'sand', label: 'Sand' })
    .add({ runtime: sandRuntime, particleCount: sandRuntime.particleCount })
    .add({ embedding: sandEmbedding, clock: sandClock, embeddedMode: 'sand' })
    .push({ id: 'gravity-granular-local-rules', collisions: 'granular',
      advance({ encoder, elapsed }) {
        const timing = sandClock.advance(elapsed);
        if (timing.steps > 0) sandRuntime.stepMany(encoder, timing.steps,
          sandRuntime.policy.timeStep);
      }, reset() { sandRuntime.reset(); sandClock.reset(); } });
  worldStack.flip(material);
  window.__materialWheel = { worldStack, waterWorld, sandWorld,
    waterRuntime, sandRuntime, waterEmbedding, sandEmbedding };
  window.__materialWheelResult = null;
  requestAnimationFrame(renderFrame);
} catch (error) { fail(error); }
