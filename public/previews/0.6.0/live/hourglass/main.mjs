import {createGranularApertureWorldGpu} from './vf-granular-aperture-world-gpu.mjs';
import {createGranularApertureEmbeddingGpu} from './vf-granular-aperture-embedding-gpu.mjs';

const canvas = document.querySelector('#scene');
const status = document.querySelector('#status');
const play = document.querySelector('#play');
const reset = document.querySelector('#reset');
const flip = document.querySelector('#flip');
const audit = document.querySelector('#audit');
const throat = document.querySelector('#throat');
const width = document.querySelector('#width');
const angle = document.querySelector('#angle');
const speed = document.querySelector('#speed');
const widthValue = document.querySelector('#width-value');
const angleValue = document.querySelector('#angle-value');
const speedValue = document.querySelector('#speed-value');
const throatValue = document.querySelector('#throat-value');

async function start() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  device.lost.then(info => { status.textContent = `GPU device lost: ${info.message}`; });
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('WebGPU canvas context is unavailable.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  const world = await createGranularApertureWorldGpu(device);
  const embedding = await createGranularApertureEmbeddingGpu(device, canvas,
    context, format, world);
  const resize = () => {
    const scale = Math.min(window.devicePixelRatio || 1, 1.5);
    const nextWidth = Math.max(1, Math.round(canvas.clientWidth * scale));
    const nextHeight = Math.max(1, Math.round(canvas.clientHeight * scale));
    if (canvas.width === nextWidth && canvas.height === nextHeight) return;
    canvas.width = nextWidth;
    canvas.height = nextHeight;
    context.configure({device, format, alphaMode: 'opaque'});
  };
  let running = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  let accumulator = 0;
  let previous = performance.now();
  let lastStatus = 0;
  const refreshControls = () => {
    play.textContent = running ? 'Pause' : 'Play';
    widthValue.value = `${Number(width.value).toFixed(2)} px`;
    angleValue.value = `${angle.value}°`;
    speedValue.value = `${speed.value}×`;
    throatValue.value = `${Number(throat.value).toFixed(1)} mm`;
    world.setReposeDegrees(Number(angle.value));
    world.setOpening(Number(throat.value) / 1000);
    embedding.setFlowWidthPixels(Number(width.value));
  };
  refreshControls();
  play.addEventListener('click', () => { running = !running; refreshControls(); });
  reset.addEventListener('click', () => { world.reset(); accumulator = 0; });
  width.addEventListener('input', refreshControls);
  angle.addEventListener('input', refreshControls);
  speed.addEventListener('input', refreshControls);
  throat.addEventListener('input', refreshControls);
  audit.addEventListener('click', async () => {
    audit.disabled = true;
    try {
      const state = await world.readAudit();
      const retained = (state.retainedFraction * 100).toFixed(4);
      const targetArea = state.direction > 0 ? state.chamberBArea : state.chamberAArea;
      const profileError = ((state.pileProfileArea - targetArea)
        / world.initialArea * 100).toFixed(5);
      status.textContent = `${state.simulatedSeconds.toFixed(1)} s · ${retained}% area retained · pile profile error ${profileError}%`;
      window.__hourglassLastAudit = state;
    } catch (error) { status.textContent = `Audit failed: ${error.message}`; }
    finally { audit.disabled = false; }
  });
  const pointerAngle = event => {
    const rect = canvas.getBoundingClientRect();
    return Math.atan2(event.clientY - (rect.top + rect.height * 0.485),
      event.clientX - (rect.left + rect.width * 0.5));
  };
  let dragging = false;
  let previousPointerAngle = 0;
  canvas.addEventListener('pointerdown', event => {
    dragging = true;
    previousPointerAngle = pointerAngle(event);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const angleNow = pointerAngle(event);
    const delta = Math.atan2(Math.sin(angleNow - previousPointerAngle),
      Math.cos(angleNow - previousPointerAngle));
    world.setPoseAngle(world.poseAngle + delta);
    previousPointerAngle = angleNow;
  });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  flip.addEventListener('click', () => world.setPoseAngle(world.poseAngle + Math.PI));
  // Read-only diagnostics for reproducible browser tests; never reads the GPU
  // during ordinary frames.
  window.__hourglass = Object.freeze({world, embedding, audit: () => world.readAudit()});
  const frame = now => {
    resize();
    const elapsed = Math.max(0, (now - previous) / 1000);
    previous = now;
    if (running) accumulator += Math.min(elapsed, 0.1) * Number(speed.value);
    const steps = Math.min(48, Math.floor(accumulator * 120));
    if (steps > 0) accumulator -= steps / 120;
    const encoder = device.createCommandEncoder({label: 'Hourglass World + Embedding'});
    if (running && steps > 0) world.step(encoder, steps);
    embedding.render(encoder);
    device.queue.submit([encoder.finish()]);
    if (now - lastStatus > 200) {
      status.textContent = `${world.simulatedSeconds.toFixed(1)} s simulated · ${running ? 'flowing' : 'paused'} · GPU material World`;
      lastStatus = now;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

start().catch(error => {
  status.textContent = `Hourglass could not start: ${error.message}`;
  console.error(error);
});
