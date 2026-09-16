import { createStoneSpeciesPileReference } from './runtime/vf-stone-species-pile.mjs?v=fourier-9';

const frameId = 'rigid_rocks_fourier_5_frame';
const playButton = document.getElementById('play');
const resetButton = document.getElementById('reset');
const gravityInput = document.getElementById('gravity');
const status = document.getElementById('status');
const errorBox = document.getElementById('error');
const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
const subtract = (a, b) => a.map((value, axis) => value - b[axis]);
const scale = (value, scalar) => value.map((component) => component * scalar);
const dot = (a, b) => a.reduce((sum, value, axis) => sum + value * b[axis], 0);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (value) => scale(value, 1 / Math.max(1e-9, Math.hypot(...value)));

let playing = true;
let bodies = [];
let picked = null;
let pointer = null;
let previous = null;
let lastRender = 0;
let transformRefreshPending = false;

function makeBodies() {
  const pile = createStoneSpeciesPileReference();
  return pile.meshes.map((source, index) => {
    const individual = pile.individuals[index];
    const id = `rock-${index}`;
    const baseMatrix = [...source._modelMatrix];
    const mesh = {
      ...source,
      id,
      object_id: index + 1,
      _modelMatrix: [...baseMatrix],
      static_vertices: true,
      static_indices: true,
      pickable: false,
    };
    return {
      id,
      mesh,
      baseMatrix,
      center: [individual.center[0], individual.center[1], individual.proxyCenterZ],
      radius: individual.collisionRadius,
      lift: 0,
      velocity: 0,
    };
  });
}

const ground = Object.freeze({
  type: 'field_mesh', id: 'rocks-ground', object_id: 100, mode3d: true,
  topology: 'triangle-list', static_vertices: true, static_indices: true,
  receives_lighting: true, casts_shadow: false, receives_shadow: true,
  specular_strength: 0.02,
  vertices: new Float32Array([
    -2.35, -1.35, -0.02, 0, 0, 1, 0.12, 0.125, 0.12, 1,
    2.35, -1.35, -0.02, 0, 0, 1, 0.12, 0.125, 0.12, 1,
    2.35, 1.35, -0.02, 0, 0, 1, 0.12, 0.125, 0.12, 1,
    -2.35, 1.35, -0.02, 0, 0, 1, 0.12, 0.125, 0.12, 1,
  ]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
});

const renderMeshes = () => [ground, ...bodies.map((body) => body.mesh)];

function updateBodyMatrix(body) {
  for (let index = 0; index < 16; index += 1) {
    body.mesh._modelMatrix[index] = body.baseMatrix[index];
  }
  body.mesh._modelMatrix[14] += body.lift;
}

function requestTransformFrame() {
  if (transformRefreshPending) return;
  transformRefreshPending = true;
  const submit = () => {
    const display = window.VfDisplay;
    if (!display) { transformRefreshPending = false; return; }
    if (display.dynamicGeomFrameHasRenderBackpressure?.(frameId)
        || !display.dynamicGeomFrameCanAcceptUpdate?.(frameId)) {
      window.setTimeout(submit, 16);
      return;
    }
    transformRefreshPending = false;
    display.requestDynamicGeomFrameUpdate(frameId);
  };
  requestAnimationFrame(submit);
}

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(playing));
  previous = null;
});

resetButton.addEventListener('click', () => {
  bodies = makeBodies();
  picked = null;
  pointer = null;
  previous = null;
  lastRender = 0;
  transformRefreshPending = false;
  window.VfDisplay?.requestDynamicGeomFrameUpdate(frameId);
  window.VfDisplay?.redrawVisibleGeomFrames();
});

try {
  await window.VfRuntimeShell.ensureSceneDependencies();
  bodies = makeBodies();
  const panel = window.VfFrame.mount(document.getElementById('layer'), {
    id: frameId,
    title: 'Five generated 3D stones · vertical lift and drop',
    draggable: false, dockable: false, resizable: false, closable: false,
  });
  Object.assign(panel.root.style, { left: '0', top: '0', width: '100%', height: '100%' });
  const camera = { pos: [3, -4.35, 2.55], target: [0, 0, 0.55], up: [0, 0, 1], fov: 36 };
  window.VfDisplay.mountDynamicGeomFrame(frameId, () => ({
    meshes: renderMeshes(), camera,
    lights: [
      { id: 'key', kind: 'point', pos: [-2.8, -3.2, 5.8], target: [0, 0, 0.7],
        color: [1, 0.94, 0.84, 1], intensity: 30, range: 15 },
      { id: 'fill', kind: 'point', pos: [3.4, 1.2, 4.2], target: [0, 0, 0.6],
        color: [0.62, 0.78, 1, 1], intensity: 12, range: 14 },
      { id: 'front', kind: 'point', pos: [0, -3.8, 2.4], target: [0, 0, 0.45],
        color: [1, 0.9, 0.76, 1], intensity: 7, range: 10 },
    ],
    background: [0.12, 0.14, 0.16, 1], unified_renderer: true,
  }));

  const basis = () => {
    const forward = normalize(subtract(camera.target, camera.pos));
    const right = normalize(cross(forward, camera.up));
    return { forward, right, up: cross(right, forward) };
  };
  const projected = (body, rect) => {
    const { forward, right, up } = basis();
    const relative = subtract(
      [body.center[0], body.center[1], body.center[2] + body.lift], camera.pos,
    );
    const depth = dot(relative, forward);
    const half = Math.tan(camera.fov * Math.PI / 360);
    const nx = dot(relative, right) / (depth * half * (rect.width / rect.height));
    const ny = dot(relative, up) / (depth * half);
    return { x: rect.left + (nx + 1) * rect.width * 0.5,
      y: rect.top + (1 - ny) * rect.height * 0.5, depth };
  };

  panel.body.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    if (event.button !== 0) return;
    const rect = panel.body.getBoundingClientRect();
    const candidates = bodies.map((body) => ({ body, screen: projected(body, rect) }))
      .filter(({ screen }) => screen.depth > 0)
      .sort((a, b) => Math.hypot(a.screen.x - event.clientX, a.screen.y - event.clientY)
        - Math.hypot(b.screen.x - event.clientX, b.screen.y - event.clientY));
    const nearest = candidates[0];
    if (!nearest || Math.hypot(nearest.screen.x - event.clientX,
      nearest.screen.y - event.clientY) > Math.max(30,
      nearest.body.radius / nearest.screen.depth * rect.height * 2.4)) return;
    event.stopPropagation();
    picked = nearest.body;
    picked.velocity = 0;
    pointer = { id: event.pointerId, startY: event.clientY,
      startLift: picked.lift, depth: nearest.screen.depth };
    panel.body.setPointerCapture(event.pointerId);
  }, { capture: true });

  panel.body.addEventListener('pointermove', (event) => {
    event.preventDefault();
    if (!pointer || pointer.id !== event.pointerId || !picked) return;
    event.stopPropagation();
    const rect = panel.body.getBoundingClientRect();
    const worldPerPixel = 2 * pointer.depth * Math.tan(camera.fov * Math.PI / 360)
      / Math.max(1, rect.height);
    picked.lift = clamp(pointer.startLift
      + (pointer.startY - event.clientY) * worldPerPixel, 0, 3.2);
    picked.velocity = 0;
    updateBodyMatrix(picked);
    requestTransformFrame();
  }, { capture: true });

  const finishDrop = () => {
    if (!pointer) return;
    if (picked) picked.velocity = 0;
    picked = null;
    const pointerId = pointer.id;
    pointer = null;
    try { panel.body.releasePointerCapture(pointerId); } catch {}
  };
  const drop = (event) => {
    if (!pointer || pointer.id !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    finishDrop();
  };
  panel.body.addEventListener('pointerup', drop, { capture: true });
  panel.body.addEventListener('pointercancel', drop, { capture: true });
  window.addEventListener('pointerup', drop, { capture: true });
  window.addEventListener('pointercancel', drop, { capture: true });
  window.addEventListener('mouseup', finishDrop, { capture: true });
  window.addEventListener('touchend', finishDrop, { capture: true, passive: false });
  window.addEventListener('blur', finishDrop);

  const animate = (time) => {
    const elapsed = previous == null ? 0 : Math.min(0.05, Math.max(0, (time - previous) / 1000));
    previous = time;
    let moving = Boolean(picked);
    if (playing) {
      for (const body of bodies) {
        if (body === picked || body.lift <= 0) continue;
        body.velocity -= Number(gravityInput.value) * elapsed;
        body.lift = Math.max(0, body.lift + body.velocity * elapsed);
        if (body.lift === 0) body.velocity = 0;
        updateBodyMatrix(body);
        moving = true;
      }
    }
    if (moving && (!lastRender || time - lastRender >= 1000 / 15)) {
      lastRender = time;
      requestTransformFrame();
    }
    const lifted = bodies.filter((body) => body.lift > 0.002).length;
    status.textContent = `5 Fourier stones · ${lifted ? `${lifted} lifted` : 'ready to lift'}`;
    requestAnimationFrame(animate);
  };

  window.__rigidRocks3D = { get bodies() { return bodies; } };
  window.VfDisplay.requestDynamicGeomFrameUpdate(frameId);
  requestAnimationFrame(animate);
} catch (error) {
  errorBox.hidden = false;
  errorBox.textContent = String(error?.stack || error);
  status.textContent = 'Scene unavailable';
}
