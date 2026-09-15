import { stepRigidPolygonWorld2D } from './runtime/vf-physics-engine.mjs';
import { createStoneSpeciesPileReference } from './runtime/vf-stone-species-pile.mjs';

const canvas = document.getElementById('rocks');
const ctx = canvas.getContext('2d');
const playButton = document.getElementById('play');
const resetButton = document.getElementById('reset');
const gravityInput = document.getElementById('gravity');
const status = document.getElementById('status');
const view = { minimum: [-1.5, -0.9], maximum: [1.5, 0.9] };
const palette = ['#777873', '#9a7364', '#ada89a', '#4f5352', '#827b70', '#686d72'];
let world;
let playing = true;
let previousTime = null;
let accumulator = 0;
let picked = null;
let dragTarget = null;
let dragVelocity = [0, 0];
let lastDrag = null;

const convexHull = (points) => {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (source) => {
    const result = [];
    for (const point of source) {
      while (result.length >= 2 && cross(result.at(-2), result.at(-1), point) <= 0) result.pop();
      result.push(point);
    }
    return result;
  };
  return [...half(sorted).slice(0, -1), ...half([...sorted].reverse()).slice(0, -1)];
};

const stonePolygon = (mesh, index) => {
  const matrix = mesh._modelMatrix || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const sx = Math.hypot(matrix[0], matrix[1], matrix[2]);
  const sz = Math.hypot(matrix[8], matrix[9], matrix[10]);
  const points = [];
  for (let offset = 0; offset < mesh.vertices.length; offset += 10) {
    points.push([mesh.vertices[offset] * sx, mesh.vertices[offset + 2] * sz]);
  }
  let hull = convexHull(points);
  const center = hull.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0])
    .map((value) => value / hull.length);
  hull = hull.map((point) => [point[0] - center[0], point[1] - center[1]]);
  const targetRadius = 0.13 + index * 0.012;
  const radius = Math.max(...hull.map((point) => Math.hypot(...point)));
  return { vertices: hull.map((point) => point.map((value) => value * targetRadius / radius)),
    radius: targetRadius };
};

const createWorld = () => {
  const pile = createStoneSpeciesPileReference();
  const bodies = pile.meshes.slice(0, 6).map((mesh, index) => {
    const polygon = stonePolygon(mesh, index);
    return {
      id: `generated-rock-${index}`,
      localVertices: polygon.vertices,
      position: [-0.9 + index * 0.34, 0.54 + (index % 2) * 0.20],
      velocity: [0, 0], angle: index * 0.31, angularVelocity: (index - 2.5) * 0.08,
      density: 2700, contact_radius: polygon.radius,
      e_n: 0.24, e_t: 0, mu_s: 0.78, mu_d: 0.56, mu_r: 0.035,
      restitution_threshold: 0.42, color: palette[index],
    };
  });
  return { width: 3, height: 1.8, gravity: [0, -Number(gravityInput.value)],
    maxStep: 1 / 240, solverIterations: 8, sleepDelay: 0.6,
    sleepLinearThreshold: 0.035, sleepAngularThreshold: 0.09, bodies };
};

const resize = () => {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, Math.max(1, devicePixelRatio || 1));
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
};

const worldToScreen = ([x, y]) => [
  (x - view.minimum[0]) / (view.maximum[0] - view.minimum[0]) * canvas.width,
  (view.maximum[1] - y) / (view.maximum[1] - view.minimum[1]) * canvas.height,
];

const screenToWorld = (event) => {
  const rect = canvas.getBoundingClientRect();
  const u = (event.clientX - rect.left) / rect.width;
  const v = (event.clientY - rect.top) / rect.height;
  return [view.minimum[0] + u * (view.maximum[0] - view.minimum[0]),
    view.maximum[1] - v * (view.maximum[1] - view.minimum[1])];
};

const transformed = (body) => {
  const c = Math.cos(body.angle); const s = Math.sin(body.angle);
  return body.localVertices.map(([x, y]) => [body.position[0] + c * x - s * y,
    body.position[1] + s * x + c * y]);
};

const pointInside = (point, polygon) => {
  let sign = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]; const b = polygon[(i + 1) % polygon.length];
    const cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
    if (Math.abs(cross) < 1e-9) continue;
    if (!sign) sign = Math.sign(cross);
    else if (Math.sign(cross) !== sign) return false;
  }
  return true;
};

canvas.addEventListener('pointerdown', (event) => {
  const point = screenToWorld(event);
  for (let i = world.bodies.length - 1; i >= 0; i -= 1) {
    if (!pointInside(point, transformed(world.bodies[i]))) continue;
    picked = world.bodies[i].id;
    dragTarget = point;
    dragVelocity = [0, 0];
    lastDrag = { point, time: performance.now() };
    canvas.setPointerCapture(event.pointerId);
    canvas.dataset.dragging = 'true';
    event.preventDefault();
    break;
  }
});

canvas.addEventListener('pointermove', (event) => {
  if (!picked) return;
  const point = screenToWorld(event);
  const now = performance.now();
  const dt = Math.max(1 / 240, Math.min(0.05, (now - lastDrag.time) / 1000));
  dragVelocity = [(point[0] - lastDrag.point[0]) / dt, (point[1] - lastDrag.point[1]) / dt];
  dragTarget = point;
  lastDrag = { point, time: now };
  event.preventDefault();
});

const release = (event) => {
  if (!picked) return;
  const body = world.bodies.find((candidate) => candidate.id === picked);
  if (body) { body.velocity = dragVelocity.map((value) => Math.max(-5, Math.min(5, value))); body.sleeping = false; }
  picked = null; dragTarget = null; lastDrag = null;
  canvas.dataset.dragging = 'false';
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(playing));
  previousTime = null;
});
resetButton.addEventListener('click', () => { world = createWorld(); picked = null; previousTime = null; });
gravityInput.addEventListener('input', () => { world.gravity = [0, -Number(gravityInput.value)]; });

const step = (elapsed) => {
  accumulator = Math.min(0.08, accumulator + elapsed);
  const fixed = 1 / 120;
  while (playing && accumulator >= fixed) {
    if (picked) {
      const body = world.bodies.find((candidate) => candidate.id === picked);
      if (body) { body.position = [...dragTarget]; body.velocity = [...dragVelocity]; body.sleeping = false; body.sleep_time = 0; }
    }
    world.gravity = [0, -Number(gravityInput.value)];
    world = stepRigidPolygonWorld2D(world, fixed);
    if (picked) {
      const body = world.bodies.find((candidate) => candidate.id === picked);
      if (body) { body.position = [...dragTarget]; body.velocity = [...dragVelocity]; body.sleeping = false; }
    }
    accumulator -= fixed;
  }
};

const draw = () => {
  resize();
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, '#343732'); gradient.addColorStop(1, '#191a18');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const floorY = worldToScreen([0, -0.9])[1];
  ctx.fillStyle = '#2d2c27'; ctx.fillRect(0, floorY - 4, canvas.width, canvas.height - floorY + 4);
  for (const body of world.bodies) {
    const polygon = transformed(body).map(worldToScreen);
    ctx.beginPath(); ctx.moveTo(...polygon[0]);
    for (let i = 1; i < polygon.length; i += 1) ctx.lineTo(...polygon[i]);
    ctx.closePath();
    const center = worldToScreen(body.position);
    const radius = body.contact_radius / (view.maximum[0] - view.minimum[0]) * canvas.width;
    const fill = ctx.createRadialGradient(center[0] - radius * 0.28, center[1] - radius * 0.35,
      radius * 0.08, center[0], center[1], radius * 1.25);
    fill.addColorStop(0, '#c9c6ba'); fill.addColorStop(0.22, body.color); fill.addColorStop(1, '#292a28');
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = body.id === picked ? '#f5dfac' : '#151615';
    ctx.lineWidth = body.id === picked ? 4 : 2; ctx.stroke();
  }
  const sleeping = world.bodies.filter((body) => body.sleeping).length;
  status.textContent = `${world.bodies.length} generated rocks · ${sleeping} sleeping · ${Number(gravityInput.value).toFixed(1)} m/s²`;
};

const frame = (timestamp) => {
  const elapsed = previousTime == null ? 0 : Math.max(0, Math.min(0.08, (timestamp - previousTime) / 1000));
  previousTime = timestamp;
  step(elapsed); draw(); requestAnimationFrame(frame);
};

world = createWorld();
window.__rigidRocksWorld = () => world;
requestAnimationFrame(frame);
