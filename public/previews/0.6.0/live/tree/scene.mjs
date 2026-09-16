import { createGrassMaterialFieldReference } from './ui/vf-grass-material-field.mjs?v=lawn-11';
import { createGrassCameraDemandControllerReference } from './ui/vf-grass-camera-demand-runtime.mjs?v=lawn-11';
import { createRetainedGeometryPacketRuntimeReference } from './ui/vf-rock-camera-demand-runtime.mjs';
import { createVfLiveWorldStackReference } from '../runtime/vf-live-world-stack.mjs?v=world-stack-1';
import { treeProGenPresets, treeProGenAsset } from './runtime/vf-tree-pro-gen-presets.mjs';

const frameId = 'tree_grass_live_frame';
const status = document.getElementById('status');
const errorBox = document.getElementById('error');
const windInput = document.getElementById('wind');
const windValue = document.getElementById('wind-value');
const grassButton = document.getElementById('grass');
const windParticlesButton = document.getElementById('wind-particles');
const generationInput = document.getElementById('generation');
const requestedGeneration = new URLSearchParams(location.search).get('generation') ?? 'original';
const generation = Object.hasOwn(treeProGenPresets, requestedGeneration) ? requestedGeneration : 'original';
generationInput.value = generation;
const fitControls = () => {
  const top = `${Math.ceil(document.getElementById('controls').getBoundingClientRect().height)}px`;
  for (const id of ['layer', 'vf-screen-canvas']) document.getElementById(id).style.top = top;
};
new ResizeObserver(fitControls).observe(document.getElementById('controls'));
fitControls();
generationInput.addEventListener('change', () => {
  const url = new URL(location.href);
  url.searchParams.set('generation', generationInput.value);
  location.replace(url.href);
});
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

let grassVisible = true;
let windParticlesVisible = false;
let windParticleMesh = null;
let grassPackets = [];
let baseGrassPacket = null;
let baseGrassWords = null;
let treeMeshes = [];
let baseVertices = [];
let vertexCells = [];
let vertexCompliance = [];
let minimum = [Infinity, Infinity, Infinity];
let maximum = [-Infinity, -Infinity, -Infinity];
let lastWindFrame = 0;
let grassRevision = 1;
let renderPending = false;
let grassWindDivider = 0;
let treeWorld = null;

const GX = 10;
const GY = 10;
const GZ = 14;
const CELLS = GX * GY * GZ;
const WIND_PARCEL_COUNT = 384;
const fieldDisplacement = new Float32Array(CELLS * 2);
const fieldVelocity = new Float32Array(CELLS * 2);
const fieldForce = new Float32Array(CELLS * 2);
const fieldPrior = new Float32Array(CELLS * 2);
const treeCollisionCells = new Uint8Array(CELLS);
const parcels = Array.from({ length: WIND_PARCEL_COUNT }, (_, index) => ({
  x: (index * 0.61803398875) % 1,
  y: (index * 0.38196601125 + 0.17) % 1,
  z: (index * 0.754877666 + 0.31) % 1,
  phase: index * 1.713,
  speedScale: 0.90 + (index % 9) * 0.025,
  vx: 0, vy: 0, vz: 0, visibleFor: 0, collisionCooldown: 0,
}));

const cellIndex = (x, y, z) => clamp(Math.floor(x * GX), 0, GX - 1)
  + GX * (clamp(Math.floor(y * GY), 0, GY - 1)
    + GY * clamp(Math.floor(z * GZ), 0, GZ - 1));

function requestFrame() {
  if (renderPending) return;
  renderPending = true;
  const submit = () => {
    const display = window.VfDisplay;
    if (!display) { renderPending = false; return; }
    if (display.dynamicGeomFrameHasRenderBackpressure?.(frameId)
        || !display.dynamicGeomFrameCanAcceptUpdate?.(frameId)) {
      window.setTimeout(submit, 18);
      return;
    }
    renderPending = false;
    display.requestDynamicGeomFrameUpdate(frameId);
  };
  requestAnimationFrame(submit);
}

async function loadCachedTree() {
  const response = await fetch(`./assets/${treeProGenAsset(generation)}?v=pro-gen-1`);
  if (!response.ok) throw new Error(`Tree asset ${response.status}`);
  if (typeof DecompressionStream !== 'function') {
    throw new Error('This browser cannot decode the cached tree asset.');
  }
  const buffer = await new Response(response.body
    .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const view = new DataView(buffer);
  const decoder = new TextDecoder();
  let offset = 0;
  if (decoder.decode(new Uint8Array(buffer, 0, 8)) !== 'VFTREE02') {
    throw new Error('Invalid cached tree asset');
  }
  offset = 8;
  const meshCount = view.getUint32(offset, true); offset += 4;
  const vertexCount = view.getUint32(offset, true); offset += 4;
  const indexCount = view.getUint32(offset, true); offset += 4;
  const meshes = [];
  for (let index = 0; index < meshCount; index += 1) {
    const metaLength = view.getUint32(offset, true); offset += 4;
    const verticesLength = view.getUint32(offset, true); offset += 4;
    const indicesLength = view.getUint32(offset, true); offset += 4;
    const uvsLength = view.getUint32(offset, true); offset += 4;
    const roughnessLength = view.getUint32(offset, true); offset += 4;
    const meta = JSON.parse(decoder.decode(new Uint8Array(buffer, offset, metaLength)));
    offset += metaLength;
    offset += (4 - (metaLength % 4)) % 4;
    const vertices = new Float32Array(buffer, offset, verticesLength).slice();
    offset += verticesLength * 4;
    const indices = new Uint32Array(buffer, offset, indicesLength).slice();
    offset += indicesLength * 4;
    const uvs = new Float32Array(buffer, offset, uvsLength).slice();
    offset += uvsLength * 4;
    const roughness = new Float32Array(buffer, offset, roughnessLength).slice();
    offset += roughnessLength * 4;
    meshes.push({ ...meta, vertices, indices, ...(uvsLength ? { uvs } : {}),
      ...(roughnessLength ? { roughness } : {}), static_vertices: false });
  }
  let low = Infinity;
  let high = -Infinity;
  for (const mesh of meshes) {
    for (let index = 2; index < mesh.vertices.length; index += 10) {
      low = Math.min(low, mesh.vertices[index]);
      high = Math.max(high, mesh.vertices[index]);
    }
  }
  const metresPerUnit = 8 / Math.max(1e-6, high - low);
  for (const mesh of meshes) {
    for (let index = 0; index < mesh.vertices.length; index += 10) {
      mesh.vertices[index] *= metresPerUnit;
      mesh.vertices[index + 1] *= metresPerUnit;
      mesh.vertices[index + 2] = (mesh.vertices[index + 2] - low) * metresPerUnit;
    }
  }
  return { meshes, vertexCount, indexCount };
}

function boundsAndMappings() {
  for (const mesh of treeMeshes) {
    for (let offset = 0; offset < mesh.vertices.length; offset += 10) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], mesh.vertices[offset + axis]);
        maximum[axis] = Math.max(maximum[axis], mesh.vertices[offset + axis]);
      }
    }
  }
  const span = maximum.map((value, axis) => Math.max(1e-6, value - minimum[axis]));
  baseVertices = treeMeshes.map((mesh) => new Float32Array(mesh.vertices));
  vertexCells = treeMeshes.map((mesh) => {
    const map = new Uint16Array(mesh.vertices.length / 10);
    for (let offset = 0, index = 0; offset < mesh.vertices.length; offset += 10, index += 1) {
      map[index] = cellIndex((mesh.vertices[offset] - minimum[0]) / span[0],
        (mesh.vertices[offset + 1] - minimum[1]) / span[1],
        (mesh.vertices[offset + 2] - minimum[2]) / span[2]);
      treeCollisionCells[map[index]] = 1;
    }
    return map;
  });
  vertexCompliance = treeMeshes.map((mesh, meshIndex) => {
    const values = new Float32Array(mesh.vertices.length / 10);
    for (let offset = 0, index = 0; offset < mesh.vertices.length; offset += 10, index += 1) {
      const height = clamp((mesh.vertices[offset + 2] - minimum[2]) / span[2], 0, 1);
      values[index] = (meshIndex === 0 ? 0.46 : 1.24) * height * height;
    }
    return values;
  });
  return span;
}

function depositParcel(parcel, forceScale, time) {
  const centerX = clamp(Math.floor(parcel.x * GX), 0, GX - 1);
  const centerY = clamp(Math.floor(parcel.y * GY), 0, GY - 1);
  const centerZ = clamp(Math.floor(parcel.z * GZ), 0, GZ - 1);
  for (let z = Math.max(0, centerZ - 1); z <= Math.min(GZ - 1, centerZ + 1); z += 1) {
    for (let y = Math.max(0, centerY - 1); y <= Math.min(GY - 1, centerY + 1); y += 1) {
      for (let x = Math.max(0, centerX - 1); x <= Math.min(GX - 1, centerX + 1); x += 1) {
        const dx = (x + 0.5) / GX - parcel.x;
        const dy = (y + 0.5) / GY - parcel.y;
        const dz = ((z + 0.5) / GZ - parcel.z) * 0.72;
        const weight = Math.exp(-(dx * dx + dy * dy + dz * dz) / 0.012);
        const cell = (x + GX * (y + GY * z)) * 2;
        const impact = parcel.visibleFor > 0 ? 1.15 * (parcel.visibleFor / 0.25) : 0;
        fieldForce[cell] += weight
          * (0.11 + 0.035 * Math.sin(parcel.phase + time * 0.001) + impact)
          * forceScale;
        fieldForce[cell + 1] += weight
          * (0.05 * Math.sin(parcel.phase * 1.9 + time * 0.0013 + parcel.z * 4)
            + impact * parcel.vy * 0.8) * forceScale;
      }
    }
  }
}

function advanceWind(dt, speedMetersPerSecond, time) {
  const spanX = Math.max(1e-6, maximum[0] - minimum[0]);
  const forceScale = 0.12 + 0.68 * (speedMetersPerSecond / 8) ** 2;
  fieldForce.fill(0);
  for (let index = 0; index < parcels.length; index += 1) {
    const parcel = parcels[index];
    const targetVx = speedMetersPerSecond * parcel.speedScale / spanX;
    const oldX = parcel.x;
    const oldY = parcel.y;
    const oldZ = parcel.z;
    parcel.visibleFor = Math.max(0, parcel.visibleFor - dt);
    parcel.collisionCooldown = Math.max(0, parcel.collisionCooldown - dt);
    parcel.vx += (targetVx - parcel.vx) * Math.min(1, dt * 7);
    parcel.vy += (Math.sin(time * 0.00047 + parcel.phase) * 0.012 - parcel.vy)
      * Math.min(1, dt * 4);
    parcel.vz += (Math.cos(time * 0.00039 + parcel.phase * 1.3) * 0.010 - parcel.vz)
      * Math.min(1, dt * 4);
    parcel.x += parcel.vx * dt;
    parcel.y = clamp(parcel.y + parcel.vy * dt, 0.012, 0.988);
    parcel.z += parcel.vz * dt;
    let collided = parcel.z <= 0.006;
    if (collided) { parcel.z = 0.006; parcel.vz = Math.abs(parcel.vz) * 0.35 + 0.003; }
    const inside = parcel.x >= 0 && parcel.x <= 1 && parcel.y >= 0
      && parcel.y <= 1 && parcel.z >= 0 && parcel.z <= 1;
    const hitTree = inside && treeCollisionCells[cellIndex(parcel.x, parcel.y, parcel.z)] === 1;
    if (hitTree && parcel.collisionCooldown <= 0) {
      collided = true;
      parcel.x = oldX; parcel.y = oldY; parcel.z = oldZ;
      parcel.vx = -Math.abs(parcel.vx) * 0.30;
      parcel.vy += (index % 2 ? 1 : -1) * targetVx * 0.045;
      parcel.vz += targetVx * 0.026;
    }
    if (collided && parcel.collisionCooldown <= 0) {
      parcel.visibleFor = 0.25;
      parcel.collisionCooldown = 0.16;
    }
    if (parcel.x > 1) {
      parcel.x = 0;
      parcel.y = (index * 0.38196601125 + time * 0.000031) % 1;
      parcel.z = (index * 0.754877666 + time * 0.000019) % 1;
      parcel.vx = targetVx;
      parcel.visibleFor = 0;
      parcel.collisionCooldown = 0;
    }
    if (parcel.x < 0) parcel.x = 0;
    depositParcel(parcel, forceScale, time);
  }

  fieldPrior.set(fieldDisplacement);
  for (let z = 0; z < GZ; z += 1) {
    for (let y = 0; y < GY; y += 1) {
      for (let x = 0; x < GX; x += 1) {
        const index = x + GX * (y + GY * z);
        for (let axis = 0; axis < 2; axis += 1) {
          let sum = 0;
          let count = 0;
          for (const [nx, ny, nz] of [[x - 1, y, z], [x + 1, y, z],
            [x, y - 1, z], [x, y + 1, z], [x, y, z - 1], [x, y, z + 1]]) {
            if (nx >= 0 && nx < GX && ny >= 0 && ny < GY && nz >= 0 && nz < GZ) {
              sum += fieldPrior[(nx + GX * (ny + GY * nz)) * 2 + axis];
              count += 1;
            }
          }
          const offset = index * 2 + axis;
          const elasticCoupling = count ? ((sum / count) - fieldPrior[offset]) * 8.5 : 0;
          fieldVelocity[offset] += (fieldForce[offset] * 0.62
            - fieldDisplacement[offset] * 11 - fieldVelocity[offset] * 3.4
            + elasticCoupling) * dt;
          fieldDisplacement[offset] = clamp(fieldDisplacement[offset]
            + fieldVelocity[offset] * dt, -0.11, 0.11);
        }
      }
    }
  }
}

function deformTree() {
  const worldSpan = Math.max(...maximum.map((value, axis) => value - minimum[axis]));
  for (let meshIndex = 0; meshIndex < treeMeshes.length; meshIndex += 1) {
    const mesh = treeMeshes[meshIndex];
    const base = baseVertices[meshIndex];
    const cells = vertexCells[meshIndex];
    const compliance = vertexCompliance[meshIndex];
    for (let offset = 0, vertex = 0; offset < base.length; offset += 10, vertex += 1) {
      const cell = cells[vertex] * 2;
      const flexibility = compliance[vertex] * worldSpan * 0.23;
      mesh.vertices[offset] = base[offset] + fieldDisplacement[cell] * flexibility;
      mesh.vertices[offset + 1] = base[offset + 1]
        + fieldDisplacement[cell + 1] * flexibility;
    }
  }
}

function windAtWorld(x, y, z = 0) {
  const nx = (x - minimum[0]) / Math.max(1e-6, maximum[0] - minimum[0]);
  const ny = (y - minimum[1]) / Math.max(1e-6, maximum[1] - minimum[1]);
  const nz = (z - minimum[2]) / Math.max(1e-6, maximum[2] - minimum[2]);
  const offset = cellIndex(nx, ny, nz) * 2;
  return [fieldDisplacement[offset], fieldDisplacement[offset + 1]];
}

function updateGrassWind() {
  if (!baseGrassPacket || !baseGrassWords) return;
  const words = new Uint32Array(baseGrassWords);
  const signed = new Int32Array(words.buffer);
  const floats = new Float32Array(words.buffer);
  for (let base = 0; base < words.length; base += 12) {
    const wind = windAtWorld(signed[base] + 0.5, signed[base + 1] + 0.5, 0);
    floats[base + 7] = wind[0] * 2.5;
    floats[base + 11] = wind[1] * 2.5;
  }
  grassPackets = [{ ...baseGrassPacket,
    grass_gpu: { ...baseGrassPacket.grass_gpu, cell_records: words } }];
}

function createWindParticleMesh() {
  const vertices = new Float32Array(parcels.length * 6 * 10);
  const indices = new Uint32Array(parcels.length * 8 * 3);
  for (let parcel = 0; parcel < parcels.length; parcel += 1) {
    const vertex = parcel * 6;
    indices.set([vertex, vertex + 2, vertex + 4, vertex + 2, vertex + 1, vertex + 4,
      vertex + 1, vertex + 3, vertex + 4, vertex + 3, vertex, vertex + 4,
      vertex + 2, vertex, vertex + 5, vertex + 1, vertex + 2, vertex + 5,
      vertex + 3, vertex + 1, vertex + 5, vertex, vertex + 3, vertex + 5], parcel * 24);
  }
  return { id: 'visible-wind-impacts', type: 'field_mesh', object_id: 990,
    mode3d: true, topology: 'triangle-list', static_vertices: false, static_indices: true,
    transparent: true, receives_lighting: false, casts_shadow: false,
    receives_shadow: false, cull_backfaces: false, vertices, indices };
}

function updateWindParticleMesh() {
  if (!windParticleMesh || !windParticlesVisible) return;
  const span = maximum.map((value, axis) => value - minimum[axis]);
  const radius = Math.max(...span) * 0.0022;
  for (let parcelIndex = 0; parcelIndex < parcels.length; parcelIndex += 1) {
    const parcel = parcels[parcelIndex];
    const alpha = 0.5 * clamp(parcel.visibleFor / 0.25, 0, 1);
    const color = [0.42 * alpha, 0.87 * alpha, alpha, alpha];
    const center = [minimum[0] + parcel.x * span[0], minimum[1] + parcel.y * span[1],
      minimum[2] + parcel.z * span[2]];
    const points = [[center[0] + radius, center[1], center[2]],
      [center[0] - radius, center[1], center[2]],
      [center[0], center[1] + radius, center[2]],
      [center[0], center[1] - radius, center[2]],
      [center[0], center[1], center[2] + radius],
      [center[0], center[1], center[2] - radius]];
    for (let pointIndex = 0; pointIndex < 6; pointIndex += 1) {
      const offset = (parcelIndex * 6 + pointIndex) * 10;
      const normal = [(points[pointIndex][0] - center[0]) / radius,
        (points[pointIndex][1] - center[1]) / radius,
        (points[pointIndex][2] - center[2]) / radius];
      windParticleMesh.vertices.set([...points[pointIndex], ...normal, ...color], offset);
    }
  }
}

function groundMesh(center, span) {
  const radius = Math.max(7, Math.max(...span) * 2.5);
  const z = -0.018;
  return { id: 'tree-grass-ground', type: 'field_mesh',
    vertices: new Float32Array([
      center[0] - radius, center[1] - radius, z, 0, 0, 1, 0.10, 0.18, 0.07, 1,
      center[0] + radius, center[1] - radius, z, 0, 0, 1, 0.10, 0.18, 0.07, 1,
      center[0] + radius, center[1] + radius, z, 0, 0, 1, 0.10, 0.18, 0.07, 1,
      center[0] - radius, center[1] + radius, z, 0, 0, 1, 0.10, 0.18, 0.07, 1,
    ]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), cull_backfaces: false,
    casts_shadow: false, receives_shadow: true, specular_strength: 0.02 };
}

function sunMesh(center, span) {
  const size = Math.max(...span);
  const origin = [center[0] - size * 0.58, center[1] + size * 0.34,
    maximum[2] - size * 0.08];
  const radius = size * 0.11;
  const latitudeBands = 8;
  const longitudeBands = 16;
  const vertices = new Float32Array((latitudeBands + 1) * (longitudeBands + 1) * 10);
  const indices = new Uint32Array(latitudeBands * longitudeBands * 6);
  for (let latitude = 0; latitude <= latitudeBands; latitude += 1) {
    const phi = Math.PI * latitude / latitudeBands;
    for (let longitude = 0; longitude <= longitudeBands; longitude += 1) {
      const theta = Math.PI * 2 * longitude / longitudeBands;
      const normal = [Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta),
        Math.cos(phi)];
      const offset = (latitude * (longitudeBands + 1) + longitude) * 10;
      vertices.set([origin[0] + normal[0] * radius, origin[1] + normal[1] * radius,
        origin[2] + normal[2] * radius, ...normal, 1.0, 0.76, 0.22, 1], offset);
    }
  }
  let write = 0;
  for (let latitude = 0; latitude < latitudeBands; latitude += 1) {
    for (let longitude = 0; longitude < longitudeBands; longitude += 1) {
      const a = latitude * (longitudeBands + 1) + longitude;
      const b = a + longitudeBands + 1;
      indices.set([a, b, a + 1, b, b + 1, a + 1], write);
      write += 6;
    }
  }
  return { id: 'tree-sun', type: 'field_mesh', object_id: 991, mode3d: true,
    topology: 'triangle-list', static_vertices: true, static_indices: true,
    no_lighting: true, receives_lighting: false, casts_shadow: false,
    vertices, indices };
}

grassButton.addEventListener('click', () => {
  grassVisible = !grassVisible;
  grassButton.setAttribute('aria-pressed', String(grassVisible));
  requestFrame();
});
windParticlesButton.addEventListener('click', () => {
  windParticlesVisible = !windParticlesVisible;
  windParticlesButton.setAttribute('aria-pressed', String(windParticlesVisible));
  if (windParticlesVisible) updateWindParticleMesh();
  requestFrame();
});
windInput.addEventListener('input', () => {
  windValue.value = `${Number(windInput.value).toFixed(1)} m/s`;
});

try {
  const started = performance.now();
  const [tree] = await Promise.all([
    loadCachedTree(),
    window.VfRuntimeShell.ensureSceneDependencies(),
  ]);
  treeMeshes = tree.meshes;
  const span = boundsAndMappings();
  const center = minimum.map((value, axis) => (value + maximum[axis]) * 0.5);
  const ground = groundMesh(center, span);
  const sun = sunMesh(center, span);
  const camera = { pos: [center[0] + Math.max(...span) * 0.78,
    center[1] - Math.max(...span) * 1.65, center[2] + Math.max(...span) * 0.32],
  target: [center[0], center[1], center[2] * 0.88], up: [0, 0, 1], fov: 42 };
  windParticleMesh = createWindParticleMesh();
  const panel = window.VfFrame.mount(document.getElementById('layer'), {
    id: frameId, title: '8 m generated tree · dense grass · local parcel wind',
    draggable: false, dockable: false, resizable: false, closable: false,
  });
  Object.assign(panel.root.style, { left: '0', top: '0', width: '100%', height: '100%' });

  let orbitDrag = null;
  panel.body.style.touchAction = 'none';
  panel.body.style.overscrollBehavior = 'none';
  for (const type of ['touchstart', 'touchmove', 'touchend', 'touchcancel']) {
    panel.body.addEventListener(type, (event) => event.preventDefault(),
      { capture: true, passive: false });
  }
  panel.body.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    if (event.button !== 0) return;
    orbitDrag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    panel.body.setPointerCapture(event.pointerId);
  }, { capture: true });
  panel.body.addEventListener('pointermove', (event) => {
    event.preventDefault();
    if (!orbitDrag || orbitDrag.id !== event.pointerId || (event.buttons & 1) === 0) return;
    const dx = event.clientX - orbitDrag.x;
    const dy = event.clientY - orbitDrag.y;
    orbitDrag.x = event.clientX;
    orbitDrag.y = event.clientY;
    const vx = camera.pos[0] - camera.target[0];
    const vy = camera.pos[1] - camera.target[1];
    const vz = camera.pos[2] - camera.target[2];
    const distance = Math.hypot(vx, vy, vz);
    const yaw = Math.atan2(vy, vx) - dx * 0.008;
    const pitch = clamp(Math.asin(vz / distance) + dy * 0.006, -1.22, 1.22);
    const cosine = Math.cos(pitch);
    camera.pos = [camera.target[0] + Math.cos(yaw) * cosine * distance,
      camera.target[1] + Math.sin(yaw) * cosine * distance,
      camera.target[2] + Math.sin(pitch) * distance];
    requestFrame();
  }, { capture: true });

  const grassRuntime = createRetainedGeometryPacketRuntimeReference({
    requestRender(packets) {
      baseGrassPacket = packets[0];
      baseGrassWords = baseGrassPacket?.grass_gpu?.cell_records
        ? new Uint32Array(baseGrassPacket.grass_gpu.cell_records) : null;
      grassPackets = packets;
      updateGrassWind();
      requestFrame();
    },
  });
  const grassController = createGrassCameraDemandControllerReference({
    field: createGrassMaterialFieldReference({ generator: 'vkf.conditioned', version: 1,
      seed: [0x01234567, 0x89abcdef], domain: 'material',
      hierarchy: ['world:boreal', 'grass-field:tree'], lod: 0, channel: 'surface' }),
    runtime: grassRuntime, planeZ: 0,
    maximumDistance: Math.max(10, Math.max(...span) * 2.2),
    cellBudget: 96, bladeBudget: 49152,
  });
  const requestGrass = async () => {
    const rect = panel.body.getBoundingClientRect();
    await grassController.request({ revision: grassRevision++, camera: {
      eye: camera.pos, target: camera.target, up: camera.up,
      verticalFovRadians: camera.fov * Math.PI / 180,
      viewportWidth: Math.max(640, rect.width), viewportHeight: Math.max(360, rect.height),
    } });
  };
  const worldStack = createVfLiveWorldStackReference();
  treeWorld = worldStack.append({ id: 'tree-grass-wind', label: 'Tree · grass · wind' })
    .add({ treeMeshes, grassRuntime, windParcels: parcels })
    .add({ fieldDisplacement, fieldVelocity, fieldForce })
    .push({ id: 'advected-local-wind', collisions: 'tree-and-ground',
      advance({ dt, speedMetersPerSecond, time }) {
        advanceWind(dt, speedMetersPerSecond, time);
        deformTree();
        grassWindDivider = (grassWindDivider + 1) % 2;
        if (grassWindDivider === 0) updateGrassWind();
        if (windParticlesVisible) updateWindParticleMesh();
      } });
  const endOrbit = (event) => {
    event.preventDefault();
    if (!orbitDrag || orbitDrag.id !== event.pointerId) return;
    orbitDrag = null;
    try { panel.body.releasePointerCapture(event.pointerId); } catch {}
    requestGrass().catch(() => {});
  };
  panel.body.addEventListener('pointerup', endOrbit, { capture: true });
  panel.body.addEventListener('pointercancel', endOrbit, { capture: true });

  window.VfDisplay.mountDynamicGeomFrame(frameId, () => ({
    meshes: [ground, sun, ...(grassVisible ? grassPackets : []),
      ...(windParticlesVisible ? [windParticleMesh] : []), ...treeMeshes],
    camera,
    lights: [{ id: 'sun_key', kind: 'point',
      pos: [center[0] - Math.max(...span) * 0.45,
        center[1] - Math.max(...span) * 0.55,
        maximum[2] + Math.max(...span) * 0.35],
      target: center, color: [1, 0.91, 0.72, 1],
      intensity: Math.max(...span) ** 2 * 1.95, range: Math.max(...span) * 3 },
    { id: 'sky_fill', kind: 'point',
      pos: [center[0] + Math.max(...span) * 0.55,
        center[1] - Math.max(...span) * 0.5,
        maximum[2] + Math.max(...span) * 0.08],
      target: center, color: [0.62, 0.78, 0.64, 1],
      intensity: Math.max(...span) ** 2 * 1.1, range: Math.max(...span) * 2.5 }],
    background: [0.12, 0.22, 0.30, 1], unified_renderer: true,
  }));

  await requestGrass();
  const grassCount = grassRuntime.packets()[0]?.instance_count ?? 0;
  status.textContent = `Ready in ${((performance.now() - started) / 1000).toFixed(1)}s`
    + ` · 8 m tree · ${grassCount.toLocaleString()} grass blades`
    + ` · ${generation} branching + leaf variation · ${WIND_PARCEL_COUNT} wind parcels · swipe to orbit`;
  window.__treeGrassResult = { outcome: 'ready', tree, grassRuntime,
    worldStack, treeWorld, windModel: 'advected-parcels-local-transfer',
    parcels, treeCollisionCells, generation };
  requestFrame();

  const animate = (time) => {
    if (time - lastWindFrame >= 92) {
      const dt = lastWindFrame ? Math.min(0.12, (time - lastWindFrame) / 1000) : 0.092;
      lastWindFrame = time;
      treeWorld.advance({ dt, speedMetersPerSecond: Number(windInput.value), time });
      requestFrame();
    }
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
} catch (error) {
  errorBox.hidden = false;
  errorBox.textContent = String(error?.stack || error);
  status.textContent = 'Scene unavailable';
  window.__treeGrassResult = { outcome: 'fail', error: errorBox.textContent };
}
