import { createForestPopulationReference, realizeForestPatchesReference } from './runtime/vf-forest-population.mjs';
import { createTreeGeometryPlannerReference, planTreeGeometryReference } from './runtime/vf-tree-geometry-plan.mjs';
import { createTreeMaterialFieldReference, realizeTreeMaterialsReference } from './runtime/vf-tree-material-field.mjs';
import { adaptTreeWorkingSetsToRetainedPacketsReference } from './runtime/vf-tree-renderer-packets.mjs';
import { adaptTreeRenderPacketToWebGpuMeshesReference } from './runtime/vf-tree-webgpu-packets.mjs';
import { createGrassMaterialFieldReference } from './ui/vf-grass-material-field.mjs';
import { createGrassCameraDemandControllerReference } from './ui/vf-grass-camera-demand-runtime.mjs';
import { createRetainedGeometryPacketRuntimeReference } from './ui/vf-rock-camera-demand-runtime.mjs';

const frameId = 'tree_grass_live_frame';
const status = document.getElementById('status');
const errorBox = document.getElementById('error');
const windInput = document.getElementById('wind');
const grassButton = document.getElementById('grass');
let grassVisible = true;
let grassPackets = [];
let treeMeshes = [];
let baseVertices = [];
let minimumZ = 0;
let maximumZ = 1;
let lastWindFrame = 0;

grassButton.addEventListener('click', () => {
  grassVisible = !grassVisible;
  grassButton.setAttribute('aria-pressed', String(grassVisible));
  window.VfDisplay?.requestDynamicGeomFrameUpdate(frameId);
});

const identity = Object.freeze({
  generator: 'vkf.conditioned', version: 1, seed: Object.freeze([0x1f83d9ab, 269]),
  domain: 'material', hierarchy: Object.freeze(['world:boreal', 'tree:webgpu-demo']),
  lod: 0, channel: 'population',
});

const groundMesh = (center, span) => {
  const radius = Math.max(7, span * 2.5);
  const z = -0.018;
  return Object.freeze({ id: 'tree-grass-ground', type: 'field_mesh',
    vertices: new Float32Array([
      center[0] - radius, center[1] - radius, z, 0, 0, 1, .11, .16, .08, 1,
      center[0] + radius, center[1] - radius, z, 0, 0, 1, .11, .16, .08, 1,
      center[0] + radius, center[1] + radius, z, 0, 0, 1, .11, .16, .08, 1,
      center[0] - radius, center[1] + radius, z, 0, 0, 1, .11, .16, .08, 1,
    ]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), cull_backfaces: false,
    casts_shadow: false, receives_shadow: true, specular_strength: 0.02 });
};

try {
  await window.VfRuntimeShell.ensureSceneDependencies();
  const forest = realizeForestPatchesReference(createForestPopulationReference(identity),
    { patches: [[0, 0]], treeBudget: 1 });
  const geometry = planTreeGeometryReference(createTreeGeometryPlannerReference(identity, {
    splitDepth: 7, lateralShoots: true, trunkShoots: false, foliageDensity: 0.42,
    scaffoldBranches: 2,
  }), forest, { treeIndices: [0], detailLevels: [2], primitiveBudget: 2400 });
  const materials = realizeTreeMaterialsReference(createTreeMaterialFieldReference(identity),
    forest, geometry, { materialBudget: 2400 });
  const retained = adaptTreeWorkingSetsToRetainedPacketsReference(geometry, materials);
  const tree = adaptTreeRenderPacketToWebGpuMeshesReference(retained.packets[0],
    { vertexBudget: 393216, indexBudget: 2359296, leafContact: true });

  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const mesh of tree.meshes) {
    for (let offset = 0; offset < mesh.vertices.length; offset += 10) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], mesh.vertices[offset + axis]);
        maximum[axis] = Math.max(maximum[axis], mesh.vertices[offset + axis]);
      }
    }
  }
  const center = minimum.map((value, axis) => (value + maximum[axis]) * 0.5);
  const span = Math.max(...minimum.map((value, axis) => maximum[axis] - value));
  minimumZ = minimum[2]; maximumZ = maximum[2];
  baseVertices = tree.meshes.map((mesh) => new Float32Array(mesh.vertices));
  treeMeshes = tree.meshes.map((mesh, index) => ({ ...mesh,
    vertices: new Float32Array(baseVertices[index]) }));
  const ground = groundMesh(center, span);
  const camera = { pos: [center[0] + span * 0.78, center[1] - span * 1.65,
    center[2] + span * 0.32], target: [center[0], center[1], center[2] * 0.88],
    up: [0, 0, 1], fov: 42 };

  const panel = window.VfFrame.mount(document.getElementById('layer'), {
    id: frameId, title: 'Generated tree · live grass · light wind', draggable: false,
    dockable: false, resizable: false, closable: false,
  });
  panel.root.style.left = '0'; panel.root.style.top = '0';
  panel.root.style.width = '100%'; panel.root.style.height = '100%';

  window.VfDisplay.mountDynamicGeomFrame(frameId, () => ({
    meshes: [ground, ...(grassVisible ? grassPackets : []), ...treeMeshes], camera,
    lights: [
      { id: 'tree_key', kind: 'point', pos: [center[0] - span * .45,
        center[1] - span * .55, maximum[2] + span * .35], target: center,
        color: [1, .91, .72, 1], intensity: span * span * 1.95, range: span * 3,
        casts_shadow: false },
      { id: 'tree_fill', kind: 'point', pos: [center[0] + span * .55,
        center[1] - span * .5, maximum[2] + span * .08], target: center,
        color: [.62, .78, .64, 1], intensity: span * span * 1.1, range: span * 2.5 },
    ], background: [.065, .105, .15, 1], unified_renderer: true,
  }));

  const grassRuntime = createRetainedGeometryPacketRuntimeReference({ requestRender(packets) {
    grassPackets = packets;
    window.VfDisplay.requestDynamicGeomFrameUpdate(frameId);
  } });
  const grassController = createGrassCameraDemandControllerReference({
    field: createGrassMaterialFieldReference({ generator: 'vkf.conditioned', version: 1,
      seed: [0x01234567, 0x89abcdef], domain: 'material',
      hierarchy: ['world:boreal', 'grass-field:tree'], lod: 0, channel: 'surface' }),
    runtime: grassRuntime, planeZ: 0, maximumDistance: Math.max(10, span * 2.2),
    cellBudget: 64, bladeBudget: 1024,
  });
  const rect = panel.body.getBoundingClientRect();
  await grassController.request({ revision: 1, camera: { eye: camera.pos, target: camera.target,
    up: camera.up, verticalFovRadians: camera.fov * Math.PI / 180,
    viewportWidth: Math.max(640, rect.width), viewportHeight: Math.max(360, rect.height) } });
  status.textContent = `${tree.vertexCount.toLocaleString()} tree vertices · ${grassRuntime.packets()[0]?.instance_count?.toLocaleString() || 0} grass blades`;
  window.__treeGrassResult = { outcome: 'ready', tree, grassRuntime };
  window.VfDisplay.requestDynamicGeomFrameUpdate(frameId);

  const animate = (time) => {
    if (time - lastWindFrame >= 180) {
      lastWindFrame = time;
      const strength = Number(windInput.value) * span * 0.0105;
      const phase = time * 0.00115;
      const height = Math.max(1e-6, maximumZ - minimumZ);
      // Wind acts on the leaf embedding while the exact woody network stays
      // fixed at its generated joints. This keeps the trunk topology intact.
      treeMeshes.slice(1).forEach((mesh, localIndex) => {
        const meshIndex = localIndex + 1;
        const base = baseVertices[meshIndex];
        for (let offset = 0; offset < base.length; offset += 10) {
          const normalizedHeight = Math.max(0, Math.min(1, (base[offset + 2] - minimumZ) / height));
          const bend = normalizedHeight * normalizedHeight;
          const localPhase = phase + base[offset + 2] * 0.34 + base[offset] * 0.11;
          mesh.vertices[offset] = base[offset] + Math.sin(localPhase) * strength * bend;
          mesh.vertices[offset + 1] = base[offset + 1]
            + Math.cos(localPhase * 0.83) * strength * 0.48 * bend;
        }
      });
      window.VfDisplay.requestDynamicGeomFrameUpdate(frameId);
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
