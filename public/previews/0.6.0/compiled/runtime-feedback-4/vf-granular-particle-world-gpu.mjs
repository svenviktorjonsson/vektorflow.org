import {
  createCohesionlessGranularContactLawReference,
  createGranularStateReference,
  enumerateGranularSphereContactPairsReference,
} from './vf-physics-granular-core-reference.mjs';
import {
  createTransportBoxReference,
  hashPhysicsStateBufferReference,
} from './vf-physics-transport-core-reference.mjs';
import { SWEPT_WHEEL_CONTACT_WGSL } from './vf-swept-wheel-contact-gpu.mjs';

const WORKGROUP_SIZE = 128;
const MAXIMUM_PARTICLE_AXIS_COUNT = 4096;
const MAXIMUM_PARTICLE_COUNT = 262144;
const MAXIMUM_GRID_AXIS_COUNT = 4096;
const MAXIMUM_GRID_CELL_COUNT = 1048576;

export const GRANULAR_PARTICLE_WORLD_GPU_ABI = Object.freeze({
  particleStrideBytes: 32,
  parameterBytes: 112,
  telemetryAtomicCount: 6,
  particleFields: Object.freeze([
    'position',
    'velocity',
    'previous_position',
    'id',
    'contact_count',
  ]),
});

export const GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY = Object.freeze({
  atomicCount: GRANULAR_PARTICLE_WORLD_GPU_ABI.telemetryAtomicCount,
  fields: Object.freeze({
    peakSpeedSquaredBits: 0,
    nonFiniteFlag: 1,
    peakCellOccupancy: 2,
    gridOverflowCount: 3,
    maximumPenetrationBits: 4,
    maximumResidualPenetrationBits: 5,
  }),
});

export const GRANULAR_PARTICLE_WORLD_GPU_POLICY = Object.freeze({
  dimension: 2,
  columns: 72,
  rows: 56,
  grainRadius: 0.004,
  seedGap: 0.002,
  seedMinimum: Object.freeze([-0.315, -0.08]),
  seed: 0x51a7,
  particleDensity: 1600,
  gravity: Object.freeze([0, -9.82]),
  timeStep: 1 / 240,
  friction: 0.90,
  boundaryFriction: 0.78,
  rollingResistance: 0.20,
  restitution: 0.005,
  projectionRelaxation: 0.98,
  contactIterations: 10,
  gridRebuildInterval: 1,
  contactSlop: 0.000002,
  linearDamping: 0.18,
  maximumParticlesPerCell: 16,
  worldMinimum: Object.freeze([-0.70, -0.22]),
  worldMaximum: Object.freeze([0.70, 0.86]),
  viewMinimum: Object.freeze([-0.62, -0.20]),
  viewMaximum: Object.freeze([0.62, 0.84]),
  openTop: false,
});

const finiteF32 = (value) => Number.isFinite(value) && Number.isFinite(Math.fround(value));

export function normalizeGranularParticleWorldGpuPolicy(overrides = {}) {
  if (overrides === null || typeof overrides !== 'object' || Array.isArray(overrides)) {
    throw new TypeError('Granular particle world policy overrides must be an object');
  }
  for (const key of Reflect.ownKeys(overrides)) {
    if (!Object.hasOwn(GRANULAR_PARTICLE_WORLD_GPU_POLICY, key)) {
      throw new TypeError(`Unknown policy key for Granular particle world: ${String(key)}`);
    }
  }
  const policy = { ...GRANULAR_PARTICLE_WORLD_GPU_POLICY, ...overrides };
  const finitePair = (name, value) => {
    if (!Array.isArray(value) || value.length !== 2 || !value.every(finiteF32)) {
      throw new TypeError(`Granular particle world ${name} must be a finite two-vector`);
    }
    return Object.freeze([...value]);
  };
  policy.gravity = finitePair('gravity', overrides.gravity
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.gravity);
  policy.seedMinimum = finitePair('seedMinimum', overrides.seedMinimum
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.seedMinimum);
  policy.worldMinimum = finitePair('worldMinimum', overrides.worldMinimum
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.worldMinimum);
  policy.worldMaximum = finitePair('worldMaximum', overrides.worldMaximum
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.worldMaximum);
  policy.viewMinimum = finitePair('viewMinimum', overrides.viewMinimum
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.viewMinimum);
  policy.viewMaximum = finitePair('viewMaximum', overrides.viewMaximum
    ?? GRANULAR_PARTICLE_WORLD_GPU_POLICY.viewMaximum);

  if (policy.dimension !== 2) {
    throw new RangeError('This GPU granular kernel specialization is two-dimensional; '
      + 'a 3D shader specialization must be generated from the shared Granular rules');
  }
  for (const name of ['columns', 'rows', 'contactIterations', 'gridRebuildInterval',
    'maximumParticlesPerCell']) {
    if (!Number.isSafeInteger(policy[name]) || policy[name] < 1) {
      throw new RangeError(`Granular particle world ${name} must be a positive integer`);
    }
  }
  for (const name of ['columns', 'rows']) {
    if (policy[name] > MAXIMUM_PARTICLE_AXIS_COUNT) {
      throw new RangeError(`Granular particle world ${name} must be at most `
        + MAXIMUM_PARTICLE_AXIS_COUNT);
    }
  }
  if (policy.contactIterations > 32) {
    throw new RangeError('Granular particle world contactIterations must be from 1 through 32');
  }
  if (policy.gridRebuildInterval > policy.contactIterations) {
    throw new RangeError('Granular gridRebuildInterval cannot exceed contactIterations');
  }
  if (policy.maximumParticlesPerCell > 64) {
    throw new RangeError('Granular maximumParticlesPerCell must be at most 64');
  }
  if (!Number.isSafeInteger(policy.seed) || policy.seed < 0 || policy.seed > 0xffffffff) {
    throw new RangeError('Granular particle world seed must be a u32');
  }
  for (const name of ['grainRadius', 'particleDensity', 'timeStep']) {
    if (!finiteF32(policy[name]) || Math.fround(policy[name]) <= 0) {
      throw new RangeError(`Granular particle world ${name} must be positive`);
    }
  }
  for (const name of ['seedGap', 'contactSlop', 'linearDamping']) {
    if (!finiteF32(policy[name]) || Math.fround(policy[name]) < 0) {
      throw new RangeError(`Granular particle world ${name} must be nonnegative`);
    }
  }
  for (const name of ['friction', 'boundaryFriction', 'rollingResistance', 'restitution']) {
    if (!finiteF32(policy[name]) || policy[name] < 0 || policy[name] > 1) {
      throw new RangeError(`Granular particle world ${name} must be from zero through one`);
    }
  }
  if (!finiteF32(policy.projectionRelaxation) || policy.projectionRelaxation <= 0
      || policy.projectionRelaxation > 1) {
    throw new RangeError('Granular projectionRelaxation must be positive through one');
  }
  if (typeof policy.openTop !== 'boolean') {
    throw new TypeError('Granular particle world openTop must be boolean');
  }
  for (let axis = 0; axis < 2; axis += 1) {
    if (!(policy.worldMinimum[axis] < policy.worldMaximum[axis])) {
      throw new RangeError('Granular particle world bounds must increase on every axis');
    }
    if (!(policy.viewMinimum[axis] < policy.viewMaximum[axis])
        || policy.viewMinimum[axis] < policy.worldMinimum[axis]
        || policy.viewMaximum[axis] > policy.worldMaximum[axis]) {
      throw new RangeError('Granular particle view must increase and stay inside the world');
    }
  }
  const diameter = policy.grainRadius * 2;
  if (!finiteF32(diameter) || diameter >= Math.min(
    policy.worldMaximum[0] - policy.worldMinimum[0],
    policy.worldMaximum[1] - policy.worldMinimum[1])) {
    throw new RangeError('Granular grain diameter must fit inside the world');
  }
  if (policy.contactSlop >= policy.grainRadius) {
    throw new RangeError('Granular contactSlop must be smaller than grainRadius');
  }
  const particleCount = policy.columns * policy.rows;
  if (!Number.isSafeInteger(particleCount) || particleCount > MAXIMUM_PARTICLE_COUNT) {
    throw new RangeError(`Granular particle count must be at most ${MAXIMUM_PARTICLE_COUNT}`);
  }
  const gridDimensions = policy.worldMaximum.map((maximum, axis) =>
    Math.ceil((maximum - policy.worldMinimum[axis]) / diameter));
  for (const [axis, name] of ['columns', 'rows'].entries()) {
    if (!Number.isSafeInteger(gridDimensions[axis]) || gridDimensions[axis] < 1
        || gridDimensions[axis] > MAXIMUM_GRID_AXIS_COUNT) {
      throw new RangeError(`Granular spatial grid ${name} must be from 1 through `
        + MAXIMUM_GRID_AXIS_COUNT);
    }
  }
  const gridCellCount = gridDimensions[0] * gridDimensions[1];
  if (!Number.isSafeInteger(gridCellCount) || gridCellCount > MAXIMUM_GRID_CELL_COUNT) {
    throw new RangeError(`Granular spatial grid cell count must be at most `
      + MAXIMUM_GRID_CELL_COUNT);
  }
  const spacing = diameter * (1 + policy.seedGap);
  const rowPitch = spacing * Math.sqrt(3) * 0.5;
  const maximumSeed = [
    policy.seedMinimum[0] + (policy.columns - 1) * spacing + spacing * 0.5,
    policy.seedMinimum[1] + (policy.rows - 1) * rowPitch,
  ];
  for (let axis = 0; axis < 2; axis += 1) {
    const lower = policy.worldMinimum[axis] + policy.grainRadius;
    const upper = policy.worldMaximum[axis] - policy.grainRadius;
    if (policy.seedMinimum[axis] < lower || maximumSeed[axis] > upper) {
      throw new RangeError('Granular overlap-free seed extent must stay inside the world');
    }
  }
  return Object.freeze(policy);
}

export function createGranularParticleWorldGpuSeedReference(options = {}) {
  const policy = normalizeGranularParticleWorldGpuPolicy(options);
  const diameter = policy.grainRadius * 2;
  const spacing = diameter * (1 + policy.seedGap);
  const rowPitch = spacing * Math.sqrt(3) * 0.5;
  const initialWheelAngle = 0.22;
  const contactRadius = policy.grainRadius + 0.012;
  const baffles = [
    [[0.500, 0.000], [0.350, 0.000]], [[0.000, 0.500], [0.000, 0.350]],
    [[-0.500, 0.000], [-0.350, 0.000]], [[0.000, -0.500], [0.000, -0.350]],
    [[-0.220, 0.175], [-0.075, 0.135]], [[0.075, -0.055], [0.215, -0.115]],
    [[-0.105, -0.250], [0.020, -0.155]],
  ];
  const rotate = ([x, y]) => [Math.cos(initialWheelAngle) * x
    - Math.sin(initialWheelAngle) * y,
  Math.sin(initialWheelAngle) * x + Math.cos(initialWheelAngle) * y];
  const distanceToSegment = ([x, y], a, b) => {
    const dx = b[0] - a[0]; const dy = b[1] - a[1];
    const along = Math.max(0, Math.min(1,
      ((x - a[0]) * dx + (y - a[1]) * dy) / Math.max(dx * dx + dy * dy, 1e-12)));
    return Math.hypot(x - (a[0] + dx * along), y - (a[1] + dy * along));
  };
  const candidates = [];
  for (let row = 0; row < policy.rows; row += 1) {
    for (let column = 0; column < policy.columns; column += 1) {
      const point = [policy.seedMinimum[0] + column * spacing
        + (row & 1) * spacing * 0.5,
      policy.seedMinimum[1] + row * rowPitch];
      if (Math.hypot(point[0], point[1] - 0.32) > 0.50 - contactRadius) continue;
      const overlapsBaffle = baffles.some(([localA, localB]) => {
        const rotatedA = rotate(localA); const rotatedB = rotate(localB);
        return distanceToSegment(point, [rotatedA[0], rotatedA[1] + 0.32],
          [rotatedB[0], rotatedB[1] + 0.32]) < contactRadius;
      });
      if (!overlapsBaffle) candidates.push(point);
    }
  }
  const count = candidates.length;
  if (count < 1) throw new RangeError('Granular drum seed contains no valid grains');
  const particleMass = policy.particleDensity * Math.PI * policy.grainRadius ** 2;
  const contactLaw = createCohesionlessGranularContactLawReference({
    friction: policy.friction,
    rollingResistance: policy.rollingResistance,
    restitution: policy.restitution,
  });
  const referenceState = createGranularStateReference({
    kind: 'granular-particle-world-gpu-seed-state-reference:v1',
    seed: policy.seed,
    count,
    precision: 'f32',
    includeIds: true,
    gravity: [policy.gravity[0], 0, policy.gravity[1]],
    timeStep: policy.timeStep,
    particleMass,
  });
  const boundaryBox = createTransportBoxReference({
    minimum: [policy.worldMinimum[0], -policy.grainRadius, policy.worldMinimum[1]],
    maximum: [policy.worldMaximum[0], policy.grainRadius, policy.worldMaximum[1]],
    openFaces: policy.openTop ? ['maximum-z'] : [],
  });
  const stride = GRANULAR_PARTICLE_WORLD_GPU_ABI.particleStrideBytes / 4;
  const bytes = new ArrayBuffer(count * GRANULAR_PARTICLE_WORLD_GPU_ABI.particleStrideBytes);
  const floats = new Float32Array(bytes);
  const integers = new Uint32Array(bytes);
  for (let grain = 0; grain < candidates.length; grain += 1) {
      const [x, y] = candidates[grain];
      const offset = grain * stride;
      floats[offset] = x;
      floats[offset + 1] = y;
      floats[offset + 2] = 0;
      floats[offset + 3] = 0;
      floats[offset + 4] = x;
      floats[offset + 5] = y;
      integers[offset + 6] = grain;
      integers[offset + 7] = 0;
      const referenceOffset = grain * 3;
      referenceState.positions[referenceOffset] = x;
      referenceState.positions[referenceOffset + 1] = 0;
      referenceState.positions[referenceOffset + 2] = y;
      referenceState.ids[grain] = grain;
  }
  referenceState.stateHash = hashPhysicsStateBufferReference(referenceState);
  const initialContacts = enumerateGranularSphereContactPairsReference({
    positions: referenceState.positions,
    count,
    maximumDiameter: diameter,
    radiusAt: () => policy.grainRadius,
  });
  if (initialContacts.length !== 0) {
    throw new Error('Granular GPU seed construction produced overlapping grains');
  }
  return Object.freeze({
    kind: 'granular-particle-world-gpu-seed-reference:v1',
    policy,
    count,
    stride,
    bytes,
    floats,
    integers,
    diameter,
    spacing,
    rowPitch,
    filteredWheelOverlapCount: policy.columns * policy.rows - count,
    minimumSeedSeparation: spacing,
    initialOverlapPairCount: initialContacts.length,
    particleMass,
    contactLaw,
    referenceState,
    boundaryBox,
  });
}

// This is the 2D lowering of the shared Granular/Transport semantics. All
// neighborhood work and state evolution remain in compute; render pipelines
// only consume the stable public particle buffer after finalize_state.
export const GRANULAR_PARTICLE_WORLD_GPU_WGSL = /* wgsl */`
struct Grain {
  position: vec2<f32>,
  velocity: vec2<f32>,
  previous_position: vec2<f32>,
  id: u32,
  contact_count: u32,
};

struct Params {
  counts: vec4<u32>,
  flags: vec4<u32>,
  world: vec4<f32>,
  material: vec4<f32>,
  force: vec4<f32>,
  solver: vec4<f32>,
  motion: vec4<f32>,
};

struct WheelContact {
  is_active: u32,
  kind: u32,
  normal: vec2<f32>,
  surface_velocity: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> source_grains: array<Grain>;
@group(0) @binding(1) var<storage, read_write> target_grains: array<Grain>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_items: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

const PI: f32 = 3.141592653589793;
const MAX_F32: f32 = 3.4028234663852886e+38;
const WHEEL_CENTER: vec2<f32> = vec2<f32>(0.0, 0.32);
const WHEEL_RADIUS: f32 = 0.50;
const WHEEL_BAR_HALF_WIDTH: f32 = 0.012;

fn rotate_local(point: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle); let s = sin(angle);
  return vec2<f32>(c * point.x - s * point.y, s * point.x + c * point.y);
}

fn baffle(index: u32) -> vec4<f32> {
  if (index == 0u) { return vec4<f32>(0.500, 0.000, 0.350, 0.000); }
  if (index == 1u) { return vec4<f32>(0.000, 0.500, 0.000, 0.350); }
  if (index == 2u) { return vec4<f32>(-0.500, 0.000, -0.350, 0.000); }
  if (index == 3u) { return vec4<f32>(0.000, -0.500, 0.000, -0.350); }
  if (index == 4u) { return vec4<f32>(-0.220, 0.175, -0.075, 0.135); }
  if (index == 5u) { return vec4<f32>(0.075, -0.055, 0.215, -0.115); }
  return vec4<f32>(-0.105, -0.250, 0.020, -0.155);
}

fn telemetry_base() -> u32 { return params.counts.y * params.counts.z; }

fn finite_scalar(value: f32) -> bool { return abs(value) <= MAX_F32; }

fn record_nonfinite(grain: Grain) {
  if (!finite_scalar(grain.position.x) || !finite_scalar(grain.position.y)
      || !finite_scalar(grain.velocity.x) || !finite_scalar(grain.velocity.y)
      || !finite_scalar(grain.previous_position.x)
      || !finite_scalar(grain.previous_position.y)) {
    atomicOr(&cell_counts[telemetry_base() + 1u], 1u);
  }
}

fn record_speed(velocity: vec2<f32>) {
  let speed_squared = dot(velocity, velocity);
  if (!finite_scalar(speed_squared) || speed_squared < 0.0) {
    atomicOr(&cell_counts[telemetry_base() + 1u], 1u);
    return;
  }
  atomicMax(&cell_counts[telemetry_base()], bitcast<u32>(speed_squared));
}

fn record_penetration(penetration: f32) {
  if (!finite_scalar(penetration)) {
    atomicOr(&cell_counts[telemetry_base() + 1u], 1u);
    return;
  }
  if (penetration > 0.0) {
    atomicMax(&cell_counts[telemetry_base() + 4u], bitcast<u32>(penetration));
  }
}

fn record_residual_penetration(penetration: f32) {
  if (!finite_scalar(penetration)) {
    atomicOr(&cell_counts[telemetry_base() + 1u], 1u);
    return;
  }
  if (penetration > 0.0) {
    atomicMax(&cell_counts[telemetry_base() + 5u], bitcast<u32>(penetration));
  }
}

fn mix_u32(input: u32) -> u32 {
  var value = input;
  value = value ^ (value >> 16u);
  value = value * 0x7feb352du;
  value = value ^ (value >> 15u);
  value = value * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn coincident_normal(own_id: u32, other_id: u32) -> vec2<f32> {
  let lower = min(own_id, other_id);
  let upper = max(own_id, other_id);
  let bits = mix_u32(params.flags.x ^ (lower * 0x9e3779b1u) ^ (upper * 0x85ebca77u));
  let angle = f32(bits & 0xffffu) / 65535.0 * 2.0 * PI;
  let basis = vec2<f32>(cos(angle), sin(angle));
  return select(-basis, basis, own_id == lower);
}

fn cell_coordinate(position: vec2<f32>) -> vec2<i32> {
  let diameter = params.material.x * 2.0;
  return vec2<i32>(floor((position - params.world.xy) / diameter));
}

fn valid_cell(cell: vec2<i32>) -> bool {
  return cell.x >= 0 && cell.y >= 0
    && cell.x < i32(params.counts.y) && cell.y < i32(params.counts.z);
}

fn cell_index(cell: vec2<i32>) -> u32 {
  return u32(cell.y) * params.counts.y + u32(cell.x);
}

fn project_world(position_input: vec2<f32>) -> vec2<f32> {
  let radius = params.material.x;
  let lower = params.world.xy + vec2<f32>(radius);
  let upper = params.world.zw - vec2<f32>(radius);
  var position = position_input;
  record_penetration(max(0.0, lower.x - position.x));
  record_penetration(max(0.0, position.x - upper.x));
  record_penetration(max(0.0, lower.y - position.y));
  position.x = clamp(position.x, lower.x, upper.x);
  position.y = max(position.y, lower.y);
  if (params.flags.y == 0u) {
    record_penetration(max(0.0, position.y - upper.y));
    position.y = min(position.y, upper.y);
  }
  return position;
}

fn wheel_penetration(position: vec2<f32>) -> f32 {
  let contact_radius = params.material.x + WHEEL_BAR_HALF_WIDTH;
  var maximum = 0.0;
  for (var segment = 0u; segment < 7u; segment = segment + 1u) {
    let local = baffle(segment);
    let a = WHEEL_CENTER + rotate_local(local.xy, params.solver.z);
    let b = WHEEL_CENTER + rotate_local(local.zw, params.solver.z);
    let edge = b - a;
    let along = clamp(dot(position - a, edge) / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
    let closest = a + edge * along;
    maximum = max(maximum, contact_radius - length(position - closest));
  }
  let radial_length = length(position - WHEEL_CENTER);
  return max(maximum, radial_length - (WHEEL_RADIUS - contact_radius));
}

fn project_wheel_at(position_input: vec2<f32>, angle:f32) -> vec2<f32> {
  var position = position_input;
  let contact_radius = params.material.x + WHEEL_BAR_HALF_WIDTH;
  for (var projection = 0u; projection < 4u; projection = projection + 1u) {
    for (var segment = 0u; segment < 7u; segment = segment + 1u) {
      let local = baffle(segment);
      let a = WHEEL_CENTER + rotate_local(local.xy, angle);
      let b = WHEEL_CENTER + rotate_local(local.zw, angle);
      let edge = b - a;
      let along = clamp(dot(position - a, edge) / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
      let closest = a + edge * along;
      let separation = position - closest;
      let distance = length(separation);
      if (distance < contact_radius) {
        let tangent = normalize(edge);
        let normal = select(vec2<f32>(-tangent.y, tangent.x),
          separation / max(distance, 1.0e-8), distance > 1.0e-8);
        let penetration = contact_radius - distance;
        record_penetration(penetration);
        position = position + normal * penetration;
      }
    }
    let radial = position - WHEEL_CENTER;
    let radial_length = length(radial);
    let rim_limit = WHEEL_RADIUS - contact_radius;
    if (radial_length > rim_limit) {
      let outward = radial / max(radial_length, 1.0e-8);
      let penetration = radial_length - rim_limit;
      record_penetration(penetration);
      position = WHEEL_CENTER + outward * rim_limit;
    }
  }
  return position;
}

fn project_wheel(position:vec2<f32>)->vec2<f32>{return project_wheel_at(position,params.solver.z);}
fn sweep_particle(origin:vec2<f32>,destination:vec2<f32>)->vec2<f32>{
  let radius=params.material.x+WHEEL_BAR_HALF_WIDTH;
  return swept_wheel_motion(project_wheel(origin),destination,vec2<f32>(0.0),WHEEL_CENTER,
    WHEEL_RADIUS-radius,radius,params.solver.z).position;
}
@compute @workgroup_size(128)
fn sweep_wheel(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.x){return;}var grain=source_grains[i];
  let swept=swept_wheel_rotation(grain.position,grain.velocity,WHEEL_CENTER,
    params.material.x+WHEEL_BAR_HALF_WIDTH,params.motion.x,params.motion.y,params.motion.w);
  grain.position=project_wheel(swept.position);grain.velocity=swept.velocity;
  grain.previous_position=grain.position;target_grains[i]=grain;
}

fn sample_wheel_contact(position: vec2<f32>, tolerance: f32) -> WheelContact {
  let contact_radius = params.material.x + WHEEL_BAR_HALF_WIDTH;
  var best_gap = MAX_F32;
  var result = WheelContact(0u, 0u, vec2<f32>(0.0), vec2<f32>(0.0));
  for (var segment = 0u; segment < 7u; segment = segment + 1u) {
    let local = baffle(segment);
    let a = WHEEL_CENTER + rotate_local(local.xy, params.solver.z);
    let b = WHEEL_CENTER + rotate_local(local.zw, params.solver.z);
    let edge = b - a;
    let tangent = normalize(edge);
    let along = clamp(dot(position - a, edge) / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
    let closest = a + edge * along;
    let separation = position - closest;
    let distance = length(separation);
    let gap = distance - contact_radius;
    if (gap <= tolerance && gap < best_gap) {
      let normal = select(vec2<f32>(-tangent.y, tangent.x),
        separation / max(distance, 1.0e-8), distance > 1.0e-8);
      let radius_vector = closest - WHEEL_CENTER;
      let surface_velocity = params.solver.w
        * vec2<f32>(-radius_vector.y, radius_vector.x);
      result = WheelContact(1u, 2u, normal, surface_velocity);
      best_gap = gap;
    }
  }
  let radial = position - WHEEL_CENTER;
  let radial_length = length(radial);
  let rim_limit = WHEEL_RADIUS - contact_radius;
  let rim_gap = rim_limit - radial_length;
  if (rim_gap <= tolerance && rim_gap < best_gap) {
    let outward = radial / max(radial_length, 1.0e-8);
    let radius_vector = outward * WHEEL_RADIUS;
    let surface_velocity = params.solver.w
      * vec2<f32>(-radius_vector.y, radius_vector.x);
    result = WheelContact(1u, 1u, -outward, surface_velocity);
  }
  return result;
}

@compute @workgroup_size(128)
fn clear_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let cell_count = params.counts.y * params.counts.z;
  let cell=invocation.x;if(cell>=cell_count){return;}
  let previous=atomicLoad(&cell_counts[cell]);atomicStore(&cell_counts[cell],0u);
  if(previous>0u){for(var slot=0u;slot<params.counts.w;slot++){
    atomicStore(&cell_items[cell*params.counts.w+slot],0xffffffffu);
  }
  }
}

@compute @workgroup_size(128)
fn fill_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  let grain = source_grains[index];
  record_nonfinite(grain);
  record_speed(grain.velocity);
  let cell = cell_coordinate(grain.position);
  if (!valid_cell(cell)) {
    atomicOr(&cell_counts[telemetry_base() + 1u], 1u);
    return;
  }
  let bucket = cell_index(cell);
  let occupancy = atomicAdd(&cell_counts[bucket], 1u) + 1u;
  atomicMax(&cell_counts[telemetry_base() + 2u], occupancy);
  if (occupancy > params.counts.w) {
    atomicAdd(&cell_counts[telemetry_base() + 3u], 1u);
  }
  // Atomic-min cascade retains the same lowest stable IDs regardless of
  // invocation order. sort_cells then fixes floating-point traversal order.
  var candidate = index;
  for (var slot = 0u; slot < params.counts.w; slot = slot + 1u) {
    let address = bucket * params.counts.w + slot;
    let previous = atomicMin(&cell_items[address], candidate);
    if (previous == 0xffffffffu) { break; }
    candidate = max(candidate, previous);
  }
}

@compute @workgroup_size(128)
fn sort_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let bucket = invocation.x;
  let cell_count = params.counts.y * params.counts.z;
  if (bucket >= cell_count) { return; }
  let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
  let base = bucket * params.counts.w;
  for (var index = 1u; index < count; index = index + 1u) {
    let key = atomicLoad(&cell_items[base + index]);
    var cursor = index;
    while (cursor > 0u) {
      let previous = atomicLoad(&cell_items[base + cursor - 1u]);
      if (previous <= key) { break; }
      atomicStore(&cell_items[base + cursor], previous);
      cursor = cursor - 1u;
    }
    atomicStore(&cell_items[base + cursor], key);
  }
}

@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  var grain = source_grains[index];
  let previous = grain.position;
  grain.velocity = (grain.velocity + params.force.xy * params.force.z)
    * exp(-params.force.w * params.force.z);
  var candidate = previous + grain.velocity * params.force.z;
  let radius = params.material.x;
  let lower = params.world.xy + vec2<f32>(radius);
  let upper = params.world.zw - vec2<f32>(radius);
  var contacts = 0u;
  if (candidate.x < lower.x) {
    record_penetration(lower.x - candidate.x); candidate.x = lower.x;
    if (grain.velocity.x < 0.0) { grain.velocity.x = -grain.velocity.x * params.material.z; }
    grain.velocity.y = grain.velocity.y * max(0.0, 1.0 - params.solver.y * 0.08);
    contacts = contacts + 1u;
  } else if (candidate.x > upper.x) {
    record_penetration(candidate.x - upper.x); candidate.x = upper.x;
    if (grain.velocity.x > 0.0) { grain.velocity.x = -grain.velocity.x * params.material.z; }
    grain.velocity.y = grain.velocity.y * max(0.0, 1.0 - params.solver.y * 0.08);
    contacts = contacts + 1u;
  }
  if (candidate.y < lower.y) {
    record_penetration(lower.y - candidate.y); candidate.y = lower.y;
    if (grain.velocity.y < 0.0) { grain.velocity.y = -grain.velocity.y * params.material.z; }
    grain.velocity.x = grain.velocity.x * max(0.0, 1.0 - params.solver.y * 0.08);
    contacts = contacts + 1u;
  } else if (params.flags.y == 0u && candidate.y > upper.y) {
    record_penetration(candidate.y - upper.y); candidate.y = upper.y;
    if (grain.velocity.y > 0.0) { grain.velocity.y = -grain.velocity.y * params.material.z; }
    grain.velocity.x = grain.velocity.x * max(0.0, 1.0 - params.solver.y * 0.08);
    contacts = contacts + 1u;
  }
  candidate = sweep_particle(previous,candidate);
  grain.previous_position = previous;
  grain.position = candidate;
  grain.contact_count = contacts;
  record_nonfinite(grain);
  record_speed(grain.velocity);
  target_grains[index] = grain;
}

@compute @workgroup_size(128)
fn project_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  var grain = source_grains[index];
  let diameter = params.material.x * 2.0 * 1.001;
  let diameter_squared = diameter * diameter;
  let center_cell = cell_coordinate(grain.position);
  var correction = vec2<f32>(0.0);
  var pair_count = 0u;
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other_index = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other_index >= params.counts.x || other_index == index) { continue; }
        let other = source_grains[other_index];
        let separation = grain.position - other.position;
        let distance_squared = dot(separation, separation);
        if (distance_squared >= diameter_squared) { continue; }
        let distance = sqrt(max(0.0, distance_squared));
        let normal = select(coincident_normal(grain.id, other.id),
          separation / max(distance, 1.0e-12), distance > 1.0e-12);
        let penetration = diameter - distance;
        record_penetration(penetration);
        let projected_depth = max(0.0, penetration - params.solver.x);
        if (projected_depth <= 0.0) { continue; }
        let normal_correction = projected_depth * 0.5;
        var pair_correction = normal * normal_correction;
        // Position-based Coulomb friction: tangential correction is bounded
        // by mu times this contact's unilateral normal correction.
        let own_motion = grain.position - grain.previous_position;
        let other_motion = other.position - other.previous_position;
        let relative_motion = own_motion - other_motion;
        let tangent_motion = relative_motion - normal * dot(relative_motion, normal);
        let tangent_length = length(tangent_motion);
        if (tangent_length > 1.0e-12) {
          let tangent_limit = params.material.y * normal_correction;
          pair_correction = pair_correction - tangent_motion / tangent_length
            * min(tangent_length * 0.5, tangent_limit);
        }
        correction = correction + pair_correction;
        pair_count = pair_count + 1u;
      }
    }
  }
  let denominator = max(1.0, f32(pair_count) * 0.5);
  grain.position = sweep_particle(grain.position,project_world(grain.position
    + correction * params.material.w / denominator));
  grain.contact_count = grain.contact_count + pair_count;
  record_nonfinite(grain);
  target_grains[index] = grain;
}

@compute @workgroup_size(128)
fn finalize_state(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  var grain = source_grains[index];
  // Cancel positional drift below the wall's static Coulomb yield. Velocity
  // friction alone resets speed but still creeps by g*timestep² every step.
  let resting=sample_wheel_contact(grain.position,params.solver.x*2.0+1.0e-7);
  if(resting.is_active!=0u){
    let motion=grain.position-grain.previous_position-resting.surface_velocity*params.force.z;
    let tangent=motion-resting.normal*dot(motion,resting.normal);let distance=length(tangent);
    let load=max(0.0,-dot(params.force.xy,resting.normal))*params.force.z*params.force.z;
    if(distance>1.0e-12){grain.position=project_wheel(grain.position-tangent/distance*min(distance,params.solver.y*load));}
  }
  // Geometric overlap repair is not an impulse. Reconstructing it as velocity
  // launches a compressed pile after a kinematic baffle sweeps through it.
  var velocity = grain.velocity;
  let diameter = params.material.x * 2.0;
  // The normal solve's safety gap must still count as a resting friction
  // contact; it is not cohesion or attraction between separated grains.
  let contact_band = diameter * 1.002 + params.solver.x * 2.0;
  let contact_band_squared = contact_band * contact_band;
  let center_cell = cell_coordinate(grain.position);
  var velocity_delta = vec2<f32>(0.0);
  var impulse_contact_count = 0u;
  var support_load = 0.0;
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other_index = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other_index >= params.counts.x || other_index == index) { continue; }
        let other = source_grains[other_index];
        let separation = grain.position - other.position;
        let distance_squared = dot(separation, separation);
        if (distance_squared > contact_band_squared) { continue; }
        let distance = sqrt(max(0.0, distance_squared));
        let normal = select(coincident_normal(grain.id, other.id),
          separation / max(distance, 1.0e-12), distance > 1.0e-12);
        let other_velocity = other.velocity;
        let relative_velocity = velocity - other_velocity;
        let normal_speed = dot(relative_velocity, normal);
        var normal_impulse = 0.0;
        if (normal_speed < 0.0) {
          normal_impulse = -0.5 * (1.0 + params.material.z) * normal_speed;
          velocity_delta = velocity_delta + normal * normal_impulse;
        }
        let gravity_support = max(0.0,
          -dot(params.force.xy * params.force.z, normal)) * 0.5;
        support_load = support_load + gravity_support;
        let tangent_velocity = relative_velocity - normal * normal_speed;
        let tangent_speed = length(tangent_velocity);
        if (tangent_speed > 1.0e-12) {
          // Resting grains still carry normal load. Including the gravity load
          // gives the contact network a true static Coulomb yield threshold,
          // so a heap keeps its angle of repose instead of flowing like water.
          let support_impulse = max(normal_impulse, gravity_support);
          let tangent_impulse = min(tangent_speed * 0.5,
            params.material.y * support_impulse);
          velocity_delta = velocity_delta - tangent_velocity / tangent_speed * tangent_impulse;
        }
        impulse_contact_count = impulse_contact_count + 1u;
      }
    }
  }
  // Jacobi averaging prevents a dense contact fan from summing several full
  // pair impulses into artificial kinetic energy.
  velocity = velocity + velocity_delta / max(1.0, f32(impulse_contact_count));
  let radius = params.material.x;
  let lower = params.world.xy + vec2<f32>(radius);
  let upper = params.world.zw - vec2<f32>(radius);
  let tolerance = params.solver.x * 2.0 + 1.0e-7;
  if (grain.position.x <= lower.x + tolerance && velocity.x < 0.0) {
    let normal_speed = -velocity.x;
    velocity.x = normal_speed * params.material.z;
    velocity.y = velocity.y * max(0.0, 1.0 - params.solver.y);
  } else if (grain.position.x >= upper.x - tolerance && velocity.x > 0.0) {
    let normal_speed = velocity.x;
    velocity.x = -normal_speed * params.material.z;
    velocity.y = velocity.y * max(0.0, 1.0 - params.solver.y);
  }
  if (grain.position.y <= lower.y + tolerance && velocity.y < 0.0) {
    let normal_speed = -velocity.y;
    velocity.y = normal_speed * params.material.z;
    velocity.x = velocity.x * max(0.0, 1.0 - params.solver.y);
  } else if (params.flags.y == 0u
      && grain.position.y >= upper.y - tolerance && velocity.y > 0.0) {
    let normal_speed = velocity.y;
    velocity.y = -normal_speed * params.material.z;
    velocity.x = velocity.x * max(0.0, 1.0 - params.solver.y);
  }
  let wheel_contact = sample_wheel_contact(grain.position, tolerance);
  if (wheel_contact.is_active != 0u) {
    var relative_velocity = velocity - wheel_contact.surface_velocity;
    let normal_speed = dot(relative_velocity, wheel_contact.normal);
    var normal_impulse = 0.0;
    if (normal_speed < 0.0) {
      normal_impulse = -(1.0 + params.material.z) * normal_speed;
      relative_velocity = relative_velocity
        + wheel_contact.normal * normal_impulse;
    }
    let tangent_velocity = relative_velocity
      - wheel_contact.normal * dot(relative_velocity, wheel_contact.normal);
    let tangent_speed = length(tangent_velocity);
    let gravity_support = max(0.0,
      -dot(params.force.xy * params.force.z, wheel_contact.normal));
    support_load = support_load + gravity_support;
    if (tangent_speed > 1.0e-12) {
      // Coulomb friction is proportional to actual normal load. It arrests
      // grains on the floor without creating adhesion on a vertical wall.
      let support_impulse = max(normal_impulse, gravity_support);
      relative_velocity = relative_velocity - tangent_velocity / tangent_speed
        * min(tangent_speed, params.solver.y * support_impulse);
    }
    velocity = wheel_contact.surface_velocity + relative_velocity;
  }
  // Rolling resistance is represented as low-speed contact damping in this
  // 2D sphere specialization. It arrests settled piles without damping grains
  // that are airborne or avalanching above the yield threshold.
  if (support_load > 1.0e-7 && length(velocity) < 0.22) {
    velocity = velocity * 0.58;
  }
  grain.velocity = velocity;
  record_nonfinite(grain);
  record_speed(velocity);
  target_grains[index] = grain;
}

@compute @workgroup_size(128)
fn audit_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  let grain = source_grains[index];
  let radius = params.material.x;
  let diameter = radius * 2.0;
  let diameter_squared = diameter * diameter;
  let lower = params.world.xy + vec2<f32>(radius);
  let upper = params.world.zw - vec2<f32>(radius);
  record_residual_penetration(max(0.0, lower.x - grain.position.x));
  record_residual_penetration(max(0.0, grain.position.x - upper.x));
  record_residual_penetration(max(0.0, lower.y - grain.position.y));
  if (params.flags.y == 0u) {
    record_residual_penetration(max(0.0, grain.position.y - upper.y));
  }
  record_residual_penetration(max(0.0, wheel_penetration(grain.position)));
  let center_cell = cell_coordinate(grain.position);
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other_index = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other_index <= index || other_index >= params.counts.x) { continue; }
        let separation = grain.position - source_grains[other_index].position;
        let distance_squared = dot(separation, separation);
        if (distance_squared < diameter_squared) {
          record_residual_penetration(diameter - sqrt(max(0.0, distance_squared)));
        }
      }
    }
  }
}
`;

const requireDevice = (device) => {
  if (!device || typeof device.createBuffer !== 'function'
      || typeof device.createComputePipelineAsync !== 'function') {
    throw new TypeError('WebGPU device required for Granular particle world');
  }
  return device;
};

const ceilDiv = (value, divisor) => Math.ceil(value / divisor);

function createBuffer(device, label, size, usage, data = null) {
  const buffer = device.createBuffer({ label, size: Math.max(4, size), usage,
    mappedAtCreation: data !== null });
  if (data !== null) {
    const source = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    new Uint8Array(buffer.getMappedRange()).set(source);
    buffer.unmap();
  }
  return buffer;
}

export async function createGranularParticleWorldGpuRuntime(deviceArgument, options = {}) {
  const device = requireDevice(deviceArgument);
  const seed = options.initialState ?? createGranularParticleWorldGpuSeedReference(options);
  const { policy } = seed;
  const resetBytes = seed.bytes.slice(0);
  const diameter = policy.grainRadius * 2;
  const gridColumns = Math.ceil(
    (policy.worldMaximum[0] - policy.worldMinimum[0]) / diameter);
  const gridRows = Math.ceil(
    (policy.worldMaximum[1] - policy.worldMinimum[1]) / diameter);
  const gridCellCount = gridColumns * gridRows;
  const telemetryAtomicCount = GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY.atomicCount;
  const cellCountsAtomicCount = gridCellCount + telemetryAtomicCount;
  const cellItemCount = gridCellCount * policy.maximumParticlesPerCell;
  const maximumDispatch = device.limits?.maxComputeWorkgroupsPerDimension
    ?? Number.MAX_SAFE_INTEGER;
  for (const [name, count] of [['particles', seed.count], ['grid', gridCellCount],
    ['grid storage', cellItemCount]]) {
    if (ceilDiv(count, WORKGROUP_SIZE) > maximumDispatch) {
      throw new RangeError(`Granular ${name} dispatch exceeds this GPU device limit`);
    }
  }
  const maximumBufferBytes = Math.min(device.limits?.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
    device.limits?.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER);
  for (const [name, size] of [
    ['particle', seed.bytes.byteLength],
    ['cell counts and telemetry', cellCountsAtomicCount * 4],
    ['cell items', cellItemCount * 4],
  ]) {
    if (!Number.isSafeInteger(size) || size > maximumBufferBytes) {
      throw new RangeError(`Granular ${name} buffer exceeds this GPU device limit`);
    }
  }

  const storage = GPUBufferUsage.STORAGE;
  const copyDst = GPUBufferUsage.COPY_DST;
  const copySrc = GPUBufferUsage.COPY_SRC;
  const vertex = GPUBufferUsage.VERTEX ?? 0;
  // A is the stable render surface. B and C are private Jacobi work buffers;
  // finalize_state always writes the completed step back to A.
  const particleBuffer = createBuffer(device, 'VKF Granular particles (stable)',
    seed.bytes.byteLength, storage | copyDst | copySrc | vertex, new Uint8Array(seed.bytes));
  const workBufferB = createBuffer(device, 'VKF Granular particles (work B)',
    seed.bytes.byteLength, storage | copySrc);
  const workBufferC = createBuffer(device, 'VKF Granular particles (work C)',
    seed.bytes.byteLength, storage | copySrc);
  const cellCountsBuffer = createBuffer(device, 'VKF Granular cell counts + telemetry',
    cellCountsAtomicCount * 4, storage | copyDst | copySrc);
  const cellItemsBuffer = createBuffer(device, 'VKF Granular cell items',
    cellItemCount * 4, storage | copyDst);
  device.queue.writeBuffer(cellItemsBuffer,0,new Uint32Array(cellItemCount).fill(0xffffffff));
  const paramsBuffer = createBuffer(device, 'VKF Granular world parameters',
    GRANULAR_PARTICLE_WORLD_GPU_ABI.parameterBytes, GPUBufferUsage.UNIFORM | copyDst);
  const shader = device.createShaderModule({
    label: 'VKF 2D Granular particle world GPU specialization',
    code: options.shaderSource ?? (GRANULAR_PARTICLE_WORLD_GPU_WGSL + SWEPT_WHEEL_CONTACT_WGSL),
  });
  if (typeof shader.getCompilationInfo === 'function') {
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === 'error');
    if (errors.length) {
      throw new Error(errors.map((message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
    }
  }
  const layout = device.createBindGroupLayout({
    label: 'VKF Granular particle world compute bindings',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const entryPoints = ['clear_cells', 'fill_cells', 'sort_cells', 'integrate',
    'project_contacts', 'finalize_state', 'audit_contacts', 'sweep_wheel'];
  if (typeof device.pushErrorScope === 'function') device.pushErrorScope('validation');
  let pipelineEntries;
  try {
    pipelineEntries = await Promise.all(entryPoints.map(async (entryPoint) => [entryPoint,
      await device.createComputePipelineAsync({ label: `VKF Granular ${entryPoint}`,
        layout: pipelineLayout, compute: { module: shader, entryPoint } })]));
  } finally {
    if (typeof device.popErrorScope === 'function') {
      const validationError = await device.popErrorScope();
      if (validationError) throw new Error(validationError.message);
    }
  }
  const pipelines = Object.freeze(Object.fromEntries(pipelineEntries));
  const makeBindGroup = (source, target) => device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: source } },
    { binding: 1, resource: { buffer: target } },
    { binding: 2, resource: { buffer: cellCountsBuffer } },
    { binding: 3, resource: { buffer: cellItemsBuffer } },
    { binding: 4, resource: { buffer: paramsBuffer } },
  ] });
  const bindGroups = Object.freeze({
    AB: makeBindGroup(particleBuffer, workBufferB),
    BC: makeBindGroup(workBufferB, workBufferC),
    CB: makeBindGroup(workBufferC, workBufferB),
    BA: makeBindGroup(workBufferB, particleBuffer),
    CA: makeBindGroup(workBufferC, particleBuffer),
  });

  const paramsBytes = new ArrayBuffer(GRANULAR_PARTICLE_WORLD_GPU_ABI.parameterBytes);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  let wheelAngle = 0;
  let wheelAngularVelocity = 0;
  const updateParams = () => {
    paramsU32.set([seed.count, gridColumns, gridRows,
      policy.maximumParticlesPerCell], 0);
    paramsU32.set([policy.seed, policy.openTop ? 1 : 0,
      policy.contactIterations, policy.gridRebuildInterval], 4);
    paramsF32.set([...policy.worldMinimum, ...policy.worldMaximum], 8);
    paramsF32.set([policy.grainRadius, policy.friction,
      policy.restitution, policy.projectionRelaxation], 12);
    paramsF32.set([...policy.gravity, policy.timeStep, policy.linearDamping], 16);
    paramsF32.set([policy.contactSlop, policy.boundaryFriction,
      wheelAngle, wheelAngularVelocity], 20);
    device.queue.writeBuffer(paramsBuffer, 0, paramsBytes);
  };
  const dispatch = (pass, entryPoint, bindGroup, count) => {
    pass.setPipeline(pipelines[entryPoint]);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(ceilDiv(count, WORKGROUP_SIZE));
  };
  const buildGrid = (pass, bindGroup) => {
    dispatch(pass, 'clear_cells', bindGroup, gridCellCount);
    dispatch(pass, 'fill_cells', bindGroup, seed.count);
    dispatch(pass, 'sort_cells', bindGroup, gridCellCount);
  };
  let frameIndex = 0;
  const encodeFixedStep = (pass) => {
    dispatch(pass, 'integrate', bindGroups.AB, seed.count);
    let current = 'B';
    for (let iteration = 0; iteration < policy.contactIterations; iteration += 1) {
      const projection = current === 'B' ? bindGroups.BC : bindGroups.CB;
      if (iteration % policy.gridRebuildInterval === 0) buildGrid(pass, projection);
      dispatch(pass, 'project_contacts', projection, seed.count);
      current = current === 'B' ? 'C' : 'B';
    }
    const finalize = current === 'B' ? bindGroups.BA : bindGroups.CA;
    dispatch(pass, 'finalize_state', finalize, seed.count);
    // Rebuild from finalized A, then measure actual residual overlap. Attempted
    // penetration and post-projection residual are intentionally separate.
    buildGrid(pass, bindGroups.AB);
    dispatch(pass, 'audit_contacts', bindGroups.AB, seed.count);
  };
  const stepMany = (encoder, count, timeStep = policy.timeStep) => {
    if (!encoder || typeof encoder.beginComputePass !== 'function') {
      throw new TypeError('Granular GPU steps require a GPU command encoder');
    }
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new RangeError('Granular GPU step count must be a nonnegative safe integer');
    }
    if (timeStep !== policy.timeStep) {
      throw new RangeError('Granular GPU steps use the fixed policy time step');
    }
    if (count === 0) return;
    updateParams();
    const pass = encoder.beginComputePass({
      label: `VKF Granular particle world ${count}-step batch`,
    });
    for (let index = 0; index < count; index += 1) encodeFixedStep(pass);
    pass.end();
    frameIndex += count;
  };
  const step = (encoder, timeStep = policy.timeStep) => stepMany(encoder, 1, timeStep);
  let encodedWheelAngle=null;
  const sweepWheel=(encoder,elapsed)=>{
    if(encodedWheelAngle===null){encodedWheelAngle=wheelAngle;return;}
    const delta=wheelAngle-encodedWheelAngle;if(Math.abs(delta)<1e-10)return;
    updateParams();const base=paramsBytes.slice(0),motion=paramsBytes.slice(0);
    new Float32Array(motion).set([encodedWheelAngle,delta,delta/elapsed,elapsed],24);
    const staging=createBuffer(device,'VKF immutable grain sweep parameters',base.byteLength*2,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(staging,0,motion);device.queue.writeBuffer(staging,base.byteLength,base);
    encoder.copyBufferToBuffer(staging,0,paramsBuffer,0,base.byteLength);
    const pass=encoder.beginComputePass({label:'Swept rigid baffles, grains'});dispatch(pass,'sweep_wheel',bindGroups.AB,seed.count);pass.end();
    // A kinematic boundary sweep must solve the grain contact network before
    // its compressed state is used for advection (also while Play is paused).
    const contacts=encoder.beginComputePass({label:'Swept grain exclusion network'});
    let current='B';
    for(let iteration=0;iteration<policy.contactIterations*2;iteration++){
      const projection=current==='B'?bindGroups.BC:bindGroups.CB;
      buildGrid(contacts,projection);dispatch(contacts,'project_contacts',projection,seed.count);current=current==='B'?'C':'B';
    }
    contacts.end();
    encoder.copyBufferToBuffer(current==='B'?workBufferB:workBufferC,0,particleBuffer,0,seed.bytes.byteLength);
    encoder.copyBufferToBuffer(staging,base.byteLength,paramsBuffer,0,base.byteLength);
    queueMicrotask(()=>device.queue.onSubmittedWorkDone().then(()=>staging.destroy()));encodedWheelAngle=wheelAngle;
  };
  const telemetryByteOffset = gridCellCount * 4;
  const telemetryByteLength = telemetryAtomicCount * 4;
  const telemetryZero = new Uint32Array(telemetryAtomicCount);
  const resetTelemetry = () => {
    device.queue.writeBuffer(cellCountsBuffer, telemetryByteOffset, telemetryZero);
  };
  const readTelemetry = async () => {
    const readback = device.createBuffer({
      label: 'VKF Granular telemetry readback',
      size: telemetryByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder({ label: 'VKF Granular telemetry copy' });
      encoder.copyBufferToBuffer(cellCountsBuffer, telemetryByteOffset,
        readback, 0, telemetryByteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange().slice(0));
      const asFloat = (field) => new Float32Array(new Uint32Array([
        words[GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY.fields[field]],
      ]).buffer)[0];
      const peakSpeedSquared = asFloat('peakSpeedSquaredBits');
      const maximumPenetration = asFloat('maximumPenetrationBits');
      const maximumResidualPenetration = asFloat('maximumResidualPenetrationBits');
      const nonFiniteFlag = words[
        GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY.fields.nonFiniteFlag];
      const peakCellOccupancy = words[
        GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakCellOccupancy];
      const gridOverflowCount = words[
        GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY.fields.gridOverflowCount];
      return Object.freeze({
        kind: 'granular-particle-world-gpu-telemetry:v1',
        frameIndex,
        peakSpeedSquared,
        peakSpeed: Math.sqrt(peakSpeedSquared),
        nonFinite: nonFiniteFlag !== 0,
        nonFiniteFlag,
        peakCellOccupancy,
        gridOverflow: gridOverflowCount !== 0,
        gridOverflowCount,
        maximumPenetration,
        maximumPenetrationInRadii: maximumPenetration / policy.grainRadius,
        maximumResidualPenetration,
        maximumResidualPenetrationInRadii:
          maximumResidualPenetration / policy.grainRadius,
        residualWithinTolerance: maximumResidualPenetration <= policy.contactSlop,
        raw: Object.freeze(Array.from(words)),
      });
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      readback.destroy();
    }
  };
  const reset = () => {
    encodedWheelAngle=null;
    device.queue.writeBuffer(particleBuffer, 0, resetBytes);
    resetTelemetry();
    frameIndex = 0;
    updateParams();
  };
  const setWheel = ({ angle = wheelAngle,
    angularVelocity = wheelAngularVelocity } = {}) => {
    if (!Number.isFinite(angle) || !Number.isFinite(angularVelocity)) {
      throw new TypeError('Granular wheel state requires finite angle and angularVelocity');
    }
    wheelAngle = angle;
    wheelAngularVelocity = angularVelocity;
    updateParams();
  };
  const destroy = () => {
    for (const buffer of [particleBuffer, workBufferB, workBufferC,
      cellCountsBuffer, cellItemsBuffer, paramsBuffer]) buffer.destroy();
  };
  reset();
  return Object.freeze({
    kind: 'granular-particle-world-gpu-runtime:v1',
    device,
    policy,
    seed,
    abi: GRANULAR_PARTICLE_WORLD_GPU_ABI,
    particleBuffer,
    particleCount: seed.count,
    primaryCount: seed.count,
    paramsBuffer,
    gridCellCount,
    contactLaw: seed.contactLaw,
    boundaryBox: seed.boundaryBox,
    pipelines,
    step,
    stepMany,
    sweepWheel,
    telemetry: Object.freeze({
      buffer: cellCountsBuffer,
      byteOffset: telemetryByteOffset,
      byteLength: telemetryByteLength,
      ...GRANULAR_PARTICLE_WORLD_GPU_TELEMETRY,
    }),
    readTelemetry,
    resetTelemetry,
    reset,
    setWheel,
    wheel: Object.freeze({ center: Object.freeze(options.geometry?.center ?? [0, 0.32]),
      radius: options.geometry?.radius ?? 0.50, barHalfWidth: options.geometry?.half_width ?? 0.012 }),
    destroy,
    get frameIndex() { return frameIndex; },
  });
}
