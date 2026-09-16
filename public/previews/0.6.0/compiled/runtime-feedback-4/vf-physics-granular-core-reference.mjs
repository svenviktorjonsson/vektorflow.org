import { createPhysicsStateBufferReference, createTransportSpatialGridReference } from
  './vf-physics-transport-core-reference.mjs';
import { verifyGranularRegimeReference }
  from './vf-physics-regime-verifier-reference.mjs';

export function characterizeGranularRegimeReference(options) {
  return verifyGranularRegimeReference(options);
}

export function granularMix32Reference(value) {
  let word = value >>> 0;
  word ^= word >>> 16; word = Math.imul(word, 0x7feb352d) >>> 0;
  word ^= word >>> 15; word = Math.imul(word, 0x846ca68b) >>> 0;
  word ^= word >>> 16;
  return word >>> 0;
}

export function granularUnitReference(seed, lane) {
  return granularMix32Reference(seed ^ Math.imul(lane + 1, 0x9e3779b1)) / 0x100000000;
}

export function granularStreamUnitReference(seed, grain, stream) {
  return granularMix32Reference((seed >>> 0) ^ Math.imul(grain + 1, 0x9e3779b1)
    ^ Math.imul(stream + 1, 0x85ebca77)) / 0x100000000;
}

export function granularStreamSignedReference(seed, grain, stream) {
  return granularStreamUnitReference(seed, grain, stream) * 2 - 1;
}

export function createGranularMineralPaletteReference({
  mineralSeed = 0x68bc21eb, toneSeed = 0x9e3779b9,
} = {}) {
  if (!Number.isSafeInteger(mineralSeed) || mineralSeed < 0 || mineralSeed > 0xffffffff
      || !Number.isSafeInteger(toneSeed) || toneSeed < 0 || toneSeed > 0xffffffff) {
    throw new RangeError('granular mineral palette seeds must be u32');
  }
  return Object.freeze({ kind: 'granular-mineral-palette-reference:v1',
    mineralSeed: mineralSeed >>> 0, toneSeed: toneSeed >>> 0,
    quartzThreshold: 0.11, ironThreshold: 0.94,
    colors: Object.freeze({ quartz: Object.freeze([0.88, 0.79, 0.61]),
      warmSand: Object.freeze([0.76, 0.57, 0.3]),
      ironDark: Object.freeze([0.34, 0.24, 0.14]) }),
    opticalOnly: true });
}

const requireMineralPalette = (palette) => {
  if (!palette || palette.kind !== 'granular-mineral-palette-reference:v1') {
    throw new TypeError('granular mineral palette required');
  }
};

const granularLow16Unit = (value) => (granularMix32Reference(value) & 0xffff) / 0xffff;

export function granularMineralSampleReference(palette, id) {
  requireMineralPalette(palette);
  if (!Number.isSafeInteger(id) || id < 0 || id > 0xffffffff) {
    throw new RangeError('granular mineral id must be a u32');
  }
  const mineral = granularLow16Unit((id >>> 0) ^ palette.mineralSeed);
  const species = mineral < palette.quartzThreshold ? 'quartz'
    : mineral > palette.ironThreshold ? 'ironDark' : 'warmSand';
  const tone = 0.8 + 0.2 * granularLow16Unit((id >>> 0) ^ palette.toneSeed);
  return Object.freeze({ species, mineral, tone,
    color: Object.freeze(palette.colors[species].map((channel) => channel * tone)) });
}

export function createGranularMineralPaletteWgslReference(palette) {
  requireMineralPalette(palette);
  const vector = (values) => `vec3f(${values.join(', ')})`;
  return `
fn granularMineralHash(value: u32) -> f32 { var x = value; x ^= x >> 16u;
  x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return f32(x & 65535u) / 65535.0; }
fn granularMineralColor(id: u32) -> vec3f {
  let quartz = ${vector(palette.colors.quartz)};
  let ironDark = ${vector(palette.colors.ironDark)};
  let warmSand = ${vector(palette.colors.warmSand)};
  let mineral = granularMineralHash(id ^ 0x${palette.mineralSeed.toString(16)}u);
  var grainColor = warmSand;
  if (mineral < ${palette.quartzThreshold}) { grainColor = quartz; }
  else if (mineral > ${palette.ironThreshold}) { grainColor = ironDark; }
  let tone = 0.8 + 0.2 * granularMineralHash(id ^ 0x${palette.toneSeed.toString(16)}u);
  return grainColor * tone;
}`;
}

const finiteUnit = (name, value, { allowZero = true } = {}) => {
  if (!Number.isFinite(value) || value < 0 || value > 1 || (!allowZero && value === 0)) {
    throw new RangeError(`${name} must be ${allowZero ? 'from zero through one' : 'positive through one'}`);
  }
  return value;
};

export function createCohesionlessGranularContactLawReference({ friction,
  rollingResistance = 0, restitution, stiffness = 0, damping = 0 }) {
  finiteUnit('granular friction', friction);
  finiteUnit('granular rolling resistance', rollingResistance);
  finiteUnit('granular restitution', restitution);
  if (!Number.isFinite(stiffness) || stiffness < 0) throw new RangeError('granular stiffness must be nonnegative');
  if (!Number.isFinite(damping) || damping < 0) throw new RangeError('granular damping must be nonnegative');
  return Object.freeze({ cohesion: 0, friction, rollingResistance, restitution, stiffness, damping });
}

export function createGranularStateReference({ kind, seed, count, precision,
  includeIds = false, gravity, timeStep, particleMass }) {
  return createPhysicsStateBufferReference({ kind, seed, count, precision, includeIds,
    gravity, timeStep, conserved: { name: 'mass', perElement: particleMass },
    adapter: 'cohesionless-dem' });
}

export function enumerateGranularSphereContactPairsReference({
  positions, count, maximumDiameter, radiusAt,
}) {
  if (!(positions instanceof Float32Array || positions instanceof Float64Array)
      || !Number.isInteger(count) || count < 1 || positions.length < count * 3) {
    throw new TypeError('granular contact positions and positive count required');
  }
  if (!Number.isFinite(maximumDiameter) || maximumDiameter <= 0) {
    throw new RangeError('granular maximum diameter must be positive');
  }
  if (typeof radiusAt !== 'function') {
    throw new TypeError('granular contact radius accessor required');
  }
  const radii = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const radius = radiusAt(index);
    if (!Number.isFinite(radius) || radius <= 0 || radius * 2 > maximumDiameter) {
      throw new RangeError('granular contact radius must be positive and fit maximum diameter');
    }
    radii[index] = radius;
  }
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < count; index += 1) for (let axis = 0; axis < 3; axis += 1) {
    const value = positions[index * 3 + axis];
    if (!Number.isFinite(value)) throw new TypeError('granular contact positions must be finite');
    minimum[axis] = Math.min(minimum[axis], value);
    maximum[axis] = Math.max(maximum[axis], value);
  }
  const dimensions = minimum.map((value, axis) =>
    Math.max(1, Math.floor((maximum[axis] - value) / maximumDiameter) + 1));
  const grid = createTransportSpatialGridReference({ size: dimensions,
    cellSize: maximumDiameter, origin: minimum });
  const buckets = Array.from({ length: dimensions[0] * dimensions[1] * dimensions[2] },
    () => []);
  const pairs = [];
  for (let second = 0; second < count; second += 1) {
    const secondOffset = second * 3;
    const position = [...positions.subarray(secondOffset, secondOffset + 3)];
    const secondRadius = radii[second];
    for (const key of grid.neighborKeys(position)) for (const first of buckets[key]) {
      const firstOffset = first * 3;
      const firstRadius = radii[first];
      const dx = positions[secondOffset] - positions[firstOffset];
      const dy = positions[secondOffset + 1] - positions[firstOffset + 1];
      const dz = positions[secondOffset + 2] - positions[firstOffset + 2];
      const distance = Math.hypot(dx, dy, dz);
      const pairRadius = firstRadius + secondRadius;
      if (distance <= pairRadius) {
        pairs.push(Object.freeze({ first, second, distance, pairRadius }));
      }
    }
    buckets[grid.key(position)].push(second);
  }
  return Object.freeze(pairs);
}

const coincidentGranularNormal = (seed, first, second) => {
  const identity = Math.imul(first + 1, 0x9e3779b1) ^ Math.imul(second + 1, 0x85ebca77);
  const direction = [0, 1, 2].map((axis) =>
    granularStreamSignedReference(seed ^ identity, second, axis));
  const length = Math.hypot(...direction);
  return length > 1e-12 ? direction.map((value) => value / length) : [1, 0, 0];
};

export function projectGranularSphereContactsReference({
  positions, count, maximumDiameter, radiusAt, seed = 0, iterations = 4,
}) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError('granular contact projection seed must be u32');
  }
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 32) {
    throw new RangeError('granular contact projection iterations must be from 1 through 32');
  }
  let projectionCount = 0;
  let maximumPenetrationBefore = 0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const pairs = enumerateGranularSphereContactPairsReference({
      positions, count, maximumDiameter, radiusAt,
    });
    for (const pair of pairs) {
      const penetration = pair.pairRadius - pair.distance;
      if (penetration <= 0) continue;
      if (iteration === 0) maximumPenetrationBefore = Math.max(
        maximumPenetrationBefore, penetration);
      const firstOffset = pair.first * 3;
      const secondOffset = pair.second * 3;
      let normal;
      if (pair.distance > 1e-12) {
        normal = [
          (positions[firstOffset] - positions[secondOffset]) / pair.distance,
          (positions[firstOffset + 1] - positions[secondOffset + 1]) / pair.distance,
          (positions[firstOffset + 2] - positions[secondOffset + 2]) / pair.distance,
        ];
      } else {
        normal = coincidentGranularNormal(seed, pair.first, pair.second);
      }
      const correction = penetration * 0.5;
      for (let axis = 0; axis < 3; axis += 1) {
        positions[firstOffset + axis] += normal[axis] * correction;
        positions[secondOffset + axis] -= normal[axis] * correction;
      }
      projectionCount += 1;
    }
  }
  const remaining = enumerateGranularSphereContactPairsReference({
    positions, count, maximumDiameter, radiusAt,
  });
  const maximumPenetrationAfter = remaining.reduce((maximum, pair) =>
    Math.max(maximum, pair.pairRadius - pair.distance), 0);
  return Object.freeze({ iterations, projectionCount,
    maximumPenetrationBefore, maximumPenetrationAfter });
}

export function fillGranularSourceVolumeReference({
  positions, count, radius, minimum, maximum, seed = 0,
}) {
  if (!(positions instanceof Float32Array || positions instanceof Float64Array)
      || !Number.isInteger(count) || count < 1 || positions.length < count * 3) {
    throw new TypeError('granular source positions and positive count required');
  }
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('granular source radius must be positive');
  }
  if (!Array.isArray(minimum) || minimum.length !== 3 || !Array.isArray(maximum)
      || maximum.length !== 3 || minimum.some((value) => !Number.isFinite(value))
      || maximum.some((value) => !Number.isFinite(value))) {
    throw new TypeError('granular source bounds must contain three finite values');
  }
  if (minimum.some((value, axis) => maximum[axis] - value < radius * 2)) {
    throw new RangeError('granular source volume cannot fit grain radius');
  }
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError('granular source seed must be u32');
  }
  const lower = minimum.map((value) => value + radius);
  const span = lower.map((value, axis) => maximum[axis] - radius - value);
  const diameter = radius * 2;
  const dimensions = minimum.map((value, axis) =>
    Math.max(1, Math.ceil((maximum[axis] - value) / diameter)));
  const grid = createTransportSpatialGridReference({ size: dimensions,
    cellSize: diameter, origin: minimum });
  const buckets = Array.from({ length: dimensions[0] * dimensions[1] * dimensions[2] },
    () => []);
  const diameterSquared = (radius * 2) ** 2;
  const attemptLimit = 64;
  let rejectedCandidateCount = 0;
  let distanceEvaluationCount = 0;
  let bruteForceDistanceEvaluationCount = 0;
  for (let grain = 0; grain < count; grain += 1) {
    let placed = false;
    for (let attempt = 0; attempt < attemptLimit; attempt += 1) {
      const attemptSeed = (seed ^ Math.imul(attempt + 1, 0x85ebca77)) >>> 0;
      const normalized = [
        (granularStreamUnitReference(attemptSeed, grain, 0)
          + granularStreamUnitReference(attemptSeed, grain, 1)) * 0.5,
        (granularStreamUnitReference(attemptSeed, grain, 2)
          + granularStreamUnitReference(attemptSeed, grain, 3)) * 0.5,
        granularStreamUnitReference(attemptSeed, grain, 4),
      ];
      const candidate = normalized.map((value, axis) => lower[axis] + span[axis] * value);
      bruteForceDistanceEvaluationCount += grain;
      let separated = true;
      for (const key of grid.neighborKeys(candidate)) {
        for (const previous of buckets[key]) {
          distanceEvaluationCount += 1;
          const previousOffset = previous * 3;
          const dx = candidate[0] - positions[previousOffset];
          const dy = candidate[1] - positions[previousOffset + 1];
          const dz = candidate[2] - positions[previousOffset + 2];
          if (dx * dx + dy * dy + dz * dz <= diameterSquared) {
            separated = false;
            break;
          }
        }
        if (!separated) break;
      }
      if (!separated) {
        rejectedCandidateCount += 1;
        continue;
      }
      const offset = grain * 3;
      for (let axis = 0; axis < 3; axis += 1) positions[offset + axis] = candidate[axis];
      buckets[grid.key(candidate)].push(grain);
      placed = true;
      break;
    }
    if (!placed) {
      throw new RangeError('granular source volume exhausted deterministic placement attempts');
    }
  }
  return Object.freeze({ placedCount: count, rejectedCandidateCount, attemptLimit,
    distanceEvaluationCount, bruteForceDistanceEvaluationCount });
}

export function resolveGranularFrictionRollingReference({ tangent, tangentSpeed,
  normalImpulse, friction, normal, radius, otherRadius, rollingResistance }) {
  const tangentImpulse = Math.min(tangentSpeed * 0.5, friction * normalImpulse);
  const scale = tangentSpeed > 1e-12 ? tangentImpulse / tangentSpeed : 0;
  const [tx, ty, tz] = tangent; const [nx, ny, nz] = normal;
  const torque = Object.freeze([ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx]);
  return Object.freeze({ tangentImpulse, scale, torque,
    spin: scale * 0.36 / Math.max(radius, otherRadius),
    rolling: granularRollingRetentionReference(rollingResistance, 0.5) });
}

export function granularRollingRetentionReference(rollingResistance, contactFraction = 1) {
  finiteUnit('granular rolling resistance', rollingResistance);
  finiteUnit('granular rolling contact fraction', contactFraction);
  return 1 - rollingResistance * contactFraction;
}

export function createGranularGpuSortReference({ kind, count, workgroupSize, bins,
  radixShifts, grid, storageBytes, shaders }) {
  if (count % workgroupSize !== 0) throw new RangeError('granular sort count must fill workgroups');
  return Object.freeze({ kind, count, workgroupSize, workgroups: count / workgroupSize,
    radix: Object.freeze({ bits: Math.log2(bins), bins, shifts: Object.freeze([...radixShifts]) }),
    grid, storageBytes, cellKeyShader: shaders.cellKey, histogramShader: shaders.histogram,
    scanShader: shaders.scan, scatterShader: shaders.scatter, compactShader: shaders.compact,
    allShaders: [shaders.cellKey, shaders.histogram, shaders.scan,
      shaders.scatter, shaders.compact].join('\n') });
}

export function createGranularGpuContactReference({ kind, count, workgroupSize,
  law, radius, neighborOrder, shader }) {
  return Object.freeze({ kind, count, workgroupSize, workgroups: count / workgroupSize,
    cohesion: law.cohesion, restitution: law.restitution,
    contact: Object.freeze({ radius, stiffness: law.stiffness, damping: law.damping,
      tangentialFriction: law.friction, neighborOrder }), shader });
}

export function createGranularPointPacketReference(state, { id, objectId, baseColor }) {
  const vertices = new Float32Array(state.count * 10);
  for (let grain = 0; grain < state.count; grain += 1) {
    const source = grain * 3; const target = grain * 10;
    vertices[target] = state.positions[source];
    vertices[target + 1] = state.positions[source + 1];
    vertices[target + 2] = state.positions[source + 2];
    vertices[target + 5] = 1;
    const tone = 0.88 + 0.2 * (granularMix32Reference(state.seed ^ grain) / 0x100000000);
    vertices[target + 6] = Math.min(1, baseColor[0] * tone);
    vertices[target + 7] = Math.min(1, baseColor[1] * tone);
    vertices[target + 8] = Math.min(1, baseColor[2] * tone);
    vertices[target + 9] = 1;
  }
  return Object.freeze({
    type: 'field_mesh', id, object_id: objectId, mode3d: true,
    topology: 'point-list', primitive_kind: 'independent-sand-material-points',
    vertex_count: state.count, grain_count: state.count, instance_count: 1,
    vertex_radius: 0, vertices, static_vertices: false, transparent: false,
    depth_write: true, receives_lighting: true, casts_shadow: false, receives_shadow: true,
    specular_strength: 0.015,
    renderContract: Object.freeze({ projectedGrainPixels: 1, geometryAmplification: 1,
      oneVertexPerGrain: true, depthOccludesInterior: true, visibilityMergesGrains: false }),
    state_link: Object.freeze({ particleCount: state.count, steps: state.steps }),
    vectorBytes: vertices.byteLength,
  });
}

function granularSphereTemplate(latitudeSegments = 8, longitudeSegments = 12) {
  const row = longitudeSegments + 1;
  const vertices = new Float32Array((latitudeSegments + 1) * row * 10); let offset = 0;
  for (let latitude = 0; latitude <= latitudeSegments; latitude += 1) {
    const phi = latitude / latitudeSegments * Math.PI;
    for (let longitude = 0; longitude <= longitudeSegments; longitude += 1) {
      const theta = longitude / longitudeSegments * Math.PI * 2;
      const nx = Math.sin(phi) * Math.cos(theta); const ny = Math.sin(phi) * Math.sin(theta);
      const nz = Math.cos(phi);
      vertices.set([nx, ny, nz, nx, ny, nz, 1, 1, 1, 1], offset); offset += 10;
    }
  }
  const indices = new Uint32Array(latitudeSegments * longitudeSegments * 6); offset = 0;
  for (let latitude = 0; latitude < latitudeSegments; latitude += 1) {
    for (let longitude = 0; longitude < longitudeSegments; longitude += 1) {
      const a = latitude * row + longitude; const b = a + 1; const c = a + row; const d = c + 1;
      indices.set([a, c, b, b, c, d], offset); offset += 6;
    }
  }
  return { vertices, indices };
}

export function syncGranularEllipsoidPacketReference(world, packet) {
  const instances = packet.instances;
  for (let index = 0; index < world.count; index += 1) {
    const source = index * 3; const target = index * 8;
    instances[target] = world.state.positions[source];
    instances[target + 1] = world.state.positions[source + 1];
    instances[target + 2] = world.state.positions[source + 2];
    instances[target + 3] = world.state.aggregated[index] ? 0
      : world.radius * world.state.sizeScales[index] * (world.state.aspects[source]
        + world.state.aspects[source + 1] + world.state.aspects[source + 2]) / 3;
  }
  return packet;
}

export function createGranularEllipsoidPacketReference(world) {
  const template = granularSphereTemplate(); const instances = new Float32Array(world.count * 8);
  for (let index = 0; index < world.count; index += 1) {
    const target = index * 8; const warm = granularUnitReference(world.seed, index * 13);
    instances[target + 4] = 0.58 + warm * 0.16;
    instances[target + 5] = 0.42 + warm * 0.13;
    instances[target + 6] = 0.20 + warm * 0.08; instances[target + 7] = 1;
  }
  const packet = { type: 'field_mesh', id: 'sand:active-grains', object_id: 1, mode3d: true,
    topology: 'triangle-list', instance_kind: 'sphere-list', instance_count: world.count,
    transparent: false, depth_write: true, receives_lighting: true, casts_shadow: true,
    receives_shadow: false, specular_strength: 0.09,
    vertices: template.vertices, indices: template.indices, instances };
  world.render.instances = instances;
  return syncGranularEllipsoidPacketReference(world, packet);
}
