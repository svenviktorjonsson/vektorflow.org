import { appendPhysicsStateBufferReference, createPhysicsStateBufferReference,
  hashPhysicsStateBufferReference, selectPhysicsStateBufferReference }
  from './vf-physics-transport-core-reference.mjs';

const EPSILON = 1e-12;
const DEFAULT_DIVERGENCE_ITERATIONS = 24;
const DEFAULT_DENSITY_ITERATIONS = 48;
const DEFAULT_DIVERGENCE_TOLERANCE = 1e-3;
const DEFAULT_DENSITY_TOLERANCE = 1e-4;
const DEFAULT_MAXIMUM_COURANT_NUMBER = 0.1;
const DEFAULT_MAXIMUM_CFL_SUBSTEPS = 256;
const NEIGHBOR_CELL_OFFSETS = Object.freeze(Array.from({ length: 27 }, (_, index) => {
  const dx = index % 3 - 1;
  const dy = Math.floor(index / 3) % 3 - 1;
  const dz = Math.floor(index / 9) - 1;
  return Object.freeze([dx, dy, dz]);
}));

const requirePositiveFinite = (name, value) => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be positive`);
  }
  return value;
};

const requireNonnegativeFinite = (name, value) => {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be nonnegative`);
  }
  return value;
};

const requireAlignedThreeVector = (name, values, count = null) => {
  if ((!Array.isArray(values) && !ArrayBuffer.isView(values))
      || values.length % 3 !== 0 || !Array.from(values).every(Number.isFinite)) {
    throw new TypeError(`${name} must be a finite vector aligned to three components`);
  }
  const inferredCount = values.length / 3;
  if (count !== null && inferredCount !== count) {
    throw new RangeError(`${name} must contain ${count} aligned elements`);
  }
  return inferredCount;
};

const requireDimension = (dimension) => {
  if (dimension !== 2 && dimension !== 3) {
    throw new RangeError('local particle Liquid dimension must be two or three');
  }
  return dimension;
};

const axesForDimension = (dimension) => dimension === 2
  ? Object.freeze([0, 2]) : Object.freeze([0, 1, 2]);

const requireAlignedDimensionVector = (name, values, dimension, count = null) => {
  if ((!Array.isArray(values) && !ArrayBuffer.isView(values))
      || values.length % dimension !== 0
      || !Array.from(values).every(Number.isFinite)) {
    throw new TypeError(`${name} must be a finite vector aligned to ${dimension} components`);
  }
  const inferredCount = values.length / dimension;
  if (count !== null && inferredCount !== count) {
    throw new RangeError(`${name} must contain ${count} aligned elements`);
  }
  return inferredCount;
};

function expandDimensionVector(values, dimension) {
  if (dimension === 3) return Float64Array.from(values);
  const expanded = new Float64Array(values.length / 2 * 3);
  for (let element = 0; element < values.length / 2; element += 1) {
    expanded[element * 3] = values[element * 2];
    expanded[element * 3 + 2] = values[element * 2 + 1];
  }
  return expanded;
}

function contractDimensionVector(values, dimension) {
  if (dimension === 3) return Float64Array.from(values);
  const contracted = new Float64Array(values.length / 3 * 2);
  for (let element = 0; element < values.length / 3; element += 1) {
    contracted[element * 2] = values[element * 3];
    contracted[element * 2 + 1] = values[element * 3 + 2];
  }
  return contracted;
}

const requireParticleState = (state) => {
  if (!state || !Number.isSafeInteger(state.count) || state.count <= 0
      || !ArrayBuffer.isView(state.positions) || !ArrayBuffer.isView(state.velocities)
      || state.positions.length !== state.count * 3
      || state.velocities.length !== state.count * 3) {
    throw new TypeError('dense Physics particle state required');
  }
  return state;
};

const cellCoordinate = (value, cellSize) => Math.floor(value / cellSize);
const cellKey = (x, y, z) => `${x},${y},${z}`;

function buildDeterministicBuckets(positions, count, cellSize) {
  const buckets = new Map();
  for (let particle = 0; particle < count; particle += 1) {
    const offset = particle * 3;
    const key = cellKey(cellCoordinate(positions[offset], cellSize),
      cellCoordinate(positions[offset + 1], cellSize),
      cellCoordinate(positions[offset + 2], cellSize));
    const bucket = buckets.get(key);
    if (bucket) bucket.push(particle);
    else buckets.set(key, [particle]);
  }
  return buckets;
}

function queryDeterministicBucketNeighbors(positions, particle, supportRadius,
  supportRadiusSquared, buckets) {
  const offset = particle * 3;
  const px = positions[offset]; const py = positions[offset + 1];
  const pz = positions[offset + 2];
  const cx = cellCoordinate(px, supportRadius);
  const cy = cellCoordinate(py, supportRadius);
  const cz = cellCoordinate(pz, supportRadius);
  const neighbors = [];
  for (const [dx, dy, dz] of NEIGHBOR_CELL_OFFSETS) {
    const bucket = buckets.get(cellKey(cx + dx, cy + dy, cz + dz));
    if (!bucket) continue;
    for (const candidate of bucket) {
      const candidateOffset = candidate * 3;
      const rx = px - positions[candidateOffset];
      const ry = py - positions[candidateOffset + 1];
      const rz = pz - positions[candidateOffset + 2];
      if (rx * rx + ry * ry + rz * rz
          <= supportRadiusSquared * (1 + 16 * Number.EPSILON)) {
        neighbors.push(candidate);
      }
    }
  }
  // Buckets are traversed by geometric cell order. The solver contract is source order,
  // so sort once here instead of depending on Map insertion or particle motion.
  neighbors.sort((left, right) => left - right);
  return neighbors;
}

export function createDeterministicLiquidNeighborhoodReference(state,
  { supportRadius, dimension = state?.dimension ?? 3 } = {}) {
  requireParticleState(state);
  requireDimension(dimension);
  requirePositiveFinite('liquid support radius', supportRadius);
  const buckets = buildDeterministicBuckets(state.positions, state.count, supportRadius);
  const rows = new Array(state.count);
  const offsets = new Uint32Array(state.count + 1);
  let entryCount = 0;
  const supportRadiusSquared = supportRadius ** 2;
  for (let particle = 0; particle < state.count; particle += 1) {
    rows[particle] = queryDeterministicBucketNeighbors(state.positions, particle,
      supportRadius, supportRadiusSquared, buckets);
    entryCount += rows[particle].length;
    offsets[particle + 1] = entryCount;
  }
  const indices = new Uint32Array(entryCount);
  let cursor = 0;
  for (const row of rows) for (const neighbor of row) indices[cursor++] = neighbor;
  return Object.freeze({ kind: 'deterministic-liquid-neighborhood-reference:v1',
    supportRadius, dimension, count: state.count, bucketCount: buckets.size,
    offsets, indices, sourceOrder: true });
}

// Wendland C2 parameterized by full compact support radius H. Its normalization is
// 7/(pi*H^2) in 2-D and 21/(2*pi*H^3) in 3-D.
export function liquidWendlandC2KernelReference(distance, supportRadius, dimension = 3) {
  requireNonnegativeFinite('liquid kernel distance', distance);
  requirePositiveFinite('liquid support radius', supportRadius);
  requireDimension(dimension);
  if (distance >= supportRadius) return 0;
  const q = distance / supportRadius;
  const oneMinusQ = 1 - q;
  const normalization = dimension === 2 ? 7 / (Math.PI * supportRadius ** 2)
    : 21 / (2 * Math.PI * supportRadius ** 3);
  return normalization
    * oneMinusQ ** 4 * (1 + 4 * q);
}

function liquidWendlandC2GradientUnchecked(rx, ry, rz, supportRadius, dimension) {
  const distanceSquared = rx * rx + ry * ry + rz * rz;
  if (distanceSquared <= EPSILON ** 2 || distanceSquared >= supportRadius ** 2) {
    return [0, 0, 0];
  }
  const distance = Math.sqrt(distanceSquared);
  const q = distance / supportRadius;
  const derivative = (dimension === 2 ? -140 / (Math.PI * supportRadius ** 3)
    : -210 / (Math.PI * supportRadius ** 4)) * q * (1 - q) ** 3;
  const scale = derivative / distance;
  return [rx * scale, ry * scale, rz * scale];
}

export function liquidWendlandC2GradientReference(displacement, supportRadius, dimension = 3) {
  requireAlignedThreeVector('liquid kernel displacement', displacement, 1);
  requirePositiveFinite('liquid support radius', supportRadius);
  requireDimension(dimension);
  return Object.freeze(liquidWendlandC2GradientUnchecked(
    displacement[0], displacement[1], displacement[2], supportRadius, dimension));
}

export function calibrateUniformLocalLiquidParticleMassReference({ dimension = 3,
  spacing, supportRadius, restDensity } = {}) {
  requireDimension(dimension);
  requirePositiveFinite('liquid lattice spacing', spacing);
  requirePositiveFinite('liquid support radius', supportRadius);
  requirePositiveFinite('liquid rest density', restDensity);
  const extent = Math.ceil(supportRadius / spacing);
  let kernelSum = 0;
  for (let z = dimension === 2 ? 0 : -extent; z <= (dimension === 2 ? 0 : extent); z += 1) {
    for (let y = -extent; y <= extent; y += 1) for (let x = -extent; x <= extent; x += 1) {
      const distance = spacing * Math.hypot(x, y, z);
      kernelSum += liquidWendlandC2KernelReference(distance, supportRadius, dimension);
    }
  }
  const nominalParticleMeasure = spacing ** dimension;
  const kernelPartition = nominalParticleMeasure * kernelSum;
  const particleMeasure = 1 / kernelSum;
  return Object.freeze({ kind: 'uniform-local-liquid-particle-mass-calibration-reference:v1',
    dimension, spacing, supportRadius, restDensity, kernelPartition,
    nominalParticleMeasure, particleMeasure,
    particleMass: restDensity * particleMeasure });
}

function queryPointNeighbors(positions, count, point, supportRadius, buckets) {
  const cx = cellCoordinate(point[0], supportRadius);
  const cy = cellCoordinate(point[1], supportRadius);
  const cz = cellCoordinate(point[2], supportRadius);
  const supportRadiusSquared = supportRadius ** 2;
  const neighbors = [];
  for (const [dx, dy, dz] of NEIGHBOR_CELL_OFFSETS) {
    const bucket = buckets.get(cellKey(cx + dx, cy + dy, cz + dz));
    if (!bucket) continue;
    for (const candidate of bucket) {
      if (candidate >= count) continue;
      const offset = candidate * 3;
      const rx = point[0] - positions[offset];
      const ry = point[1] - positions[offset + 1];
      const rz = point[2] - positions[offset + 2];
      if (rx * rx + ry * ry + rz * rz
          <= supportRadiusSquared * (1 + 16 * Number.EPSILON)) {
        neighbors.push(candidate);
      }
    }
  }
  neighbors.sort((left, right) => left - right);
  return neighbors;
}

export function createAkinciLiquidSolidBoundaryReference({ dimension = 3, positions,
  velocities = null, volumes = null, supportRadius, closedPolygon = null,
  closedPolygons = null } = {}) {
  requireDimension(dimension);
  const count = requireAlignedDimensionVector('solid boundary positions', positions, dimension);
  if (count === 0) throw new RangeError('solid boundary positions must not be empty');
  requirePositiveFinite('solid boundary support radius', supportRadius);
  const boundaryPositions = expandDimensionVector(positions, dimension);
  const boundaryVelocities = velocities === null ? new Float64Array(count * 3)
    : expandDimensionVector(velocities, dimension);
  if (velocities !== null) {
    requireAlignedDimensionVector('solid boundary velocities', velocities, dimension, count);
  }
  let boundaryVolumes;
  if (volumes !== null) {
    if ((!Array.isArray(volumes) && !ArrayBuffer.isView(volumes))
        || volumes.length !== count
        || !Array.from(volumes).every((value) => Number.isFinite(value) && value > 0)) {
      throw new TypeError(`solid boundary volumes must contain ${count} positive values`);
    }
    boundaryVolumes = Float64Array.from(volumes);
  } else {
    const buckets = buildDeterministicBuckets(boundaryPositions, count, supportRadius);
    const supportRadiusSquared = supportRadius ** 2;
    boundaryVolumes = new Float64Array(count);
    for (let boundary = 0; boundary < count; boundary += 1) {
      const neighbors = queryDeterministicBucketNeighbors(boundaryPositions, boundary,
        supportRadius, supportRadiusSquared, buckets);
      const offset = boundary * 3;
      let kernelSum = 0;
      for (const neighbor of neighbors) {
        const neighborOffset = neighbor * 3;
        kernelSum += liquidWendlandC2KernelReference(Math.hypot(
          boundaryPositions[offset] - boundaryPositions[neighborOffset],
          boundaryPositions[offset + 1] - boundaryPositions[neighborOffset + 1],
          boundaryPositions[offset + 2] - boundaryPositions[neighborOffset + 2]),
        supportRadius, dimension);
      }
      boundaryVolumes[boundary] = 1 / Math.max(kernelSum, EPSILON);
    }
  }
  const spatialBuckets = buildDeterministicBuckets(boundaryPositions, count, supportRadius);
  if (closedPolygon !== null && closedPolygons !== null) {
    throw new TypeError('use closedPolygon or closedPolygons, not both');
  }
  const polygonInputs = closedPolygons ?? (closedPolygon === null ? [] : [closedPolygon]);
  const polygons = [];
  for (const polygonInput of polygonInputs) {
    if (dimension !== 2) {
      throw new TypeError('closed solid boundary polygon is available only in two dimensions');
    }
    const polygonCount = requireAlignedDimensionVector(
      'closed solid boundary polygon', polygonInput, 2);
    if (polygonCount < 3) throw new RangeError('closed solid boundary polygon needs three points');
    polygons.push(Float64Array.from(polygonInput));
  }
  return Object.freeze({ kind: 'akinci-liquid-solid-boundary-reference:v1', count,
    dimension, activeAxes: axesForDimension(dimension), supportRadius,
    positions: boundaryPositions, velocities: boundaryVelocities,
    volumes: boundaryVolumes, spatialBuckets,
    closedPolygon: polygons.length === 1 ? polygons[0] : null,
    closedPolygons: Object.freeze(polygons),
    coupling: 'volume-weighted-pressure-support', penaltyForce: false });
}

function sampleClosedPolygonContact(point, polygon) {
  const count = polygon.length / 2;
  let twiceArea = 0;
  let minimumDistanceSquared = Number.POSITIVE_INFINITY;
  let closestPoint = [0, 0]; let closestEdge = 0; let inside = false;
  for (let edge = 0, previous = count - 1; edge < count; previous = edge, edge += 1) {
    const ax = polygon[previous * 2]; const ay = polygon[previous * 2 + 1];
    const bx = polygon[edge * 2]; const by = polygon[edge * 2 + 1];
    twiceArea += ax * by - bx * ay;
    const ex = bx - ax; const ey = by - ay;
    const lengthSquared = ex * ex + ey * ey;
    const projection = lengthSquared <= EPSILON ? 0 : Math.max(0, Math.min(1,
      ((point[0] - ax) * ex + (point[1] - ay) * ey) / lengthSquared));
    const projected = [ax + projection * ex, ay + projection * ey];
    const dx = point[0] - projected[0]; const dy = point[1] - projected[1];
    const distanceSquared = dx * dx + dy * dy;
    if (distanceSquared < minimumDistanceSquared) {
      minimumDistanceSquared = distanceSquared; closestPoint = projected; closestEdge = edge;
    }
    if ((ay > point[1]) !== (by > point[1])) {
      const crossingX = ax + (point[1] - ay) * (bx - ax) / (by - ay);
      if (point[0] < crossingX) inside = !inside;
    }
  }
  const previous = (closestEdge + count - 1) % count;
  const ex = polygon[closestEdge * 2] - polygon[previous * 2];
  const ey = polygon[closestEdge * 2 + 1] - polygon[previous * 2 + 1];
  const inverseLength = 1 / Math.max(EPSILON, Math.hypot(ex, ey));
  const outwardNormal = twiceArea >= 0
    ? [ey * inverseLength, -ex * inverseLength]
    : [-ey * inverseLength, ex * inverseLength];
  const distance = Math.sqrt(minimumDistanceSquared);
  return { signedDistance: inside ? -distance : distance,
    closestPoint, outwardNormal, inside };
}

export function sampleAkinciLiquidSolidContactReference(boundary, point) {
  if (boundary?.kind !== 'akinci-liquid-solid-boundary-reference:v1'
      || boundary.dimension !== 2 || boundary.closedPolygons.length === 0) {
    throw new TypeError('two-dimensional Akinci boundary with closed polygons required');
  }
  requireAlignedDimensionVector('solid signed-distance point', point, 2, 1);
  const contacts = boundary.closedPolygons.map((polygon) =>
    sampleClosedPolygonContact(point, polygon));
  const containing = contacts.filter((contact) => contact.inside);
  const selected = containing.length > 0
    ? containing.reduce((nearest, contact) => contact.signedDistance > nearest.signedDistance
      ? contact : nearest)
    : contacts.reduce((nearest, contact) => contact.signedDistance < nearest.signedDistance
      ? contact : nearest);
  return Object.freeze({ kind: 'akinci-liquid-solid-contact-sample-reference:v1',
    signedDistance: selected.signedDistance,
    closestPoint: Object.freeze(selected.closestPoint),
    outwardNormal: Object.freeze(selected.outwardNormal), inside: selected.inside });
}

export function sampleAkinciLiquidSolidSignedDistanceReference(boundary, point) {
  return sampleAkinciLiquidSolidContactReference(boundary, point).signedDistance;
}

const crossTwo = (ax, ay, bx, by) => ax * by - ay * bx;

function polygonOutwardNormal(polygon, previous, edge) {
  let twiceArea = 0;
  const count = polygon.length / 2;
  for (let vertex = 0, prior = count - 1; vertex < count; prior = vertex, vertex += 1) {
    twiceArea += polygon[prior * 2] * polygon[vertex * 2 + 1]
      - polygon[vertex * 2] * polygon[prior * 2 + 1];
  }
  const ex = polygon[edge * 2] - polygon[previous * 2];
  const ey = polygon[edge * 2 + 1] - polygon[previous * 2 + 1];
  const inverseLength = 1 / Math.max(EPSILON, Math.hypot(ex, ey));
  return twiceArea >= 0
    ? [ey * inverseLength, -ex * inverseLength]
    : [-ey * inverseLength, ex * inverseLength];
}

// Exact point-segment time of impact against static closed polygon edges. Only
// entering faces participate: this is a zero-clearance unilateral constraint,
// not a force field or collision-padding proxy.
export function sweepAkinciLiquidSolidContactReference(boundary, start, end) {
  if (boundary?.kind !== 'akinci-liquid-solid-boundary-reference:v1'
      || boundary.dimension !== 2 || boundary.closedPolygons.length === 0) {
    throw new TypeError('two-dimensional Akinci boundary with closed polygons required');
  }
  requireAlignedDimensionVector('solid sweep start', start, 2, 1);
  requireAlignedDimensionVector('solid sweep end', end, 2, 1);
  const rx = end[0] - start[0]; const ry = end[1] - start[1];
  const scale = Math.max(1, Math.abs(start[0]), Math.abs(start[1]),
    Math.abs(end[0]), Math.abs(end[1]));
  const tolerance = 128 * Number.EPSILON * scale;
  let earliest = Number.POSITIVE_INFINITY; const hits = [];
  for (let polygonIndex = 0;
    polygonIndex < boundary.closedPolygons.length; polygonIndex += 1) {
    const polygon = boundary.closedPolygons[polygonIndex];
    const count = polygon.length / 2;
    for (let edge = 0, previous = count - 1;
      edge < count; previous = edge, edge += 1) {
      const ax = polygon[previous * 2]; const ay = polygon[previous * 2 + 1];
      const sx = polygon[edge * 2] - ax; const sy = polygon[edge * 2 + 1] - ay;
      const denominator = crossTwo(rx, ry, sx, sy);
      if (Math.abs(denominator) <= tolerance) continue;
      const qx = ax - start[0]; const qy = ay - start[1];
      const fraction = crossTwo(qx, qy, sx, sy) / denominator;
      const edgeFraction = crossTwo(qx, qy, rx, ry) / denominator;
      if (fraction < -tolerance || fraction > 1 + tolerance
          || edgeFraction < -tolerance || edgeFraction > 1 + tolerance) continue;
      const outwardNormal = polygonOutwardNormal(polygon, previous, edge);
      // Ignore exits and exact tangential travel. A start-on-wall inward step has t=0.
      if (rx * outwardNormal[0] + ry * outwardNormal[1] >= -tolerance) continue;
      const clampedFraction = Math.max(0, Math.min(1, fraction));
      if (clampedFraction < earliest - tolerance) {
        earliest = clampedFraction; hits.length = 0;
      }
      if (Math.abs(clampedFraction - earliest) <= tolerance) {
        hits.push({ polygonIndex, edge, outwardNormal });
      }
    }
  }
  if (!Number.isFinite(earliest)) return null;
  hits.sort((left, right) => left.polygonIndex - right.polygonIndex
    || left.edge - right.edge);
  return Object.freeze({ kind: 'akinci-liquid-solid-sweep-contact-reference:v1',
    fraction: earliest,
    point: Object.freeze([start[0] + earliest * rx, start[1] + earliest * ry]),
    activeNormals: Object.freeze(hits.map((hit) => Object.freeze({
      polygonIndex: hit.polygonIndex, edge: hit.edge,
      outwardNormal: Object.freeze(hit.outwardNormal) }))) });
}

function requireParticleScalarVector(name, values, count) {
  if ((!Array.isArray(values) && !ArrayBuffer.isView(values))
      || values.length !== count
      || !Array.from(values).every((value) => Number.isFinite(value) && value > 0)) {
    throw new TypeError(`${name} must contain ${count} positive values`);
  }
  return Float64Array.from(values);
}

const sumParticleMass = (masses) => {
  let total = 0;
  for (const mass of masses) total += mass;
  return total;
};

const particleMomentum = (state, masses = state.masses) => {
  const momentum = [0, 0, 0];
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      momentum[axis] += masses[particle] * state.velocities[offset + axis];
    }
  }
  return momentum;
};

export function createLocalParticleLiquidStateReference({ dimension = 2, positions,
  velocities, masses, supportRadii, restDensity = 998.207,
  gravity = dimension === 2 ? [0, -9.82] : [0, 0, -9.82],
  timeStep = 1 / 240, seed = 0, ids = null } = {}) {
  requireDimension(dimension);
  const count = requireAlignedDimensionVector('local liquid positions', positions, dimension);
  if (count === 0) throw new RangeError('local liquid positions must not be empty');
  requireAlignedDimensionVector('local liquid velocities', velocities, dimension, count);
  const resolvedMasses = requireParticleScalarVector('local liquid masses', masses, count);
  const resolvedSupportRadii = Number.isFinite(supportRadii)
    ? new Float64Array(count).fill(requirePositiveFinite(
      'local liquid support radius', supportRadii))
    : requireParticleScalarVector('local liquid support radii', supportRadii, count);
  requirePositiveFinite('local liquid rest density', restDensity);
  requireAlignedDimensionVector('local liquid gravity', gravity, dimension, 1);
  const activeAxes = axesForDimension(dimension);
  const expandedGravity = dimension === 2 ? [gravity[0], 0, gravity[1]] : Array.from(gravity);
  const equalMass = resolvedMasses.every((mass) => mass === resolvedMasses[0]);
  if (!equalMass) {
    throw new RangeError('mutable local liquid state currently requires equal particle masses');
  }
  const state = createPhysicsStateBufferReference({
    kind: 'local-particle-liquid-state-reference:v1', seed, count, precision: 'f64',
    includeIds: true, gravity: expandedGravity, timeStep,
    conserved: { name: 'mass', perElement: resolvedMasses[0] },
    adapter: 'local-particle-liquid',
  });
  state.positions.set(expandDimensionVector(positions, dimension));
  state.velocities.set(expandDimensionVector(velocities, dimension));
  if (ids === null) {
    for (let particle = 0; particle < count; particle += 1) state.ids[particle] = particle;
  } else {
    if ((!Array.isArray(ids) && !ArrayBuffer.isView(ids)) || ids.length !== count
        || !Array.from(ids).every((id) => Number.isSafeInteger(id) && id >= 0)) {
      throw new TypeError(`local liquid ids must contain ${count} nonnegative integers`);
    }
    state.ids.set(ids);
  }
  state.dimension = dimension; state.activeAxes = activeAxes;
  state.masses = resolvedMasses; state.supportRadii = resolvedSupportRadii;
  state.densities = new Float64Array(count); state.restDensity = restDensity;
  state.density = restDensity; state.particleMass = resolvedMasses[0];
  state.time = 0;
  state.localFluxLedger = { appendedMass: 0, removedMass: 0,
    appendedMomentum: [0, 0, 0], removedMomentum: [0, 0, 0] };
  state.stateHash = hashPhysicsStateBufferReference(state);
  return state;
}

function appendScalarVector(current, additions) {
  const next = new current.constructor(current.length + additions.length);
  next.set(current); next.set(additions, current.length);
  return next;
}

export function appendLocalParticleLiquidStateReference(state, {
  positions, velocities, masses, supportRadii, ids,
} = {}) {
  requireParticleState(state);
  if (state.kind !== 'local-particle-liquid-state-reference:v1') {
    throw new TypeError('local particle Liquid state required for append');
  }
  const addedCount = requireAlignedDimensionVector(
    'appended local liquid positions', positions, state.dimension);
  requireAlignedDimensionVector(
    'appended local liquid velocities', velocities, state.dimension, addedCount);
  const appendedMasses = requireParticleScalarVector(
    'appended local liquid masses', masses, addedCount);
  if (!appendedMasses.every((mass) => mass === state.particleMass)) {
    throw new RangeError('appended local liquid masses must match the state particle mass');
  }
  const appendedRadii = Number.isFinite(supportRadii)
    ? new Float64Array(addedCount).fill(requirePositiveFinite(
      'appended local liquid support radius', supportRadii))
    : requireParticleScalarVector(
      'appended local liquid support radii', supportRadii, addedCount);
  const expandedVelocities = expandDimensionVector(velocities, state.dimension);
  const resolvedIds = ids ?? Array.from({ length: addedCount }, (_, index) =>
    Math.max(...state.ids) + 1 + index);
  const edit = appendPhysicsStateBufferReference(state, {
    positions: expandDimensionVector(positions, state.dimension),
    velocities: expandedVelocities, ids: resolvedIds,
  });
  state.masses = appendScalarVector(state.masses, appendedMasses);
  state.supportRadii = appendScalarVector(state.supportRadii, appendedRadii);
  state.densities = appendScalarVector(state.densities, new Float64Array(addedCount));
  state.localFluxLedger.appendedMass += sumParticleMass(appendedMasses);
  for (let particle = 0; particle < addedCount; particle += 1) {
    const offset = particle * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      state.localFluxLedger.appendedMomentum[axis] += appendedMasses[particle]
        * expandedVelocities[offset + axis];
    }
  }
  return edit;
}

export function selectLocalParticleLiquidStateReference(state, retainedIndices) {
  requireParticleState(state);
  if (state.kind !== 'local-particle-liquid-state-reference:v1') {
    throw new TypeError('local particle Liquid state required for selection');
  }
  const retained = Array.from(retainedIndices);
  const retainedSet = new Set(retained);
  const oldMasses = state.masses; const oldSupportRadii = state.supportRadii;
  const oldDensities = state.densities; const oldVelocities = state.velocities;
  let removedMass = 0; const removedMomentum = [0, 0, 0];
  for (let particle = 0; particle < state.count; particle += 1) {
    if (retainedSet.has(particle)) continue;
    removedMass += oldMasses[particle];
    const offset = particle * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      removedMomentum[axis] += oldMasses[particle] * oldVelocities[offset + axis];
    }
  }
  const selectScalar = (source) => Float64Array.from(retained.map((index) => source[index]));
  const edit = selectPhysicsStateBufferReference(state, retained);
  state.masses = selectScalar(oldMasses); state.supportRadii = selectScalar(oldSupportRadii);
  state.densities = selectScalar(oldDensities);
  state.localFluxLedger.removedMass += removedMass;
  for (let axis = 0; axis < 3; axis += 1) {
    state.localFluxLedger.removedMomentum[axis] += removedMomentum[axis];
  }
  return edit;
}

function resolveParticleMasses(core) {
  const { state, particleMasses, particleMass } = core;
  if (state.masses instanceof Float64Array) {
    if (state.masses.length !== state.count) {
      throw new RangeError('liquid state masses must stay aligned with the entity axis');
    }
    return state.masses;
  }
  if (particleMasses !== null) {
    if (particleMasses.length !== state.count) {
      throw new RangeError('liquid particle masses must stay aligned with the entity axis');
    }
    return particleMasses;
  }
  return null;
}

const particleMassAt = (core, masses, particle) => masses === null
  ? core.particleMass : masses[particle];

function maximumParticleSupportRadius(state, fallback) {
  let maximum = fallback;
  if (state.supportRadii !== undefined) {
    for (const supportRadius of state.supportRadii) maximum = Math.max(maximum, supportRadius);
  }
  return maximum;
}

function minimumParticleSupportRadius(state, fallback) {
  let minimum = fallback;
  if (state.supportRadii !== undefined) {
    for (const supportRadius of state.supportRadii) minimum = Math.min(minimum, supportRadius);
  }
  return minimum;
}

function createCoreLiquidNeighborhood(core) {
  // For variable h, h_ij=(h_i+h_j)/2 <= max(h). The broad phase must therefore
  // query at least max(h), even though the exact pair filter below uses h_ij.
  return createDeterministicLiquidNeighborhoodReference(core.state, {
    supportRadius: maximumParticleSupportRadius(core.state, core.supportRadius),
    dimension: core.dimension,
  });
}

function requireBoundaryCompatibility(boundary, supportRadius, dimension) {
  if (boundary === null) return null;
  if (boundary?.kind !== 'akinci-liquid-solid-boundary-reference:v1') {
    throw new TypeError('Akinci volume-weighted solid boundary required');
  }
  if (Math.abs(boundary.supportRadius - supportRadius)
      > supportRadius * 16 * Number.EPSILON) {
    throw new RangeError('solid boundary and liquid must share one support radius');
  }
  if (boundary.dimension !== dimension) {
    throw new RangeError('solid boundary and liquid must share one dimension');
  }
  return boundary;
}

export function createLocalParticleLiquidCoreReference(state, {
  dimension = state?.dimension ?? 3,
  restDensity = state?.density ?? 998.207,
  particleMass = state?.particleMass,
  particleMasses = state?.masses ?? null,
  supportRadius = state?.supportRadii?.[0],
  solidBoundary = null,
  maximumDivergenceIterations = DEFAULT_DIVERGENCE_ITERATIONS,
  maximumDensityIterations = DEFAULT_DENSITY_ITERATIONS,
  divergenceTolerance = DEFAULT_DIVERGENCE_TOLERANCE,
  densityTolerance = DEFAULT_DENSITY_TOLERANCE,
  maximumCourantNumber = DEFAULT_MAXIMUM_COURANT_NUMBER,
  maximumCflSubsteps = DEFAULT_MAXIMUM_CFL_SUBSTEPS,
} = {}) {
  requireParticleState(state);
  requireDimension(dimension);
  requirePositiveFinite('liquid rest density', restDensity);
  requirePositiveFinite('liquid support radius', supportRadius);
  if (particleMasses !== null) {
    if ((!Array.isArray(particleMasses) && !ArrayBuffer.isView(particleMasses))
        || particleMasses.length !== state.count
        || !Array.from(particleMasses).every((value) => Number.isFinite(value) && value > 0)) {
      throw new TypeError(`liquid particle masses must contain ${state.count} positive values`);
    }
  } else requirePositiveFinite('liquid particle mass', particleMass);
  for (const [name, value] of [['maximum divergence iterations', maximumDivergenceIterations],
    ['maximum density iterations', maximumDensityIterations]]) {
    if (!Number.isSafeInteger(value) || value < 1 || value > 256) {
      throw new RangeError(`${name} must be an integer from one through 256`);
    }
  }
  requireNonnegativeFinite('liquid divergence tolerance', divergenceTolerance);
  requireNonnegativeFinite('liquid density tolerance', densityTolerance);
  if (!Number.isFinite(maximumCourantNumber)
      || maximumCourantNumber <= 0 || maximumCourantNumber > 1) {
    throw new RangeError('maximum liquid Courant number must be above zero through one');
  }
  if (!Number.isSafeInteger(maximumCflSubsteps)
      || maximumCflSubsteps < 1 || maximumCflSubsteps > 4096) {
    throw new RangeError('maximum liquid CFL substeps must be an integer from one through 4096');
  }
  const resolvedBoundary = requireBoundaryCompatibility(solidBoundary, supportRadius, dimension);
  const resolvedMasses = particleMasses === null
    ? new Float64Array(state.count).fill(particleMass) : Float64Array.from(particleMasses);
  const initialMass = sumParticleMass(resolvedMasses);
  const initialMomentum = particleMomentum(state, resolvedMasses);
  if (dimension === 2 && resolvedBoundary?.closedPolygons.length > 0) {
    const numericalTolerance = supportRadius * 1e-10;
    for (let particle = 0; particle < state.count; particle += 1) {
      const offset = particle * 3;
      const distance = sampleAkinciLiquidSolidSignedDistanceReference(resolvedBoundary,
        [state.positions[offset], state.positions[offset + 2]]);
      if (distance < -numericalTolerance) {
        throw new RangeError(`local liquid particle ${particle} starts inside a closed solid`);
      }
    }
  }
  return { kind: 'local-particle-liquid-core-reference:v1', state, dimension,
    activeAxes: axesForDimension(dimension), restDensity,
    particleMass: particleMasses === null ? particleMass : null,
    particleMasses: resolvedMasses,
    supportRadius, solidBoundary: resolvedBoundary,
    maximumDivergenceIterations, maximumDensityIterations,
    divergenceTolerance, densityTolerance, maximumCourantNumber, maximumCflSubsteps,
    pressureModel: 'local-density-and-divergence-jacobi-tracer',
    researchDirection: 'dfsph-style-two-constraint-projection',
    boundaryModel: resolvedBoundary === null ? null : 'akinci-volume-weighted-particles',
    contactModel: dimension === 2 && resolvedBoundary?.closedPolygons.length > 0
      ? 'swept-zero-clearance-unilateral-static-polygon' : null,
    viscosityModel: null, capillaryModel: null,
    adaptiveResolution: 'gated-until-density-consistent-2-to-1-transition',
    time: state.time ?? state.steps * state.timeStep,
    initialMass, initialMomentum: Object.freeze(initialMomentum),
    accumulatedGravityImpulse: [0, 0, 0],
    accumulatedFluidBoundaryImpulse: [0, 0, 0],
    accumulatedPressureBoundaryImpulse: [0, 0, 0],
    accumulatedContactBoundaryImpulse: [0, 0, 0],
    authoredState: Object.freeze({ positions: state.positions.slice(),
      velocities: state.velocities.slice(), masses: resolvedMasses.slice(),
      supportRadii: state.supportRadii?.slice() ?? new Float64Array(state.count).fill(supportRadius),
      densities: state.densities?.slice() ?? new Float64Array(state.count),
      ids: state.ids?.slice() ?? null, conservation: Object.freeze({ ...state.conservation }),
      localFluxLedger: Object.freeze({ appendedMass: 0, removedMass: 0,
        appendedMomentum: Object.freeze([0, 0, 0]),
        removedMomentum: Object.freeze([0, 0, 0]) }) }),
    metrics: null };
}

function buildBoundaryNeighborRows(core) {
  const { state, solidBoundary: boundary, supportRadius } = core;
  if (boundary === null) return new Array(state.count).fill(null).map(() => []);
  const rows = new Array(state.count);
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    rows[particle] = queryPointNeighbors(boundary.positions, boundary.count,
      [state.positions[offset], state.positions[offset + 1], state.positions[offset + 2]],
      supportRadius, boundary.spatialBuckets);
  }
  return rows;
}

function buildLocalConstraints(core, neighborhood) {
  const { state, restDensity, supportRadius, solidBoundary: boundary } = core;
  const maximumPairSupportRadius = maximumParticleSupportRadius(state, supportRadius);
  if (neighborhood?.kind !== 'deterministic-liquid-neighborhood-reference:v1'
      || neighborhood.count !== state.count
      || neighborhood.supportRadius + maximumPairSupportRadius * 16 * Number.EPSILON
        < maximumPairSupportRadius
      || neighborhood.dimension !== core.dimension) {
    throw new TypeError('matching deterministic liquid neighborhood required');
  }
  const masses = resolveParticleMasses(core);
  const inverseRestDensity = 1 / restDensity;
  const boundaryRows = buildBoundaryNeighborRows(core);
  const constraints = new Array(state.count);
  const densities = new Float64Array(state.count);
  let boundaryNeighborCount = 0;
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    const particleSupportRadius = state.supportRadii?.[particle] ?? supportRadius;
    let density = particleMassAt(core, masses, particle)
      * liquidWendlandC2KernelReference(0, particleSupportRadius, core.dimension);
    const centerGradient = [0, 0, 0];
    const fluidTerms = [];
    const fluidStart = neighborhood.offsets[particle];
    const fluidEnd = neighborhood.offsets[particle + 1];
    for (let cursor = fluidStart; cursor < fluidEnd; cursor += 1) {
      const neighbor = neighborhood.indices[cursor];
      if (neighbor === particle) continue;
      const neighborOffset = neighbor * 3;
      const rx = state.positions[offset] - state.positions[neighborOffset];
      const ry = state.positions[offset + 1] - state.positions[neighborOffset + 1];
      const rz = state.positions[offset + 2] - state.positions[neighborOffset + 2];
      const distance = Math.hypot(rx, ry, rz);
      const neighborMass = particleMassAt(core, masses, neighbor);
      const neighborSupportRadius = state.supportRadii?.[neighbor] ?? supportRadius;
      const pairSupportRadius = (particleSupportRadius + neighborSupportRadius) / 2;
      if (distance >= pairSupportRadius) continue;
      density += neighborMass * liquidWendlandC2KernelReference(
        distance, pairSupportRadius, core.dimension);
      const kernelGradient = liquidWendlandC2GradientUnchecked(
        rx, ry, rz, pairSupportRadius, core.dimension);
      const gradient = [kernelGradient[0] * neighborMass * inverseRestDensity,
        kernelGradient[1] * neighborMass * inverseRestDensity,
        kernelGradient[2] * neighborMass * inverseRestDensity];
      centerGradient[0] += gradient[0]; centerGradient[1] += gradient[1];
      centerGradient[2] += gradient[2];
      fluidTerms.push({ neighbor, gradient });
    }
    const boundaryTerms = [];
    if (boundary !== null) for (const boundaryParticle of boundaryRows[particle]) {
      const boundaryOffset = boundaryParticle * 3;
      const rx = state.positions[offset] - boundary.positions[boundaryOffset];
      const ry = state.positions[offset + 1] - boundary.positions[boundaryOffset + 1];
      const rz = state.positions[offset + 2] - boundary.positions[boundaryOffset + 2];
      const distance = Math.hypot(rx, ry, rz);
      const boundarySupportRadius = boundary.supportRadius;
      if (distance >= boundarySupportRadius) continue;
      const volume = boundary.volumes[boundaryParticle];
      density += restDensity * volume
        * liquidWendlandC2KernelReference(distance, boundarySupportRadius, core.dimension);
      const kernelGradient = liquidWendlandC2GradientUnchecked(
        rx, ry, rz, boundarySupportRadius, core.dimension);
      const gradient = [kernelGradient[0] * volume, kernelGradient[1] * volume,
        kernelGradient[2] * volume];
      centerGradient[0] += gradient[0]; centerGradient[1] += gradient[1];
      centerGradient[2] += gradient[2];
      boundaryTerms.push({ boundaryParticle, gradient });
      boundaryNeighborCount += 1;
    }
    const inverseParticleMass = 1 / particleMassAt(core, masses, particle);
    let denominator = inverseParticleMass * (centerGradient[0] ** 2
      + centerGradient[1] ** 2 + centerGradient[2] ** 2);
    for (const term of fluidTerms) {
      const inverseNeighborMass = 1 / particleMassAt(core, masses, term.neighbor);
      denominator += inverseNeighborMass * (term.gradient[0] ** 2
        + term.gradient[1] ** 2 + term.gradient[2] ** 2);
    }
    densities[particle] = density;
    constraints[particle] = { centerGradient, fluidTerms, boundaryTerms,
      denominator: Math.max(denominator, EPSILON) };
  }
  return { constraints, densities, masses, boundaryNeighborCount };
}

function constraintDensityRate(core, constraint) {
  const { state, solidBoundary: boundary } = core;
  const particle = constraint.particle;
  const offset = particle * 3;
  let rate = constraint.centerGradient[0] * state.velocities[offset]
    + constraint.centerGradient[1] * state.velocities[offset + 1]
    + constraint.centerGradient[2] * state.velocities[offset + 2];
  for (const term of constraint.fluidTerms) {
    const neighborOffset = term.neighbor * 3;
    rate -= term.gradient[0] * state.velocities[neighborOffset]
      + term.gradient[1] * state.velocities[neighborOffset + 1]
      + term.gradient[2] * state.velocities[neighborOffset + 2];
  }
  if (boundary !== null) for (const term of constraint.boundaryTerms) {
    const boundaryOffset = term.boundaryParticle * 3;
    rate -= term.gradient[0] * boundary.velocities[boundaryOffset]
      + term.gradient[1] * boundary.velocities[boundaryOffset + 1]
      + term.gradient[2] * boundary.velocities[boundaryOffset + 2];
  }
  return rate;
}

function applyConstraintJacobiIteration(core, data, residualForConstraint) {
  const { state } = core;
  const deltaVelocities = new Float64Array(state.count * 3);
  let activeConstraintCount = 0; let maximumResidual = 0; let residualSum = 0;
  const fluidBoundaryImpulse = [0, 0, 0];
  for (let particle = 0; particle < state.count; particle += 1) {
    const constraint = data.constraints[particle]; constraint.particle = particle;
    const residual = residualForConstraint(constraint, particle);
    maximumResidual = Math.max(maximumResidual, residual);
    residualSum += residual;
    if (residual <= 0) continue;
    activeConstraintCount += 1;
    const lambda = -residual / constraint.denominator;
    const inverseParticleMass = 1 / particleMassAt(core, data.masses, particle);
    const offset = particle * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      const correction = inverseParticleMass * lambda * constraint.centerGradient[axis];
      deltaVelocities[offset + axis] += correction;
    }
    for (const term of constraint.fluidTerms) {
      const neighborOffset = term.neighbor * 3;
      const inverseNeighborMass = 1 / particleMassAt(core, data.masses, term.neighbor);
      for (let axis = 0; axis < 3; axis += 1) {
        const correction = -inverseNeighborMass * lambda * term.gradient[axis];
        deltaVelocities[neighborOffset + axis] += correction;
      }
    }
    // Internal fluid-pair terms cancel exactly. The remaining impulse is lambda
    // times the authored boundary gradient, calculated directly rather than
    // inferred from the measured total momentum change.
    for (const term of constraint.boundaryTerms) for (let axis = 0; axis < 3; axis += 1) {
      fluidBoundaryImpulse[axis] += lambda * term.gradient[axis];
    }
  }
  for (let component = 0; component < state.velocities.length; component += 1) {
    state.velocities[component] += deltaVelocities[component];
  }
  return { activeConstraintCount, maximumResidual,
    meanResidual: residualSum / state.count, fluidBoundaryImpulse };
}

function projectLocalConstraintReference(core, data, {
  kind, maximumIterations, tolerance, densityTimeStep = null,
}) {
  // The reported residual is the amount still above the requested physical
  // tolerance. Resolve it to 0.1% of that tolerance, with a numerical floor.
  const convergenceThreshold = Math.max(EPSILON, (densityTimeStep === null
    ? tolerance : tolerance / densityTimeStep) * 1e-3);
  const momentumBefore = particleMomentum(core.state, data.masses
    ?? new Float64Array(core.state.count).fill(core.particleMass));
  const explicitFluidBoundaryImpulse = [0, 0, 0];
  let iterations = 0; let latest = null;
  do {
    latest = applyConstraintJacobiIteration(core, data, (constraint, particle) => {
      const rate = constraintDensityRate(core, constraint);
      if (densityTimeStep === null) return Math.max(0, rate - tolerance);
      const currentDensityError = data.densities[particle] / core.restDensity - 1;
      // Clamp only the final predicted compression. Keeping a real free-surface
      // density deficit avoids inventing pressure before that deficit is filled.
      return Math.max(0, currentDensityError + densityTimeStep * rate - tolerance)
        / densityTimeStep;
    });
    for (let axis = 0; axis < 3; axis += 1) {
      explicitFluidBoundaryImpulse[axis] += latest.fluidBoundaryImpulse[axis];
    }
    iterations += 1;
    if (latest.maximumResidual <= convergenceThreshold
        || latest.activeConstraintCount === 0) break;
  } while (iterations < maximumIterations);
  const momentumAfter = particleMomentum(core.state, data.masses
    ?? new Float64Array(core.state.count).fill(core.particleMass));
  const measuredFluidMomentumDelta = momentumAfter.map(
    (component, axis) => component - momentumBefore[axis]);
  const impulseClosureResidual = measuredFluidMomentumDelta.map(
    (component, axis) => component - explicitFluidBoundaryImpulse[axis]);
  return Object.freeze({ kind, iterations,
    converged: latest.maximumResidual <= convergenceThreshold
      || latest.activeConstraintCount === 0,
    convergenceThreshold,
    maximumResidual: latest.maximumResidual,
    meanResidual: latest.meanResidual,
    activeConstraintCount: latest.activeConstraintCount,
    fluidBoundaryImpulse: Object.freeze(explicitFluidBoundaryImpulse),
    solidReactionImpulse: Object.freeze(explicitFluidBoundaryImpulse.map((value) => -value)),
    measuredFluidMomentumDelta: Object.freeze(measuredFluidMomentumDelta),
    impulseClosureResidual: Object.freeze(impulseClosureResidual) });
}

export function measureLocalParticleLiquidDensityReference(core, neighborhood = null) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  const resolvedNeighborhood = neighborhood
    ?? createCoreLiquidNeighborhood(core);
  const data = buildLocalConstraints(core, resolvedNeighborhood);
  let maximumRelativeCompression = 0; let maximumRelativeError = 0;
  let maximumRelativeUnderdensity = 0; let meanRelativeCompression = 0;
  let meanRelativeUnderdensity = 0; let meanDensity = 0;
  let freeSurfaceParticleCount = 0; let freeSurfaceUnderdensitySum = 0;
  for (const density of data.densities) {
    const relativeDifference = density / core.restDensity - 1;
    const compression = Math.max(0, relativeDifference);
    const underdensity = Math.max(0, -relativeDifference);
    maximumRelativeCompression = Math.max(maximumRelativeCompression, compression);
    maximumRelativeUnderdensity = Math.max(maximumRelativeUnderdensity, underdensity);
    maximumRelativeError = Math.max(maximumRelativeError,
      Math.abs(relativeDifference));
    meanRelativeCompression += compression / core.state.count;
    meanRelativeUnderdensity += underdensity / core.state.count;
    if (density / core.restDensity < 0.94) {
      freeSurfaceParticleCount += 1; freeSurfaceUnderdensitySum += underdensity;
    }
    meanDensity += density / core.state.count;
  }
  return Object.freeze({ kind: 'local-particle-liquid-density-reference:v1',
    densities: data.densities, meanDensity,
    maximumRelativeCompression: Math.max(0, maximumRelativeCompression),
    meanRelativeCompression, maximumRelativeUnderdensity, meanRelativeUnderdensity,
    freeSurfaceParticleCount,
    freeSurfaceMeanRelativeUnderdensity: freeSurfaceParticleCount === 0 ? 0
      : freeSurfaceUnderdensitySum / freeSurfaceParticleCount,
    maximumRelativeError, boundaryNeighborCount: data.boundaryNeighborCount });
}

export function projectLocalParticleLiquidDivergenceReference(core,
  neighborhood = null) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  const resolvedNeighborhood = neighborhood
    ?? createCoreLiquidNeighborhood(core);
  const data = buildLocalConstraints(core, resolvedNeighborhood);
  return projectLocalConstraintReference(core, data, {
    kind: 'local-particle-liquid-divergence-projection-reference:v1',
    maximumIterations: core.maximumDivergenceIterations,
    tolerance: core.divergenceTolerance,
  });
}

export function projectLocalParticleLiquidDensityReference(core,
  neighborhood = null, timeStep = core?.state?.timeStep) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  requirePositiveFinite('local particle Liquid timestep', timeStep);
  const resolvedNeighborhood = neighborhood
    ?? createCoreLiquidNeighborhood(core);
  const data = buildLocalConstraints(core, resolvedNeighborhood);
  return projectLocalConstraintReference(core, data, {
    kind: 'local-particle-liquid-density-projection-reference:v1',
    maximumIterations: core.maximumDensityIterations,
    tolerance: core.densityTolerance,
    densityTimeStep: timeStep,
  });
}

function maximumParticleSpeed(state) {
  let maximum = 0;
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    maximum = Math.max(maximum, Math.hypot(state.velocities[offset],
      state.velocities[offset + 1], state.velocities[offset + 2]));
  }
  return maximum;
}

function projectStaticFreeSlipVelocityTwoDimensions(velocity, activeNormals) {
  const candidates = [[...velocity], [0, 0]];
  for (const active of activeNormals) {
    const normal = active.outwardNormal;
    const normalSpeed = velocity[0] * normal[0] + velocity[1] * normal[1];
    candidates.push([velocity[0] - normalSpeed * normal[0],
      velocity[1] - normalSpeed * normal[1]]);
  }
  let best = null; let bestChangeSquared = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    const feasible = activeNormals.every((active) => candidate[0] * active.outwardNormal[0]
      + candidate[1] * active.outwardNormal[1] >= -1e-13);
    if (!feasible) continue;
    const changeSquared = (candidate[0] - velocity[0]) ** 2
      + (candidate[1] - velocity[1]) ** 2;
    if (changeSquared < bestChangeSquared) {
      best = candidate; bestChangeSquared = changeSquared;
    }
  }
  return best ?? [0, 0];
}

export function localParticleLiquidCflSubstepsReference(core,
  timeStep = core?.state?.timeStep) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  requirePositiveFinite('local particle Liquid CFL timestep', timeStep);
  const { state } = core;
  const minimumSupportRadius = minimumParticleSupportRadius(state, core.supportRadius);
  let maximumPredictedSpeed = 0;
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    maximumPredictedSpeed = Math.max(maximumPredictedSpeed, Math.hypot(
      state.velocities[offset] + state.gravity[0] * timeStep,
      state.velocities[offset + 1] + state.gravity[1] * timeStep,
      state.velocities[offset + 2] + state.gravity[2] * timeStep));
  }
  const unrestrictedCourantNumber = maximumPredictedSpeed * timeStep
    / minimumSupportRadius;
  const requiredSubsteps = Math.max(1,
    Math.ceil(unrestrictedCourantNumber / core.maximumCourantNumber - 1e-14));
  return Object.freeze({ kind: 'local-particle-liquid-cfl-reference:v1', timeStep,
    minimumSupportRadius, maximumPredictedSpeed, unrestrictedCourantNumber,
    maximumCourantNumber: core.maximumCourantNumber, requiredSubsteps,
    maximumSubsteps: core.maximumCflSubsteps,
    withinSubstepLimit: requiredSubsteps <= core.maximumCflSubsteps });
}

function advanceLocalParticlesWithSweptSolidReference(core, timeStep) {
  const { state, solidBoundary: boundary } = core;
  const masses = resolveParticleMasses(core)
    ?? new Float64Array(state.count).fill(core.particleMass);
  const fluidBoundaryImpulse = [0, 0, 0];
  const sweptSegments = [];
  let contactCount = 0; let numericalProjectionCount = 0;
  let maximumRejectedEndpointPenetration = 0;
  let minimumSignedDistance = boundary?.closedPolygons.length > 0
    ? Number.POSITIVE_INFINITY : null;
  if (core.dimension !== 2 || boundary === null || boundary.closedPolygons.length === 0) {
    for (let particle = 0; particle < state.count; particle += 1) {
      const offset = particle * 3;
      const from = [state.positions[offset], state.positions[offset + 1],
        state.positions[offset + 2]];
      for (let axis = 0; axis < 3; axis += 1) {
        state.positions[offset + axis] += state.velocities[offset + axis] * timeStep;
      }
      sweptSegments.push(Object.freeze({ particle,
        from: Object.freeze(from),
        to: Object.freeze(Array.from(state.positions.subarray(offset, offset + 3))),
        contact: false }));
    }
    return Object.freeze({ kind: 'local-particle-liquid-swept-advection-reference:v1',
      timeStep, contactCount, numericalProjectionCount,
      maximumRejectedEndpointPenetration, minimumSignedDistance,
      fluidBoundaryImpulse: Object.freeze(fluidBoundaryImpulse),
      solidReactionImpulse: Object.freeze(fluidBoundaryImpulse.map((value) => -value)),
      sweptSegments: Object.freeze(sweptSegments) });
  }

  const numericalClearance = core.supportRadius * 1e-10;
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    let position = [state.positions[offset], state.positions[offset + 2]];
    let velocity = [state.velocities[offset], state.velocities[offset + 2]];
    let remainingTime = timeStep; let contactIterations = 0;
    const startDistance = sampleAkinciLiquidSolidSignedDistanceReference(boundary, position);
    if (startDistance < -numericalClearance) {
      throw new RangeError(`local liquid particle ${particle} begins advection inside solid`);
    }
    while (remainingTime > timeStep * 1e-14 && contactIterations < 16) {
      const intendedEnd = [position[0] + velocity[0] * remainingTime,
        position[1] + velocity[1] * remainingTime];
      const intendedDistance = sampleAkinciLiquidSolidSignedDistanceReference(
        boundary, intendedEnd);
      maximumRejectedEndpointPenetration = Math.max(maximumRejectedEndpointPenetration,
        Math.max(0, -intendedDistance));
      const hit = sweepAkinciLiquidSolidContactReference(boundary, position, intendedEnd);
      if (hit === null) {
        sweptSegments.push(Object.freeze({ particle,
          from: Object.freeze([...position]), to: Object.freeze(intendedEnd), contact: false }));
        position = intendedEnd; remainingTime = 0; break;
      }
      sweptSegments.push(Object.freeze({ particle,
        from: Object.freeze([...position]), to: hit.point, contact: true,
        activeEdges: Object.freeze(hit.activeNormals.map((active) => Object.freeze({
          polygonIndex: active.polygonIndex, edge: active.edge }))) }));
      position = [...hit.point];
      const velocityBefore = [...velocity];
      // Exact 2-D active-set projection at an edge/corner. Static wall, zero
      // restitution, free slip: choose the closest feasible velocity and retain
      // the tangential component wherever one face alone is active.
      velocity = projectStaticFreeSlipVelocityTwoDimensions(velocity, hit.activeNormals);
      fluidBoundaryImpulse[0] += masses[particle] * (velocity[0] - velocityBefore[0]);
      fluidBoundaryImpulse[2] += masses[particle] * (velocity[1] - velocityBefore[1]);
      contactCount += 1; contactIterations += 1;
      remainingTime *= Math.max(0, 1 - hit.fraction);
      let nx = 0; let ny = 0;
      for (const active of hit.activeNormals) {
        nx += active.outwardNormal[0]; ny += active.outwardNormal[1];
      }
      const inverseNormalLength = 1 / Math.max(EPSILON, Math.hypot(nx, ny));
      // This is only a floating-point separation (1e-10 H), reported explicitly;
      // it is not a physical collision radius or dry halo.
      position[0] += nx * inverseNormalLength * numericalClearance;
      position[1] += ny * inverseNormalLength * numericalClearance;
    }
    if (remainingTime > timeStep * 1e-14) {
      // Corner active set exhausted: fail closed at the last valid contact point.
      remainingTime = 0;
    }
    let finalContact = sampleAkinciLiquidSolidContactReference(boundary, position);
    if (finalContact.signedDistance < 0) {
      if (finalContact.signedDistance < -numericalClearance) {
        throw new RangeError(`swept local liquid contact penetrated solid at particle ${particle}`);
      }
      position = [finalContact.closestPoint[0]
        + finalContact.outwardNormal[0] * numericalClearance,
      finalContact.closestPoint[1] + finalContact.outwardNormal[1] * numericalClearance];
      numericalProjectionCount += 1;
      finalContact = sampleAkinciLiquidSolidContactReference(boundary, position);
    }
    minimumSignedDistance = Math.min(minimumSignedDistance, finalContact.signedDistance);
    state.positions[offset] = position[0]; state.positions[offset + 2] = position[1];
    state.velocities[offset] = velocity[0]; state.velocities[offset + 2] = velocity[1];
  }
  return Object.freeze({ kind: 'local-particle-liquid-swept-advection-reference:v1',
    timeStep, contactCount, numericalProjectionCount, numericalClearance,
    maximumRejectedEndpointPenetration, minimumSignedDistance,
    fluidBoundaryImpulse: Object.freeze(fluidBoundaryImpulse),
    solidReactionImpulse: Object.freeze(fluidBoundaryImpulse.map((value) => -value)),
    sweptSegments: Object.freeze(sweptSegments) });
}

function runLocalParticleLiquidSubsteps(core, logicalTimeStep, substepCount) {
  const { state } = core;
  const substepTimeStep = logicalTimeStep / substepCount;
  const gravityImpulse = [0, 0, 0];
  const pressureBoundaryImpulse = [0, 0, 0];
  const contactBoundaryImpulse = [0, 0, 0];
  const substeps = [];
  let allPressureSolvesConverged = true; let maximumObservedCourantNumber = 0;
  let maximumDivergenceResidual = 0; let maximumDensityResidual = 0;
  let maximumPressureImpulseClosureResidual = 0;
  let minimumSignedDistanceAfterSubsteps = null;
  let contactCount = 0; let maximumRejectedEndpointPenetration = 0;
  const sweptSegments = [];
  const currentMass = sumParticleMass(resolveParticleMasses(core)
    ?? new Float64Array(state.count).fill(core.particleMass));
  for (let substep = 0; substep < substepCount; substep += 1) {
    const neighborhood = createCoreLiquidNeighborhood(core);
    const divergence = projectLocalParticleLiquidDivergenceReference(core, neighborhood);
    for (let particle = 0; particle < state.count; particle += 1) {
      const offset = particle * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        state.velocities[offset + axis] += state.gravity[axis] * substepTimeStep;
      }
    }
    for (let axis = 0; axis < 3; axis += 1) {
      gravityImpulse[axis] += currentMass * state.gravity[axis] * substepTimeStep;
    }
    const density = projectLocalParticleLiquidDensityReference(
      core, neighborhood, substepTimeStep);
    const preContactSpeed = maximumParticleSpeed(state);
    const minimumSupportRadius = minimumParticleSupportRadius(state, core.supportRadius);
    const courantNumber = preContactSpeed * substepTimeStep / minimumSupportRadius;
    maximumObservedCourantNumber = Math.max(maximumObservedCourantNumber, courantNumber);
    const advection = advanceLocalParticlesWithSweptSolidReference(core, substepTimeStep);
    for (let axis = 0; axis < 3; axis += 1) {
      pressureBoundaryImpulse[axis] += divergence.fluidBoundaryImpulse[axis]
        + density.fluidBoundaryImpulse[axis];
      contactBoundaryImpulse[axis] += advection.fluidBoundaryImpulse[axis];
    }
    allPressureSolvesConverged = allPressureSolvesConverged
      && divergence.converged && density.converged;
    maximumDivergenceResidual = Math.max(maximumDivergenceResidual,
      divergence.maximumResidual);
    maximumDensityResidual = Math.max(maximumDensityResidual, density.maximumResidual);
    maximumPressureImpulseClosureResidual = Math.max(
      maximumPressureImpulseClosureResidual,
      ...divergence.impulseClosureResidual.map(Math.abs),
      ...density.impulseClosureResidual.map(Math.abs));
    if (advection.minimumSignedDistance !== null) {
      minimumSignedDistanceAfterSubsteps = minimumSignedDistanceAfterSubsteps === null
        ? advection.minimumSignedDistance
        : Math.min(minimumSignedDistanceAfterSubsteps, advection.minimumSignedDistance);
    }
    contactCount += advection.contactCount;
    maximumRejectedEndpointPenetration = Math.max(maximumRejectedEndpointPenetration,
      advection.maximumRejectedEndpointPenetration);
    sweptSegments.push(...advection.sweptSegments);
    substeps.push(Object.freeze({ index: substep, timeStep: substepTimeStep,
      courantNumber, divergence, density, advection }));
  }
  return { substepCount, substepTimeStep, gravityImpulse, pressureBoundaryImpulse,
    contactBoundaryImpulse, allPressureSolvesConverged, maximumObservedCourantNumber,
    maximumDivergenceResidual, maximumDensityResidual,
    maximumPressureImpulseClosureResidual,
    minimumSignedDistanceAfterSubsteps, contactCount,
    maximumRejectedEndpointPenetration,
    sweptSegments: Object.freeze(sweptSegments), substeps: Object.freeze(substeps) };
}

export function stepLocalParticleLiquidCoreReference(core, steps = 1) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  if (!Number.isSafeInteger(steps) || steps < 0 || steps > 1000) {
    throw new RangeError('local particle Liquid steps must be an integer from zero through 1000');
  }
  const { state } = core;
  let divergence = null; let density = null;
  const stepFluidBoundaryImpulse = [0, 0, 0];
  const stepPressureBoundaryImpulse = [0, 0, 0];
  const stepContactBoundaryImpulse = [0, 0, 0];
  const allSubsteps = []; const allSweptSegments = [];
  let solverSubsteps = 0; let cflRetryCount = 0;
  let maximumObservedCourantNumber = 0; let allPressureSolvesConverged = true;
  let maximumDivergenceResidual = 0; let maximumDensityResidual = 0;
  let maximumPressureImpulseClosureResidual = 0;
  let minimumSignedDistanceAfterSubsteps = null; let contactCount = 0;
  let maximumRejectedEndpointPenetration = 0;
  for (let step = 0; step < steps; step += 1) {
    const cfl = localParticleLiquidCflSubstepsReference(core, state.timeStep);
    if (!cfl.withinSubstepLimit) {
      throw new RangeError(`liquid CFL needs ${cfl.requiredSubsteps} substeps; limit is ${core.maximumCflSubsteps}`);
    }
    const positionsBefore = state.positions.slice();
    const velocitiesBefore = state.velocities.slice();
    let substepCount = cfl.requiredSubsteps; let attempt;
    while (true) {
      state.positions.set(positionsBefore); state.velocities.set(velocitiesBefore);
      attempt = runLocalParticleLiquidSubsteps(core, state.timeStep, substepCount);
      if (attempt.maximumObservedCourantNumber
          <= core.maximumCourantNumber * (1 + 1e-10)) break;
      const refinedCount = Math.max(substepCount + 1, Math.ceil(substepCount
        * attempt.maximumObservedCourantNumber / core.maximumCourantNumber * (1 + 1e-10)));
      if (refinedCount > core.maximumCflSubsteps) {
        state.positions.set(positionsBefore); state.velocities.set(velocitiesBefore);
        throw new RangeError(`liquid CFL pressure correction needs ${refinedCount} substeps; limit is ${core.maximumCflSubsteps}`);
      }
      substepCount = refinedCount; cflRetryCount += 1;
    }
    solverSubsteps += attempt.substepCount;
    allPressureSolvesConverged = allPressureSolvesConverged
      && attempt.allPressureSolvesConverged;
    maximumObservedCourantNumber = Math.max(maximumObservedCourantNumber,
      attempt.maximumObservedCourantNumber);
    maximumDivergenceResidual = Math.max(maximumDivergenceResidual,
      attempt.maximumDivergenceResidual);
    maximumDensityResidual = Math.max(maximumDensityResidual,
      attempt.maximumDensityResidual);
    maximumPressureImpulseClosureResidual = Math.max(maximumPressureImpulseClosureResidual,
      attempt.maximumPressureImpulseClosureResidual);
    if (attempt.minimumSignedDistanceAfterSubsteps !== null) {
      minimumSignedDistanceAfterSubsteps = minimumSignedDistanceAfterSubsteps === null
        ? attempt.minimumSignedDistanceAfterSubsteps
        : Math.min(minimumSignedDistanceAfterSubsteps,
          attempt.minimumSignedDistanceAfterSubsteps);
    }
    contactCount += attempt.contactCount;
    maximumRejectedEndpointPenetration = Math.max(maximumRejectedEndpointPenetration,
      attempt.maximumRejectedEndpointPenetration);
    allSubsteps.push(...attempt.substeps); allSweptSegments.push(...attempt.sweptSegments);
    divergence = attempt.substeps.at(-1).divergence;
    density = attempt.substeps.at(-1).density;
    for (let axis = 0; axis < 3; axis += 1) {
      const pressureImpulse = attempt.pressureBoundaryImpulse[axis];
      const contactImpulse = attempt.contactBoundaryImpulse[axis];
      const boundaryImpulse = pressureImpulse + contactImpulse;
      core.accumulatedGravityImpulse[axis] += attempt.gravityImpulse[axis];
      core.accumulatedPressureBoundaryImpulse[axis] += pressureImpulse;
      core.accumulatedContactBoundaryImpulse[axis] += contactImpulse;
      core.accumulatedFluidBoundaryImpulse[axis] += boundaryImpulse;
      stepPressureBoundaryImpulse[axis] += pressureImpulse;
      stepContactBoundaryImpulse[axis] += contactImpulse;
      stepFluidBoundaryImpulse[axis] += boundaryImpulse;
    }
    state.steps += 1; core.time += state.timeStep; state.time = core.time;
  }
  const stepFlux = state.localFluxLedger ?? { appendedMass: 0, removedMass: 0 };
  state.conservation.total = sumParticleMass(resolveParticleMasses(core)
    ?? new Float64Array(state.count).fill(core.particleMass));
  state.conservation.error = Math.abs(state.conservation.total
    - (core.initialMass + stepFlux.appendedMass - stepFlux.removedMass));
  state.stateHash = hashPhysicsStateBufferReference(state);
  const densityReceipt = measureLocalParticleLiquidDensityReference(core);
  if (state.densities?.length === densityReceipt.densities.length) {
    state.densities.set(densityReceipt.densities);
  }
  const masses = resolveParticleMasses(core)
    ?? new Float64Array(state.count).fill(core.particleMass);
  const currentMass = sumParticleMass(masses);
  const currentMomentum = particleMomentum(state, masses);
  const flux = state.localFluxLedger ?? { appendedMass: 0, removedMass: 0,
    appendedMomentum: [0, 0, 0], removedMomentum: [0, 0, 0] };
  const expectedMass = core.initialMass + flux.appendedMass - flux.removedMass;
  const expectedMomentum = core.initialMomentum.map((component, axis) => component
    + flux.appendedMomentum[axis] - flux.removedMomentum[axis]
    + core.accumulatedGravityImpulse[axis]
    + core.accumulatedFluidBoundaryImpulse[axis]);
  const momentumResidual = currentMomentum.map(
    (component, axis) => component - expectedMomentum[axis]);
  const finalMaximumParticleSpeed = maximumParticleSpeed(state);
  core.metrics = Object.freeze({ kind: 'local-particle-liquid-step-reference:v1',
    steps: state.steps, time: core.time, divergence, density,
    allPressureSolvesConverged: steps === 0 ? true : allPressureSolvesConverged,
    solverSubsteps, cflRetryCount, maximumCourantNumber: core.maximumCourantNumber,
    maximumObservedCourantNumber, maximumDivergenceResidual, maximumDensityResidual,
    maximumPressureImpulseClosureResidual,
    minimumSignedDistanceAfterSubsteps, contactCount,
    maximumRejectedEndpointPenetration,
    maximumParticleSpeed: finalMaximumParticleSpeed,
    finalMaximumRelativeCompression: densityReceipt.maximumRelativeCompression,
    finalMaximumRelativeDensityError: densityReceipt.maximumRelativeError,
    initialMass: core.initialMass, currentMass, expectedMass,
    massResidual: currentMass - expectedMass,
    initialMomentum: core.initialMomentum,
    currentMomentum: Object.freeze(currentMomentum),
    accumulatedGravityImpulse: Object.freeze([...core.accumulatedGravityImpulse]),
    accumulatedFluidBoundaryImpulse:
      Object.freeze([...core.accumulatedFluidBoundaryImpulse]),
    accumulatedPressureBoundaryImpulse:
      Object.freeze([...core.accumulatedPressureBoundaryImpulse]),
    accumulatedContactBoundaryImpulse:
      Object.freeze([...core.accumulatedContactBoundaryImpulse]),
    stepFluidBoundaryImpulse: Object.freeze(stepFluidBoundaryImpulse),
    stepPressureBoundaryImpulse: Object.freeze(stepPressureBoundaryImpulse),
    stepContactBoundaryImpulse: Object.freeze(stepContactBoundaryImpulse),
    appendedMass: flux.appendedMass, removedMass: flux.removedMass,
    appendedMomentum: Object.freeze([...flux.appendedMomentum]),
    removedMomentum: Object.freeze([...flux.removedMomentum]),
    momentumResidual: Object.freeze(momentumResidual),
    conservationError: Math.max(Math.abs(currentMass - expectedMass),
      ...momentumResidual.map(Math.abs)),
    substeps: Object.freeze(allSubsteps),
    sweptSegments: Object.freeze(allSweptSegments),
    stateHash: state.stateHash });
  return core.metrics;
}

export function snapshotLocalParticleLiquidCoreReference(core) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  const { state } = core;
  const density = measureLocalParticleLiquidDensityReference(core);
  if (state.densities?.length === density.densities.length) state.densities.set(density.densities);
  const masses = resolveParticleMasses(core)
    ?? new Float64Array(state.count).fill(core.particleMass);
  let signedDistances = null; let minimumSignedDistance = null;
  if (core.dimension === 2 && core.solidBoundary?.closedPolygons.length > 0) {
    signedDistances = new Float64Array(state.count);
    minimumSignedDistance = Number.POSITIVE_INFINITY;
    for (let particle = 0; particle < state.count; particle += 1) {
      const offset = particle * 3;
      signedDistances[particle] = sampleAkinciLiquidSolidSignedDistanceReference(
        core.solidBoundary, [state.positions[offset], state.positions[offset + 2]]);
      minimumSignedDistance = Math.min(minimumSignedDistance, signedDistances[particle]);
    }
  }
  const flux = state.localFluxLedger ?? { appendedMass: 0, removedMass: 0,
    appendedMomentum: [0, 0, 0], removedMomentum: [0, 0, 0] };
  const currentMass = sumParticleMass(masses);
  const currentMomentum = particleMomentum(state, masses);
  state.stateHash = hashPhysicsStateBufferReference(state);
  const boundarySnapshot = core.solidBoundary === null ? null : Object.freeze({
    positions: contractDimensionVector(core.solidBoundary.positions, core.dimension),
    velocities: contractDimensionVector(core.solidBoundary.velocities, core.dimension),
    volumes: core.solidBoundary.volumes.slice(),
    supportRadius: core.solidBoundary.supportRadius,
    closedPolygon: core.solidBoundary.closedPolygon?.slice() ?? null,
    closedPolygons: Object.freeze(core.solidBoundary.closedPolygons
      .map((polygon) => polygon.slice())),
  });
  return Object.freeze({ kind: 'local-particle-liquid-snapshot-reference:v1',
    dimension: core.dimension, activeAxes: core.activeAxes, count: state.count,
    step: state.steps, time: core.time, stateHash: state.stateHash,
    restDensity: core.restDensity,
    positions: contractDimensionVector(state.positions, core.dimension),
    velocities: contractDimensionVector(state.velocities, core.dimension),
    densities: density.densities.slice(), masses: masses.slice(),
    supportRadii: state.supportRadii?.slice()
      ?? new Float64Array(state.count).fill(core.supportRadius),
    ids: state.ids?.slice() ?? null,
    stableIds: state.ids === undefined ? null : Object.freeze(Array.from(state.ids)),
    initialMass: core.initialMass, currentMass,
    currentMomentum: Object.freeze(currentMomentum),
    accumulatedGravityImpulse: Object.freeze([...core.accumulatedGravityImpulse]),
    accumulatedFluidBoundaryImpulse:
      Object.freeze([...core.accumulatedFluidBoundaryImpulse]),
    accumulatedPressureBoundaryImpulse:
      Object.freeze([...core.accumulatedPressureBoundaryImpulse]),
    accumulatedContactBoundaryImpulse:
      Object.freeze([...core.accumulatedContactBoundaryImpulse]),
    appendedMass: flux.appendedMass, removedMass: flux.removedMass,
    appendedMomentum: Object.freeze([...flux.appendedMomentum]),
    removedMomentum: Object.freeze([...flux.removedMomentum]),
    signedDistances, minimumSignedDistance,
    boundary: boundarySnapshot, solidBoundary: boundarySnapshot,
    density, diagnostics: core.metrics });
}

export function resetLocalParticleLiquidCoreReference(core) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  const { state, authoredState: authored } = core;
  state.positions = authored.positions.slice(); state.velocities = authored.velocities.slice();
  state.masses = authored.masses.slice(); state.supportRadii = authored.supportRadii.slice();
  state.densities = authored.densities.slice();
  if (authored.ids !== null) state.ids = authored.ids.slice();
  state.count = state.positions.length / 3; state.steps = 0; state.time = 0;
  state.conservation = { ...authored.conservation };
  state.localFluxLedger = { appendedMass: 0, removedMass: 0,
    appendedMomentum: [0, 0, 0], removedMomentum: [0, 0, 0] };
  state.vectorBytes = state.positions.byteLength + state.velocities.byteLength
    + (state.ids?.byteLength ?? 0);
  state.stateHash = hashPhysicsStateBufferReference(state);
  core.time = 0; core.accumulatedGravityImpulse.fill(0);
  core.accumulatedFluidBoundaryImpulse.fill(0);
  core.accumulatedPressureBoundaryImpulse.fill(0);
  core.accumulatedContactBoundaryImpulse.fill(0); core.metrics = null;
  return core;
}

export function splitConservativeLocalLiquidParticleReference({
  dimension = position?.length ?? 3, position, velocity, mass, volume,
  supportRadius, stableId = 0, solidBoundary = null,
} = {}) {
  requireDimension(dimension);
  requireAlignedDimensionVector('split particle position', position, dimension, 1);
  requireAlignedDimensionVector('split particle velocity', velocity, dimension, 1);
  requirePositiveFinite('split particle mass', mass);
  requirePositiveFinite('split particle volume', volume);
  requirePositiveFinite('split particle support radius', supportRadius);
  if ((!Number.isSafeInteger(stableId) && typeof stableId !== 'string')
      || (Number.isSafeInteger(stableId) && stableId < 0)) {
    throw new TypeError('split particle stable id must be a nonnegative integer or string');
  }
  const childCount = 2 ** dimension;
  const childMass = mass / childCount;
  const childVolume = volume / childCount;
  const childSupportRadius = supportRadius / 2;
  const offsetMagnitude = supportRadius / 8;
  const positions = new Float64Array(childCount * dimension);
  const velocities = new Float64Array(childCount * dimension);
  const masses = new Float64Array(childCount);
  const volumes = new Float64Array(childCount);
  const supportRadii = new Float64Array(childCount);
  const stableIds = new Array(childCount);
  for (let child = 0; child < childCount; child += 1) {
    const offset = child * dimension;
    for (let axis = 0; axis < dimension; axis += 1) {
      positions[offset + axis] = position[axis]
        + (child & (1 << axis) ? offsetMagnitude : -offsetMagnitude);
    }
    velocities.set(velocity, offset);
    masses[child] = childMass; volumes[child] = childVolume;
    supportRadii[child] = childSupportRadius;
    stableIds[child] = `${stableId}/split:${child}`;
  }
  if (solidBoundary !== null) {
    if (dimension !== 2 || solidBoundary?.dimension !== 2
        || solidBoundary.closedPolygons.length === 0) {
      throw new TypeError('solid-aware conservative split requires 2-D closed polygons');
    }
    for (let child = 0; child < childCount; child += 1) {
      const childOffset = child * dimension;
      if (sampleAkinciLiquidSolidSignedDistanceReference(solidBoundary,
        [positions[childOffset], positions[childOffset + 1]]) < 0) {
        throw new RangeError('conservative split refused: a child stencil enters the solid');
      }
    }
  }
  const childMomentum = new Array(dimension).fill(0);
  let resolvedMass = 0; let resolvedVolume = 0;
  for (let child = 0; child < childCount; child += 1) {
    resolvedMass += masses[child]; resolvedVolume += volumes[child];
    const offset = child * dimension;
    for (let axis = 0; axis < dimension; axis += 1) {
      childMomentum[axis] += masses[child] * velocities[offset + axis];
    }
  }
  const parentMomentum = velocity.map((component) => mass * component);
  return Object.freeze({ kind: 'conservative-local-liquid-particle-split-reference:v1',
    parent: Object.freeze({ position: Object.freeze(Array.from(position)),
      velocity: Object.freeze(Array.from(velocity)), mass, volume, supportRadius, stableId }),
    childCount, positions, velocities, masses, volumes, supportRadii,
    stableIds: Object.freeze(stableIds),
    receipt: Object.freeze({ massError: Math.abs(resolvedMass - mass),
      volumeError: Math.abs(resolvedVolume - volume),
      linearMomentumError: Object.freeze(parentMomentum.map(
        (component, axis) => childMomentum[axis] - component)),
      centerOfMassPreserved: true,
      dimension,
      triggerContract: Object.freeze(['free-surface', 'solid-proximity', 'high-strain']),
      solverTransitionSupported: false,
      coarseningImplemented: false }) });
}

export function classifyLocalParticleLiquidRefinementDemandReference(core, {
  freeSurfaceDensityRatio = 0.94,
  solidProximity = core?.supportRadius,
  highStrainRate = 20,
  minimumSupportRadius = core?.supportRadius / 4,
} = {}) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  if (!Number.isFinite(freeSurfaceDensityRatio) || freeSurfaceDensityRatio <= 0
      || freeSurfaceDensityRatio >= 1) {
    throw new RangeError('free-surface density ratio must be between zero and one');
  }
  requirePositiveFinite('solid refinement proximity', solidProximity);
  requirePositiveFinite('high-strain refinement rate', highStrainRate);
  requirePositiveFinite('minimum refined support radius', minimumSupportRadius);
  const { state } = core;
  const neighborhood = createCoreLiquidNeighborhood(core);
  const density = measureLocalParticleLiquidDensityReference(core, neighborhood);
  const reasonMasks = new Uint8Array(state.count);
  const solidDistances = core.solidBoundary === null ? null : new Float64Array(state.count);
  const maximumStrainRates = new Float64Array(state.count);
  for (let particle = 0; particle < state.count; particle += 1) {
    const support = state.supportRadii?.[particle] ?? core.supportRadius;
    if (support / 2 < minimumSupportRadius) continue;
    if (density.densities[particle] / core.restDensity < freeSurfaceDensityRatio) {
      reasonMasks[particle] |= 1;
    }
    const offset = particle * 3;
    if (core.solidBoundary !== null) {
      let distance;
      if (core.dimension === 2 && core.solidBoundary.closedPolygons.length > 0) {
        distance = sampleAkinciLiquidSolidSignedDistanceReference(core.solidBoundary,
          [state.positions[offset], state.positions[offset + 2]]);
      } else {
        distance = Number.POSITIVE_INFINITY;
        for (let boundaryParticle = 0;
          boundaryParticle < core.solidBoundary.count; boundaryParticle += 1) {
          const boundaryOffset = boundaryParticle * 3;
          distance = Math.min(distance, Math.hypot(
            state.positions[offset] - core.solidBoundary.positions[boundaryOffset],
            state.positions[offset + 1] - core.solidBoundary.positions[boundaryOffset + 1],
            state.positions[offset + 2] - core.solidBoundary.positions[boundaryOffset + 2]));
        }
      }
      solidDistances[particle] = distance;
      if (distance <= solidProximity) reasonMasks[particle] |= 2;
    }
    const begin = neighborhood.offsets[particle];
    const end = neighborhood.offsets[particle + 1];
    for (let cursor = begin; cursor < end; cursor += 1) {
      const neighbor = neighborhood.indices[cursor];
      if (neighbor === particle) continue;
      const neighborOffset = neighbor * 3;
      const distance = Math.hypot(
        state.positions[offset] - state.positions[neighborOffset],
        state.positions[offset + 1] - state.positions[neighborOffset + 1],
        state.positions[offset + 2] - state.positions[neighborOffset + 2]);
      if (distance <= EPSILON) continue;
      const relativeSpeed = Math.hypot(
        state.velocities[offset] - state.velocities[neighborOffset],
        state.velocities[offset + 1] - state.velocities[neighborOffset + 1],
        state.velocities[offset + 2] - state.velocities[neighborOffset + 2]);
      maximumStrainRates[particle] = Math.max(
        maximumStrainRates[particle], relativeSpeed / distance);
    }
    if (maximumStrainRates[particle] >= highStrainRate) reasonMasks[particle] |= 4;
  }
  const indices = [];
  for (let particle = 0; particle < state.count; particle += 1) {
    if (reasonMasks[particle] !== 0) indices.push(particle);
  }
  return Object.freeze({ kind: 'local-particle-liquid-refinement-demand-reference:v1',
    indices: Object.freeze(indices), reasonMasks, solidDistances,
    maximumStrainRates, densities: density.densities,
    reasonBits: Object.freeze({ freeSurface: 1, solidProximity: 2, highStrain: 4 }),
    thresholds: Object.freeze({ freeSurfaceDensityRatio, solidProximity,
      highStrainRate, minimumSupportRadius }) });
}

export function refineLocalParticleLiquidStateReference(core, demand, {
  maximumSplits = 256,
} = {}) {
  if (core?.kind !== 'local-particle-liquid-core-reference:v1') {
    throw new TypeError('local particle Liquid core required');
  }
  if (demand?.kind !== 'local-particle-liquid-refinement-demand-reference:v1') {
    throw new TypeError('local particle Liquid refinement demand required');
  }
  if (!Number.isSafeInteger(maximumSplits) || maximumSplits < 0 || maximumSplits > 65536) {
    throw new RangeError('maximum local liquid splits must be an integer from zero through 65536');
  }
  const { state, dimension } = core;
  const selected = demand.indices.slice(0, maximumSplits);
  if (!selected.every((particle, index) => Number.isSafeInteger(particle)
      && particle >= 0 && particle < state.count
      && (index === 0 || particle > selected[index - 1]))) {
    throw new TypeError('refinement indices must be unique ascending particle indices');
  }
  const selectedSet = new Set(selected);
  if (selected.length > 0) {
    const error = new RangeError('adaptive liquid refinement is gated until a density-consistent 2:1 transition stencil is implemented');
    error.code = 'VF_LIQUID_REFINEMENT_TRANSITION_UNSUPPORTED';
    throw error;
  }
  const nextPositions = []; const nextVelocities = [];
  const nextMasses = []; const nextSupportRadii = []; const nextIds = [];
  let nextId = state.ids === undefined ? 0 : Math.max(...state.ids) + 1;
  const massBefore = sumParticleMass(state.masses);
  const momentumBefore = particleMomentum(state, state.masses);
  for (let particle = 0; particle < state.count; particle += 1) {
    const offset = particle * 3;
    const position = dimension === 2
      ? [state.positions[offset], state.positions[offset + 2]]
      : Array.from(state.positions.subarray(offset, offset + 3));
    const velocity = dimension === 2
      ? [state.velocities[offset], state.velocities[offset + 2]]
      : Array.from(state.velocities.subarray(offset, offset + 3));
    if (!selectedSet.has(particle)) {
      nextPositions.push(...position); nextVelocities.push(...velocity);
      nextMasses.push(state.masses[particle]);
      nextSupportRadii.push(state.supportRadii[particle]);
      nextIds.push(state.ids?.[particle] ?? nextId++);
      continue;
    }
    const split = splitConservativeLocalLiquidParticleReference({ dimension,
      position, velocity, mass: state.masses[particle],
      volume: state.masses[particle] / core.restDensity,
      supportRadius: state.supportRadii[particle], stableId: state.ids?.[particle] ?? particle,
      solidBoundary: core.solidBoundary });
    nextPositions.push(...split.positions); nextVelocities.push(...split.velocities);
    nextMasses.push(...split.masses); nextSupportRadii.push(...split.supportRadii);
    for (let child = 0; child < split.childCount; child += 1) nextIds.push(nextId++);
  }
  state.count = nextMasses.length;
  state.positions = expandDimensionVector(nextPositions, dimension);
  state.velocities = expandDimensionVector(nextVelocities, dimension);
  state.masses = Float64Array.from(nextMasses);
  state.supportRadii = Float64Array.from(nextSupportRadii);
  state.densities = new Float64Array(state.count);
  state.ids = Uint32Array.from(nextIds);
  state.vectorBytes = state.positions.byteLength + state.velocities.byteLength
    + state.ids.byteLength;
  state.conservation.total = sumParticleMass(state.masses);
  state.conservation.error = Math.abs(state.conservation.total - massBefore);
  state.stateHash = hashPhysicsStateBufferReference(state);
  const momentumAfter = particleMomentum(state, state.masses);
  const receipt = Object.freeze({ kind: 'local-particle-liquid-refinement-reference:v1',
    splitCount: selected.length, previousCount: state.count
      - selected.length * (2 ** dimension - 1), count: state.count,
    dimension, childCountPerSplit: 2 ** dimension,
    massError: state.conservation.error,
    linearMomentumError: Object.freeze(momentumAfter.map(
      (component, axis) => component - momentumBefore[axis])),
    sourceFluxChanged: false, selectedIndices: Object.freeze(selected),
    variableSupportRadius: true, coarseningImplemented: false });
  core.refinementReceipt = receipt;
  return receipt;
}
