const UINT32_BYTES = 4;

export function hashDynamicParticleGridCoordinate2D([x, y], bucketMask) {
  if (![x, y].every((value) => Number.isInteger(value)
      && value >= -0x80000000 && value <= 0x7fffffff)) {
    throw new RangeError('Dynamic particle grid coordinates must fit signed i32');
  }
  if (!Number.isInteger(bucketMask) || bucketMask < 0
      || ((bucketMask + 1) & bucketMask) !== 0) {
    throw new RangeError('Dynamic particle grid bucket mask must describe a power of two');
  }
  let hash = (Math.imul(x | 0, 0x8da6b343)
    ^ Math.imul(y | 0, 0xd8163841)) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash = (hash ^ (hash >>> 15)) >>> 0;
  return hash & bucketMask;
}

const nextPowerOfTwo = (value) => {
  let capacity = 1;
  while (capacity < value) capacity *= 2;
  return capacity;
};

// Production particle-neighborhood allocation. Capacity follows simulated
// material, never the enclosing world volume. Hash collisions share a bucket,
// but consumers must retain exact signed cell identity while traversing it.
export function createDynamicParticleGridGpuPlan({
  particleCapacity,
  maximumParticlesPerCell,
  expectedParticlesPerCell = 1,
  targetBucketLoad = 0.5,
  minimumBucketCapacity = 16,
  telemetryAtomicCount = 0,
} = {}) {
  for (const [name, value] of [
    ['particleCapacity', particleCapacity],
    ['maximumParticlesPerCell', maximumParticlesPerCell],
    ['minimumBucketCapacity', minimumBucketCapacity],
    ['telemetryAtomicCount', telemetryAtomicCount],
  ]) {
    if (!Number.isSafeInteger(value) || value < (name === 'telemetryAtomicCount' ? 0 : 1)) {
      throw new RangeError(`Dynamic particle grid ${name} is invalid`);
    }
  }
  if (!Number.isFinite(expectedParticlesPerCell) || expectedParticlesPerCell <= 0) {
    throw new RangeError('Dynamic particle grid expectedParticlesPerCell must be positive');
  }
  if (!Number.isFinite(targetBucketLoad) || targetBucketLoad <= 0
      || targetBucketLoad > 0.75) {
    throw new RangeError('Dynamic particle grid targetBucketLoad must be above zero through 0.75');
  }
  const requiredBuckets = Math.ceil(
    particleCapacity / expectedParticlesPerCell / targetBucketLoad);
  const bucketCapacity = nextPowerOfTwo(Math.max(minimumBucketCapacity, requiredBuckets));
  if (!Number.isSafeInteger(bucketCapacity) || bucketCapacity > 0x40000000) {
    throw new RangeError('Dynamic particle grid bucket capacity exceeds u32 indexing');
  }
  const itemCapacity = bucketCapacity * maximumParticlesPerCell;
  const countAtomicCount = bucketCapacity + telemetryAtomicCount;
  if (!Number.isSafeInteger(itemCapacity) || !Number.isSafeInteger(countAtomicCount)) {
    throw new RangeError('Dynamic particle grid storage exceeds safe indexing');
  }
  const bytes = Object.freeze({
    countsAndTelemetry: countAtomicCount * UINT32_BYTES,
    items: itemCapacity * UINT32_BYTES,
  });
  return Object.freeze({
    kind: 'dynamic-particle-grid-gpu-plan:v1',
    particleCapacity,
    maximumParticlesPerCell,
    expectedParticlesPerCell,
    targetBucketLoad,
    bucketCapacity,
    bucketMask: bucketCapacity - 1,
    itemCapacity,
    countAtomicCount,
    telemetryAtomicCount,
    telemetryByteOffset: bucketCapacity * UINT32_BYTES,
    bytes,
    residentBytes: bytes.countsAndTelemetry + bytes.items,
    allocationBasis: 'particle-capacity',
    worldExtentIndependent: true,
    coordinateIdentity: 'exact-signed-i32',
    collisionPolicy: 'hash-bucket+exact-coordinate-filter+overflow-receipt',
    routineReadbackBytes: 0,
  });
}

export function createDynamicParticleGridWgsl2D({
  originExpression,
  cellSizeExpression,
  particleExpression,
} = {}) {
  for (const [name, value] of Object.entries({
    originExpression, cellSizeExpression, particleExpression,
  })) {
    if (typeof value !== 'string' || value.length === 0) {
      throw new TypeError(`Dynamic particle grid ${name} must be WGSL source`);
    }
  }
  return /* wgsl */`
fn cell_coordinate(position: vec2<f32>) -> vec2<i32> {
  return vec2<i32>(floor((position - ${originExpression}) / ${cellSizeExpression}));
}

fn valid_cell(cell: vec2<i32>) -> bool {
  return true;
}

fn cell_index(cell: vec2<i32>) -> u32 {
  var hash = bitcast<u32>(cell.x) * 0x8da6b343u;
  hash = hash ^ (bitcast<u32>(cell.y) * 0xd8163841u);
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x7feb352du;
  hash = hash ^ (hash >> 15u);
  return hash & (params.counts.y - 1u);
}

fn particle_occupies_cell(index: u32, cell: vec2<i32>) -> bool {
  return all(cell_coordinate(${particleExpression}) == cell);
}
`;
}

export function createDynamicParticleGridAuditWgsl2D({
  particleExpression,
  occupiedBucketTelemetryOffset,
  activeCellTelemetryOffset,
} = {}) {
  if (typeof particleExpression !== 'string' || particleExpression.length === 0) {
    throw new TypeError('Dynamic particle grid audit particleExpression must be WGSL source');
  }
  for (const [name, value] of Object.entries({
    occupiedBucketTelemetryOffset, activeCellTelemetryOffset,
  })) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`Dynamic particle grid audit ${name} is invalid`);
    }
  }
  return /* wgsl */`
fn dynamic_grid_particle_coordinate(index: u32) -> vec2<i32> {
  return cell_coordinate(${particleExpression});
}

fn dynamic_grid_item_precedes(left: u32, right: u32) -> bool {
  let left_cell = dynamic_grid_particle_coordinate(left);
  let right_cell = dynamic_grid_particle_coordinate(right);
  if (left_cell.y != right_cell.y) { return left_cell.y < right_cell.y; }
  if (left_cell.x != right_cell.x) { return left_cell.x < right_cell.x; }
  return left < right;
}

fn audit_sorted_bucket(bucket: u32, count: u32) {
  if (count == 0u) { return; }
  atomicAdd(&cell_counts[telemetry_base() + ${occupiedBucketTelemetryOffset}u], 1u);
  let base = bucket * params.counts.w;
  var exact_cells = 1u;
  var previous_coordinate = dynamic_grid_particle_coordinate(
    atomicLoad(&cell_items[base]));
  for (var index = 0u; index < count; index = index + 1u) {
    let particle = atomicLoad(&cell_items[base + index]);
    if (particle >= params.counts.x) { continue; }
    let coordinate = dynamic_grid_particle_coordinate(particle);
    if (index > 0u && any(coordinate != previous_coordinate)) {
      exact_cells = exact_cells + 1u;
    }
    previous_coordinate = coordinate;
  }
  atomicAdd(&cell_counts[telemetry_base() + ${activeCellTelemetryOffset}u], exact_cells);
}
`;
}

export function auditDynamicParticleGrid2D({
  positions,
  particleCount,
  stride,
  origin = [0, 0],
  cellSize,
  plan,
} = {}) {
  if (plan?.kind !== 'dynamic-particle-grid-gpu-plan:v1') {
    throw new TypeError('Dynamic particle grid plan required');
  }
  if (!ArrayBuffer.isView(positions) || !Number.isSafeInteger(particleCount)
      || particleCount < 0 || particleCount > plan.particleCapacity
      || !Number.isSafeInteger(stride) || stride < 2
      || positions.length < particleCount * stride) {
    throw new RangeError('Dynamic particle grid audit particle layout is invalid');
  }
  if ((!Array.isArray(origin) && !ArrayBuffer.isView(origin)) || origin.length !== 2
      || !Array.from(origin).every(Number.isFinite)
      || !Number.isFinite(cellSize) || cellSize <= 0) {
    throw new RangeError('Dynamic particle grid audit transform is invalid');
  }
  const buckets = Array.from({ length: plan.bucketCapacity }, () => []);
  const exactCoordinates = new Set();
  for (let particle = 0; particle < particleCount; particle += 1) {
    const offset = particle * stride;
    const coordinate = [0, 1].map((axis) => Math.floor(
      (positions[offset + axis] - origin[axis]) / cellSize));
    exactCoordinates.add(coordinate.join(','));
    buckets[hashDynamicParticleGridCoordinate2D(
      coordinate, plan.bucketMask)].push({ particle, coordinate });
  }
  let occupiedBuckets = 0;
  let retainedActiveCells = 0;
  let peakBucketOccupancy = 0;
  let overflowCount = 0;
  for (const bucket of buckets) {
    if (bucket.length === 0) continue;
    occupiedBuckets += 1;
    peakBucketOccupancy = Math.max(peakBucketOccupancy, bucket.length);
    overflowCount += Math.max(0, bucket.length - plan.maximumParticlesPerCell);
    bucket.sort((left, right) => left.particle - right.particle);
    retainedActiveCells += new Set(bucket.slice(0, plan.maximumParticlesPerCell)
      .map(({ coordinate }) => coordinate.join(','))).size;
  }
  const activeCells = exactCoordinates.size;
  const complete = overflowCount === 0 && retainedActiveCells === activeCells;
  return Object.freeze({
    kind: 'dynamic-grid-audit:v1',
    particleCount,
    activeCells,
    retainedActiveCells,
    occupiedBuckets,
    allocatedGridCells: plan.bucketCapacity,
    gridHashCollisionCount: activeCells - occupiedBuckets,
    gridCapacityUtilization: activeCells / plan.bucketCapacity,
    peakBucketOccupancy,
    overflowCount,
    complete,
  });
}
