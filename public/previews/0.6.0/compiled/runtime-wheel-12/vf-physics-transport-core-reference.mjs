const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

const hashBytes = (values) => {
  let hash = FNV_OFFSET;
  for (const value of values) { hash ^= value; hash = Math.imul(hash, FNV_PRIME); }
  return (hash >>> 0).toString(16).padStart(8, '0');
};

const requireVector = (name, value) => {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) {
    throw new TypeError(`${name} must be a finite three-vector`);
  }
  return Object.freeze([...value]);
};

export function hashPhysicsStateBufferReference(state) {
  let hash = FNV_OFFSET;
  for (const values of [state.positions, state.velocities]) {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    for (const value of bytes) { hash ^= value; hash = Math.imul(hash, FNV_PRIME); }
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Canonical data-only Transport boundary packet for an oriented open 2-D
 * polyline. Runtime adapters copy the same f32 vertices; neither a GPU solver
 * nor a renderer is allowed to reconstruct a separate analytic boundary.
 */
export function createTransportPolylineBoundaryPacketReference({
  points, stableVertexIds = null, approximationTolerance = 0,
  orientation = 'left-to-right-solid-below',
} = {}) {
  if ((!Array.isArray(points) && !ArrayBuffer.isView(points))
      || points.length < 4 || points.length % 2 !== 0
      || !Array.from(points).every(Number.isFinite)) {
    throw new TypeError('Transport polyline boundary points must contain finite 2D vertices');
  }
  if (orientation !== 'left-to-right-solid-below') {
    throw new RangeError('Transport polyline boundary orientation is unsupported');
  }
  if (!Number.isFinite(approximationTolerance) || approximationTolerance < 0) {
    throw new RangeError('Transport polyline approximation tolerance must be nonnegative');
  }
  const vertices = Float32Array.from(points);
  const vertexCount = vertices.length / 2;
  for (let vertex = 1; vertex < vertexCount; vertex += 1) {
    if (!(vertices[vertex * 2] > vertices[(vertex - 1) * 2])) {
      throw new RangeError('Transport solid-below polyline x coordinates must strictly increase');
    }
  }
  const minimumX = vertices[0];
  const maximumX = vertices[(vertexCount - 1) * 2];
  const uniformStepX = (maximumX - minimumX) / (vertexCount - 1);
  let maximumUniformXError = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    maximumUniformXError = Math.max(maximumUniformXError,
      Math.abs(vertices[vertex * 2] - (minimumX + vertex * uniformStepX)));
  }
  const uniformXTolerance = Math.max(1e-7,
    16 * Number.EPSILON * Math.max(1, Math.abs(minimumX), Math.abs(maximumX)));
  const uniformX = maximumUniformXError <= uniformXTolerance;
  const ids = stableVertexIds === null
    ? Array.from({ length: vertexCount }, (_, index) => index)
    : Array.from(stableVertexIds);
  if (ids.length !== vertexCount || !ids.every((value, index) =>
    Number.isSafeInteger(value) && value >= 0 && (index === 0 || value > ids[index - 1]))) {
    throw new TypeError('Transport polyline stable vertex IDs must be unique ascending integers');
  }
  const stateHash = hashBytes(new Uint8Array(vertices.buffer));
  return Object.freeze({
    kind: 'transport-polyline-boundary-packet-reference:v1',
    dimension: 2,
    orientation,
    openEnds: true,
    vertexCount,
    segmentCount: vertexCount - 1,
    minimumX,
    maximumX,
    uniformX,
    uniformStepX,
    maximumUniformXError,
    stableVertexIds: Object.freeze(ids),
    stableSegmentIds: Object.freeze(Array.from(
      { length: vertexCount - 1 }, (_, index) => index)),
    approximationTolerance,
    stateHash,
    copyVertices: () => vertices.slice(),
  });
}

const TRANSPORT_POLYLINE_SWEEP_EPSILON = 256 * Number.EPSILON;

const transportPolylineRequireVector2 = (name, value) => {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value))
      || value.length !== 2 || !Array.from(value).every(Number.isFinite)) {
    throw new TypeError(`${name} must be a finite two-vector`);
  }
  return [Number(value[0]), Number(value[1])];
};

const transportPolylineRequirePacket = (packet) => {
  if (packet?.kind !== 'transport-polyline-boundary-packet-reference:v1'
      || packet.dimension !== 2 || packet.openEnds !== true
      || typeof packet.copyVertices !== 'function') {
    throw new TypeError('Transport polyline boundary packet required');
  }
};

const transportPolylineDot2 = (left, right) => left[0] * right[0] + left[1] * right[1];

const transportPolylineCanonicalVector2 = (value) => [
  Object.is(value[0], -0) ? 0 : value[0],
  Object.is(value[1], -0) ? 0 : value[1],
];

const transportPolylineFeatureOrder = (feature) => feature.featureKind === 'vertex-endcap'
  ? feature.featureIndex * 2 : feature.featureIndex * 2 + 1;

const transportPolylineCompareFeatures = (left, right) =>
  transportPolylineFeatureOrder(left) - transportPolylineFeatureOrder(right);

const transportPolylineFeatureIdentity = (packet, featureKind, featureIndex) => {
  const vertex = featureKind === 'vertex-endcap';
  const stableId = vertex
    ? packet.stableVertexIds[featureIndex]
    : packet.stableSegmentIds[featureIndex];
  return {
    featureKind,
    featureIndex,
    stableFeatureId: stableId,
    featureKey: `${vertex ? 'vertex' : 'segment'}:${stableId}`,
  };
};

// Geometry tolerances scale with geometry. A unit floor makes the same model
// expressed in millimetres behave differently from one expressed in metres,
// and is especially destructive for the quadratic endcap solve below.
const transportPolylineScale = (...values) => Math.max(Number.MIN_VALUE,
  ...values.flat().filter(Number.isFinite).map(Math.abs));

const transportPolylineTolerances = (start, end, radius, vertices) => {
  const spatialScale = transportPolylineScale(start, end, radius, vertices);
  const displacement = [end[0] - start[0], end[1] - start[1]];
  const travel = Math.hypot(...displacement);
  const spatial = TRANSPORT_POLYLINE_SWEEP_EPSILON * spatialScale;
  return {
    spatial,
    squared: TRANSPORT_POLYLINE_SWEEP_EPSILON * spatialScale * spatialScale,
    // dot(displacement, unitNormal) has units L. Include cancellation error
    // from forming end-start at the ambient coordinate scale plus travel error.
    direction: spatial + TRANSPORT_POLYLINE_SWEEP_EPSILON * travel,
    fraction: TRANSPORT_POLYLINE_SWEEP_EPSILON
      * Math.max(1, spatialScale / Math.max(travel, spatial)),
  };
};

const transportPolylineSegment = (vertices, index) => {
  const from = [vertices[index * 2], vertices[index * 2 + 1]];
  const to = [vertices[(index + 1) * 2], vertices[(index + 1) * 2 + 1]];
  const edge = [to[0] - from[0], to[1] - from[1]];
  const length = Math.hypot(...edge);
  const tangent = [edge[0] / length, edge[1] / length];
  return { from, to, length, tangent, orientedNormal: [-tangent[1], tangent[0]] };
};

const transportPolylineVertexFallbackNormal = (vertices, vertexIndex) => {
  const vertexCount = vertices.length / 2;
  const normals = [];
  if (vertexIndex > 0) {
    normals.push(transportPolylineSegment(vertices, vertexIndex - 1).orientedNormal);
  }
  if (vertexIndex + 1 < vertexCount) {
    normals.push(transportPolylineSegment(vertices, vertexIndex).orientedNormal);
  }
  const summed = normals.reduce((value, normal) =>
    [value[0] + normal[0], value[1] + normal[1]], [0, 0]);
  const length = Math.hypot(...summed);
  return length > TRANSPORT_POLYLINE_SWEEP_EPSILON
    ? [summed[0] / length, summed[1] / length]
    : [...normals[0]];
};

const transportPolylineFreezeFeature = (feature) => Object.freeze({
  featureKind: feature.featureKind,
  featureIndex: feature.featureIndex,
  stableFeatureId: feature.stableFeatureId,
  featureKey: feature.featureKey,
  closestPoint: Object.freeze(transportPolylineCanonicalVector2(feature.closestPoint)),
  outwardNormal: Object.freeze(transportPolylineCanonicalVector2(feature.outwardNormal)),
});

const transportPolylineUniqueNormals = (features, tolerance) => {
  const normals = [];
  for (const feature of features) {
    const normal = feature.outwardNormal;
    if (normals.some((existing) =>
      (existing[0] - normal[0]) ** 2 + (existing[1] - normal[1]) ** 2
        <= tolerance ** 2)) continue;
    normals.push(Object.freeze(transportPolylineCanonicalVector2(normal)));
  }
  return Object.freeze(normals);
};

const transportPolylineClosestFeatures = (packet, vertices, point, tolerances) => {
  const candidates = [];
  for (let segmentIndex = 0; segmentIndex < packet.segmentCount; segmentIndex += 1) {
    const segment = transportPolylineSegment(vertices, segmentIndex);
    const relative = [point[0] - segment.from[0], point[1] - segment.from[1]];
    const along = transportPolylineDot2(relative, segment.tangent);
    if (along > 0 && along < segment.length) {
      const closestPoint = [segment.from[0] + segment.tangent[0] * along,
        segment.from[1] + segment.tangent[1] * along];
      const delta = [point[0] - closestPoint[0], point[1] - closestPoint[1]];
      const distance = Math.hypot(...delta);
      candidates.push({
        ...transportPolylineFeatureIdentity(packet, 'segment-body', segmentIndex),
        squaredDistance: distance ** 2,
        closestPoint,
        outwardNormal: distance > tolerances.spatial
          ? [delta[0] / distance, delta[1] / distance]
          : segment.orientedNormal,
      });
    }
  }
  for (let vertexIndex = 0; vertexIndex < packet.vertexCount; vertexIndex += 1) {
    const closestPoint = [vertices[vertexIndex * 2], vertices[vertexIndex * 2 + 1]];
    const delta = [point[0] - closestPoint[0], point[1] - closestPoint[1]];
    const distance = Math.hypot(...delta);
    candidates.push({
      ...transportPolylineFeatureIdentity(packet, 'vertex-endcap', vertexIndex),
      squaredDistance: distance ** 2,
      closestPoint,
      outwardNormal: distance > tolerances.spatial
        ? [delta[0] / distance, delta[1] / distance]
        : transportPolylineVertexFallbackNormal(vertices, vertexIndex),
    });
  }
  const minimumSquared = candidates.reduce((minimum, candidate) =>
    Math.min(minimum, candidate.squaredDistance), Number.POSITIVE_INFINITY);
  return candidates
    .filter((candidate) => candidate.squaredDistance <= minimumSquared + tolerances.squared)
    .sort(transportPolylineCompareFeatures);
};

/**
 * Exact closest-feature clearance for a disk against the finite capsule union
 * of an open Transport polyline. Beyond either endpoint only that endpoint's
 * circular cap exists: no last-segment half-plane is extrapolated.
 */
export function sampleDiskAgainstTransportPolylineBoundaryReference(packet, center,
  { radius } = {}) {
  transportPolylineRequirePacket(packet);
  const point = transportPolylineRequireVector2('Transport disk center', center);
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('Transport disk radius must be positive');
  }
  const vertices = packet.copyVertices();
  const tolerances = transportPolylineTolerances(point, point, radius, vertices);
  const closest = transportPolylineClosestFeatures(packet, vertices, point, tolerances);
  const primary = closest[0];
  const distance = Math.sqrt(Math.max(0, primary.squaredDistance));
  const clearance = distance - radius;
  const overlap = clearance < -tolerances.spatial;
  const exactContact = Math.abs(clearance) <= tolerances.spatial;
  const projectedCenter = [primary.closestPoint[0] + primary.outwardNormal[0] * radius,
    primary.closestPoint[1] + primary.outwardNormal[1] * radius];
  const activeFeatures = Object.freeze(closest.map(transportPolylineFreezeFeature));
  return Object.freeze({
    kind: 'transport-polyline-disk-sample-reference:v1',
    center: Object.freeze(point),
    radius,
    signedClearance: clearance,
    minimumClearance: clearance,
    overlap,
    exactContact,
    penetrationDepth: overlap ? -clearance : 0,
    featureKind: primary.featureKind,
    featureIndex: primary.featureIndex,
    stableFeatureId: primary.stableFeatureId,
    featureKey: primary.featureKey,
    closestPoint: Object.freeze([...primary.closestPoint]),
    outwardNormal: Object.freeze([...primary.outwardNormal]),
    projectedCenter: Object.freeze(projectedCenter),
    activeFeatures,
    activeNormals: transportPolylineUniqueNormals(closest, tolerances.spatial),
    tolerance: tolerances.spatial,
  });
}

const transportPolylineLowerBound = (vertices, target) => {
  let lower = 0; let upper = vertices.length / 2;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (vertices[middle * 2] < target) lower = middle + 1;
    else upper = middle;
  }
  return lower;
};

const transportPolylineUpperBound = (vertices, target) => {
  let lower = 0; let upper = vertices.length / 2;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (vertices[middle * 2] <= target) lower = middle + 1;
    else upper = middle;
  }
  return lower;
};

/** Candidate features whose x extent can intersect the swept disk AABB. */
export function transportPolylineDiskSweepCandidateRangeReference(packet, start, end,
  { radius, search = 'bounded' } = {}) {
  transportPolylineRequirePacket(packet);
  const from = transportPolylineRequireVector2('Transport disk sweep start', start);
  const to = transportPolylineRequireVector2('Transport disk sweep end', end);
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('Transport disk radius must be positive');
  }
  if (search !== 'bounded' && search !== 'full') {
    throw new RangeError('Transport polyline sweep search must be bounded or full');
  }
  const vertices = packet.copyVertices();
  if (search === 'full') return Object.freeze({
    kind: 'transport-polyline-disk-candidate-range-reference:v1',
    search,
    empty: false,
    firstSegment: 0,
    lastSegment: packet.segmentCount - 1,
    firstVertex: 0,
    lastVertex: packet.vertexCount - 1,
  });
  const tolerances = transportPolylineTolerances(from, to, radius, vertices);
  const lowX = Math.min(from[0], to[0]) - radius - tolerances.spatial;
  const highX = Math.max(from[0], to[0]) + radius + tolerances.spatial;
  let firstSegment; let lastSegment; let rangeStrategy;
  if (packet.uniformX && packet.uniformStepX > 0) {
    const error = packet.maximumUniformXError + tolerances.spatial;
    firstSegment = Math.ceil((lowX - error - packet.minimumX) / packet.uniformStepX) - 1;
    lastSegment = Math.floor((highX + error - packet.minimumX) / packet.uniformStepX);
    rangeStrategy = 'uniform-x';
  } else {
    firstSegment = transportPolylineLowerBound(vertices, lowX) - 1;
    lastSegment = transportPolylineUpperBound(vertices, highX) - 1;
    rangeStrategy = 'binary-x';
  }
  firstSegment = Math.max(0, firstSegment);
  lastSegment = Math.min(packet.segmentCount - 1, lastSegment);
  const empty = lastSegment < firstSegment || highX < packet.minimumX - tolerances.spatial
    || lowX > packet.maximumX + tolerances.spatial;
  return Object.freeze({
    kind: 'transport-polyline-disk-candidate-range-reference:v1',
    search,
    rangeStrategy,
    empty,
    firstSegment: empty ? null : firstSegment,
    lastSegment: empty ? null : lastSegment,
    firstVertex: empty ? null : firstSegment,
    lastVertex: empty ? null : lastSegment + 1,
  });
}

const transportPolylinePointSegmentDistanceSquared = (point, from, to) => {
  const edge = [to[0] - from[0], to[1] - from[1]];
  const denominator = transportPolylineDot2(edge, edge);
  if (denominator === 0) {
    return (point[0] - from[0]) ** 2 + (point[1] - from[1]) ** 2;
  }
  const fraction = Math.max(0, Math.min(1,
    transportPolylineDot2([point[0] - from[0], point[1] - from[1]], edge) / denominator));
  const closest = [from[0] + edge[0] * fraction, from[1] + edge[1] * fraction];
  return (point[0] - closest[0]) ** 2 + (point[1] - closest[1]) ** 2;
};

const transportPolylineCross2 = (left, right) => left[0] * right[1] - left[1] * right[0];

const transportPolylineSegmentsIntersect = (a, b, c, d, spatialTolerance) => {
  const ab = [b[0] - a[0], b[1] - a[1]];
  const cd = [d[0] - c[0], d[1] - c[1]];
  const ac = [c[0] - a[0], c[1] - a[1]];
  const abLength = Math.hypot(...ab);
  const cdLength = Math.hypot(...cd);
  const denominator = transportPolylineCross2(ab, cd);
  // Cross products have units L^2. Comparing one directly with a spatial
  // tolerance (L) made small, parallel, separated segments appear to cross.
  const orientationTolerance = TRANSPORT_POLYLINE_SWEEP_EPSILON
    * Math.max(abLength * cdLength, Number.MIN_VALUE);
  if (Math.abs(denominator) <= orientationTolerance) {
    const collinearityTolerance = spatialTolerance * abLength;
    if (Math.abs(transportPolylineCross2(ac, ab)) > collinearityTolerance) return false;
    const dominant = Math.abs(ab[0]) >= Math.abs(ab[1]) ? 0 : 1;
    const a0 = Math.min(a[dominant], b[dominant]);
    const a1 = Math.max(a[dominant], b[dominant]);
    const c0 = Math.min(c[dominant], d[dominant]);
    const c1 = Math.max(c[dominant], d[dominant]);
    return Math.max(a0, c0) <= Math.min(a1, c1) + spatialTolerance;
  }
  const t = transportPolylineCross2(ac, cd) / denominator;
  const u = transportPolylineCross2(ac, ab) / denominator;
  const fractionTolerance = spatialTolerance
    / Math.max(Math.min(abLength, cdLength), spatialTolerance);
  return t >= -fractionTolerance && t <= 1 + fractionTolerance
    && u >= -fractionTolerance && u <= 1 + fractionTolerance;
};

const transportPolylineSegmentSegmentDistanceSquared = (a, b, c, d, tolerance) => {
  const abSquared = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2;
  const cdSquared = (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2;
  if (abSquared === 0) return transportPolylinePointSegmentDistanceSquared(a, c, d);
  if (cdSquared === 0) return transportPolylinePointSegmentDistanceSquared(c, a, b);
  if (transportPolylineSegmentsIntersect(a, b, c, d, tolerance)) return 0;
  return Math.min(
    transportPolylinePointSegmentDistanceSquared(a, c, d),
    transportPolylinePointSegmentDistanceSquared(b, c, d),
    transportPolylinePointSegmentDistanceSquared(c, a, b),
    transportPolylinePointSegmentDistanceSquared(d, a, b),
  );
};

const transportPolylineMinimumSweepClearance = (vertices, start, end, radius, tolerance) => {
  let minimumSquared = Number.POSITIVE_INFINITY;
  for (let segmentIndex = 0; segmentIndex < vertices.length / 2 - 1; segmentIndex += 1) {
    const from = [vertices[segmentIndex * 2], vertices[segmentIndex * 2 + 1]];
    const to = [vertices[(segmentIndex + 1) * 2], vertices[(segmentIndex + 1) * 2 + 1]];
    minimumSquared = Math.min(minimumSquared,
      transportPolylineSegmentSegmentDistanceSquared(start, end, from, to, tolerance));
  }
  const clearance = Math.sqrt(Math.max(0, minimumSquared)) - radius;
  return Math.abs(clearance) <= tolerance ? 0 : clearance;
};

const transportPolylineStableCircleRoots = (a, b, c) => {
  if (a === 0) return [];
  let discriminant = b * b - a * c;
  const discriminantTolerance = TRANSPORT_POLYLINE_SWEEP_EPSILON
    * Math.max(Number.MIN_VALUE, b * b, Math.abs(a * c));
  if (discriminant < -discriminantTolerance) return [];
  discriminant = Math.max(0, discriminant);
  const squareRoot = Math.sqrt(discriminant);
  const signedSquareRoot = b < 0 ? -squareRoot : squareRoot;
  const q = -b - signedSquareRoot;
  const qTolerance = TRANSPORT_POLYLINE_SWEEP_EPSILON
    * Math.max(Number.MIN_VALUE, Math.abs(a), Math.abs(b), Math.abs(c), squareRoot);
  const roots = Math.abs(q) > qTolerance
    ? [q / a, c / q]
    : [-b / a];
  return roots.filter(Number.isFinite).sort((left, right) => left - right);
};

const transportPolylineAddSweepCandidate = (candidates, candidate, displacement,
  tolerances) => {
  if (candidate.fraction < -tolerances.fraction
      || candidate.fraction > 1 + tolerances.fraction) return;
  const inward = transportPolylineDot2(displacement, candidate.outwardNormal);
  // Exact-contact outward and tangent travel are not impacts.
  if (inward >= -tolerances.direction) return;
  candidates.push({ ...candidate,
    fraction: Math.max(0, Math.min(1, candidate.fraction)) });
};

/**
 * Analytic point-vs-capsule-union TOI for a disk swept across a finite open
 * polyline. Segment bodies own only their strict interiors; every vertex owns
 * exactly one circular endcap, including the two finite open ends.
 */
export function sweepDiskAgainstTransportPolylineBoundaryReference(packet, start, end,
  { radius, search = 'bounded' } = {}) {
  transportPolylineRequirePacket(packet);
  const from = transportPolylineRequireVector2('Transport disk sweep start', start);
  const to = transportPolylineRequireVector2('Transport disk sweep end', end);
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('Transport disk radius must be positive');
  }
  const vertices = packet.copyVertices();
  const tolerances = transportPolylineTolerances(from, to, radius, vertices);
  const displacement = [to[0] - from[0], to[1] - from[1]];
  const initial = sampleDiskAgainstTransportPolylineBoundaryReference(packet, from, { radius });
  const candidateRange = transportPolylineDiskSweepCandidateRangeReference(
    packet, from, to, { radius, search });
  const requestedMinimumClearance = transportPolylineMinimumSweepClearance(
    vertices, from, to, radius, tolerances.spatial);
  if (initial.overlap) {
    return Object.freeze({
      kind: 'transport-polyline-disk-sweep-reference:v1',
      hit: true,
      initialOverlap: true,
      overlap: initial,
      fraction: 0,
      point: Object.freeze([...from]),
      projectedPoint: initial.projectedCenter,
      minimumClearance: initial.signedClearance,
      requestedMinimumClearance,
      featureKind: initial.featureKind,
      featureIndex: initial.featureIndex,
      stableFeatureId: initial.stableFeatureId,
      featureKey: initial.featureKey,
      activeFeatures: initial.activeFeatures,
      activeNormals: initial.activeNormals,
      candidateRange,
    });
  }
  const candidates = [];
  if (!candidateRange.empty) {
    for (let segmentIndex = candidateRange.firstSegment;
      segmentIndex <= candidateRange.lastSegment; segmentIndex += 1) {
      const segment = transportPolylineSegment(vertices, segmentIndex);
      const relative = [from[0] - segment.from[0], from[1] - segment.from[1]];
      const normalStart = transportPolylineDot2(relative, segment.orientedNormal);
      const normalTravel = transportPolylineDot2(displacement, segment.orientedNormal);
      if (Math.abs(normalTravel) <= tolerances.direction) continue;
      for (const side of [-1, 1]) {
        const fraction = (side * radius - normalStart) / normalTravel;
        const point = [from[0] + displacement[0] * fraction,
          from[1] + displacement[1] * fraction];
        const along = transportPolylineDot2(
          [point[0] - segment.from[0], point[1] - segment.from[1]], segment.tangent);
        // Endpoint ownership belongs exclusively to the one vertex circle.
        if (along <= tolerances.spatial || along >= segment.length - tolerances.spatial) continue;
        const outwardNormal = [segment.orientedNormal[0] * side,
          segment.orientedNormal[1] * side];
        transportPolylineAddSweepCandidate(candidates, {
          ...transportPolylineFeatureIdentity(packet, 'segment-body', segmentIndex),
          fraction,
          point,
          closestPoint: [segment.from[0] + segment.tangent[0] * along,
            segment.from[1] + segment.tangent[1] * along],
          outwardNormal,
        }, displacement, tolerances);
      }
    }
    const a = transportPolylineDot2(displacement, displacement);
    for (let vertexIndex = candidateRange.firstVertex;
      vertexIndex <= candidateRange.lastVertex; vertexIndex += 1) {
      const vertex = [vertices[vertexIndex * 2], vertices[vertexIndex * 2 + 1]];
      const relative = [from[0] - vertex[0], from[1] - vertex[1]];
      const b = transportPolylineDot2(relative, displacement);
      const c = transportPolylineDot2(relative, relative) - radius ** 2;
      const roots = transportPolylineStableCircleRoots(a, b, c);
      for (const fraction of roots) {
        const point = [from[0] + displacement[0] * fraction,
          from[1] + displacement[1] * fraction];
        const radial = [point[0] - vertex[0], point[1] - vertex[1]];
        const radialLength = Math.hypot(...radial);
        const outwardNormal = radialLength > tolerances.spatial
          ? [radial[0] / radialLength, radial[1] / radialLength]
          : transportPolylineVertexFallbackNormal(vertices, vertexIndex);
        transportPolylineAddSweepCandidate(candidates, {
          ...transportPolylineFeatureIdentity(packet, 'vertex-endcap', vertexIndex),
          fraction,
          point,
          closestPoint: vertex,
          outwardNormal,
        }, displacement, tolerances);
      }
    }
  }
  if (candidates.length === 0) return Object.freeze({
    kind: 'transport-polyline-disk-sweep-reference:v1',
    hit: false,
    initialOverlap: false,
    overlap: null,
    fraction: 1,
    point: Object.freeze([...to]),
    projectedPoint: Object.freeze([...to]),
    minimumClearance: requestedMinimumClearance,
    requestedMinimumClearance,
    featureKind: null,
    featureIndex: null,
    stableFeatureId: null,
    featureKey: null,
    activeFeatures: Object.freeze([]),
    activeNormals: Object.freeze([]),
    candidateRange,
  });
  const earliest = candidates.reduce((minimum, candidate) =>
    Math.min(minimum, candidate.fraction), Number.POSITIVE_INFINITY);
  const active = candidates
    .filter((candidate) => candidate.fraction <= earliest + tolerances.fraction)
    .sort(transportPolylineCompareFeatures);
  const primary = active[0];
  const impactPoint = [from[0] + displacement[0] * earliest,
    from[1] + displacement[1] * earliest];
  const minimumClearance = transportPolylineMinimumSweepClearance(
    vertices, from, impactPoint, radius, tolerances.spatial);
  const activeFeatures = Object.freeze(active.map(transportPolylineFreezeFeature));
  return Object.freeze({
    kind: 'transport-polyline-disk-sweep-reference:v1',
    hit: true,
    initialOverlap: false,
    overlap: null,
    fraction: earliest,
    point: Object.freeze(transportPolylineCanonicalVector2(impactPoint)),
    projectedPoint: Object.freeze(transportPolylineCanonicalVector2(impactPoint)),
    minimumClearance,
    requestedMinimumClearance,
    featureKind: primary.featureKind,
    featureIndex: primary.featureIndex,
    stableFeatureId: primary.stableFeatureId,
    featureKey: primary.featureKey,
    activeFeatures,
    activeNormals: transportPolylineUniqueNormals(active, tolerances.spatial),
    candidateRange,
  });
}

const transportPolylineProjectOntoContactCone = (displacement, normals, tolerance) => {
  const feasible = (candidate) => normals.every((normal) =>
    transportPolylineDot2(candidate, normal) >= -tolerance);
  if (feasible(displacement)) return [...displacement];
  const candidates = [[0, 0]];
  for (const normal of normals) {
    const inward = transportPolylineDot2(displacement, normal);
    const candidate = [displacement[0] - inward * normal[0],
      displacement[1] - inward * normal[1]];
    if (feasible(candidate)) candidates.push(candidate);
  }
  candidates.sort((left, right) => {
    const leftError = (left[0] - displacement[0]) ** 2 + (left[1] - displacement[1]) ** 2;
    const rightError = (right[0] - displacement[0]) ** 2 + (right[1] - displacement[1]) ** 2;
    return leftError - rightError || left[0] - right[0] || left[1] - right[1];
  });
  return candidates[0];
};

// Minimum-norm local correction which clears every co-nearest capsule feature.
// A single-feature projection is insufficient at an acute joint: projecting
// onto one arm can leave the center inside the other arm's capsule.
const transportPolylineOverlapCorrection = (sample, radius) => {
  const constraints = sample.activeFeatures.map((feature) => {
    const separation = [sample.center[0] - feature.closestPoint[0],
      sample.center[1] - feature.closestPoint[1]];
    const distance = Math.hypot(...separation);
    return { normal: feature.outwardNormal, depth: Math.max(0, radius - distance) };
  });
  const feasibleTolerance = sample.tolerance;
  const feasible = (candidate) => constraints.every(({ normal, depth }) =>
    transportPolylineDot2(candidate, normal) >= depth - feasibleTolerance);
  const candidates = [];
  for (const { normal, depth } of constraints) {
    const candidate = [normal[0] * depth, normal[1] * depth];
    if (feasible(candidate)) candidates.push(candidate);
  }
  for (let left = 0; left < constraints.length; left += 1) {
    for (let right = left + 1; right < constraints.length; right += 1) {
      const a = constraints[left]; const b = constraints[right];
      const determinant = transportPolylineCross2(a.normal, b.normal);
      if (Math.abs(determinant) <= TRANSPORT_POLYLINE_SWEEP_EPSILON) continue;
      const candidate = [
        (a.depth * b.normal[1] - a.normal[1] * b.depth) / determinant,
        (a.normal[0] * b.depth - a.depth * b.normal[0]) / determinant,
      ];
      if (candidate.every(Number.isFinite) && feasible(candidate)) candidates.push(candidate);
    }
  }
  if (candidates.length === 0) {
    const primary = constraints[0];
    return [primary.normal[0] * primary.depth, primary.normal[1] * primary.depth];
  }
  candidates.sort((left, right) => transportPolylineDot2(left, left)
    - transportPolylineDot2(right, right) || left[0] - right[0] || left[1] - right[1]);
  return candidates[0];
};

/**
 * Resolves a complete displacement with exact swept contacts and frictionless
 * cone projection. Contact-budget exhaustion is an explicit failed receipt;
 * it never masquerades as a successfully consumed frame.
 */
export function resolveDiskMotionAgainstTransportPolylineBoundaryReference(
  packet, start, displacement, { radius, maximumContacts = 8,
    throwOnContactBudgetExhaustion = false, search = 'bounded' } = {}) {
  transportPolylineRequirePacket(packet);
  const origin = transportPolylineRequireVector2('Transport disk motion start', start);
  const requested = transportPolylineRequireVector2(
    'Transport disk requested displacement', displacement);
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new RangeError('Transport disk radius must be positive');
  }
  if (!Number.isInteger(maximumContacts) || maximumContacts < 0) {
    throw new RangeError('Transport disk maximum contacts must be a nonnegative integer');
  }
  const vertices = packet.copyVertices();
  const intendedEnd = [origin[0] + requested[0], origin[1] + requested[1]];
  const tolerances = transportPolylineTolerances(origin, intendedEnd, radius, vertices);
  const initial = sampleDiskAgainstTransportPolylineBoundaryReference(packet, origin, { radius });
  let position = [...origin];
  let remaining = [...requested];
  let remainingFraction = 1;
  let consumedFraction = 0;
  let minimumClearance = initial.signedClearance;
  const contacts = [];
  let contactBudgetExhausted = false;
  let blockingContact = null;
  const depenetrate = () => {
    let sample = sampleDiskAgainstTransportPolylineBoundaryReference(
      packet, position, { radius });
    while (sample.overlap) {
      minimumClearance = Math.min(minimumClearance, sample.signedClearance);
      if (contacts.length >= maximumContacts) {
        contactBudgetExhausted = true;
        blockingContact = sample;
        return sample;
      }
      const correction = transportPolylineOverlapCorrection(sample, radius);
      if (!correction.every(Number.isFinite)
          || Math.hypot(...correction) <= tolerances.spatial) {
        contactBudgetExhausted = true;
        blockingContact = sample;
        return sample;
      }
      position[0] += correction[0]; position[1] += correction[1];
      contacts.push(sample);
      sample = sampleDiskAgainstTransportPolylineBoundaryReference(
        packet, position, { radius });
    }
    return sample;
  };
  depenetrate();
  while (!contactBudgetExhausted && remainingFraction > tolerances.fraction
      && Math.hypot(...remaining) > tolerances.spatial) {
    const target = [position[0] + remaining[0], position[1] + remaining[1]];
    const hit = sweepDiskAgainstTransportPolylineBoundaryReference(
      packet, position, target, { radius, search });
    minimumClearance = Math.min(minimumClearance, hit.minimumClearance);
    if (!hit.hit) {
      position = target;
      consumedFraction += remainingFraction;
      remainingFraction = 0;
      remaining = [0, 0];
      break;
    }
    if (contacts.length >= maximumContacts) {
      contactBudgetExhausted = true;
      blockingContact = hit;
      break;
    }
    if (hit.initialOverlap) {
      const correction = transportPolylineOverlapCorrection(hit.overlap, radius);
      if (!correction.every(Number.isFinite)
          || Math.hypot(...correction) <= tolerances.spatial) {
        contactBudgetExhausted = true;
        blockingContact = hit;
        break;
      }
      position[0] += correction[0]; position[1] += correction[1];
      contacts.push(hit);
      continue;
    }
    position = [...hit.point];
    consumedFraction += remainingFraction * hit.fraction;
    remainingFraction *= Math.max(0, 1 - hit.fraction);
    const afterImpact = [remaining[0] * (1 - hit.fraction),
      remaining[1] * (1 - hit.fraction)];
    remaining = transportPolylineProjectOntoContactCone(
      afterImpact, hit.activeNormals, tolerances.direction);
    contacts.push(hit);
    if (Math.hypot(...remaining) <= tolerances.spatial) {
      consumedFraction += remainingFraction;
      remainingFraction = 0;
      remaining = [0, 0];
    }
  }
  if (!contactBudgetExhausted && remainingFraction > 0) {
    position[0] += remaining[0];
    position[1] += remaining[1];
    consumedFraction += remainingFraction;
    remainingFraction = 0;
    remaining = [0, 0];
  }
  let finalSample = sampleDiskAgainstTransportPolylineBoundaryReference(
    packet, position, { radius });
  if (!contactBudgetExhausted && finalSample.overlap) finalSample = depenetrate();
  if (finalSample.overlap && !contactBudgetExhausted) {
    contactBudgetExhausted = true;
    blockingContact = finalSample;
  }
  minimumClearance = Math.min(minimumClearance, finalSample.signedClearance);
  const receipt = Object.freeze({
    kind: 'transport-polyline-disk-motion-reference:v1',
    complete: !contactBudgetExhausted && !finalSample.overlap,
    initialOverlap: initial.overlap,
    initialOverlapReceipt: initial.overlap ? initial : null,
    position: Object.freeze(position),
    requestedDisplacement: Object.freeze(requested),
    acceptedDisplacement: Object.freeze([position[0] - origin[0], position[1] - origin[1]]),
    remainingDisplacement: Object.freeze(remaining),
    consumedFraction: Math.max(0, Math.min(1, consumedFraction)),
    minimumClearance,
    finalClearance: finalSample.signedClearance,
    finalOverlap: finalSample.overlap,
    contactCount: contacts.length,
    maximumContacts,
    contactBudgetExhausted,
    blockingContact,
    contacts: Object.freeze(contacts),
  });
  if (contactBudgetExhausted && throwOnContactBudgetExhaustion) {
    const error = new RangeError('Transport polyline disk contact budget exhausted');
    error.receipt = receipt;
    throw error;
  }
  return receipt;
}

export function centerPhysicsStateBufferReference(state) {
  const center = [0, 0, 0];
  if (state.count === 0) return center;
  for (let element = 0; element < state.count; element += 1) {
    const offset = element * 3;
    for (let axis = 0; axis < 3; axis += 1) center[axis] += state.positions[offset + axis];
  }
  return center.map((value) => value / state.count);
}

export function createPhysicsStateBufferReference({ kind, seed = 0, count, precision = 'f32',
  affineComponents = 0, includeIds = false, gravity = [0, 0, -9.81], timeStep,
  conserved = null, adapter = 'unspecified' }) {
  if (!Number.isInteger(count) || count <= 0) throw new RangeError('Physics state buffer count must be positive');
  if (precision !== 'f32' && precision !== 'f64') throw new RangeError('Physics state buffer precision must be f32 or f64');
  if (!Number.isFinite(timeStep) || timeStep <= 0) throw new RangeError('Physics state buffer timestep must be positive');
  const ArrayType = precision === 'f64' ? Float64Array : Float32Array;
  const positions = new ArrayType(count * 3); const velocities = new ArrayType(count * 3);
  const affine = affineComponents ? new ArrayType(count * affineComponents) : null;
  const ids = includeIds ? new Uint32Array(count) : null;
  const perElement = conserved?.perElement ?? 0;
  const state = { kind, seed: seed >>> 0, count, positions, velocities,
    affineComponents,
    gravity: requireVector('Physics state buffer gravity', gravity), timeStep, steps: 0,
    transport: Object.freeze({ core: 'transport-core-reference:v1', adapter }),
    conservation: { name: conserved?.name ?? null, perElement, initial: perElement * count,
      inflow: 0, outflow: 0, total: perElement * count, error: 0 } };
  if (affine) state.affine = affine;
  if (ids) state.ids = ids;
  state.vectorBytes = positions.byteLength + velocities.byteLength
    + (affine?.byteLength ?? 0) + (ids?.byteLength ?? 0);
  state.stateHash = hashPhysicsStateBufferReference(state);
  return state;
}

export function snapshotPhysicsStateBufferReference(state) {
  return Object.freeze({ positions: state.positions.slice(), velocities: state.velocities.slice(),
    affine: state.affine?.slice() ?? null, ids: state.ids?.slice() ?? null, count: state.count,
    steps: state.steps, conservation: Object.freeze({ ...state.conservation }),
    ...(!Object.hasOwn(state, 'totalVolume') ? {} : { totalVolume: state.totalVolume }) });
}

export function restorePhysicsStateBufferReference(state, snapshot) {
  state.positions = snapshot.positions.slice(); state.velocities = snapshot.velocities.slice();
  if (state.affine && snapshot.affine) state.affine = snapshot.affine.slice();
  if (state.ids && snapshot.ids) state.ids = snapshot.ids.slice();
  state.count = snapshot.count ?? snapshot.positions.length / 3;
  state.steps = snapshot.steps;
  state.conservation = snapshot.conservation ? { ...snapshot.conservation }
    : { ...state.conservation, total: state.conservation.perElement * state.count, error: 0 };
  if (Object.hasOwn(snapshot, 'totalVolume')) state.totalVolume = snapshot.totalVolume;
  state.vectorBytes = state.positions.byteLength + state.velocities.byteLength
    + (state.affine?.byteLength ?? 0) + (state.ids?.byteLength ?? 0);
  state.stateHash = hashPhysicsStateBufferReference(state);
  return state;
}

const requireAlignedVector = (name, values, components, count = null) => {
  if ((!Array.isArray(values) && !ArrayBuffer.isView(values))
      || values.length % components !== 0
      || !Array.from(values).every(Number.isFinite)) {
    throw new TypeError(`${name} must be a finite vector aligned to ${components} components`);
  }
  const elementCount = values.length / components;
  if (count !== null && elementCount !== count) {
    throw new RangeError(`${name} must contain ${count} aligned elements`);
  }
  return elementCount;
};

const replaceWithAppended = (current, additions) => {
  const replacement = new current.constructor(current.length + additions.length);
  replacement.set(current); replacement.set(additions, current.length);
  return replacement;
};

const updateAxisEditAccounting = (state, addedCount, removedCount) => {
  state.conservation.initial ??= state.conservation.total;
  const amountIn = addedCount * state.conservation.perElement;
  const amountOut = removedCount * state.conservation.perElement;
  state.conservation.inflow = (state.conservation.inflow ?? 0) + amountIn;
  state.conservation.outflow = (state.conservation.outflow ?? 0) + amountOut;
  state.conservation.total = state.conservation.perElement * state.count;
  state.conservation.error = Math.abs(state.conservation.initial + state.conservation.inflow
    - state.conservation.outflow - state.conservation.total);
  if (Object.hasOwn(state, 'totalVolume')) state.totalVolume = state.conservation.total;
  state.vectorBytes = state.positions.byteLength + state.velocities.byteLength
    + (state.affine?.byteLength ?? 0) + (state.ids?.byteLength ?? 0);
  state.stateHash = hashPhysicsStateBufferReference(state);
};

export function appendPhysicsStateBufferReference(state,
  { positions, velocities, affine = null, ids = null }) {
  const addedCount = requireAlignedVector('appended positions', positions, 3);
  requireAlignedVector('appended velocities', velocities, 3, addedCount);
  if (state.affine) requireAlignedVector('appended affine', affine,
    state.affineComponents, addedCount);
  else if (affine !== null) throw new TypeError('state does not own an affine vector');
  if (state.ids) {
    if ((!Array.isArray(ids) && !ArrayBuffer.isView(ids)) || ids.length !== addedCount
        || !Array.from(ids).every((value) => Number.isSafeInteger(value) && value >= 0)) {
      throw new TypeError(`appended ids must contain ${addedCount} nonnegative integers`);
    }
  } else if (ids !== null) throw new TypeError('state does not own an id vector');
  const previousCount = state.count;
  state.positions = replaceWithAppended(state.positions, positions);
  state.velocities = replaceWithAppended(state.velocities, velocities);
  if (state.affine) state.affine = replaceWithAppended(state.affine, affine);
  if (state.ids) state.ids = replaceWithAppended(state.ids, ids);
  state.count += addedCount;
  updateAxisEditAccounting(state, addedCount, 0);
  return Object.freeze({ kind: 'physics-state-axis-edit-reference:v1', operation: 'append',
    previousCount, addedCount, removedCount: 0, count: state.count,
    conservationError: state.conservation.error, stateHash: state.stateHash });
}

export function selectPhysicsStateBufferReference(state, retainedIndices) {
  if ((!Array.isArray(retainedIndices) && !ArrayBuffer.isView(retainedIndices))
      || !Array.from(retainedIndices).every((value, index, values) => Number.isSafeInteger(value)
        && value >= 0 && value < state.count && (index === 0 || value > values[index - 1]))) {
    throw new TypeError('retained indices must be unique, ascending, and inside the entity axis');
  }
  const previousCount = state.count; const count = retainedIndices.length;
  const select = (source, components) => {
    const selected = new source.constructor(count * components);
    for (let target = 0; target < count; target += 1) {
      const origin = retainedIndices[target] * components;
      selected.set(source.subarray(origin, origin + components), target * components);
    }
    return selected;
  };
  state.positions = select(state.positions, 3); state.velocities = select(state.velocities, 3);
  if (state.affine) state.affine = select(state.affine, state.affineComponents);
  if (state.ids) state.ids = select(state.ids, 1);
  state.count = count;
  const removedCount = previousCount - count;
  updateAxisEditAccounting(state, 0, removedCount);
  return Object.freeze({ kind: 'physics-state-axis-edit-reference:v1', operation: 'select',
    previousCount, addedCount: 0, removedCount, count,
    retainedIndices: Object.freeze(Array.from(retainedIndices)),
    conservationError: state.conservation.error, stateHash: state.stateHash });
}

export function stepPhysicsStateBufferReference(state, steps = 1, {
  maximumSteps = 5000, diagnostic = 'transport steps must be an integer from 0 through 5000',
  resolveBoundary = null, beforeStep = null, afterStep = null, applyGravity = true,
  accelerations = null, prescribed = null } = {}) {
  if (!Number.isInteger(steps) || steps < 0 || steps > maximumSteps) throw new RangeError(diagnostic);
  if (accelerations !== null) requireAlignedVector('accelerations', accelerations, 3, state.count);
  if (prescribed !== null) {
    for (const key of ['positionMask', 'velocityMask', 'accelerationMask',
      'positions', 'velocities', 'accelerations']) {
      requireAlignedVector(`prescribed ${key}`, prescribed[key], 3, state.count);
    }
    for (const key of ['positionMask', 'velocityMask', 'accelerationMask']) {
      if (!Array.from(prescribed[key]).every((value) => value === 0 || value === 1)) {
        throw new TypeError(`prescribed ${key} must contain only 0 or 1`);
      }
    }
  }
  for (let step = 0; step < steps; step += 1) {
    beforeStep?.(state, (state.steps + 1) * state.timeStep);
    const previous = prescribed ? state.positions.slice() : null;
    for (let element = 0; element < state.count; element += 1) {
      const offset = element * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const index = offset + axis;
        if (prescribed?.positionMask[index]) {
          state.positions[index] = prescribed.positions[index];
          state.velocities[index] = (state.positions[index] - previous[index]) / state.timeStep;
        } else if (prescribed?.velocityMask[index]) {
          state.velocities[index] = prescribed.velocities[index];
          state.positions[index] += state.velocities[index] * state.timeStep;
        } else {
          if (prescribed?.accelerationMask[index]) {
            state.velocities[index] += prescribed.accelerations[index] * state.timeStep;
          } else {
            // Keep the original arithmetic order for existing exact replay receipts.
            if (applyGravity) state.velocities[index] += state.gravity[axis] * state.timeStep;
            if (accelerations) state.velocities[index] += accelerations[index] * state.timeStep;
          }
          state.positions[index] += state.velocities[index] * state.timeStep;
        }
      }
      resolveBoundary?.(state, offset);
    }
    state.steps += 1; afterStep?.(state);
    // Contacts/constraints are lower priority than prescribed placement/motion.
    if (prescribed) for (let index = 0; index < state.positions.length; index += 1) {
      if (prescribed.positionMask[index]) {
        state.positions[index] = prescribed.positions[index];
        state.velocities[index] = (state.positions[index] - previous[index]) / state.timeStep;
      } else if (prescribed.velocityMask[index]) state.velocities[index] = prescribed.velocities[index];
    }
  }
  state.conservation.error = Math.abs(state.conservation.total
    - state.conservation.perElement * state.count);
  state.stateHash = hashPhysicsStateBufferReference(state);
  return state;
}

export function createTransportBoxReference({ minimum, maximum, openFaces = [] }) {
  const open = Object.freeze([...openFaces]);
  return Object.freeze({ minimum: requireVector('transport box minimum', minimum),
    maximum: requireVector('transport box maximum', maximum), openFaces: open,
    openTop: open.includes('maximum-z') });
}

export function resolveTransportBoxReference(state, offset, box, { padding,
  restitution = [0.12, 0.12, 0.04], floorTangentialDamping = 0.998 } = {}) {
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = box.minimum[axis] + padding; const upper = box.maximum[axis] - padding;
    const lowerFace = ['minimum-x', 'minimum-y', 'minimum-z'][axis];
    const upperFace = ['maximum-x', 'maximum-y', 'maximum-z'][axis];
    const position = state.positions[offset + axis];
    if (position < lower && !box.openFaces.includes(lowerFace)) {
      state.maximumAttemptedBoundaryPenetration = Math.max(
        state.maximumAttemptedBoundaryPenetration, lower - position);
      state.positions[offset + axis] = lower;
      if (state.velocities[offset + axis] < 0) state.velocities[offset + axis] *= -restitution[axis];
      if (axis === 2) { state.velocities[offset] *= floorTangentialDamping;
        state.velocities[offset + 1] *= floorTangentialDamping; }
      state.boundaryCollisionCount += 1;
    } else if (position > upper && !box.openFaces.includes(upperFace)) {
      state.maximumAttemptedBoundaryPenetration = Math.max(
        state.maximumAttemptedBoundaryPenetration, position - upper);
      state.positions[offset + axis] = upper;
      if (state.velocities[offset + axis] > 0) state.velocities[offset + axis] *= -restitution[axis];
      state.boundaryCollisionCount += 1;
    }
  }
}

export function createTransportSpatialGridReference({ size, cellSize, origin }) {
  const scalarSize = Array.isArray(size) ? null : size;
  const dimensions = Array.isArray(size) ? size : [size, size, size];
  if (dimensions.length !== 3 || !dimensions.every((value) => Number.isInteger(value) && value > 0)) {
    throw new RangeError('transport spatial grid size must contain three positive integers');
  }
  const frozenSize = Object.freeze([...dimensions]); const frozenOrigin = requireVector('transport grid origin', origin);
  const key = (position) => {
    const coordinate = position.map((value, axis) => Math.max(0, Math.min(frozenSize[axis] - 1,
      Math.floor((value - frozenOrigin[axis]) / cellSize))));
    return (coordinate[2] * frozenSize[1] + coordinate[1]) * frozenSize[0] + coordinate[0];
  };
  const neighborKeys = (position) => {
    const coordinate = position.map((value, axis) => Math.max(0, Math.min(frozenSize[axis] - 1,
      Math.floor((value - frozenOrigin[axis]) / cellSize))));
    const keys = [];
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const cell = [coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz];
        if (cell.some((value, axis) => value < 0 || value >= frozenSize[axis])) continue;
        keys.push((cell[2] * frozenSize[1] + cell[1]) * frozenSize[0] + cell[0]);
      }
    }
    return keys;
  };
  return Object.freeze({ size: scalarSize ?? frozenSize, dimensions: frozenSize,
    cellSize, origin: frozenOrigin, key, neighborKeys });
}
