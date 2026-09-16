import { calibrateUniformLocalLiquidParticleMassReference }
  from './vf-physics-liquid-local-particle-reference.mjs';

const requirePositive = (name, value) => {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
};

const requireFinite = (name, value) => {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`);
  return value;
};

const pointSegmentDistance = (x, y, ax, ay, bx, by) => {
  const ex = bx - ax; const ey = by - ay;
  const denominator = ex * ex + ey * ey;
  const t = denominator <= Number.EPSILON ? 0 : Math.max(0, Math.min(1,
    ((x - ax) * ex + (y - ay) * ey) / denominator));
  return Math.hypot(x - (ax + t * ex), y - (ay + t * ey));
};

export function sampleClosedPolygonSignedDistance2DReference(polygon, point) {
  if ((!Array.isArray(polygon) && !ArrayBuffer.isView(polygon))
      || polygon.length < 6 || polygon.length % 2 !== 0
      || !Array.from(polygon).every(Number.isFinite)) {
    throw new TypeError('closed 2D polygon must contain at least three finite points');
  }
  if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
    throw new TypeError('closed 2D polygon sample point must be finite');
  }
  const count = polygon.length / 2;
  let distance = Number.POSITIVE_INFINITY; let inside = false;
  for (let current = 0, previous = count - 1; current < count;
    previous = current, current += 1) {
    const ax = polygon[previous * 2]; const ay = polygon[previous * 2 + 1];
    const bx = polygon[current * 2]; const by = polygon[current * 2 + 1];
    distance = Math.min(distance, pointSegmentDistance(point[0], point[1], ax, ay, bx, by));
    if ((ay > point[1]) !== (by > point[1])
        && point[0] < ax + (point[1] - ay) * (bx - ax) / (by - ay)) inside = !inside;
  }
  return inside ? -distance : distance;
}

/**
 * A side-view channel whose bed and attached stone are one exact closed solid.
 * The open left/right ends are deliberately absent from boundarySamples.
 */
export function createSlopedStoneChannelGeometryReference({
  minimumX = -1.5, maximumX = 1.7, minimumY = -0.24,
  bedAtStone = 0.04, bedSlope = -0.055,
  stoneCenterX = 0, stoneHalfWidth = 0.23, stoneHeight = 0.18,
  profileSegments = 160, boundarySpacing = 0.024,
} = {}) {
  requireFinite('channel minimum x', minimumX); requireFinite('channel maximum x', maximumX);
  requireFinite('channel minimum y', minimumY); requireFinite('channel bed height', bedAtStone);
  requireFinite('channel bed slope', bedSlope); requireFinite('stone center x', stoneCenterX);
  requirePositive('stone half width', stoneHalfWidth); requirePositive('stone height', stoneHeight);
  requirePositive('channel boundary spacing', boundarySpacing);
  if (maximumX <= minimumX) throw new RangeError('channel x extent must increase');
  if (!Number.isSafeInteger(profileSegments) || profileSegments < 32 || profileSegments > 4096) {
    throw new RangeError('channel profile segments must be an integer from 32 through 4096');
  }
  if (stoneCenterX - stoneHalfWidth <= minimumX
      || stoneCenterX + stoneHalfWidth >= maximumX) {
    throw new RangeError('stone must be strictly inside the open channel');
  }
  const bedHeightAt = (x) => bedAtStone + bedSlope * (x - stoneCenterX);
  const stoneHeightAt = (x) => {
    const q = (x - stoneCenterX) / stoneHalfWidth;
    if (Math.abs(q) >= 1) return 0;
    // Smooth compact cap: height and tangent both meet the bed continuously.
    const cap = (1 - q * q) ** 2;
    const asymmetry = 1 - 0.09 * q + 0.035 * (2 * q * q - 1);
    return stoneHeight * cap * asymmetry;
  };
  const surfaceHeightAt = (x) => bedHeightAt(x) + stoneHeightAt(x);
  const surface = [];
  for (let segment = 0; segment <= profileSegments; segment += 1) {
    const x = minimumX + (maximumX - minimumX) * segment / profileSegments;
    surface.push(x, surfaceHeightAt(x));
  }
  const closedSolidPolygon = Float64Array.from([
    ...surface,
    maximumX, minimumY,
    minimumX, minimumY,
  ]);
  const stoneProfile = [];
  const stoneSegments = Math.max(32,
    Math.ceil(profileSegments * stoneHalfWidth * 2 / (maximumX - minimumX)));
  for (let segment = 0; segment <= stoneSegments; segment += 1) {
    const x = stoneCenterX - stoneHalfWidth
      + stoneHalfWidth * 2 * segment / stoneSegments;
    stoneProfile.push(x, surfaceHeightAt(x));
  }
  const boundarySamples = [];
  // Arc-length resampling gives the steep shoulders the same pressure support as
  // the flat bed. Two interior layers avoid a one-sided kernel-density deficit.
  const denseCount = Math.max(profileSegments * 4,
    Math.ceil((maximumX - minimumX) / boundarySpacing) * 4);
  const dense = [];
  let totalLength = 0;
  for (let sample = 0; sample <= denseCount; sample += 1) {
    const x = minimumX + (maximumX - minimumX) * sample / denseCount;
    const y = surfaceHeightAt(x);
    if (dense.length > 0) {
      const previous = dense[dense.length - 1];
      totalLength += Math.hypot(x - previous.x, y - previous.y);
    }
    dense.push({ x, y, arc: totalLength });
  }
  const sampleAtArc = (arc) => {
    let low = 0; let high = dense.length - 1;
    while (low + 1 < high) {
      const middle = (low + high) >> 1;
      if (dense[middle].arc < arc) low = middle; else high = middle;
    }
    const first = dense[low]; const second = dense[high];
    const width = Math.max(Number.EPSILON, second.arc - first.arc);
    const t = Math.max(0, Math.min(1, (arc - first.arc) / width));
    return { x: first.x + (second.x - first.x) * t,
      y: first.y + (second.y - first.y) * t };
  };
  const boundaryCount = Math.ceil(totalLength / boundarySpacing) + 1;
  for (let sample = 0; sample < boundaryCount; sample += 1) {
    const center = sampleAtArc(totalLength * sample / (boundaryCount - 1));
    const before = sampleAtArc(Math.max(0,
      totalLength * sample / (boundaryCount - 1) - boundarySpacing * 0.2));
    const after = sampleAtArc(Math.min(totalLength,
      totalLength * sample / (boundaryCount - 1) + boundarySpacing * 0.2));
    const tx = after.x - before.x; const ty = after.y - before.y;
    const length = Math.max(Number.EPSILON, Math.hypot(tx, ty));
    // The solid lies below the authored surface, so its inward normal is the
    // clockwise tangent normal.
    const inwardX = ty / length; const inwardY = -tx / length;
    for (let layer = 0; layer < 2; layer += 1) {
      const depth = layer * boundarySpacing * 0.8;
      boundarySamples.push(center.x + inwardX * depth, center.y + inwardY * depth);
    }
  }
  return Object.freeze({
    kind: 'sloped-stone-channel-geometry-reference:v1',
    bounds: Object.freeze({ minimum: Object.freeze([minimumX, minimumY]),
      maximum: Object.freeze([maximumX, bedAtStone + stoneHeight + 0.48]) }),
    minimumX, maximumX, minimumY, bedAtStone, bedSlope,
    stone: Object.freeze({ centerX: stoneCenterX, halfWidth: stoneHalfWidth,
      height: stoneHeight, minimumX: stoneCenterX - stoneHalfWidth,
      maximumX: stoneCenterX + stoneHalfWidth,
      profile: Float64Array.from(stoneProfile) }),
    surface: Float64Array.from(surface),
    closedSolidPolygon,
    boundarySamples: Float64Array.from(boundarySamples),
    boundarySpacing,
    bedHeightAt, stoneHeightAt, surfaceHeightAt,
  });
}

export function seedWater2DAboveChannelReference(channel, {
  particleSpacing = 0.034, freeSurfaceY = 0.43, upstreamVelocity = 0.82,
  minimumParticleX = channel?.minimumX + 0.08,
  maximumParticleX = channel?.maximumX - 0.42,
  restDensity = 998.207,
  supportScale = 2.35,
} = {}) {
  if (channel?.kind !== 'sloped-stone-channel-geometry-reference:v1') {
    throw new TypeError('sloped stone channel geometry required');
  }
  requirePositive('water 2D particle spacing', particleSpacing);
  requireFinite('water 2D free-surface height', freeSurfaceY);
  requireFinite('water 2D upstream velocity', upstreamVelocity);
  requirePositive('water 2D rest density', restDensity);
  requirePositive('water 2D support scale', supportScale);
  if (maximumParticleX <= minimumParticleX) {
    throw new RangeError('water 2D seed x extent must increase');
  }
  const positions = []; const velocities = [];
  const supportRadius = particleSpacing * supportScale;
  const massCalibration = calibrateUniformLocalLiquidParticleMassReference({
    dimension: 2, spacing: particleSpacing, supportRadius, restDensity,
  });
  const particleMass = massCalibration.particleMass;
  const tangentNorm = Math.hypot(1, channel.bedSlope);
  const vx = upstreamVelocity / tangentNorm;
  const vy = upstreamVelocity * channel.bedSlope / tangentNorm;
  let row = 0;
  for (let y = channel.minimumY + particleSpacing; y <= freeSurfaceY;
    y += particleSpacing, row += 1) {
    const stagger = row % 2 === 0 ? 0 : particleSpacing * 0.5;
    for (let x = minimumParticleX + particleSpacing + stagger;
      x <= maximumParticleX; x += particleSpacing) {
      if (sampleClosedPolygonSignedDistance2DReference(
        channel.closedSolidPolygon, [x, y]) < particleSpacing) continue;
      positions.push(x, y); velocities.push(vx, vy);
    }
  }
  const count = positions.length / 2;
  if (count === 0) throw new Error('water 2D seed produced no particles');
  return Object.freeze({ kind: 'water-2d-channel-seed-reference:v1', count,
    positions: Float64Array.from(positions), velocities: Float64Array.from(velocities),
    masses: new Float64Array(count).fill(particleMass),
    supportRadii: new Float64Array(count).fill(supportRadius),
    particleSpacing, particleMass, supportRadius, restDensity,
    massCalibration, freeSurfaceY, upstreamVelocity });
}
