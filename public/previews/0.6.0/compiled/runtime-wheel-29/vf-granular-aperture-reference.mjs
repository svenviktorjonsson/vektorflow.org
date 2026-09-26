// CPU reference for a GPU granular aperture Law, not a second material World.
// Areas are 2-D cross-sectional material volume per metre of depth.
// 2-D Beverloo scaling: https://doi.org/10.3389/frsfm.2024.1340744
// The 32° default is an example dry-sand calibration, not a universal constant.
// A river-sand experiment measured 31.91°: https://doi.org/10.1038/s41598-025-93909-2
// Discharge and empty-annulus coefficients below are diagnostic placeholders;
// they must be calibrated to the actual sand before claiming predictive flow.
import {granularDischargeAreaRate, relaxGranularDischargeRate}
  from './vf-granular-discharge-law.mjs';
export function createGranularApertureReference({
  guideDiameter = 0.015, grainDiameter = 0.0015, opening = 0.008,
  initialArea = 0.08, gravity = 9.82, fallHeight = 0.44,
  chamberWidth = 0.5, chamberHeight = 0.41,
  reposeDegrees = 32, dischargeCoefficient = 0.5,
  emptyAnnulus = 1, streamCells = 64, pileCells = 128,
  settlingSeconds = 0.35, releaseResponseSeconds = 0.25,
} = {}) {
  const positive = (name, value) => {
    if (!Number.isFinite(value) || value <= 0)
      throw new RangeError(`${name} must be positive`);
  };
  for (const [name, value] of Object.entries({guideDiameter, grainDiameter,
    opening, initialArea, gravity, fallHeight, chamberWidth, chamberHeight,
    dischargeCoefficient, emptyAnnulus, settlingSeconds,
    releaseResponseSeconds})) positive(name, value);
  if (!Number.isFinite(reposeDegrees) || reposeDegrees < 0 || reposeDegrees >= 90)
    throw new RangeError('Repose angle must be between 0 and 90 degrees');
  if (!Number.isSafeInteger(streamCells) || streamCells < 2 ||
      !Number.isSafeInteger(pileCells) || pileCells < 4)
    throw new RangeError('Field resolution is invalid');
  const stream = new Float64Array(streamCells);
  const pile = new Float64Array(pileCells);
  const dy = fallHeight / streamCells;
  const dx = chamberWidth / pileCells;
  const reposeSlope = Math.tan(reposeDegrees * Math.PI / 180);
  const halfWidth = chamberWidth / 2;
  const wallSlope = chamberHeight / (halfWidth - opening / 2);
  const dischargeAreaPerSecond = granularDischargeAreaRate({opening,
    grainDiameter, gravity, dischargeCoefficient, emptyAnnulus});
  let upperArea = initialArea, settlingArea = 0, time = 0, flowRate = 0;
  const pileArea = () => pile.reduce((sum, height) => sum + height * dx, 0);
  const streamArea = () => stream.reduce((sum, area) => sum + area, 0);
  const conservedArea = () => upperArea + streamArea() + settlingArea + pileArea();
  const settlePile = () => {
    // Repose projection respects the glass capacity. Material outside the
    // chamber cannot be counted as part of an invisible pile behind its wall.
    const area = pileArea();
    let low = 0, high = chamberHeight;
    for (let iteration = 0; iteration < 24; iteration++) {
      const peak = (low + high) / 2;
      let capacity = peak * peak / reposeSlope;
      if (peak > reposeSlope * halfWidth) {
        const crossing = Math.max(opening / 2, Math.min(halfWidth,
          (chamberHeight + wallSlope * opening / 2 - peak)
            / (wallSlope - reposeSlope)));
        capacity = 2 * (peak * crossing
          - reposeSlope * crossing * crossing / 2
          + wallSlope * (halfWidth - crossing) ** 2 / 2);
      }
      if (capacity < area) low = peak; else high = peak;
    }
    const peak = (low + high) / 2;
    let sampled = 0;
    for (let index = 0; index < pileCells; index++) {
      const x = Math.abs((index + 0.5) * dx - halfWidth);
      const wallCapacity = chamberHeight * Math.max(0,
        Math.min(1, (halfWidth - x) / (halfWidth - opening / 2)));
      pile[index] = Math.min(Math.max(0, peak - reposeSlope * x), wallCapacity);
      sampled += pile[index] * dx;
    }
    if (sampled === 0 && area > 0) {
      pile[pileCells / 2 - 1] = area / (2 * dx);
      pile[pileCells / 2] = area / (2 * dx);
      sampled = area;
    }
    const scale = area / Math.max(sampled, 1e-30);
    for (let index = 0; index < pileCells; index++) pile[index] *= scale;
  };
  const step = dt => {
    positive('Timestep', dt);
    flowRate = relaxGranularDischargeRate(flowRate,
      upperArea > 0 ? dischargeAreaPerSecond : 0, dt, releaseResponseSeconds);
    const released = Math.min(upperArea, flowRate * dt);
    upperArea -= released;
    stream[0] += released;
    const impactIndex = Math.min(streamCells, Math.max(1,
      Math.ceil((fallHeight - pile[Math.floor(pileCells / 2)]) / dy)));
    for (let index = streamCells - 1; index >= 0; index--) {
      if (index >= impactIndex) {
        settlingArea += stream[index];
        stream[index] = 0;
        continue;
      }
      const speed = Math.sqrt(2 * gravity * (index + 0.5) * dy);
      const travel = speed * dt / dy;
      const whole = Math.floor(travel), fraction = travel - whole;
      const parcel = stream[index];
      stream[index] = 0;
      const near = parcel * (1 - fraction), far = parcel - near;
      const nearIndex = index + whole, farIndex = nearIndex + 1;
      if (nearIndex >= impactIndex) settlingArea += near;
      else stream[nearIndex] += near;
      if (farIndex >= impactIndex) settlingArea += far;
      else stream[farIndex] += far;
    }
    const deposited = settlingArea * (1 - Math.exp(-dt / settlingSeconds));
    settlingArea -= deposited;
    pile[Math.floor(pileCells / 2)] += deposited / dx;
    settlePile();
    const nextImpact = Math.min(streamCells, Math.max(1,
      Math.ceil((fallHeight - pile[Math.floor(pileCells / 2)]) / dy)));
    for (let index = nextImpact; index < streamCells; index++) {
      settlingArea += stream[index];
      stream[index] = 0;
    }
    time += dt;
    return snapshot();
  };
  const snapshot = () => Object.freeze({time, upperArea, streamArea:streamArea(),
    settlingArea, pileArea:pileArea(), totalArea:conservedArea(),
    retainedFraction:conservedArea() / initialArea,
    dischargeAreaPerSecond, guideDiameter, grainDiameter, opening,
    flowRate, reposeDegrees, pile:Object.freeze([...pile]),
    stream:Object.freeze([...stream])});
  return Object.freeze({step, snapshot});
}
