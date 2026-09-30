// Saturation is a material state, not a replacement material type. Capillary
// bridges peak at intermediate wetness and vanish in dry or flooded sand.
export function sandCapillaryCohesion(wetness) {
  if (!Number.isFinite(wetness) || wetness < 0 || wetness > 1)
    throw new RangeError('Sand wetness must be between zero and one');
  return Math.min(1, 6.75 * wetness * (1 - wetness) ** 2);
}

export function sandConstitutiveState(wetness, grainRadius = 0.0075) {
  const cohesion = sandCapillaryCohesion(wetness);
  if (!Number.isFinite(grainRadius) || grainRadius <= 0)
    throw new RangeError('Effective grain radius must be positive');
  return Object.freeze({wetness, cohesion,
    // Physical guide particles stand for many grains: a short bridge range
    // captures unresolved capillary contacts without long-range attraction.
    bridgeFraction: 0.12 * cohesion,
    cohesionAcceleration: 40 * cohesion * 0.0075 / grainRadius,
    frictionBoost: 0.18 * cohesion});
}
