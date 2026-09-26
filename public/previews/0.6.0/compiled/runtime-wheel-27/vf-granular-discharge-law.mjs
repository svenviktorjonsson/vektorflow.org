// Material Law: a coarse parcel can release fractional area through an
// aperture. The represented grain diameter, not the parcel support diameter,
// controls the empty-annulus cutoff. 2-D Beverloo form; coefficients require
// calibration for a particular sand and throat.
export function granularDischargeAreaRate({opening, grainDiameter, gravity = 9.82,
  dischargeCoefficient = 0.5, emptyAnnulus = 1, axialGravity = 1}) {
  if (![opening, grainDiameter, gravity, dischargeCoefficient, emptyAnnulus,
    axialGravity].every(Number.isFinite)) throw new RangeError('Law parameters must be finite');
  if (opening < 0 || grainDiameter <= 0 || gravity <= 0 ||
    dischargeCoefficient < 0 || emptyAnnulus < 0)
    throw new RangeError('Invalid granular discharge Law parameters');
  return dischargeCoefficient * Math.sqrt(gravity * Math.max(0, axialGravity))
    * Math.max(0, opening - emptyAnnulus * grainDiameter) ** 1.5;
}

export function relaxGranularDischargeRate(previousRate, targetRate, dt,
  responseSeconds = 0.25) {
  if (![previousRate, targetRate, dt, responseSeconds].every(Number.isFinite) ||
    previousRate < 0 || targetRate < 0 || dt < 0 || responseSeconds <= 0)
    throw new RangeError('Invalid discharge response Law parameters');
  return previousRate + (targetRate - previousRate)
    * -Math.expm1(-dt / responseSeconds);
}

export const GRANULAR_DISCHARGE_LAW_WGSL = `
fn granular_discharge_area_rate(opening: f32, grain_diameter: f32,
    gravity: f32, coefficient: f32, empty_annulus: f32,
    axial_gravity: f32) -> f32 {
  return coefficient * sqrt(gravity * max(0.0, axial_gravity))
    * pow(max(0.0, opening - empty_annulus * grain_diameter), 1.5);
}
fn relax_granular_discharge_rate(previous_rate: f32, target_rate: f32,
    dt: f32, response_seconds: f32) -> f32 {
  return previous_rate + (target_rate - previous_rate)
    * (1.0 - exp(-dt / response_seconds));
}
`;
