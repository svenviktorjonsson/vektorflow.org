const requirePositive = (name, value) => {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
};

const requireNonnegative = (name, value) => {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be nonnegative`);
  return value;
};

const requireContactAngle = (value) => {
  if (!Number.isFinite(value) || value < 0 || value > Math.PI) {
    throw new RangeError('contact angle must be from zero through pi');
  }
  return value;
};

export function characterizeDimensionlessRegimeReference({ density, dynamicViscosity,
  surfaceTension, gravityMagnitude, speed, length, contactAngleRadians = Math.PI / 2 }) {
  requirePositive('density', density); requirePositive('dynamic viscosity', dynamicViscosity);
  requirePositive('surface tension', surfaceTension);
  requireNonnegative('gravity magnitude', gravityMagnitude);
  requireNonnegative('speed', speed); requirePositive('characteristic length', length);
  requireContactAngle(contactAngleRadians);
  return Object.freeze({
    reynolds: density * speed * length / dynamicViscosity,
    weber: density * speed ** 2 * length / surfaceTension,
    ohnesorge: dynamicViscosity / Math.sqrt(density * surfaceTension * length),
    bond: density * gravityMagnitude * length ** 2 / surfaceTension,
    capillary: dynamicViscosity * speed / surfaceTension,
    contactAngleRadians, wetting: contactAngleRadians <= Math.PI / 2,
    youngLaplacePressure: (radius1, radius2) => surfaceTension
      * (1 / requirePositive('first curvature radius', radius1)
        + 1 / requirePositive('second curvature radius', radius2)),
    rayleighPlateauUnstable: (radius, wavelength) =>
      requirePositive('jet wavelength', wavelength) > 2 * Math.PI
        * requirePositive('jet radius', radius),
  });
}

export function measureCapillaryEquilibriumReference({ density, surfaceTension,
  gravityMagnitude, contactAngleRadians, tubeRadius, curvatureRadii }) {
  requirePositive('density', density); requirePositive('surface tension', surfaceTension);
  requireNonnegative('gravity magnitude', gravityMagnitude);
  requireContactAngle(contactAngleRadians); requirePositive('tube radius', tubeRadius);
  if (!Array.isArray(curvatureRadii) || curvatureRadii.length !== 2) {
    throw new TypeError('two curvature radii are required');
  }
  const radius1 = requirePositive('first curvature radius', curvatureRadii[0]);
  const radius2 = requirePositive('second curvature radius', curvatureRadii[1]);
  return Object.freeze({
    youngLaplacePressure: surfaceTension * (1 / radius1 + 1 / radius2),
    riseHeight: gravityMagnitude === 0 ? Infinity
      : 2 * surfaceTension * Math.cos(contactAngleRadians)
        / (density * gravityMagnitude * tubeRadius),
    wetting: contactAngleRadians <= Math.PI / 2,
  });
}

export function classifyRayleighPlateauReference({ radius, wavelength,
  initialAmplitude, growthRate, elapsedTime }) {
  requirePositive('jet radius', radius); requirePositive('jet wavelength', wavelength);
  requireNonnegative('initial perturbation amplitude', initialAmplitude);
  requireNonnegative('growth rate', growthRate); requireNonnegative('elapsed time', elapsedTime);
  const unstable = wavelength > 2 * Math.PI * radius;
  const amplitude = unstable ? initialAmplitude * Math.exp(growthRate * elapsedTime)
    : initialAmplitude;
  const classification = !unstable ? 'stable-wavelength'
    : amplitude >= radius ? 'breakup' : 'growing-neck';
  return Object.freeze({ unstable, amplitude, classification });
}

export function measureRayleighPlateauTransitionReference({ radius, density,
  dynamicViscosity, surfaceTension, initialAmplitude, elapsedTime,
  inviscidOhnesorgeLimit = 0.1 }) {
  requirePositive('jet radius', radius); requirePositive('density', density);
  requireNonnegative('dynamic viscosity', dynamicViscosity);
  requirePositive('surface tension', surfaceTension);
  requireNonnegative('initial perturbation amplitude', initialAmplitude);
  requireNonnegative('elapsed time', elapsedTime);
  requirePositive('inviscid Ohnesorge limit', inviscidOhnesorgeLimit);
  const fastestWavenumberRadius = 0.697;
  const fastestWavelength = 2 * Math.PI * radius / fastestWavenumberRadius;
  const maximumGrowthRate = 0.343 * Math.sqrt(surfaceTension / (density * radius ** 3));
  const transition = classifyRayleighPlateauReference({ radius,
    wavelength: fastestWavelength, initialAmplitude,
    growthRate: maximumGrowthRate, elapsedTime });
  const breakupTime = initialAmplitude === 0 ? Infinity
    : Math.max(0, Math.log(radius / initialAmplitude) / maximumGrowthRate);
  const jetSegmentVolume = Math.PI * radius ** 2 * fastestWavelength;
  const equivalentDropRadius = Math.cbrt(3 * jetSegmentVolume / (4 * Math.PI));
  const ohnesorge = dynamicViscosity / Math.sqrt(density * surfaceTension * radius);
  return Object.freeze({ ...transition, radius, fastestWavenumberRadius, fastestWavelength,
    maximumGrowthRate, breakupTime, jetSegmentVolume, dropVolume: jetSegmentVolume,
    equivalentDropRadius, equivalentDropDiameter: 2 * equivalentDropRadius,
    ohnesorge, inviscidOhnesorgeLimit,
    inviscidApproximationValid: ohnesorge <= inviscidOhnesorgeLimit });
}

export function measurePhysicalInvariantsReference(state, { perElementMass,
  perElementVolume = 0, gravity = [0, 0, 0] }) {
  if (!state || !Number.isInteger(state.count) || state.positions.length !== state.count * 3
    || state.velocities.length !== state.count * 3) throw new TypeError('dense physics state required');
  requirePositive('element mass', perElementMass); requireNonnegative('element volume', perElementVolume);
  if (!Array.isArray(gravity) || gravity.length !== 3 || !gravity.every(Number.isFinite)) {
    throw new TypeError('invariant gravity must be a finite three-vector');
  }
  const momentum = [0, 0, 0]; let kineticEnergy = 0; let potentialEnergy = 0;
  for (let element = 0; element < state.count; element += 1) {
    const offset = element * 3; let speedSquared = 0; let gravityDotPosition = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      const velocity = state.velocities[offset + axis];
      momentum[axis] += perElementMass * velocity; speedSquared += velocity ** 2;
      gravityDotPosition += gravity[axis] * state.positions[offset + axis];
    }
    kineticEnergy += 0.5 * perElementMass * speedSquared;
    potentialEnergy -= perElementMass * gravityDotPosition;
  }
  return Object.freeze({ mass: perElementMass * state.count,
    volume: perElementVolume * state.count, momentum: Object.freeze(momentum),
    kineticEnergy, potentialEnergy, mechanicalEnergy: kineticEnergy + potentialEnergy });
}

export function verifyGranularRegimeReference({ reposeAngleDegrees, reposeBounds,
  dischargedCount, activeCount, initialCount, massError, shearStress, normalStress, friction }) {
  if (!Array.isArray(reposeBounds) || reposeBounds.length !== 2
    || !reposeBounds.every(Number.isFinite) || reposeBounds[0] > reposeBounds[1]) {
    throw new TypeError('granular repose bounds must be an ordered finite pair');
  }
  finiteCount('discharged count', dischargedCount); finiteCount('active count', activeCount);
  finiteCount('initial count', initialCount); requireNonnegative('granular mass error', massError);
  requireNonnegative('granular shear stress', shearStress);
  requirePositive('granular normal stress', normalStress); requireNonnegative('granular friction', friction);
  const stressRatio = shearStress / normalStress;
  return Object.freeze({ reposeAccepted: reposeAngleDegrees >= reposeBounds[0]
      && reposeAngleDegrees <= reposeBounds[1], dischargeFraction: dischargedCount / initialCount,
    populationConserved: dischargedCount + activeCount === initialCount,
    massConserved: massError === 0, stressRatio,
    yieldClassification: stressRatio < friction ? 'below-yield'
      : stressRatio === friction ? 'at-yield' : 'flowing-yield' });
}

function finiteCount(name, value) {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${name} must be a nonnegative integer`);
}

function evidenceHash(value) {
  const text = JSON.stringify(value); let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index); hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function createPhysicalRegimeReceiptReference({ runtime, adapter, stateHash, metrics }) {
  if (!['native', 'wasm', 'gpu'].includes(runtime)) {
    throw new RangeError('physical regime runtime must be native, wasm, or gpu');
  }
  if (adapter !== 'liquid' && adapter !== 'granular') {
    throw new RangeError('physical regime adapter must be liquid or granular');
  }
  if (typeof stateHash !== 'string' || !stateHash) throw new TypeError('physical regime state hash required');
  if (!metrics || typeof metrics !== 'object') throw new TypeError('physical regime metrics required');
  const evidence = { adapter, stateHash, metrics };
  return Object.freeze({ kind: 'physical-regime-receipt:v1', runtime, ...evidence,
    evidenceHash: evidenceHash(evidence) });
}
