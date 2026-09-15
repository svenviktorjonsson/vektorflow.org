const requireFinitePositive = (value, name) => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and positive`);
  }
  return value;
};

const requirePositiveInteger = (value, name) => {
  if (!Number.isInteger(value) || value < 1 || value > 4096) {
    throw new RangeError(`${name} must be an integer from one through 4096`);
  }
  return value;
};

export const WATER_GPU_MAX_SUBSTEPS_PER_DISPLAY_FRAME = 8;
export const WATER_GPU_MAX_DEBT_STEPS = 8;
export const WATER_GPU_MAX_ELAPSED_SECONDS = 0.05;

export function createFixedStepRealtimeClock({
  timeStep,
  maximumSubstepsPerFrame = WATER_GPU_MAX_SUBSTEPS_PER_DISPLAY_FRAME,
  maximumDebtSteps = WATER_GPU_MAX_DEBT_STEPS,
  maximumElapsedSeconds = WATER_GPU_MAX_ELAPSED_SECONDS,
} = {}) {
  const dt = requireFinitePositive(timeStep, 'fixed time step');
  const substepLimit = requirePositiveInteger(maximumSubstepsPerFrame,
    'maximum substeps per frame');
  const debtStepLimit = requirePositiveInteger(maximumDebtSteps,
    'maximum debt steps');
  const elapsedLimit = requireFinitePositive(maximumElapsedSeconds,
    'maximum elapsed seconds');
  if (elapsedLimit < dt) {
    throw new RangeError('maximum elapsed seconds must cover at least one fixed step');
  }

  let debtSeconds = 0;
  let observedWallSeconds = 0;
  let simulatedSeconds = 0;
  let droppedSimulationSeconds = 0;
  let clippedWallSeconds = 0;
  let capacityDroppedSeconds = 0;
  let frameCount = 0;
  let maximumObservedDebtSeconds = 0;
  let last = Object.freeze({
    elapsedSeconds: 0,
    acceptedElapsedSeconds: 0,
    simulatedDeltaSeconds: 0,
    clippedSeconds: 0,
    capacityDroppedSeconds: 0,
    steps: 0,
  });

  const snapshot = () => Object.freeze({
    timeStep: dt,
    maximumSubstepsPerFrame: substepLimit,
    maximumDebtSteps: debtStepLimit,
    maximumElapsedSeconds: elapsedLimit,
    frameCount,
    observedWallSeconds,
    simulatedSeconds,
    debtSeconds,
    droppedSimulationSeconds,
    clippedWallSeconds,
    capacityDroppedSeconds,
    maximumObservedDebtSeconds,
    accountingErrorSeconds: observedWallSeconds - simulatedSeconds
      - debtSeconds - droppedSimulationSeconds,
    last,
  });

  const advance = (elapsedSeconds) => {
    if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) {
      throw new RangeError('elapsed seconds must be finite and nonnegative');
    }
    const acceptedElapsedSeconds = Math.min(elapsedSeconds, elapsedLimit);
    const clippedSeconds = elapsedSeconds - acceptedElapsedSeconds;
    observedWallSeconds += elapsedSeconds;
    clippedWallSeconds += clippedSeconds;
    droppedSimulationSeconds += clippedSeconds;
    debtSeconds += acceptedElapsedSeconds;

    // Eight steps give a 240 Hz solver twice the ordinary 60 Hz budget. Keep
    // bounded debt after a rare long frame so the next ordinary frame can catch
    // up, instead of throwing away the accumulator immediately.
    const epsilon = dt * 1e-9;
    const dueSteps = Math.floor((debtSeconds + epsilon) / dt);
    const steps = Math.min(dueSteps, substepLimit);
    const simulatedDeltaSeconds = steps * dt;
    debtSeconds -= simulatedDeltaSeconds;
    simulatedSeconds += simulatedDeltaSeconds;

    const pendingSteps = Math.floor((debtSeconds + epsilon) / dt);
    const discardedSteps = Math.max(0, pendingSteps - debtStepLimit);
    const discardedSeconds = discardedSteps * dt;
    debtSeconds -= discardedSeconds;
    capacityDroppedSeconds += discardedSeconds;
    droppedSimulationSeconds += discardedSeconds;
    if (Math.abs(debtSeconds) < epsilon) debtSeconds = 0;
    maximumObservedDebtSeconds = Math.max(maximumObservedDebtSeconds, debtSeconds);
    frameCount += 1;
    last = Object.freeze({
      elapsedSeconds,
      acceptedElapsedSeconds,
      simulatedDeltaSeconds,
      clippedSeconds,
      capacityDroppedSeconds: discardedSeconds,
      steps,
    });
    return Object.freeze({ ...snapshot(), steps, simulatedDeltaSeconds });
  };

  const reset = () => {
    debtSeconds = 0;
    observedWallSeconds = 0;
    simulatedSeconds = 0;
    droppedSimulationSeconds = 0;
    clippedWallSeconds = 0;
    capacityDroppedSeconds = 0;
    frameCount = 0;
    maximumObservedDebtSeconds = 0;
    last = Object.freeze({
      elapsedSeconds: 0,
      acceptedElapsedSeconds: 0,
      simulatedDeltaSeconds: 0,
      clippedSeconds: 0,
      capacityDroppedSeconds: 0,
      steps: 0,
    });
    return snapshot();
  };

  return Object.freeze({ advance, reset, snapshot });
}
