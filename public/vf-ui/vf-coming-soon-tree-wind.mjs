import { createGpuShowcase } from './vf-coming-soon-gpu.mjs';

export async function createComingSoonScenario(options) {
  const runtime = await createGpuShowcase('tree-wind', options);
  return {
    ...runtime,
    authoritativeState: runtime.state,
    renderState: runtime.state,
    execution: 'gpu-realtime',
    regenerate() { runtime.state.seed += 7919; runtime.reset(); return runtime.state.seed; },
    setParameter(name, value) {
      const slots = { 'wind-strength': 0, 'early-branching': 1, 'branching-density': 1, 'twig-leaf-density': 2 };
      const slot = slots[name];
      if (slot !== undefined) runtime.state.params[slot] = Number(value);
    },
  };
}
