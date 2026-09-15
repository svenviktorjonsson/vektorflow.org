import { createGpuShowcase } from './vf-coming-soon-gpu.mjs';

export async function createComingSoonScenario(options) {
  const runtime = await createGpuShowcase('stones-water', options);
  const slots = { 'wave-strength': 0, 'stone-size': 1, 'fall-speed': 2 };
  return {
    ...runtime,
    authoritativeState: runtime.state,
    renderState: runtime.state,
    execution: 'gpu-realtime',
    setParameter(name, value) { if (slots[name] !== undefined) runtime.state.params[slots[name]] = Number(value); },
  };
}
