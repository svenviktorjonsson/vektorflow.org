import { createGpuShowcase } from './vf-coming-soon-gpu.mjs';

export async function createComingSoonScenario(options) {
  const runtime = await createGpuShowcase('sand-drum', options);
  return {
    ...runtime,
    authoritativeState: runtime.state,
    renderState: runtime.state,
    execution: 'gpu-realtime',
    setView(view) { runtime.state.params[3] = view === 'particles' ? 1 : 0; },
    setParameter(name, value) {
      const slots = { 'rotation-speed': 0, 'fill-level': 1, glimmer: 2 };
      if (slots[name] !== undefined) runtime.state.params[slots[name]] = Number(value);
    },
  };
}
