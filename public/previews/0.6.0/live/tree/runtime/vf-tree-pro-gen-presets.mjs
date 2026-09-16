import { proGen } from './vf-pro-gen-distribution-reference.mjs';

// Cached initial conditions, not a replacement wind law or animated shape.
export const treeProGenPresets = Object.freeze({
  original: Object.freeze({ branching: {}, leafShape: {} }),
  uniform: Object.freeze({
    branching: { mainAngle: proGen.uniform(0.07, 0.22), lateralAngle: proGen.uniform(0.35, 0.85),
      areaLoss: proGen.uniform(0.79, 0.89), mainAreaShare: proGen.uniform(0.58, 0.72) },
    leafShape: { widthRatio: proGen.uniform(0.3, 0.62), roundness: proGen.uniform(0.52, 0.9),
      asymmetry: proGen.uniform(-0.13, 0.13), camberRatio: proGen.uniform(-0.06, 0.06) },
  }),
  normal: Object.freeze({
    branching: { mainAngle: proGen.normal(0.13, 0.04), lateralAngle: proGen.normal(0.6, 0.12),
      areaLoss: proGen.normal(0.84, 0.025), mainAreaShare: proGen.normal(0.65, 0.04) },
    leafShape: { widthRatio: proGen.normal(0.46, 0.1), roundness: proGen.normal(0.72, 0.1),
      asymmetry: proGen.normal(0, 0.08), camberRatio: proGen.normal(0, 0.035) },
  }),
  triangular: Object.freeze({
    branching: { mainAngle: proGen.triangular(0.07, 0.22, 0.18), lateralAngle: proGen.triangular(0.35, 0.85, 0.72),
      areaLoss: proGen.triangular(0.79, 0.89, 0.85), mainAreaShare: proGen.triangular(0.58, 0.72, 0.67) },
    leafShape: { widthRatio: proGen.triangular(0.3, 0.62, 0.5), roundness: proGen.triangular(0.52, 0.9, 0.78),
      asymmetry: proGen.triangular(-0.13, 0.13, 0), camberRatio: proGen.triangular(-0.06, 0.06, 0) },
  }),
});

export function treeProGenAsset(name) {
  if (!Object.hasOwn(treeProGenPresets, name)) throw new Error('Unknown tree distribution');
  return name === 'original' ? 'tree-mesh.bin.gz' : `tree-mesh-${name}.bin.gz`;
}
