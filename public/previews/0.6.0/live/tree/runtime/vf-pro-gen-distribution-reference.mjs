// Distribution sugar for generation, not simulation. Random identity is owned
// by the conditioned hierarchy; changing a PDF never changes entity identity.
import { sampleBoundedUniform, sampleNormalReference } from './vf-conditioned-distribution.mjs';

export function proGenDistributionReference(options = 'uniform') {
  const pdf = typeof options === 'string' ? { kind: options } : { ...options };
  if (pdf.kind === 'normal') {
    pdf.mean ??= 0; pdf.deviation ??= 0.5;
    if (!Number.isFinite(pdf.mean) || !Number.isFinite(pdf.deviation) || pdf.deviation < 0) {
      throw new Error('invalid normal distribution');
    }
  } else if (pdf.kind === 'uniform' || pdf.kind === 'triangular') {
    pdf.low ??= -1; pdf.high ??= 1;
    if (![pdf.low, pdf.high, pdf.high - pdf.low].every(Number.isFinite) || pdf.low >= pdf.high) {
      throw new Error('invalid distribution range');
    }
    if (pdf.kind === 'triangular') {
      pdf.mode ??= (pdf.low + pdf.high) / 2;
      if (!Number.isFinite(pdf.mode) || pdf.mode < pdf.low || pdf.mode > pdf.high) {
        throw new Error('invalid triangular distribution mode');
      }
    }
  } else throw new Error('unknown distribution kind');
  if (pdf.range !== undefined && (!Array.isArray(pdf.range) || pdf.range.length !== 2
    || !pdf.range.every(Number.isFinite) || pdf.range[0] > pdf.range[1])) {
    throw new Error('invalid distribution bounds');
  }
  if (pdf.range) pdf.range = Object.freeze([...pdf.range]);
  return Object.freeze(pdf);
}

export function proGenValueFromUnitReference(u, v, options) {
  const pdf = proGenDistributionReference(options);
  if (!Number.isFinite(u) || u < 0 || u >= 1 || !Number.isFinite(v) || v < 0 || v >= 1
    || (pdf.kind === 'normal' && u === 0)) throw new Error('invalid distribution unit sample');
  let value;
  if (pdf.kind === 'normal') value = pdf.mean + pdf.deviation * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  else if (pdf.kind === 'triangular') {
    const split = (pdf.mode - pdf.low) / (pdf.high - pdf.low);
    value = u < split ? pdf.low + Math.sqrt(u * (pdf.high - pdf.low) * (pdf.mode - pdf.low))
      : pdf.high - Math.sqrt((1 - u) * (pdf.high - pdf.low) * (pdf.high - pdf.mode));
  } else value = pdf.low + (pdf.high - pdf.low) * u;
  return pdf.range ? Math.max(pdf.range[0], Math.min(pdf.range[1], value)) : value;
}

export function sampleProGenDistributionReference(node, sample, options) {
  const pdf = proGenDistributionReference(options);
  if (pdf.kind === 'normal') {
    const value = sampleNormalReference(node, sample, { mean: pdf.mean, standardDeviation: pdf.deviation });
    return pdf.range ? Math.max(pdf.range[0], Math.min(pdf.range[1], value)) : value;
  }
  const u = sampleBoundedUniform(node, sample, { min: 0, max: 1 });
  return proGenValueFromUnitReference(u, 0, pdf);
}

export const proGen = Object.freeze({
  uniform: (low = -1, high = 1) => proGenDistributionReference({ kind: 'uniform', low, high }),
  normal: (mean = 0, deviation = 0.5, range) => proGenDistributionReference({ kind: 'normal', mean, deviation, range }),
  triangular: (low = -1, high = 1, mode = (low + high) / 2) => proGenDistributionReference({ kind: 'triangular', low, high, mode }),
});
