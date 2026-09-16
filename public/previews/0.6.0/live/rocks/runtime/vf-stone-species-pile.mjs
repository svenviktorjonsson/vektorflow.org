import {
  createWeatheredGraniteSpecimenReference,
} from './vf-weathered-granite-specimen.mjs?v=fourier-4';

export const STONE_SPECIES_PROFILES = Object.freeze([
  Object.freeze({ id: 'gray-granite', aspect: [1.02, 0.96, 0.98], aspectVariation: Object.freeze({ planar: [0.91, 1.10], vertical: [0.94, 1.08] }), tilt: 0.25, macroForm: Object.freeze({ facetCount: 7, facetStrength: 0, profileExponent: 1, latitudeWarp: 0, latitudeTwist: 0 }), roughness: [0.70, 0.94], albedo: [0.28, 0.82] }),
  Object.freeze({ id: 'red-granite', aspect: [1.08, 0.92, 1.00], aspectVariation: Object.freeze({ planar: [0.90, 1.11], vertical: [0.94, 1.08] }), tilt: 0.30, macroForm: Object.freeze({ facetCount: 8, facetStrength: 0, profileExponent: 1, latitudeWarp: 0, latitudeTwist: 0 }), roughness: [0.68, 0.93], albedo: [0.22, 0.74] }),
  Object.freeze({ id: 'pale-quartzite', aspect: [0.94, 1.04, 1.06], aspectVariation: Object.freeze({ planar: [0.91, 1.09], vertical: [0.95, 1.09] }), tilt: 0.22, macroForm: Object.freeze({ facetCount: 6, facetStrength: 0, profileExponent: 1, latitudeWarp: 0, latitudeTwist: 0 }), roughness: [0.66, 0.90], albedo: [0.52, 0.84] }),
  Object.freeze({ id: 'dark-basalt', aspect: [0.96, 0.92, 1.08], aspectVariation: Object.freeze({ planar: [0.90, 1.10], vertical: [0.94, 1.08] }), tilt: 0.34, macroForm: Object.freeze({ facetCount: 9, facetStrength: 0, profileExponent: 1, latitudeWarp: 0, latitudeTwist: 0 }), roughness: [0.80, 0.94], albedo: [0.10, 0.34] }),
  Object.freeze({ id: 'warm-granite', aspect: [1.06, 0.94, 0.97], aspectVariation: Object.freeze({ planar: [0.91, 1.09], vertical: [0.94, 1.08] }), tilt: 0.27, macroForm: Object.freeze({ facetCount: 5, facetStrength: 0, profileExponent: 1, latitudeWarp: 0, latitudeTwist: 0 }), roughness: [0.70, 0.90], albedo: [0.25, 0.68] }),
]);

const PILE_SPECIES_COMPOSITION = Object.freeze([0, 3, 1, 2, 0]);
const PILE_LAYOUT = Object.freeze([
  Object.freeze({ center: [-1.20, 0.02], size: 0.54, layer: 0 }),
  Object.freeze({ center: [0.00, -0.04], size: 0.68, layer: 0 }),
  Object.freeze({ center: [1.25, 0.04], size: 0.48, layer: 0 }),
  Object.freeze({ center: [-0.43, 0.01], size: 0.43, layer: 1 }),
  Object.freeze({ center: [0.44, -0.01], size: 0.35, layer: 1 }),
]);

function mix32(value) {
  let word = value >>> 0;
  word ^= word >>> 16;
  word = Math.imul(word, 0x7feb352d) >>> 0;
  word ^= word >>> 15;
  word = Math.imul(word, 0x846ca68b) >>> 0;
  word ^= word >>> 16;
  return word >>> 0;
}

function unit(seed, lane) {
  return mix32(seed ^ Math.imul(lane + 1, 0x9e3779b1)) / 0x100000000;
}

function modelMatrix(center, yaw, pitch, roll, scale) {
  const cy = Math.cos(yaw); const sy = Math.sin(yaw);
  const cp = Math.cos(pitch); const sp = Math.sin(pitch);
  const cr = Math.cos(roll); const sr = Math.sin(roll);
  return [
    cy * cp * scale[0], sy * cp * scale[0], -sp * scale[0], 0,
    (cy * sp * sr - sy * cr) * scale[1],
    (sy * sp * sr + cy * cr) * scale[1], cp * sr * scale[1], 0,
    (cy * sp * cr + sy * sr) * scale[2],
    (sy * sp * cr - cy * sr) * scale[2], cp * cr * scale[2], 0,
    center[0], center[1], center[2], 1,
  ];
}

function transformedBounds(packet, matrix) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < packet.vertices.length; offset += 10) {
    const x = packet.vertices[offset]; const y = packet.vertices[offset + 1];
    const z = packet.vertices[offset + 2];
    const point = [
      matrix[0] * x + matrix[4] * y + matrix[8] * z,
      matrix[1] * x + matrix[5] * y + matrix[9] * z,
      matrix[2] * x + matrix[6] * y + matrix[10] * z,
    ];
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  return { minimum, maximum };
}

export function createStoneSpeciesPileReference() {
  const meshes = [];
  const individuals = [];
  const speciesOccurrences = STONE_SPECIES_PROFILES.map(() => 0);
  for (let index = 0; index < PILE_LAYOUT.length; index += 1) {
    const speciesIndex = PILE_SPECIES_COMPOSITION[index];
    const individualIndex = speciesOccurrences[speciesIndex];
    speciesOccurrences[speciesIndex] += 1;
    const profile = STONE_SPECIES_PROFILES[speciesIndex];
    const seed0 = mix32(0x51f15e5d ^ Math.imul(speciesIndex + 1, 0x9e3779b1)
      ^ Math.imul(individualIndex + 1, 0x85ebca77));
    const seed1 = mix32(seed0 ^ 0xc2b2ae3d);
    const identity = Object.freeze({
      generator: 'vkf.conditioned', version: 1,
      seed: Object.freeze([seed0, seed1]), domain: 'material',
      hierarchy: Object.freeze(['world:highland', `stone:${profile.id}`, `individual:${individualIndex}`]),
      lod: 0, channel: 'geology',
    });
    const specimen = createWeatheredGraniteSpecimenReference(identity, {
      granularMicrorelief: true,
      microshadow: false,
      roundedUnderside: true,
      macroForm: profile.macroForm,
    });
    const layout = PILE_LAYOUT[index];
    const layer = layout.layer;
    const sizeRatio = (layout.size - 0.35) / (0.68 - 0.35);
    const scale = profile.aspect.map((value, axis) => {
      const bounds = axis === 2
        ? profile.aspectVariation.vertical : profile.aspectVariation.planar;
      const individualVariation = bounds[0]
        + unit(seed0, 3 + axis) * (bounds[1] - bounds[0]);
      // Small stones are modestly flatter, while the entire set stays close
      // to spherical rather than becoming the previous stack of flat slabs.
      const flattening = axis === 2 ? 0.84 + 0.18 * sizeRatio : 1;
      return value * layout.size * individualVariation * flattening;
    });
    const horizontalCenter = [
      layout.center[0] + (unit(seed1, 8) - 0.5) * 0.035,
      layout.center[1] + (unit(seed1, 10) - 0.5) * 0.035,
    ];
    const supportRadius = Math.max(
      specimen.metrics.maximumRadius * scale[0],
      specimen.metrics.maximumRadius * scale[1],
    );
    const yaw = unit(seed1, 9) * Math.PI * 2;
    const pitch = (unit(seed0, 11) * 2 - 1) * profile.tilt;
    const roll = (unit(seed1, 12) * 2 - 1) * profile.tilt;
    const orientation = modelMatrix([0, 0, 0], yaw, pitch, roll, scale);
    const bounds = transformedBounds(specimen.packet, orientation);
    const formExtents = bounds.maximum.map((value, axis) => value - bounds.minimum[axis]);
    const formAspectRatio = formExtents[2] / Math.max(formExtents[0], formExtents[1]);
    const localMidZ = (bounds.minimum[2] + bounds.maximum[2]) * 0.5;
    const halfHeight = (bounds.maximum[2] - bounds.minimum[2]) * 0.5;
    const collisionRadius = supportRadius * 0.72;
    const groundTranslation = -bounds.minimum[2];
    let translationZ = groundTranslation;
    const candidates = [];
    for (const support of individuals) {
      const distance = Math.hypot(
        horizontalCenter[0] - support.center[0], horizontalCenter[1] - support.center[1],
      );
      const horizontalSum = collisionRadius + support.collisionRadius;
      if (distance >= horizontalSum) continue;
      const horizontalPart = distance / horizontalSum;
      const verticalSum = halfHeight + support.halfHeight;
      const requiredCenterZ = support.proxyCenterZ
        + verticalSum * Math.sqrt(Math.max(0, 1 - horizontalPart * horizontalPart));
      const requiredTranslation = requiredCenterZ - localMidZ;
      candidates.push({ supportIndex: support.index, requiredTranslation });
      translationZ = Math.max(translationZ, requiredTranslation);
    }
    if (layer > 0) {
      translationZ = Math.max(groundTranslation, translationZ - layout.size * 0.42);
    }
    const center = [horizontalCenter[0], horizontalCenter[1], translationZ];
    const proxyCenterZ = translationZ + localMidZ;
    const contactSlack = layer > 0 ? layout.size * 0.43 : 2e-6;
    const contacts = candidates
      .filter(({ requiredTranslation }) => (
        Math.abs(requiredTranslation - translationZ) <= contactSlack
      ))
      .map(({ supportIndex }) => {
        const support = individuals[supportIndex];
        const horizontalPart = Math.hypot(
          center[0] - support.center[0], center[1] - support.center[1],
        ) / (collisionRadius + support.collisionRadius);
        const verticalPart = (proxyCenterZ - support.proxyCenterZ) / (halfHeight + support.halfHeight);
        return Object.freeze({ supportIndex, normalizedSeparation: Math.hypot(horizontalPart, verticalPart) });
      });
    const packet = Object.freeze({
      ...specimen.packet,
      id: `stone:pile:${profile.id}:${individualIndex}`,
      object_id: index + 1,
      _modelMatrix: modelMatrix(center, yaw, pitch, roll, scale),
      rock_material_gpu: Object.freeze({
        ...specimen.packet.rock_material_gpu,
        speciesIndex,
      }),
    });
    meshes.push(packet);
    individuals.push(Object.freeze({
      index, speciesIndex, speciesId: profile.id, individualIndex, layer, identity,
      seed: Object.freeze([seed0, seed1]), center: Object.freeze(center.slice()),
      scale: Object.freeze(scale), yaw, pitch, roll, supportRadius,
      formExtents: Object.freeze(formExtents), formAspectRatio,
      vertexCount: packet.vertices.length / 10,
      triangleCount: packet.indices.length / 3,
      baseHeightSpan: specimen.metrics.baseHeightSpan,
      undersideHeightSpan: specimen.metrics.undersideHeightSpan,
      macroFacetDisplacementSpan: specimen.metrics.macroFacetDisplacementSpan,
      latitudeContourHeightSpan: specimen.metrics.latitudeContourHeightSpan,
      latitudeProfileDisplacementMaximum: specimen.metrics.latitudeProfileDisplacementMaximum,
      thetaPeriodicityError: specimen.metrics.thetaPeriodicityError,
      phiPeriodicityError: specimen.metrics.phiPeriodicityError,
      collisionRadius, halfHeight, proxyCenterZ,
      minimumWorldZ: bounds.minimum[2] + translationZ,
      contacts: Object.freeze(contacts),
      vectorBytes: specimen.vectorBytes,
    }));
  }
  let maximumNormalizedPenetration = 0;
  for (let right = 1; right < individuals.length; right += 1) {
    for (let left = 0; left < right; left += 1) {
      const a = individuals[left]; const b = individuals[right];
      const horizontalPart = Math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
        / (a.collisionRadius + b.collisionRadius);
      const verticalPart = Math.abs(a.proxyCenterZ - b.proxyCenterZ) / (a.halfHeight + b.halfHeight);
      maximumNormalizedPenetration = Math.max(
        maximumNormalizedPenetration, Math.max(0, 1 - Math.hypot(horizontalPart, verticalPart)),
      );
    }
  }
  return Object.freeze({
    kind: 'stone-species-pile:v1',
    profiles: STONE_SPECIES_PROFILES,
    individuals: Object.freeze(individuals),
    meshes: Object.freeze(meshes),
    settlement: Object.freeze({
      maximumNormalizedPenetration,
      floatingCount: individuals.filter((item) => (
        item.minimumWorldZ > 1e-7 && item.contacts.length === 0
      )).length,
    }),
  });
}
