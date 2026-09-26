import { proGenDistributionReference, sampleProGenDistributionReference } from './vf-pro-gen-distribution-reference.mjs';
const MAX_VERTEX_BUDGET = 393_216;
const MAX_INDEX_BUDGET = 2_359_296;
const KIND_TRUNK = 0;
const KIND_CROWN = 1;
const KIND_BRANCH = 2;
const KIND_FOLIAGE = 3;
const KIND_TWIG = 4;
const LEAVES_PER_CLUSTER = 2;
const LEAF_PARAMETER_STRIDE = 9;
const LEAF_VERTEX_COUNT = 12;
const LEAF_INDEX_COUNT = 66;
const LEAF_OUTLINES = Object.freeze({
  ovate: Object.freeze([0.42, 0.76]),
  oak: Object.freeze([0.14, 0.24, 0.34, 0.45, 0.55, 0.66, 0.77, 0.88]),
  birch: Object.freeze([0.13, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89]),
  beech: Object.freeze([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80, 0.90]),
});
const LEAF_CONTACT_EPSILON = 1.0e-5;
const LEAF_COLLISION_CELL_SIZE = 0.04;
const WOOD_RING_SIDES = 12;
const DETAILED_JUNCTION_RING_SIDES = 20;
const TRUNK_BARK_RING_SIDES = 30;
const TRUNK_BARK_STEPS = 64;
const BRANCH_RING_SIDES = 10;
const TWIG_RING_SIDES = 8;
const DETAILED_TRUNK_BARK_RING_SIDES = 88;
const DETAILED_TRUNK_BARK_STEPS = 384;
const DETAILED_BRANCH_RING_SIDES = 14;
const DETAILED_TWIG_RING_SIDES = 10;

function add(left, right) {
  return left.map((value, axis) => value + right[axis]);
}

function scale(vector, amount) {
  return vector.map((value) => value * amount);
}

function subtract(left, right) {
  return left.map((value, axis) => value - right[axis]);
}

function dot(left, right) {
  return left.reduce((sum, value, axis) => sum + value * right[axis], 0);
}

function distance(left, right) {
  return Math.hypot(...subtract(left, right));
}

function angleBetween(left, right) {
  return Math.acos(clamp(dot(normalize(left), normalize(right)), -1, 1));
}

function cross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function normalize(vector) {
  const length = Math.hypot(...vector);
  if (!(length > 1.0e-12)) {
    throw new RangeError('tree WebGPU primitive direction must be non-zero');
  }
  return vector.map((value) => value / length);
}

function basis(direction) {
  const axis = Math.abs(direction[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const first = normalize(cross(axis, direction));
  return [first, normalize(cross(direction, first))];
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function hashString(value, initial = 0x811c9dc5) {
  let hash = initial >>> 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 0x01000193) >>> 0;
  }
  return hash;
}

function leafRoot(packet) {
  return createConditionedRoot({
    generator: 'vkf.conditioned',
    version: 1,
    seed: [hashString(packet.treeId), hashString(packet.id, 0x9e3779b9)],
    domain: 'material',
    hierarchy: ['tree:webgpu-leaves', packet.treeId],
    lod: packet.detailLevel,
    channel: 'leaf-mesh',
  });
}

function boundedNormal(node, lane, mean, standardDeviation, minimum, maximum) {
  return clamp(
    sampleNormalReference(node, [0, lane], { mean, standardDeviation }),
    minimum,
    maximum,
  );
}

function requirePacket(packet,allowSparse=false) {
  if (
    packet?.kind !== 'tree-render-packet:v1'
    || !Number.isSafeInteger(packet.primitiveCount)
    || packet.primitiveCount < 1
    || !Array.isArray(packet.primitiveIds)
    || !Array.isArray(packet.curves)
    || !(packet.primitiveKinds instanceof Uint8Array)
    || !(packet.detailLevels instanceof Uint8Array)
    || !(packet.parents instanceof Int32Array)
    || !(packet.transforms instanceof Float32Array)
    || !(packet.baseColors instanceof Float32Array)
    || !(packet.surfaceParams instanceof Float32Array)
    || packet.primitiveIds.length !== packet.primitiveCount
    || packet.curves.length !== packet.primitiveCount
    || packet.primitiveKinds.length !== packet.primitiveCount
    || packet.detailLevels.length !== packet.primitiveCount
    || packet.parents.length !== packet.primitiveCount
    || packet.transforms.length !== packet.primitiveCount * 8
    || packet.baseColors.length !== packet.primitiveCount * 4
    || packet.surfaceParams.length !== packet.primitiveCount * 4
  ) {
    throw new TypeError('tree render packet is required');
  }
  const counts = { trunks: 0, crowns: 0, branches: 0, twigs: 0, foliageClusters: 0 };
  for (const kind of packet.primitiveKinds) {
    if (kind === KIND_TRUNK) counts.trunks += 1;
    else if (kind === KIND_CROWN) counts.crowns += 1;
    else if (kind === KIND_BRANCH) counts.branches += 1;
    else if (kind === KIND_FOLIAGE) counts.foliageClusters += 1;
    else if (kind === KIND_TWIG) counts.twigs += 1;
    else throw new RangeError('tree WebGPU primitive kind is unsupported');
  }
  if (
    counts.trunks !== 1
    || counts.crowns !== 1
    || (allowSparse ? counts.branches > 252 : ![30, 62, 124, 126, 252].includes(counts.branches))
    || counts.twigs < (allowSparse ? 1 : 220)
    || counts.twigs > 1536
    || counts.foliageClusters < counts.twigs
    || counts.foliageClusters > counts.twigs * 9
  ) {
    throw new RangeError('complete tree detail packet is required');
  }
  for (let index = 0; index < packet.primitiveCount; index += 1) {
    const kind = packet.primitiveKinds[index];
    const parent = packet.parents[index];
    if (kind === KIND_BRANCH && !(
      parent >= 0
      && parent < index
      && (packet.primitiveKinds[parent] === KIND_TRUNK
        || packet.primitiveKinds[parent] === KIND_BRANCH)
    )) throw new RangeError('complete tree detail packet is required');
    if (kind === KIND_FOLIAGE && !(
      parent >= 0
      && parent < index
      && packet.primitiveKinds[parent] === KIND_TWIG
    )) throw new RangeError('complete tree detail packet is required');
    if (kind === KIND_TWIG && !(
      parent >= 0
      && parent < index
      && (packet.primitiveKinds[parent] === KIND_TRUNK
        || packet.primitiveKinds[parent] === KIND_BRANCH
        || packet.primitiveKinds[parent] === KIND_TWIG)
    )) throw new RangeError('complete tree detail packet is required');
  }
  const leafParents = new Map();
  for (let index = 0; index < packet.primitiveCount; index += 1) {
    if (packet.primitiveKinds[index] === KIND_FOLIAGE) {
      const parent = packet.parents[index];
      leafParents.set(parent, (leafParents.get(parent) ?? 0) + 1);
    }
  }
  for (let index = 0; index < packet.primitiveCount; index += 1) {
    if (packet.primitiveKinds[index] === KIND_TWIG) {
      const leafCount = leafParents.get(index) ?? 0;
      const isLateralShoot = packet.primitiveIds[index].includes(':branch:shoot:');
      const bounds = isLateralShoot
        ? packet.profile.twig.shootLeafCountBounds
        : packet.profile.twig.terminalLeafCountBounds;
      if (leafCount < bounds[0] || leafCount > bounds[1]) {
        throw new RangeError('complete tree detail packet is required');
      }
    }
  }
  return counts;
}

function requireBudgets(vertexBudget, indexBudget) {
  if (
    !Number.isSafeInteger(vertexBudget)
    || vertexBudget < 0
    || vertexBudget > MAX_VERTEX_BUDGET
  ) {
    throw new RangeError(`tree WebGPU vertexBudget must be from 0 through ${MAX_VERTEX_BUDGET}`);
  }
  if (
    !Number.isSafeInteger(indexBudget)
    || indexBudget < 0
    || indexBudget > MAX_INDEX_BUDGET
  ) {
    throw new RangeError(`tree WebGPU indexBudget must be from 0 through ${MAX_INDEX_BUDGET}`);
  }
}

function meshBuilder(vertexBudget, indexBudget, usage) {
  const vertices = [];
  const indices = [];
  const uvs = [];
  const roughness = [];
  function reserve(vertexCount, indexCount) {
    if (usage.vertices + vertexCount > vertexBudget) {
      throw new RangeError('tree WebGPU vertex budget is exhausted');
    }
    if (usage.indices + indexCount > indexBudget) {
      throw new RangeError('tree WebGPU index budget is exhausted');
    }
    usage.vertices += vertexCount;
    usage.indices += indexCount;
  }
  function vertex(position, normal, color, surfaceRoughness, uv = [0, 0]) {
    const index = vertices.length / 10;
    vertices.push(...position, ...normal, ...color);
    uvs.push(...uv);
    roughness.push(surfaceRoughness);
    return index;
  }
  return { vertices, indices, uvs, roughness, reserve, vertex, vertexBudget };
}

function woodTaper(kind) {
  if (kind === KIND_TRUNK) return 0.64;
  if (kind === KIND_BRANCH) return 0.46;
  if (kind === KIND_TWIG) return 0.24;
  throw new RangeError('tree WebGPU wood kind is unsupported');
}

function pathState(curve) {
  const distances = [0];
  for (let index = 1; index < curve.points.length; index += 1) {
    distances.push(distances.at(-1) + distance(curve.points[index - 1], curve.points[index]));
  }
  return { curve, distances, length: distances.at(-1) };
}

function samplePath(path, along) {
  const bounded = clamp(along, 0, path.length);
  let segment = path.distances.length - 2;
  for (let index = 0; index < path.distances.length - 1; index += 1) {
    if (bounded <= path.distances[index + 1] + 1e-12) {
      segment = index;
      break;
    }
  }
  const start = path.distances[segment];
  const span = Math.max(path.distances[segment + 1] - start, 1e-12);
  const fraction = clamp((bounded - start) / span, 0, 1);
  return {
    point: add(
      scale(path.curve.points[segment], 1 - fraction),
      scale(path.curve.points[segment + 1], fraction),
    ),
    tangent: normalize(add(
      scale(path.curve.tangents[Math.max(0, segment - 1)] ?? path.curve.tangents[segment], 1 - fraction),
      scale(path.curve.tangents[segment] ?? path.curve.tangents.at(-1), fraction),
    )),
  };
}

function closestPathDistance(path, point) {
  let best = { squared: Number.POSITIVE_INFINITY, along: 0 };
  for (let segment = 0; segment < path.curve.points.length - 1; segment += 1) {
    const start = path.curve.points[segment];
    const delta = subtract(path.curve.points[segment + 1], start);
    const squaredLength = dot(delta, delta);
    const fraction = clamp(dot(subtract(point, start), delta) / squaredLength, 0, 1);
    const candidate = add(start, scale(delta, fraction));
    const squared = dot(subtract(point, candidate), subtract(point, candidate));
    if (squared < best.squared) {
      best = {
        squared,
        along: path.distances[segment] + Math.sqrt(squaredLength) * fraction,
      };
    }
  }
  return best.along;
}

function parallelTransportBasis(path, along) {
  let tangent = samplePath(path, 0).tangent;
  let [first] = basis(tangent);
  const stations = [...path.distances.filter((value) => value > 0 && value < along), along];
  for (const station of stations) {
    tangent = samplePath(path, station).tangent;
    const projected = subtract(first, scale(tangent, dot(first, tangent)));
    first = Math.hypot(...projected) > 1e-9 ? normalize(projected) : basis(tangent)[0];
  }
  return [first, normalize(cross(tangent, first))];
}

function appendPartitionedFork(builder, ports, nodePoint, barkMaterialAt) {
  const junctionRingSides = ports[0].target.indices.length;
  if (ports.some(({ target }) => target.indices.length !== junctionRingSides)) {
    throw new RangeError('tree WebGPU junction port rings must have one resolution');
  }
  const ringSides = Array.from({ length: junctionRingSides }, (_, side) => side);
  const isLateralGraft = ports.some(({ role }) => role === 'continuation');
  const innerLoops = ports.map(({ target, record, role }) => {
    const center = target.positions.reduce((sum, point) => add(sum, point), [0, 0, 0])
      .map((value) => value / junctionRingSides);
    const lateralChild = isLateralGraft && role === 'child';
    const innerCenter = lateralChild
      ? add(scale(nodePoint, 0.34), scale(center, 0.66))
      : add(scale(nodePoint, 0.72), scale(center, 0.28));
    builder.reserve(junctionRingSides, 0);
    return ringSides.map((side) => {
      const radial = subtract(target.positions[side], center);
      const position = add(innerCenter, radial);
      const sourceVertex = target.indices[side];
      const material = barkMaterialAt(position, record, builder.roughness[sourceVertex]);
      const sourceColor = builder.vertices.slice(sourceVertex * 10 + 6, sourceVertex * 10 + 10);
      const sourceNormal = builder.vertices.slice(sourceVertex * 10 + 3, sourceVertex * 10 + 6);
      return builder.vertex(
        position,
        normalize(add(scale(sourceNormal, 0.85), scale(normalize(radial), 0.15))),
        material.color.map((value, channel) => channel === 3 ? value
          : value * 0.15 + sourceColor[channel] * 0.85),
        material.roughness * 0.15 + builder.roughness[sourceVertex] * 0.85,
        [side / junctionRingSides, target.barkV],
      );
    });
  });
  const continuationIndex = ports[1].target.radius >= ports[2].target.radius ? 1 : 2;
  const lateralIndex = continuationIndex === 1 ? 2 : 1;
  const incoming = innerLoops[0];
  const vertexPoint = (vertex) => builder.vertices.slice(vertex * 10, vertex * 10 + 3);
  function alignCycle(outer, inner) {
    const candidates = [];
    for (const source of [inner, [...inner].reverse()]) {
      for (let rotation = 0; rotation < source.length; rotation += 1) {
        candidates.push([...source.slice(rotation), ...source.slice(0, rotation)]);
      }
    }
    return candidates.reduce((best, candidate) => {
      const score = outer.reduce((sum, vertex, side) => (
        sum + distance(
          vertexPoint(vertex),
          vertexPoint(candidate[Math.floor(side * candidate.length / outer.length)]),
        )
      ), 0);
      return score < best.score ? { value: candidate, score } : best;
    }, { value: inner, score: Number.POSITIVE_INFINITY }).value;
  }
  const continuation = alignCycle(incoming, innerLoops[continuationIndex]);
  const lateral = innerLoops[lateralIndex];
  const triangles = [];
  const lateralCenter = lateral.reduce((sum, vertex) => add(sum, vertexPoint(vertex)), [0, 0, 0])
    .map((value) => value / junctionRingSides);
  let graftStart = 0;
  let graftDistance = Number.POSITIVE_INFINITY;
  for (let start = 0; start < junctionRingSides; start += 1) {
    const patchVertices = [];
    for (let offset = 0; offset <= 1; offset += 1) {
      patchVertices.push(incoming[(start + offset) % junctionRingSides]);
      patchVertices.push(continuation[(start + offset) % junctionRingSides]);
    }
    const center = patchVertices.reduce((sum, vertex) => add(sum, vertexPoint(vertex)), [0, 0, 0])
      .map((value) => value / patchVertices.length);
    const candidate = distance(center, lateralCenter);
    if (candidate < graftDistance) {
      graftDistance = candidate;
      graftStart = start;
    }
  }
  const omitted = new Set([graftStart]);
  for (let side = 0; side < junctionRingSides; side += 1) {
    if (omitted.has(side)) continue;
    const next = (side + 1) % junctionRingSides;
    triangles.push(
      [incoming[side], incoming[next], continuation[side]],
      [incoming[next], continuation[next], continuation[side]],
    );
  }
  const graftBoundary = [
    ...Array.from({ length: 2 }, (_, offset) => incoming[(graftStart + offset) % junctionRingSides]),
    ...Array.from({ length: 2 }, (_, offset) => (
      continuation[(graftStart + 1 - offset) % junctionRingSides]
    )),
  ];
  function stitch(outer, inner) {
    const alignedInner = alignCycle(outer, inner);
    let outerIndex = 0;
    let innerIndex = 0;
    while (outerIndex < outer.length || innerIndex < inner.length) {
      const outerNext = (outerIndex + 1) / outer.length;
      const innerNext = (innerIndex + 1) / inner.length;
      if (outerIndex < outer.length && (innerIndex >= inner.length || outerNext <= innerNext)) {
        triangles.push([
          outer[outerIndex % outer.length],
          outer[(outerIndex + 1) % outer.length],
          alignedInner[innerIndex % alignedInner.length],
        ]);
        outerIndex += 1;
      } else {
        triangles.push([
          outer[outerIndex % outer.length],
          alignedInner[(innerIndex + 1) % alignedInner.length],
          alignedInner[innerIndex % alignedInner.length],
        ]);
        innerIndex += 1;
      }
    }
  }
  stitch(graftBoundary, lateral);
  ports.forEach(({ target }, portIndex) => stitch(target.indices, innerLoops[portIndex]));
  const edgeTriangles = new Map();
  triangles.forEach((triangle, triangleIndex) => {
    for (let edge = 0; edge < 3; edge += 1) {
      const left = triangle[edge];
      const right = triangle[(edge + 1) % 3];
      const key = [left, right].sort((a, b) => a - b).join(':');
      if (!edgeTriangles.has(key)) edgeTriangles.set(key, []);
      edgeTriangles.get(key).push({ triangleIndex, left, right });
    }
  });
  const flips = new Array(triangles.length).fill(null);
  flips[0] = false;
  const pending = [0];
  while (pending.length > 0) {
    const triangleIndex = pending.pop();
    const triangle = triangles[triangleIndex];
    for (let edge = 0; edge < 3; edge += 1) {
      const left = triangle[edge];
      const right = triangle[(edge + 1) % 3];
      const key = [left, right].sort((a, b) => a - b).join(':');
      for (const neighbor of edgeTriangles.get(key)) {
        if (neighbor.triangleIndex === triangleIndex || flips[neighbor.triangleIndex] !== null) continue;
        const currentLeft = flips[triangleIndex] ? right : left;
        const currentRight = flips[triangleIndex] ? left : right;
        flips[neighbor.triangleIndex] = neighbor.left === currentLeft && neighbor.right === currentRight;
        pending.push(neighbor.triangleIndex);
      }
    }
  }
  const oriented = triangles.map((triangle, index) => (
    flips[index] ? [triangle[0], triangle[2], triangle[1]] : triangle
  ));
  const outwardScore = oriented.reduce((sum, triangle) => {
    const points = triangle.map(vertexPoint);
    const normal = cross(subtract(points[1], points[0]), subtract(points[2], points[0]));
    const center = points.reduce((total, point) => add(total, point), [0, 0, 0])
      .map((value) => value / 3);
    return sum + dot(normal, subtract(center, nodePoint));
  }, 0);
  if (outwardScore < 0) oriented.forEach((triangle) => triangle.splice(1, 2, triangle[2], triangle[1]));
  builder.reserve(0, oriented.length * 3);
  builder.indices.push(...oriented.flat());
  return Object.freeze({
    triangleCount: oriented.length,
    minimumRadialScale: 1,
    maximumRadialScale: 1,
    connectorComponents: 0,
    lateralGraftReachRatio: isLateralGraft
      ? distance(nodePoint, lateralCenter) / Math.max(ports[0].target.radius, 1e-12)
      : 0,
  });
}

function appendWoodyNetwork(builder, packet, bark,allowSparse=false,barkDetail=null) {
  const woody = [];
  for (let primitive = 0; primitive < packet.primitiveCount; primitive += 1) {
    const kind = packet.primitiveKinds[primitive];
    if (![KIND_TRUNK, KIND_BRANCH, KIND_TWIG].includes(kind)) continue;
    const transform = packet.transforms.subarray(primitive * 8, primitive * 8 + 8);
    const path = pathState(packet.curves[primitive]);
    woody.push({
      primitive,
      kind,
      parent: packet.parents[primitive],
      transform,
      path,
      taper: woodTaper(kind),
      endRadius: transform[7] * woodTaper(kind),
      color: Array.from(packet.baseColors.subarray(primitive * 4, primitive * 4 + 4)),
      roughness: packet.surfaceParams[primitive * 4],
      exclusions: [],
      baseBarkV: 0,
    });
  }
  const byPrimitive = new Map(woody.map((record) => [record.primitive, record]));
  const groupsByParent = new Map();
  for (const child of woody) {
    const parent = byPrimitive.get(child.parent);
    if (!parent) continue;
    const attachment = closestPathDistance(parent.path, child.path.curve.points[0]);
    child.attachment = attachment;
    child.baseBarkV = parent.baseBarkV + attachment / packet.targetPathLength;
    if (!groupsByParent.has(parent.primitive)) groupsByParent.set(parent.primitive, []);
    const groups = groupsByParent.get(parent.primitive);
    let group = groups.find((candidate) => Math.abs(candidate.along - attachment) < 1e-5);
    if (!group) {
      group = { parent, along: attachment, children: [] };
      groups.push(group);
    }
    group.children.push(child);
  }
  const junctionPlans = [];
  for (const groups of groupsByParent.values()) {
    groups.sort((left, right) => left.along - right.along);
    groups.forEach((group, index) => {
      const parentRadius = group.parent.transform[7] * (
        1 + (group.parent.taper - 1) * group.along / group.parent.path.length
      );
      const before = index === 0 ? group.along : group.along - groups[index - 1].along;
      const after = index === groups.length - 1
        ? group.parent.path.length - group.along
        : groups[index + 1].along - group.along;
      const isTerminalSplit = after < 1e-5;
      const available = Math.min(
        Math.max(before * 0.28, 1e-5),
        isTerminalSplit ? Number.POSITIVE_INFINITY : Math.max(after * 0.28, 1e-5),
      );
      const half = Math.min(parentRadius * 1.35, available, group.parent.path.length * 0.08);
      const parentStart = Math.max(0, group.along - half);
      const parentEnd = isTerminalSplit ? group.along : Math.min(group.parent.path.length, group.along + half);
      group.parent.exclusions.push([parentStart, parentEnd]);
      const childStarts = group.children.map((child) => {
        const lateralTwig = !isTerminalSplit && child.kind === KIND_TWIG;
        const cut = lateralTwig
          ? Math.min(
            Math.max(parentRadius * 0.45, child.transform[7] * 2.25),
            child.path.length * 0.24,
          )
          : Math.min(Math.max(half * 0.5, child.transform[7] * 0.75), child.path.length * 0.04);
        child.exclusions.push([0, cut]);
        return { child, cut };
      });
      junctionPlans.push({ ...group, parentStart, parentEnd, isTerminalSplit, childStarts });
    });
  }
  for (const record of woody) {
    const endpointChildren = (groupsByParent.get(record.primitive) ?? [])
      .filter((group) => record.path.length - group.along < 1e-5)
      .flatMap((group) => group.children);
    if (endpointChildren.length > 0) {
      const largestChild = Math.max(...endpointChildren.map((child) => child.transform[7]));
      record.endRadius = Math.min(
        record.transform[7] * 0.96,
        Math.max(record.endRadius, largestChild * 1.05),
      );
    }
  }
  const ringCache = new Map();
  const trunkBaseRadius = woody.find(({ kind }) => kind === KIND_TRUNK).transform[7];
  const networkBaseColor = woody.find(({ kind }) => kind === KIND_TRUNK).color;
  const networkBaseRoughness = woody.find(({ kind }) => kind === KIND_TRUNK).roughness;
  const detailedBark = barkDetail ?? builder.vertexBudget > 65_536;
  const junctionRingSides = detailedBark
    ? DETAILED_JUNCTION_RING_SIDES : WOOD_RING_SIDES;
  const noiseSeed = bark.phase * 0.15915494309189535;
  function latticeNoise(x, y, z) {
    const value = Math.sin(
      x * 127.1 + y * 311.7 + z * 74.7 + noiseSeed * 101.3,
    ) * 43758.5453123;
    return (value - Math.floor(value)) * 2 - 1;
  }
  function valueNoise3(position, frequency) {
    const x = position[0] / trunkBaseRadius * frequency;
    const y = position[1] / trunkBaseRadius * frequency;
    const z = position[2] / trunkBaseRadius * frequency;
    const ix = Math.floor(x); const iy = Math.floor(y); const iz = Math.floor(z);
    const fx = x - ix; const fy = y - iy; const fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const uz = fz * fz * (3 - 2 * fz);
    const mix = (left, right, amount) => left + (right - left) * amount;
    const lower = mix(
      mix(latticeNoise(ix, iy, iz), latticeNoise(ix + 1, iy, iz), ux),
      mix(latticeNoise(ix, iy + 1, iz), latticeNoise(ix + 1, iy + 1, iz), ux),
      uy,
    );
    const upper = mix(
      mix(latticeNoise(ix, iy, iz + 1), latticeNoise(ix + 1, iy, iz + 1), ux),
      mix(latticeNoise(ix, iy + 1, iz + 1), latticeNoise(ix + 1, iy + 1, iz + 1), ux),
      uy,
    );
    return mix(lower, upper, uz);
  }
  function spatialBark(position) {
    const field = {
      coarse: valueNoise3(position, 0.24),
      medium: valueNoise3(position, 0.72),
      fine: valueNoise3(position, 1.85),
      micro: valueNoise3(position, 5.2),
    };
    // A narrow isosurface in the shared 3-D field reads as fine bark cracking.
    // Unlike path-local stripes, it crosses trunk/branch ownership boundaries.
    field.crease = Math.pow(Math.max(0,
      1 - Math.abs(field.fine * 0.42 + field.micro * 0.58) * 8.5), 2.4);
    return field;
  }
  function sharedBarkMaterial(position, record, sourceRoughness = record.roughness) {
    const spatial = spatialBark(position);
    const breakup = spatial.coarse * 0.18 + spatial.medium * 0.26
      + spatial.fine * 0.34 + spatial.micro * 0.22;
    const radiusRatio = clamp(record.transform[7] / Math.max(trunkBaseRadius, 1e-9), 0, 1);
    const thinSmoothness = 0.16 * (1 - Math.sqrt(radiusRatio));
    return {
      color: networkBaseColor.map((value, channel) => channel === 3 ? value
        : clamp(value + bark.colorVariation * breakup - spatial.crease * 0.08, 0, 1)),
      roughness: clamp(networkBaseRoughness - thinSmoothness
        + bark.roughnessVariation * (spatial.fine * 0.34 + spatial.micro * 0.18),
      0.42, 0.98),
      spatial,
      sourceRoughness,
    };
  }
  function radiusAt(record, along) {
    return record.transform[7]
      + (record.endRadius - record.transform[7]) * along / record.path.length;
  }
  function barkSample(angle, barkAlong, position, detailStrength) {
    const detail = bark.detailProfile;
    const ridge = Math.sin(bark.ridgeCount * angle + bark.phase + barkAlong * bark.grainTurns)
      + 0.35 * Math.sin((bark.ridgeCount + 3) * angle - bark.phase * 0.7 + barkAlong * 9);
    const ridgeNoise = ridge / 1.35;
    const fissure = Math.pow(Math.max(0, Math.cos(
      (bark.ridgeCount - 1) * angle - bark.phase * 0.43
        + barkAlong * detail.fissureAxialFrequency,
    )), detail.fissureSharpness);
    const grain = Math.sin(barkAlong * detail.grainAxialFrequency + angle * 2 + bark.phase * 1.7);
    const microRidge = Math.sin(
      (bark.ridgeCount + 5) * angle + barkAlong * detail.microRidgeAxialFrequency
        - bark.phase * 0.35,
    );
    // Tree-space value noise stays continuous through forks. Frequencies are relative
    // to trunk radius, so every octave is represented by the generated geometry.
    const spatial = spatialBark(position);
    const stoneLikeBreakup = spatial.coarse * 0.18 + spatial.medium * 0.26
      + spatial.fine * 0.34 + spatial.micro * 0.22;
    return {
      ridgeNoise,
      fissure,
      materialNoise: clamp(
        ridgeNoise * 0.22 + grain * 0.08 + microRidge * detail.microRidgeWeight * 0.5
          + stoneLikeBreakup * (0.45 + detailStrength * 0.55)
          - fissure * 0.9,
        -1,
        1,
      ),
      displacement: bark.ridgeAmplitude * detailStrength * (
        ridgeNoise * 0.52 + microRidge * detail.microRidgeWeight * 0.45
          + spatial.coarse * 0.10 + spatial.medium * 0.18 + spatial.fine * 0.20
          + spatial.micro * 0.08 - spatial.crease * 0.11
          - fissure * 0.48
      ),
      spatialCoarse: spatial.coarse,
      spatialMid: spatial.medium,
      spatialFine: spatial.fine,
      spatialMicro: spatial.micro,
    };
  }
  function ring(record, along, sides = WOOD_RING_SIDES) {
    const key = `${record.primitive}:${along.toFixed(10)}:${sides}`;
    if (ringCache.has(key)) return ringCache.get(key);
    const state = samplePath(record.path, along);
    const [first, second] = parallelTransportBasis(record.path, along);
    const radius = radiusAt(record, along);
    const indices = [];
    const positions = [];
    builder.reserve(sides, 0);
    for (let side = 0; side < sides; side += 1) {
      const angle = Math.PI * 2 * side / sides;
      const radial = add(scale(first, Math.cos(angle)), scale(second, Math.sin(angle)));
      const barkAlong = record.baseBarkV + along / packet.targetPathLength;
      const baseSurface = add(state.point, scale(radial, radius));
      const radiusRatio = clamp(radius / Math.max(trunkBaseRadius, 1e-9), 0, 1);
      const detailStrength = 0.12 + 0.88 * Math.sqrt(radiusRatio);
      const field = barkSample(angle, barkAlong, baseSurface, detailStrength);
      const angleStep = 0.018;
      const alongStep = 0.0015;
      const circumferential = normalize(cross(state.tangent, radial));
      const angularSlope = (
        barkSample(angle + angleStep, barkAlong,
          add(baseSurface, scale(circumferential, radius * angleStep)), detailStrength).displacement
        - barkSample(angle - angleStep, barkAlong,
          add(baseSurface, scale(circumferential, -radius * angleStep)), detailStrength).displacement
      ) / (2 * angleStep);
      const axialSlope = (
        barkSample(angle, barkAlong + alongStep,
          add(baseSurface, scale(state.tangent, packet.targetPathLength * alongStep)), detailStrength).displacement
        - barkSample(angle, barkAlong - alongStep,
          add(baseSurface, scale(state.tangent, -packet.targetPathLength * alongStep)), detailStrength).displacement
      ) / (2 * alongStep) * radius / packet.targetPathLength;
      const surfaceNormal = normalize(add(
        add(radial, scale(circumferential, -angularSlope * bark.normalStrength)),
        scale(state.tangent, -axialSlope * bark.normalStrength),
      ));
      const ringRadius = radius * (1 + field.displacement);
      const axialOffset = 0;
      const position = add(
        add(state.point, scale(radial, ringRadius)),
        scale(state.tangent, axialOffset),
      );
      const sharedMaterial = sharedBarkMaterial(baseSurface, record);
      const vertexColor = sharedMaterial.color.map((value, channel) => (
        channel === 3 ? value : clamp(value + bark.colorVariation
          * (field.materialNoise - (field.spatialCoarse * 0.18 + field.spatialMid * 0.26
            + field.spatialFine * 0.34 + field.spatialMicro * 0.22)) * 0.38, 0, 1)
      ));
      const radiusSmoothness = 0.18 * (1 - Math.sqrt(radiusRatio));
      indices.push(builder.vertex(
        position,
        surfaceNormal,
        vertexColor,
        clamp(sharedMaterial.roughness - radiusSmoothness + bark.roughnessVariation
          * (field.fissure * 0.34 - field.ridgeNoise * 0.12
            + field.spatialFine * 0.16 + field.spatialMicro * 0.12), 0.42, 0.98),
        [side / sides, barkAlong],
      ));
      positions.push(position);
    }
    const created = { indices, positions, state, radius, barkV: record.baseBarkV + along / packet.targetPathLength };
    ringCache.set(key, created);
    return created;
  }
  function connectRings(first, second) {
    const firstCount = first.indices.length;
    const secondCount = second.indices.length;
    builder.reserve(0, (firstCount + secondCount) * 3);
    let firstSide = 0;
    let secondSide = 0;
    while (firstSide < firstCount || secondSide < secondCount) {
      const firstNext = (firstSide + 1) / firstCount;
      const secondNext = (secondSide + 1) / secondCount;
      const firstIndex = first.indices[firstSide % firstCount];
      const secondIndex = second.indices[secondSide % secondCount];
      if (Math.abs(firstNext - secondNext) < 1e-10) {
        const firstNextIndex = first.indices[(firstSide + 1) % firstCount];
        const secondNextIndex = second.indices[(secondSide + 1) % secondCount];
        builder.indices.push(
          firstIndex, firstNextIndex, secondIndex,
          firstNextIndex, secondNextIndex, secondIndex,
        );
        firstSide += 1;
        secondSide += 1;
      } else if (firstNext < secondNext) {
        builder.indices.push(firstIndex, first.indices[(firstSide + 1) % firstCount], secondIndex);
        firstSide += 1;
      } else {
        builder.indices.push(firstIndex, second.indices[(secondSide + 1) % secondCount], secondIndex);
        secondSide += 1;
      }
    }
  }
  function capRing(target, direction) {
    builder.reserve(1, target.indices.length * 3);
    const center = target.state.point;
    const centerIndex = builder.vertex(center, direction, [0.25, 0.13, 0.055, 1], 0.82, [0.5, target.barkV]);
    for (let side = 0; side < target.indices.length; side += 1) {
      const next = (side + 1) % target.indices.length;
      if (dot(direction, target.state.tangent) < 0) {
        builder.indices.push(centerIndex, target.indices[next], target.indices[side]);
      } else {
        builder.indices.push(centerIndex, target.indices[side], target.indices[next]);
      }
    }
  }
  const terminalPrimitives = new Set(woody.map(({ primitive }) => primitive));
  for (const child of woody) terminalPrimitives.delete(child.parent);
  for (const record of woody) {
    const exclusions = record.exclusions
      .sort((left, right) => left[0] - right[0]);
    const spans = [];
    let cursor = 0;
    for (const [start, end] of exclusions) {
      if (start > cursor + 1e-8) spans.push([cursor, start]);
      cursor = Math.max(cursor, end);
    }
    if (cursor < record.path.length - 1e-8) spans.push([cursor, record.path.length]);
    for (const [start, end] of spans) {
      const baselineBarkSteps = record.kind === KIND_TRUNK
        ? detailedBark ? DETAILED_TRUNK_BARK_STEPS : TRUNK_BARK_STEPS
        : record.kind === KIND_BRANCH ? detailedBark ? 4 : 2 : detailedBark ? 2 : 1;
      const adaptiveBarkSteps = detailedBark
        ? Math.min(record.kind === KIND_BRANCH ? 10 : record.kind === KIND_TWIG ? 2 : 24,
          Math.max(1, Math.ceil(
            (end - start) / Math.max(radiusAt(record, (start + end) * 0.5), 1e-9)
              * (record.kind === KIND_BRANCH ? 0.52 : record.kind === KIND_TWIG ? 0.16 : 1),
        )))
        : 1;
      const barkSteps = Math.max(baselineBarkSteps, adaptiveBarkSteps);
      const uniformStations = Array.from({ length: barkSteps - 1 }, (_, step) => (
        start + (end - start) * (step + 1) / barkSteps
      ));
      const stations = [start, ...record.path.distances.filter((value) => (
        value > start + 1e-8 && value < end - 1e-8
      )), ...uniformStations, end].sort((left, right) => left - right)
        .filter((value, index, values) => index === 0 || value - values[index - 1] > 1e-8);
      let prior = ring(record, stations[0], junctionRingSides);
      for (let stationIndex = 1; stationIndex < stations.length; stationIndex += 1) {
        const station = stations[stationIndex];
        const sides = stationIndex === stations.length - 1 ? junctionRingSides
          : record.kind === KIND_TRUNK
            ? detailedBark ? DETAILED_TRUNK_BARK_RING_SIDES : TRUNK_BARK_RING_SIDES
            : record.kind === KIND_BRANCH
              ? detailedBark ? DETAILED_BRANCH_RING_SIDES : BRANCH_RING_SIDES
              : detailedBark ? DETAILED_TWIG_RING_SIDES : TWIG_RING_SIDES;
        const next = ring(record, station, sides);
        connectRings(prior, next);
        prior = next;
      }
    }
    if (record.kind === KIND_TRUNK) capRing(ring(record, 0, junctionRingSides),
      scale(samplePath(record.path, 0).tangent, -1));
    if (terminalPrimitives.has(record.primitive)) {
      capRing(ring(record, record.path.length, junctionRingSides),
        samplePath(record.path, record.path.length).tangent);
    }
  }
  const junctions = [];
  junctionPlans.forEach((plan, junctionIndex) => {
    const ports = [];
    ports.push({ role: 'incoming', record: plan.parent, along: plan.parentStart });
    if (!plan.isTerminalSplit) {
      ports.push({ role: 'continuation', record: plan.parent, along: plan.parentEnd });
    }
    for (const { child, cut } of plan.childStarts) {
      ports.push({ role: 'child', record: child, along: cut });
    }
    const portTargets = [];
    const metadataPorts = ports.map((port, portIndex) => {
      const target = ring(port.record, port.along, junctionRingSides);
      portTargets.push(target);
      return Object.freeze({
        role: port.role,
        primitive: port.record.primitive,
        radius: target.radius,
        tangent: Object.freeze([...target.state.tangent]),
        barkV: target.barkV,
        ringVertices: Object.freeze([...target.indices]),
      });
    });
    if (portTargets.length !== 3 && !(allowSparse && portTargets.length === 2)) {
      throw new RangeError('tree WebGPU junction must have two or three connected ports');
    }
    const forkSkin = portTargets.length === 2 ? (() => {
      const before = builder.indices.length;
      connectRings(portTargets[0], portTargets[1]);
      return {triangleCount:(builder.indices.length-before)/3,
        minimumRadialScale:Math.min(...portTargets.map(port=>port.radius)),
        maximumRadialScale:Math.max(...portTargets.map(port=>port.radius)),
        connectorComponents:1,lateralGraftReachRatio:0};
    })() : appendPartitionedFork(
      builder,
      ports.map((port, index) => ({
        target: portTargets[index], record: port.record, role: port.role,
      })),
      samplePath(plan.parent.path, plan.along).point,
      sharedBarkMaterial,
    );
    const partitionedBarkV = metadataPorts.reduce((sum, port) => sum + port.barkV, 0)
      / metadataPorts.length;
    const partitionedIncoming = metadataPorts.find((port) => port.role === 'incoming').tangent;
    junctions.push(Object.freeze({
      point: Object.freeze([...samplePath(plan.parent.path, plan.along).point]),
      ports: Object.freeze(metadataPorts),
      triangles: forkSkin.triangleCount,
      minimumRadialScale: forkSkin.minimumRadialScale,
      maximumRadialScale: forkSkin.maximumRadialScale,
      connectorComponents: forkSkin.connectorComponents,
      lateralGraftReachRatio: forkSkin.lateralGraftReachRatio,
      barkV: partitionedBarkV,
      maximumTangentTurn: Math.max(...metadataPorts
        .filter((port) => port.role !== 'incoming')
        .map((port) => angleBetween(partitionedIncoming, port.tangent))),
      maximumNormalSeamAngle: 0,
    }));
  });
  const edgeUse = new Map();
  for (let offset = 0; offset < builder.indices.length; offset += 3) {
    const triangle = builder.indices.slice(offset, offset + 3);
    for (let edge = 0; edge < 3; edge += 1) {
      const key = [triangle[edge], triangle[(edge + 1) % 3]].sort((a, b) => a - b).join(':');
      edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1);
    }
  }
  return {
    junctions: Object.freeze(junctions),
    topology: Object.freeze({
      ringSides: WOOD_RING_SIDES,
      detailedJunctionRingSides: DETAILED_JUNCTION_RING_SIDES,
      trunkBarkSides: TRUNK_BARK_RING_SIDES,
      trunkBarkSteps: TRUNK_BARK_STEPS,
      branchSides: BRANCH_RING_SIDES,
      twigSides: TWIG_RING_SIDES,
      boundaryEdges: [...edgeUse.values()].filter((count) => count === 1).length,
      nonManifoldEdges: [...edgeUse.values()].filter((count) => count > 2).length,
      internalCaps: 0,
      endpointCaps: terminalPrimitives.size + 1,
      taperedSegments: woody.length,
      minimumTaper: Math.min(...woody.map((record) => record.endRadius / record.transform[7])),
      maximumTaper: Math.max(...woody.map((record) => record.endRadius / record.transform[7])),
      frameTransport: 'parallel-transport',
    }),
  };
}

function addDoubleSidedTriangle(indices, first, second, third) {
  indices.push(first, second, third, third, second, first);
}

function addDoubleSidedQuad(indices, first, second, third, fourth) {
  addDoubleSidedTriangle(indices, first, second, third);
  addDoubleSidedTriangle(indices, first, third, fourth);
}

function segmentSegmentDistance(firstStart, firstEnd, secondStart, secondEnd) {
  const first = subtract(firstEnd, firstStart);
  const second = subtract(secondEnd, secondStart);
  const relative = subtract(firstStart, secondStart);
  const aa = dot(first, first);
  const bb = dot(first, second);
  const cc = dot(second, second);
  const dd = dot(first, relative);
  const ee = dot(second, relative);
  const denominator = aa * cc - bb * bb;
  let firstParameter = denominator > 1.0e-14
    ? clamp((bb * ee - cc * dd) / denominator, 0, 1) : 0;
  let secondParameter = cc > 1.0e-14
    ? clamp((bb * firstParameter + ee) / cc, 0, 1) : 0;
  if (aa > 1.0e-14) {
    firstParameter = clamp((bb * secondParameter - dd) / aa, 0, 1);
  }
  if (cc > 1.0e-14) {
    secondParameter = clamp((bb * firstParameter + ee) / cc, 0, 1);
  }
  const firstPoint = add(firstStart, scale(first, firstParameter));
  const secondPoint = add(secondStart, scale(second, secondParameter));
  return distance(firstPoint, secondPoint);
}

function leafCell(value) {
  return Math.floor(value / LEAF_COLLISION_CELL_SIZE);
}

function leafCellKey(x, y, z) {
  return `${x}:${y}:${z}`;
}

function leafBodyCells(body) {
  const { minimum, maximum } = body;
  const cells = [];
  for (let x = leafCell(minimum[0]); x <= leafCell(maximum[0]); x += 1) {
    for (let y = leafCell(minimum[1]); y <= leafCell(maximum[1]); y += 1) {
      for (let z = leafCell(minimum[2]); z <= leafCell(maximum[2]); z += 1) {
        cells.push(leafCellKey(x, y, z));
      }
    }
  }
  return cells;
}

function leafBodyOverlaps(body, physics) {
  const possible = new Set();
  for (const key of leafBodyCells(body)) {
    for (const index of physics.cells.get(key) ?? []) possible.add(index);
  }
  for (const index of possible) {
    const other = physics.bodies[index];
    if ([0, 1, 2].some((axis) => (
      body.maximum[axis] < other.minimum[axis] - LEAF_CONTACT_EPSILON
      || body.minimum[axis] > other.maximum[axis] + LEAF_CONTACT_EPSILON
    ))) continue;
    for (const first of body.triangles) for (const second of other.triangles) {
      if (leafTrianglesIntersect(first, second)) return true;
    }
  }
  return false;
}

function leafSegmentTriangle(start, end, triangle) {
  const direction = subtract(end, start);
  const edge1 = subtract(triangle[1], triangle[0]);
  const edge2 = subtract(triangle[2], triangle[0]);
  const h = cross(direction, edge2);
  const determinant = dot(edge1, h);
  if (Math.abs(determinant) < 1.0e-10) return false;
  const inverse = 1 / determinant;
  const s = subtract(start, triangle[0]);
  const u = inverse * dot(s, h);
  if (u < LEAF_CONTACT_EPSILON || u > 1 - LEAF_CONTACT_EPSILON) return false;
  const q = cross(s, edge1);
  const v = inverse * dot(direction, q);
  if (v < LEAF_CONTACT_EPSILON || u + v > 1 - LEAF_CONTACT_EPSILON) return false;
  const t = inverse * dot(edge2, q);
  return t > LEAF_CONTACT_EPSILON && t < 1 - LEAF_CONTACT_EPSILON;
}

function leafTrianglesIntersect(first, second) {
  for (let edge = 0; edge < 3; edge += 1) {
    if (leafSegmentTriangle(first[edge], first[(edge + 1) % 3], second)) return true;
    if (leafSegmentTriangle(second[edge], second[(edge + 1) % 3], first)) return true;
  }
  return false;
}

const LEAF_CONTACT_TRIANGLES = Object.freeze([
  [0, 1, 3], [0, 3, 2], [4, 5, 7], [2, 5, 4], [3, 4, 7],
  [5, 8, 9], [5, 9, 6], [6, 9, 10], [6, 10, 7],
  [8, 11, 9], [9, 11, 10],
]);

function leafPositions(parameters, pose) {
  const { attachment, bladeBase, apex, longAxis, wideAxis, normal } = pose;
  const bladeLength = parameters[0];
  const bladeWidth = parameters[1];
  const baseRoundness = parameters[2];
  const asymmetry = parameters[3];
  const camber = parameters[5];
  const petioleHalfWidth = Math.max(bladeWidth * 0.025, bladeLength * 0.008);
  const positions = [
    add(attachment, scale(wideAxis, -petioleHalfWidth)),
    add(attachment, scale(wideAxis, petioleHalfWidth)),
    add(bladeBase, scale(wideAxis, -petioleHalfWidth)),
    add(bladeBase, scale(wideAxis, petioleHalfWidth)),
    bladeBase,
  ];
  for (const station of [0.42, 0.76]) {
    const profile = Math.pow(Math.sin(Math.PI * station), baseRoundness)
      * (1 - station * 0.12);
    const halfWidth = bladeWidth * 0.5 * profile;
    const center = add(
      add(bladeBase, scale(longAxis, bladeLength * station)),
      add(
        scale(wideAxis, bladeWidth * asymmetry * Math.sin(Math.PI * station)),
        scale(normal, camber * Math.sin(Math.PI * station)),
      ),
    );
    positions.push(add(center, scale(wideAxis, -halfWidth)), center,
      add(center, scale(wideAxis, halfWidth)));
  }
  positions.push(apex);
  return positions;
}

function leafContactBody(parameters, pose) {
  const positions = leafPositions(parameters, pose);
  return {
    triangles: LEAF_CONTACT_TRIANGLES.map((triangle) => triangle.map((index) => positions[index])),
    minimum: [0, 1, 2].map((axis) => Math.min(...positions.map((point) => point[axis]))),
    maximum: [0, 1, 2].map((axis) => Math.max(...positions.map((point) => point[axis]))),
  };
}

function addLeafBody(body, physics) {
  const index = physics.bodies.length;
  physics.bodies.push(body);
  for (const key of leafBodyCells(body)) {
    if (!physics.cells.has(key)) physics.cells.set(key, []);
    physics.cells.get(key).push(index);
  }
}

function leafParameters(leafNode, transform, controls) {
  const scaleHint = transform[7];
  const choose = (name, lane, mean, deviation, minimum, maximum) => controls[name]
    ? clamp(sampleProGenDistributionReference(leafNode, [0, lane], controls[name]), minimum, maximum)
    : boundedNormal(leafNode, lane, mean, deviation, minimum, maximum);
  const bladeLength = controls.lengthRatio ? scaleHint * choose('lengthRatio', 0, 0.9, 0.12, 0.6, 1.2)
    : boundedNormal(leafNode, 0, scaleHint * 0.9, scaleHint * 0.12, scaleHint * 0.6, scaleHint * 1.2);
  const widthRatio = choose('widthRatio', 1, 0.46, 0.06, 0.28, 0.65);
  const baseRoundness = choose('roundness', 2, 0.72, 0.07, 0.5, 0.92);
  const asymmetry = choose('asymmetry', 3, 0, 0.05, -0.16, 0.16);
  const petioleRatio = choose('petioleRatio', 4, 0.27, 0.04, 0.16, 0.4);
  const camberRatio = choose('camberRatio', 5, 0, 0.025, -0.08, 0.08);
  const attachment = boundedNormal(leafNode, 6, 0.68, 0.18, 0.12, 0.98);
  const orientationOffset = boundedNormal(leafNode, 7, 0, 0.34, -0.9, 0.9);
  const colorVariation = boundedNormal(leafNode, 8, 0, 0.025, -0.06, 0.06);
  return [
    bladeLength,
    bladeLength * widthRatio,
    baseRoundness,
    asymmetry,
    bladeLength * petioleRatio,
    bladeLength * camberRatio,
    attachment,
    orientationOffset,
    colorVariation,
  ];
}

function leafPose(transform, parameters, leafIndex, bend, twist) {
  const origin = Array.from(transform.slice(0, 3));
  const direction = normalize(Array.from(transform.slice(3, 6)));
  const [first, second] = basis(direction);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const baseAngle = leafIndex * goldenAngle + parameters[7];
  const attachmentRadial = add(
    scale(first, Math.cos(baseAngle)), scale(second, Math.sin(baseAngle)),
  );
  const angle = baseAngle + twist;
  const radial = add(scale(first, Math.cos(angle)), scale(second, Math.sin(angle)));
  const wideAxis = normalize(cross(direction, radial));
  const relaxedLongAxis = normalize(add(
    radial, scale(direction, 0.24 + 0.06 * (leafIndex % 3)),
  ));
  const relaxedNormal = normalize(cross(relaxedLongAxis, wideAxis));
  const longAxis = normalize(add(
    scale(relaxedLongAxis, Math.cos(bend)), scale(relaxedNormal, Math.sin(bend)),
  ));
  const normal = normalize(cross(longAxis, wideAxis));
  const bladeLength = parameters[0];
  const bladeWidth = parameters[1];
  const petioleLength = parameters[4];
  const camber = parameters[5];
  const attachment = add(
    add(origin, scale(direction, transform[6] * parameters[6])),
    scale(attachmentRadial, transform[7] * 0.14),
  );
  const bladeBase = add(attachment, scale(longAxis, petioleLength));
  const apex = add(
    add(bladeBase, scale(longAxis, bladeLength)),
    scale(normal, camber * 0.18),
  );
  return { attachment, bladeBase, apex, longAxis, wideAxis, normal };
}

function resolveLeafPose(transform, parameters, leafIndex, physics) {
  if (!physics.enabled) {
    return leafPose(transform, parameters, leafIndex, 0, 0);
  }
  const candidates = [{ bend: 0, twist: 0, energy: 0 }];
  for (const magnitude of [0.28, 0.56, 0.86, 1.16, 1.42]) {
    for (const sign of [-1, 1]) {
      candidates.push(
        { bend: sign * magnitude, twist: 0, energy: magnitude * magnitude },
        { bend: 0, twist: sign * magnitude, energy: magnitude * magnitude * 0.35 },
        { bend: sign * magnitude, twist: magnitude, energy: magnitude * magnitude * 1.35 },
        { bend: sign * magnitude, twist: -magnitude, energy: magnitude * magnitude * 1.35 },
      );
    }
  }
  candidates.push({ bend: 0, twist: Math.PI, energy: Math.PI * Math.PI * 0.35 });
  candidates.sort((left, right) => left.energy - right.energy);
  for (const candidate of candidates) {
    physics.candidateTests += 1;
    const pose = leafPose(
      transform, parameters, leafIndex, candidate.bend, candidate.twist,
    );
    const body = leafContactBody(parameters, pose);
    if (leafBodyOverlaps(body, physics)) continue;
    addLeafBody(body, physics);
    if (candidate.energy > 0) physics.bentLeaves += 1;
    physics.maximumBend = Math.max(physics.maximumBend, Math.abs(candidate.bend));
    physics.maximumTwist = Math.max(physics.maximumTwist, Math.abs(candidate.twist));
    return pose;
  }
  physics.rejectedLeaves += 1;
  return null;
}

function appendOvateLeaf(builder, color, roughness, parameters, pose, outline) {
  const { attachment, bladeBase, apex, longAxis, wideAxis, normal } = pose;
  const bladeLength = parameters[0];
  const bladeWidth = parameters[1];
  const baseRoundness = parameters[2];
  const asymmetry = parameters[3];
  const camber = parameters[5];
  const leafColor = color.map((value, channel) => (
    channel === 3 ? value : clamp(value + parameters[8], 0, 1)
  ));
  const veinColor = leafColor.map((value, channel) => (
    channel === 3 ? value : clamp(value + (channel === 1 ? 0.065 : -0.025), 0, 1)
  ));
  const petioleHalfWidth = Math.max(bladeWidth * 0.025, bladeLength * 0.008);
  const base = builder.vertices.length / 10;
  builder.vertex(add(attachment, scale(wideAxis, -petioleHalfWidth)), normal, veinColor, roughness - 0.04, [0.48, 0]);
  builder.vertex(add(attachment, scale(wideAxis, petioleHalfWidth)), normal, veinColor, roughness - 0.04, [0.52, 0]);
  builder.vertex(add(bladeBase, scale(wideAxis, -petioleHalfWidth)), normal, veinColor, roughness - 0.04, [0.48, 0.16]);
  builder.vertex(add(bladeBase, scale(wideAxis, petioleHalfWidth)), normal, veinColor, roughness - 0.04, [0.52, 0.16]);
  builder.vertex(bladeBase, normal, veinColor, roughness - 0.08, [0.5, 0.16]);
  const stations = LEAF_OUTLINES[outline];
  stations.forEach((station, stationIndex) => {
    const oval = Math.pow(Math.sin(Math.PI * station), baseRoundness) * (1 - station * 0.12);
    const profile = outline === 'oak'
      ? oval * (0.72 + 0.28 * Math.cos(8 * Math.PI * station) ** 2)
      : outline === 'birch'
        ? Math.max(0.03, (1 - station) ** 0.56) * (0.93 + 0.07 * Math.cos(13 * Math.PI * station))
        : outline === 'beech'
          ? oval * (0.97 + 0.03 * Math.cos(12 * Math.PI * station))
          : oval;
    const halfWidth = bladeWidth * 0.5 * profile;
    const center = add(
      add(bladeBase, scale(longAxis, bladeLength * station)),
      add(
        scale(wideAxis, bladeWidth * asymmetry * Math.sin(Math.PI * station)),
        scale(normal, camber * Math.sin(Math.PI * station)),
      ),
    );
    const secondaryVein = (stationIndex % 2 === 0 ? 1 : -1) * 0.035;
    const edgeColor = leafColor.map((value, channel) => (
      channel === 3 ? value : clamp(value + (channel === 1 ? secondaryVein : secondaryVein * 0.35), 0, 1)
    ));
    const veinNormal = normalize(add(normal, scale(wideAxis, secondaryVein * 2.4)));
    builder.vertex(add(center, scale(wideAxis, -halfWidth)), normal, edgeColor, roughness + secondaryVein, [0, 0.18 + station * 0.82]);
    builder.vertex(center, veinNormal, veinColor, roughness - 0.08, [0.5, 0.18 + station * 0.82]);
    builder.vertex(add(center, scale(wideAxis, halfWidth)), normal, edgeColor, roughness - secondaryVein, [1, 0.18 + station * 0.82]);
  });
  builder.vertex(apex, normal, veinColor, roughness - 0.06, [0.5, 1]);
  addDoubleSidedQuad(builder.indices, base, base + 1, base + 3, base + 2);
  addDoubleSidedTriangle(builder.indices, base + 4, base + 5, base + 7);
  addDoubleSidedTriangle(builder.indices, base + 2, base + 5, base + 4);
  addDoubleSidedTriangle(builder.indices, base + 3, base + 4, base + 7);
  for (let station = 0; station < stations.length - 1; station += 1) {
    const first = base + 5 + station * 3;
    const next = first + 3;
    addDoubleSidedQuad(builder.indices, first, next, next + 1, first + 1);
    addDoubleSidedQuad(builder.indices, first + 1, next + 1, next + 2, first + 2);
  }
  const lastPair = base + 5 + (stations.length - 1) * 3;
  const apexIndex = base + 5 + stations.length * 3;
  addDoubleSidedTriangle(builder.indices, lastPair, apexIndex, lastPair + 1);
  addDoubleSidedTriangle(builder.indices, lastPair + 1, apexIndex, lastPair + 2);
}

function appendLeaves(builder, transform, color, roughness, clusterNode, parameterBuffer, physics) {
  if (!(transform[6] > 0) || !(transform[7] > 0)) {
    throw new RangeError('tree WebGPU leaf dimensions must be positive');
  }
  for (let leaf = 0; leaf < LEAVES_PER_CLUSTER; leaf += 1) {
    const leafNode = conditionChild(clusterNode, {
      segment: `leaf:${leaf}`,
      channel: 'leaf-geometry',
    });
    const parameters = leafParameters(leafNode, transform, physics.shape);
    const pose = resolveLeafPose(transform, parameters, leaf, physics);
    if (!pose) continue;
    // Rejected hard-contact candidates emit no geometry and must consume no
    // retained-mesh budget. Reserve each accepted leaf exactly when emitted.
    const stationCount=LEAF_OUTLINES[physics.outline].length;
    builder.reserve(6+stationCount*3,42+24*(stationCount-1));
    parameterBuffer.push(...parameters);
    appendOvateLeaf(builder, color, roughness, parameters, pose, physics.outline);
  }
}

function finishMesh(packet, suffix, objectId, builder) {
  const meanRoughness = builder.roughness.reduce((sum, value) => sum + value, 0)
    / builder.roughness.length;
  return Object.freeze({
    type: 'field_mesh',
    id: `${packet.id}:${suffix}`,
    object_id: objectId,
    mode3d: true,
    topology: 'triangle-list',
    static_vertices: true,
    static_indices: true,
    receives_lighting: true,
    casts_shadow: true,
    receives_shadow: true,
    specular_strength: Math.max(0.02, 0.12 * (1 - meanRoughness)),
    vertices: new Float32Array(builder.vertices),
    uvs: new Float32Array(builder.uvs),
    roughness: new Float32Array(builder.roughness),
    indices: new Uint32Array(builder.indices),
  });
}

export function adaptTreeRenderPacketToWebGpuMeshesReference(
  packet,
  { vertexBudget, indexBudget, leafContact = false, leafShape = {},allowSparse=false,barkDetail=null,leafOutline='ovate' },
) {
  const sourceCounts = requirePacket(packet,allowSparse);
  requireBudgets(vertexBudget, indexBudget);
  if (!leafShape || typeof leafShape !== 'object' || Array.isArray(leafShape)) throw new Error('invalid leaf shape controls');
  if(!Object.hasOwn(LEAF_OUTLINES,leafOutline))throw new RangeError('unknown leaf outline');
  const shape = Object.freeze(Object.fromEntries(Object.entries(leafShape).map(([name, pdf]) => {
    if (!['lengthRatio', 'widthRatio', 'roundness', 'asymmetry', 'petioleRatio', 'camberRatio'].includes(name)) {
      throw new Error(`unknown leaf shape control ${name}`);
    }
    return [name, proGenDistributionReference(pdf)];
  })));
  const usage = { vertices: 0, indices: 0 };
  const wood = meshBuilder(vertexBudget, indexBudget, usage);
  const foliage = meshBuilder(vertexBudget, indexBudget, usage);
  const root = leafRoot(packet);
  const barkNode = conditionChild(root, { segment: 'bark', channel: 'bark-material' });
  const barkProfile = packet.profile?.bark;
  if (!barkProfile) throw new RangeError('tree species bark profile is required');
  const variantTotal = barkProfile.textureVariantWeights.reduce((sum, value) => sum + value, 0);
  let variantMark = hashString(packet.treeId, 0x85ebca6b) % variantTotal;
  let textureVariant = 0;
  while (variantMark >= barkProfile.textureVariantWeights[textureVariant]) {
    variantMark -= barkProfile.textureVariantWeights[textureVariant];
    textureVariant += 1;
  }
  const bark = Object.freeze({
    materialChannels: 'albedo+roughness+normal+radial-displacement',
    features: barkProfile.featureGrammar,
    coordinateSpace: 'shared-tree-space',
    junctionSampling: 'shared-field-resampled-inside-partitioned-forks',
    spatialOctaves: Object.freeze([0.24, 0.72, 1.85, 5.2]),
    detailedSampling: Object.freeze({
      trunkRingSides: DETAILED_TRUNK_BARK_RING_SIDES,
      trunkSteps: DETAILED_TRUNK_BARK_STEPS,
      branchRingSides: DETAILED_BRANCH_RING_SIDES,
      twigRingSides: DETAILED_TWIG_RING_SIDES,
    }),
    textureVariant,
    ridgeCount: Math.round(boundedNormal(
      barkNode, 0, barkProfile.ridgeCountMean, barkProfile.ridgeCountDeviation,
      ...barkProfile.ridgeCountBounds,
    )),
    ridgeAmplitude: boundedNormal(
      barkNode, 1, barkProfile.ridgeAmplitudeMean, barkProfile.ridgeAmplitudeDeviation,
      ...barkProfile.ridgeAmplitudeBounds,
    ),
    phase: boundedNormal(barkNode, 2, Math.PI, 1.4, 0, Math.PI * 2),
    grainTurns: boundedNormal(barkNode, 3, 4.5, 0.8, 2.5, 6.5),
    roughnessVariation: barkProfile.roughnessVariation,
    colorVariation: barkProfile.colorVariation,
    normalStrength: barkProfile.normalStrength,
    detailProfile: Object.freeze({
      fissureSharpness: barkProfile.fissureSharpness,
      fissureAxialFrequency: barkProfile.fissureAxialFrequency,
      grainAxialFrequency: barkProfile.grainAxialFrequency,
      microRidgeWeight: barkProfile.microRidgeWeight,
      microRidgeAxialFrequency: barkProfile.microRidgeAxialFrequency,
    }),
  });
  const woodNetwork = appendWoodyNetwork(wood, packet, bark,allowSparse,barkDetail);
  const leafParameterValues = [];
  const leafPhysics = {
    shape,
    outline:leafOutline,
    solver: 'petiole-torsion-hard-contact:v1',
    enabled: leafContact === true,
    bodies: [], cells: new Map(), candidateTests: 0, bentLeaves: 0,
    rejectedLeaves: 0, maximumBend: 0, maximumTwist: 0,
  };
  for (let primitive = 0; primitive < packet.primitiveCount; primitive += 1) {
    const kind = packet.primitiveKinds[primitive];
    const transform = packet.transforms.subarray(primitive * 8, primitive * 8 + 8);
    const color = Array.from(packet.baseColors.subarray(primitive * 4, primitive * 4 + 4));
    const roughness = packet.surfaceParams[primitive * 4];
    if (kind === KIND_FOLIAGE) {
      appendLeaves(
        foliage,
        transform,
        color,
        roughness,
        conditionChild(root, {
          segment: packet.primitiveIds[primitive],
          channel: 'leaf-cluster',
        }),
        leafParameterValues,
        leafPhysics,
      );
    }
  }
  const vertexCount = wood.vertices.length / 10 + foliage.vertices.length / 10;
  const indexCount = wood.indices.length + foliage.indices.length;
  if (vertexCount > vertexBudget) {
    throw new RangeError('tree WebGPU vertex budget is exhausted');
  }
  if (indexCount > indexBudget) {
    throw new RangeError('tree WebGPU index budget is exhausted');
  }
  return Object.freeze({
    kind: 'tree-webgpu-mesh-state:v1',
    source: packet,
    meshes: Object.freeze([
      finishMesh(packet, 'wood', packet.treeIndex * 2 + 1, wood),
      Object.freeze({...finishMesh(packet, 'foliage', packet.treeIndex * 2 + 2, foliage),
        leaf_vertex_count:6+LEAF_OUTLINES[leafOutline].length*3}),
    ]),
    counts: Object.freeze({
      trunks: sourceCounts.trunks,
      crowns: 0,
      branches: sourceCounts.branches,
      twigs: sourceCounts.twigs,
      foliageClusters: sourceCounts.foliageClusters,
      leaves: leafPhysics.enabled
        ? leafPhysics.bodies.length
        : sourceCounts.foliageClusters * LEAVES_PER_CLUSTER,
    }),
    leafParameterStride: LEAF_PARAMETER_STRIDE,
    leafParameters: new Float32Array(leafParameterValues),
    leafPhysics: Object.freeze({
      solver: leafPhysics.solver,
      enabled: leafPhysics.enabled,
      bodies: leafPhysics.bodies.length,
      candidateTests: leafPhysics.candidateTests,
      bentLeaves: leafPhysics.bentLeaves,
      rejectedLeaves: leafPhysics.rejectedLeaves,
      maximumBend: leafPhysics.maximumBend,
      maximumTwist: leafPhysics.maximumTwist,
      maximumPenetration: 0,
    }),
    bark,
    leafMaterial: Object.freeze({
      channels: 'albedo+roughness+normal',
      features: Object.freeze(['central-vein', 'secondary-veins', 'multiscale-green']),
      translucency: 'unsupported-double-sided',
    }),
    junctions: woodNetwork.junctions,
    woodTopology: woodNetwork.topology,
    vertexCount,
    indexCount,
    vertexBudget,
    indexBudget,
  });
}
import {
  conditionChild,
  createConditionedRoot,
  sampleNormalReference,
} from './vf-conditioned-distribution.mjs';
