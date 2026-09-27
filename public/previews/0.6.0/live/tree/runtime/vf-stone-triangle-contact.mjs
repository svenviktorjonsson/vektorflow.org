// Shared triangle-surface contact predicate used by tree foliage admission.

const CELL_SIZE = 0.18;
const INTERSECTION_EPSILON = 1e-9;

const subtract = (a, b) => a.map((value, axis) => value - b[axis]);
const add = (a, b) => a.map((value, axis) => value + b[axis]);
const scale = (value, amount) => value.map((component) => component * amount);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (value) => {
  const length = Math.hypot(...value);
  return value.map((component) => component / length);
};
const magnitude = (value) => Math.hypot(...value);

function transformedPoints(packet, matrix) {
  const points = [];
  for (let offset = 0; offset < packet.vertices.length; offset += 10) {
    const x = packet.vertices[offset]; const y = packet.vertices[offset + 1];
    const z = packet.vertices[offset + 2];
    points.push([
      matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
      matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
      matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    ]);
  }
  return points;
}

function triangleRecords(packet, matrix) {
  const points = transformedPoints(packet, matrix);
  const records = [];
  for (let offset = 0; offset < packet.indices.length; offset += 3) {
    const triangle = [
      points[packet.indices[offset]],
      points[packet.indices[offset + 1]],
      points[packet.indices[offset + 2]],
    ];
    records.push(Object.freeze({
      id: offset / 3,
      triangle,
      minimum: [0, 1, 2].map((axis) => Math.min(...triangle.map((point) => point[axis]))),
      maximum: [0, 1, 2].map((axis) => Math.max(...triangle.map((point) => point[axis]))),
    }));
  }
  return records;
}

const cell = (value) => Math.floor(value / CELL_SIZE);
const cellKey = (x, y, z) => `${x}:${y}:${z}`;

function triangleIndex(records) {
  const bins = new Map();
  for (const record of records) {
    const minimum = record.minimum.map(cell);
    const maximum = record.maximum.map(cell);
    for (let x = minimum[0]; x <= maximum[0]; x += 1) {
      for (let y = minimum[1]; y <= maximum[1]; y += 1) {
        for (let z = minimum[2]; z <= maximum[2]; z += 1) {
          const key = cellKey(x, y, z);
          if (!bins.has(key)) bins.set(key, []);
          bins.get(key).push(record);
        }
      }
    }
  }
  return bins;
}

function segmentTriangle(start, end, triangle) {
  const direction = subtract(end, start);
  const edge1 = subtract(triangle[1], triangle[0]);
  const edge2 = subtract(triangle[2], triangle[0]);
  const h = cross(direction, edge2);
  const determinant = dot(edge1, h);
  if (Math.abs(determinant) < 1e-10) return null;
  const inverse = 1 / determinant;
  const s = subtract(start, triangle[0]);
  const u = inverse * dot(s, h);
  if (u < -1e-8 || u > 1 + 1e-8) return null;
  const q = cross(s, edge1);
  const v = inverse * dot(direction, q);
  if (v < -1e-8 || u + v > 1 + 1e-8) return null;
  const t = inverse * dot(edge2, q);
  return t >= -1e-8 && t <= 1 + 1e-8 ? add(start, scale(direction, t)) : null;
}

const project2 = (point, droppedAxis) => droppedAxis === 0
  ? [point[1], point[2]] : droppedAxis === 1
    ? [point[0], point[2]] : [point[0], point[1]];

const orient2 = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1])
  - (b[1] - a[1]) * (c[0] - a[0]);

function pointOnSegment2(point, start, end, linearTolerance, areaTolerance) {
  return Math.abs(orient2(start, end, point)) <= areaTolerance
    && point[0] >= Math.min(start[0], end[0]) - linearTolerance
    && point[0] <= Math.max(start[0], end[0]) + linearTolerance
    && point[1] >= Math.min(start[1], end[1]) - linearTolerance
    && point[1] <= Math.max(start[1], end[1]) + linearTolerance;
}

function segmentParameter2(point, start, end) {
  const delta = [end[0] - start[0], end[1] - start[1]];
  const axis = Math.abs(delta[0]) >= Math.abs(delta[1]) ? 0 : 1;
  return Math.abs(delta[axis]) <= INTERSECTION_EPSILON
    ? 0 : (point[axis] - start[axis]) / delta[axis];
}

function segmentIntersection2(firstStart, firstEnd, secondStart, secondEnd,
  linearTolerance, areaTolerance) {
  const orientations = [
    orient2(firstStart, firstEnd, secondStart),
    orient2(firstStart, firstEnd, secondEnd),
    orient2(secondStart, secondEnd, firstStart),
    orient2(secondStart, secondEnd, firstEnd),
  ];
  const sign = (value) => value > areaTolerance ? 1 : value < -areaTolerance ? -1 : 0;
  const signs = orientations.map(sign);
  if (signs[0] * signs[1] < 0 && signs[2] * signs[3] < 0) {
    const denominator = orientations[2] - orientations[3];
    return denominator === 0 ? 0 : orientations[2] / denominator;
  }
  if (signs[0] === 0 && pointOnSegment2(
    secondStart, firstStart, firstEnd, linearTolerance, areaTolerance)) {
    return segmentParameter2(secondStart, firstStart, firstEnd);
  }
  if (signs[1] === 0 && pointOnSegment2(
    secondEnd, firstStart, firstEnd, linearTolerance, areaTolerance)) {
    return segmentParameter2(secondEnd, firstStart, firstEnd);
  }
  if (signs[2] === 0 && pointOnSegment2(
    firstStart, secondStart, secondEnd, linearTolerance, areaTolerance)) return 0;
  if (signs[3] === 0 && pointOnSegment2(
    firstEnd, secondStart, secondEnd, linearTolerance, areaTolerance)) return 1;
  return null;
}

function pointInTriangle2(point, triangle, areaTolerance) {
  const orientations = [
    orient2(triangle[0], triangle[1], point),
    orient2(triangle[1], triangle[2], point),
    orient2(triangle[2], triangle[0], point),
  ];
  return !orientations.some((value) => value > areaTolerance)
    || !orientations.some((value) => value < -areaTolerance);
}

function coplanarTriangleIntersection(first, second, normal) {
  const droppedAxis = [0, 1, 2].reduce((best, axis) =>
    Math.abs(normal[axis]) > Math.abs(normal[best]) ? axis : best, 0);
  const first2 = first.map((point) => project2(point, droppedAxis));
  const second2 = second.map((point) => project2(point, droppedAxis));
  const scale2 = Math.max(1, ...first2.flat().map(Math.abs), ...second2.flat().map(Math.abs));
  const linearTolerance = INTERSECTION_EPSILON * scale2;
  const areaTolerance = INTERSECTION_EPSILON * scale2 ** 2;
  for (let firstEdge = 0; firstEdge < 3; firstEdge += 1) {
    for (let secondEdge = 0; secondEdge < 3; secondEdge += 1) {
      const parameter = segmentIntersection2(
        first2[firstEdge], first2[(firstEdge + 1) % 3],
        second2[secondEdge], second2[(secondEdge + 1) % 3],
        linearTolerance, areaTolerance,
      );
      if (parameter === null) continue;
      return add(first[firstEdge], scale(subtract(
        first[(firstEdge + 1) % 3], first[firstEdge]), Math.max(0, Math.min(1, parameter))));
    }
  }
  if (pointInTriangle2(first2[0], second2, areaTolerance)) return first[0];
  if (pointInTriangle2(second2[0], first2, areaTolerance)) return second[0];
  return null;
}

function coplanarNormal(first, second) {
  const firstNormal = cross(subtract(first[1], first[0]), subtract(first[2], first[0]));
  const secondNormal = cross(subtract(second[1], second[0]), subtract(second[2], second[0]));
  const firstLength = magnitude(firstNormal); const secondLength = magnitude(secondNormal);
  if (firstLength <= INTERSECTION_EPSILON || secondLength <= INTERSECTION_EPSILON) return null;
  if (magnitude(cross(firstNormal, secondNormal))
      > INTERSECTION_EPSILON * firstLength * secondLength) return null;
  const coordinateScale = Math.max(1, ...first.flat().map(Math.abs), ...second.flat().map(Math.abs));
  const planeTolerance = INTERSECTION_EPSILON * coordinateScale;
  if (second.some((point) => Math.abs(dot(firstNormal, subtract(point, first[0])))
      > planeTolerance * firstLength)) return null;
  return firstNormal;
}

function triangleIntersection(first, second, includeCoplanar = false) {
  for (let edge = 0; edge < 3; edge += 1) {
    const point = segmentTriangle(
      first[edge], first[(edge + 1) % 3], second,
    );
    if (point) return Object.freeze({ point, kind: 'transverse' });
  }
  for (let edge = 0; edge < 3; edge += 1) {
    const point = segmentTriangle(
      second[edge], second[(edge + 1) % 3], first,
    );
    if (point) return Object.freeze({ point, kind: 'transverse' });
  }
  if (includeCoplanar) {
    const commonNormal = coplanarNormal(first, second);
    if (commonNormal) {
      const point = coplanarTriangleIntersection(first, second, commonNormal);
      return point ? Object.freeze({ point, kind: 'coplanar' }) : null;
    }
  }
  return null;
}

function collision(packet, matrix, supportIndex, { includeCoplanar = false } = {}) {
  const candidates = triangleRecords(packet, matrix);
  for (const candidate of candidates) {
    const possible = new Map();
    const minimum = candidate.minimum.map(cell);
    const maximum = candidate.maximum.map(cell);
    for (let x = minimum[0]; x <= maximum[0]; x += 1) {
      for (let y = minimum[1]; y <= maximum[1]; y += 1) {
        for (let z = minimum[2]; z <= maximum[2]; z += 1) {
          for (const support of supportIndex.get(cellKey(x, y, z)) ?? []) {
            possible.set(support.id, support);
          }
        }
      }
    }
    for (const support of possible.values()) {
      if ([0, 1, 2].some((axis) => (
        candidate.maximum[axis] < support.minimum[axis] - INTERSECTION_EPSILON
        || candidate.minimum[axis] > support.maximum[axis] + INTERSECTION_EPSILON
      ))) continue;
      const intersection = triangleIntersection(
        candidate.triangle, support.triangle, includeCoplanar);
      if (!intersection) continue;
      let normal = normalize(cross(
        subtract(support.triangle[1], support.triangle[0]),
        subtract(support.triangle[2], support.triangle[0]),
      ));
      if (normal[2] < 0) normal = scale(normal, -1);
      return Object.freeze({
        candidateTriangle: candidate.id,
        supportTriangle: support.id,
        point: Object.freeze(intersection.point),
        ...(includeCoplanar ? { intersectionKind: intersection.kind } : {}),
        normal: Object.freeze(normal),
      });
    }
  }
  return null;
}

export function createTriangleSurfaceAdmissionReference(packet) {
  const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  const index=triangleIndex(triangleRecords(packet,identity));
  return Object.freeze({intersects(candidate){
    return collision(candidate,identity,index,{includeCoplanar:true});
  }});
}
