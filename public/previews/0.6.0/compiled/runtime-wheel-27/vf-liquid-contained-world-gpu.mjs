import {createCheckedGpuPipeline,createOptionalGpuPipeline} from './vf-gpu-pipeline-errors.mjs';
import {
  calibrateUniformLocalLiquidParticleMassReference,
  createAkinciLiquidSolidBoundaryReference,
} from './vf-physics-liquid-local-particle-reference.mjs';
import { createTransportPolylineBoundaryPacketReference }
  from './vf-physics-transport-core-reference.mjs';
import { createSlopedStoneChannelGeometryReference }
  from './vf-water-2d-stone-channel-reference.mjs';
import { SWEPT_WHEEL_CONTACT_WGSL } from './vf-swept-wheel-contact-gpu.mjs';
import { PREVENTIVE_PARTICLE_CONTACT_WGSL } from './vf-preventive-particle-contact-gpu.mjs';
import { LIQUID_PRESSURE_GEOMETRY_CACHE_WGSL } from './vf-liquid-pressure-geometry-cache-wgsl.mjs';
import {
  createDynamicParticleGridAuditWgsl2D,
  createDynamicParticleGridGpuPlan,
  createDynamicParticleGridWgsl2D,
} from './vf-physics-dynamic-grid-gpu.mjs';

const PRIMARY_LIQUID = 0;
const DIFFUSE_SPRAY = 1;
const DIFFUSE_FOAM = 2;
const DIFFUSE_BUBBLE = 3;

export const LIQUID_PARTICLE_WORLD_GPU_ABI = Object.freeze({
  primaryStrideBytes: 48,
  diffuseStrideBytes: 32,
  diffuseRenderStrideBytes: 16,
  parameterBytes: 144,
  surfaceVertexStrideBytes: 8,
  telemetryAtomicCount: 13,
  primaryFields: Object.freeze(['position', 'velocity', 'scratch_velocity', 'lambda',
    'density', 'foam', 'neighbor_count', 'surface_class', 'volume']),
  diffuseRenderFields: Object.freeze(['position', 'radius', 'kind']),
});

export const LIQUID_PARTICLE_WORLD_GPU_TELEMETRY = Object.freeze({
  atomicCount: LIQUID_PARTICLE_WORLD_GPU_ABI.telemetryAtomicCount,
  fields: Object.freeze({
    peakSpeedSquaredBits: 0,
    nonFiniteFlag: 1,
    peakCellOccupancy: 2,
    contactBudgetExhaustedReceipt: 3,
    occupiedBucketEvents: 4,
    activeCellEvents: 5,
    peakPredictedOrPostContactSpeedSquaredBits: 6,
    peakFastInverseDenominatorBits: 7,
    peakFastDensityErrorBits: 8,
    peakSweepSpeedSquaredBits: 9,
    peakPressureSpeedSquaredBits: 10,
    peakSweepAddedSpeedBits: 11,
    peakPreSweepSpeedSquaredBits: 12,
  }),
});

export const LIQUID_PARTICLE_KIND_RULES_GPU = Object.freeze({
  primaryLiquid: Object.freeze({
    id: PRIMARY_LIQUID,
    interactions: Object.freeze(['density-projection', 'divergence-projection',
      'viscosity', 'gravity', 'unilateral-solid-contact']),
  }),
  spray: Object.freeze({
    id: DIFFUSE_SPRAY,
    interactions: Object.freeze(['gravity', 'liquid-occupancy-reentry',
      'unilateral-solid-contact', 'finite-lifetime']),
  }),
  foam: Object.freeze({
    id: DIFFUSE_FOAM,
    interactions: Object.freeze(['liquid-velocity-advection', 'surface-band-classification',
      'unilateral-solid-contact', 'finite-lifetime']),
  }),
  bubble: Object.freeze({
    id: DIFFUSE_BUBBLE,
    interactions: Object.freeze(['liquid-velocity-drag', 'buoyancy',
      'interior-classification', 'finite-lifetime']),
  }),
});

// Compute owns interaction and phase evolution. Vertex stages only consume the
// resulting buffers; duplicating neighborhood physics per rendered vertex would
// change both the model and its complexity.
export const LIQUID_PARTICLE_WORLD_GPU_WGSL = /* wgsl */`
struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  scratch_velocity: vec2<f32>,
  lambda: f32,
  density: f32,
  foam: f32,
  neighbor_count: u32,
  surface_class: u32,
  volume: f32,
};

struct DiffuseParticle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  radius: f32,
  age: f32,
  lifetime: f32,
  kind: u32,
};

struct Params {
  counts: vec4<u32>,
  limits: vec4<u32>,
  world: vec4<f32>,
  fluid: vec4<f32>,
  material: vec4<f32>,
  force: vec4<f32>,
  terrain: vec4<f32>,
  stone: vec4<f32>,
  motion: vec4<f32>,
};

struct ConstraintSample {
  density: f32,
  material_density: f32,
  density_rate: f32,
  denominator: f32,
  neighbor_count: u32,
  center_gradient: vec2<f32>,
};

struct KernelSample {
  weight: f32,
  gradient: vec2<f32>,
  supported: u32,
};

struct WheelBoundarySample {
  density_factor: f32,
  gradient: vec2<f32>,
};

struct ContactResult {
  position: vec2<f32>,
  velocity: vec2<f32>,
};

struct TerrainSample {
  signed_distance: f32,
  active_count: u32,
  closest: vec2<f32>,
  normal0: vec2<f32>,
  closest1: vec2<f32>,
  normal1: vec2<f32>,
};

struct SweepHit {
  fraction: f32,
  hit: u32,
  initial_overlap: u32,
  normal_count: u32,
  point: vec2<f32>,
  normal0: vec2<f32>,
  normal1: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cell_items: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> diffuse: array<DiffuseParticle>;
@group(0) @binding(5) var<storage, read_write> diffuse_active: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> diffuse_render: array<vec4<f32>>;
// One immutable, canonically-authored solid packet serves both exact contact
// and Akinci pressure support. Its prefix is also the renderer's polyline.
@group(0) @binding(7) var<storage, read> solid_data: array<vec4<u32>>;

const PI: f32 = 3.141592653589793;
const SPRAY: u32 = 1u;
const FOAM: u32 = 2u;
const BUBBLE: u32 = 3u;
const WHEEL_RADIUS: f32 = 0.50;
const WHEEL_BAR_HALF_WIDTH: f32 = 0.012;
const RIGID_SEGMENT_COUNT: u32 = 7u;

fn rotate_local(point: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle); let s = sin(angle);
  return vec2<f32>(c * point.x - s * point.y, s * point.x + c * point.y);
}

fn baffle(index: u32) -> vec4<f32> {
  if (index == 0u) { return vec4<f32>(0.500, 0.000, 0.350, 0.000); }
  if (index == 1u) { return vec4<f32>(0.000, 0.500, 0.000, 0.350); }
  if (index == 2u) { return vec4<f32>(-0.500, 0.000, -0.350, 0.000); }
  if (index == 3u) { return vec4<f32>(0.000, -0.500, 0.000, -0.350); }
  if (index == 4u) { return vec4<f32>(-0.220, 0.175, -0.075, 0.135); }
  if (index == 5u) { return vec4<f32>(0.075, -0.055, 0.215, -0.115); }
  return vec4<f32>(-0.105, -0.250, 0.020, -0.155);
}

fn surface_point(index: u32) -> vec2<f32> {
  let packed = solid_data[index >> 1u];
  if ((index & 1u) == 0u) {
    return vec2<f32>(bitcast<f32>(packed.x), bitcast<f32>(packed.y));
  }
  return vec2<f32>(bitcast<f32>(packed.z), bitcast<f32>(packed.w));
}

fn boundary_data_offset() -> u32 {
  // Two densely-packed vec2 surface vertices share each vec4 entry.
  return (params.limits.y + 2u) >> 1u;
}

fn boundary_particle(index: u32) -> vec3<f32> {
  let packed = solid_data[boundary_data_offset() + index];
  return vec3<f32>(bitcast<f32>(packed.x), bitcast<f32>(packed.y),
    bitcast<f32>(packed.z));
}

fn solid_word(address: u32) -> u32 {
  let packed = solid_data[address >> 2u];
  return packed[address & 3u];
}

fn boundary_grid_word_offset() -> u32 {
  return (boundary_data_offset() + params.limits.z) * 4u;
}

fn boundary_cell_count(bucket: u32) -> u32 {
  let stride = params.limits.w + 1u;
  return min(solid_word(boundary_grid_word_offset() + bucket * stride), params.limits.w);
}

fn boundary_cell_item(bucket: u32, slot: u32) -> u32 {
  let stride = params.limits.w + 1u;
  return solid_word(boundary_grid_word_offset() + bucket * stride + slot + 1u);
}

fn hash_u32(input: u32) -> u32 {
  var value = input;
  value = value ^ (value >> 16u);
  value = value * 0x7feb352du;
  value = value ^ (value >> 15u);
  value = value * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn random_unit(input: u32) -> f32 {
  return f32(hash_u32(input)) / 4294967296.0;
}

fn sample_kernel(displacement: vec2<f32>) -> KernelSample {
  let h = params.fluid.z;
  let distance_squared = dot(displacement, displacement);
  if (distance_squared >= h * h) {
    return KernelSample(0.0, vec2<f32>(0.0), 0u);
  }
  let distance = sqrt(max(distance_squared, 0.0));
  let q = distance / h;
  let one_minus_q = 1.0 - q;
  let weight = 7.0 / (PI * h * h) * one_minus_q * one_minus_q
    * one_minus_q * one_minus_q * (1.0 + 4.0 * q);
  var gradient = vec2<f32>(0.0);
  if (distance > 1.0e-7) {
    let derivative = -140.0 / (PI * h * h * h) * q
      * one_minus_q * one_minus_q * one_minus_q;
    gradient = displacement / distance * derivative;
  }
  return KernelSample(weight, gradient, 1u);
}

// A coincident parcel pair has finite density but a zero Wendland gradient.
// Its pressure constraint is singular: no pressure impulse can separate it.
// Evaluate very close pairs at a small, deterministic separation instead.
// Reversing the particle IDs reverses the direction, preserving pair impulse.
fn sample_particle_kernel(displacement: vec2<f32>, index: u32,
  other: u32) -> KernelSample {
  let minimum = params.fluid.y * 0.25;
  let distance = length(displacement);
  if (distance >= minimum) { return sample_kernel(displacement); }
  if (distance > 1.0e-7) {
    return sample_kernel(displacement * (minimum / distance));
  }
  let low = min(index, other);
  let high = max(index, other);
  let first = hash_u32((low * 0x9e3779b9u) ^ (high * 0x85ebca6bu));
  let second = hash_u32(first ^ 0xc2b2ae35u);
  let candidate = vec2<f32>(f32(first) / 2147483648.0 - 1.0,
    f32(second) / 2147483648.0 - 1.0);
  let direction = candidate / max(length(candidate), 1.0e-6);
  return sample_kernel(select(-direction, direction, index < other) * minimum);
}


fn wheel_boundary_support(position: vec2<f32>) -> WheelBoundarySample {
  var density_factor = 0.0;
  var gradient = vec2<f32>(0.0);
  let center = params.terrain.zw;
  let angle = params.terrain.x;
  let h = params.fluid.z;
  // Matches the calibrated volume of the canonical 2D Akinci boundary. The
  // samples are evaluated only near the analytic wheel, so the drum becomes a
  // pressure boundary without changing the liquid particle rules.
  let boundary_volume = 0.000112;
  let sample_spacing = 0.009;

  let radial = position - center;
  let radial_length = length(radial);
  if (radial_length > WHEEL_RADIUS - h - WHEEL_BAR_HALF_WIDTH) {
    let theta = atan2(radial.y, radial.x);
    for (var along = -3; along <= 3; along = along + 1) {
      let sample_theta = theta + f32(along) * sample_spacing / WHEEL_RADIUS;
      let outward = vec2<f32>(cos(sample_theta), sin(sample_theta));
      for (var layer = 0; layer <= 2; layer = layer + 1) {
        let solid = center + outward
          * (WHEEL_RADIUS + f32(layer) * sample_spacing);
        let kernel = sample_kernel(position - solid);
        density_factor = density_factor + boundary_volume * kernel.weight;
        gradient = gradient + boundary_volume * kernel.gradient;
      }
    }
  }

  for (var segment = 0u; segment < RIGID_SEGMENT_COUNT; segment = segment + 1u) {
    let local = baffle(segment);
    let a = center + rotate_local(local.xy, angle);
    let b = center + rotate_local(local.zw, angle);
    let edge = b - a;
    let edge_length = max(length(edge), 1.0e-8);
    let tangent = edge / edge_length;
    let raw_along = dot(position - a, tangent);
    let closest_along = clamp(raw_along, 0.0, edge_length);
    let closest = a + tangent * closest_along;
    if (length(position - closest) < h + WHEEL_BAR_HALF_WIDTH) {
      let normal = vec2<f32>(-tangent.y, tangent.x);
      let snapped_along = round(closest_along / sample_spacing) * sample_spacing;
      for (var offset = -3; offset <= 3; offset = offset + 1) {
        let sample_along = snapped_along + f32(offset) * sample_spacing;
        if (sample_along < 0.0 || sample_along > edge_length) { continue; }
        let center_sample = a + tangent * sample_along;
        for (var layer = -1; layer <= 1; layer = layer + 1) {
          let solid = center_sample + normal
            * (f32(layer) * WHEEL_BAR_HALF_WIDTH);
          let kernel = sample_kernel(position - solid);
          density_factor = density_factor + boundary_volume * kernel.weight;
          gradient = gradient + boundary_volume * kernel.gradient;
        }
      }
    }
  }
  return WheelBoundarySample(density_factor, gradient);
}

fn telemetry_base() -> u32 {
  return params.counts.y * params.counts.z;
}

fn finite_scalar(value: f32) -> bool {
  // Comparisons reject NaN; the upper bound rejects both infinities while
  // remaining exactly representable as the largest finite binary32 value.
  return abs(value) <= 3.4028234663852886e+38;
}

fn flag_nonfinite() {
  atomicOr(&cell_counts[telemetry_base() + 1u], 128u);
}

fn flag_nonfinite_reason(reason: u32) {
  atomicOr(&cell_counts[telemetry_base() + 1u], reason);
}

fn flag_contact_budget_exhausted(receipt: u32, reason: u32) {
  // Contact iteration exhaustion is a failed physics receipt, never silent
  // time loss. High bit identifies diffuse state, bits 28..30 the reason,
  // and low 28 bits the source index + 1.
  atomicMax(&cell_counts[telemetry_base() + 3u], receipt | (reason << 28u));
}

fn record_motion_telemetry(position: vec2<f32>, velocity: vec2<f32>) {
  let speed_squared = dot(velocity, velocity);
  if (!finite_scalar(position.x) || !finite_scalar(position.y)
      || !finite_scalar(velocity.x) || !finite_scalar(velocity.y)
      || !finite_scalar(speed_squared) || speed_squared < 0.0) {
    flag_nonfinite_reason(1u);
    return;
  }
  // IEEE-754 bit patterns are monotonically ordered for nonnegative f32, so
  // integer atomicMax is an exact, lock-free max reduction for speed squared.
  atomicMax(&cell_counts[telemetry_base()], bitcast<u32>(speed_squared));
}

fn record_primary_telemetry(index: u32) {
  let particle = particles[index];
  record_motion_telemetry(particle.position, particle.velocity);
  if (!finite_scalar(particle.scratch_velocity.x)
      || !finite_scalar(particle.scratch_velocity.y)) { flag_nonfinite_reason(2u); }
  if (!finite_scalar(particle.lambda)) { flag_nonfinite_reason(4u); }
  if (!finite_scalar(particle.density)) { flag_nonfinite_reason(8u); }
  if (!finite_scalar(particle.foam)) { flag_nonfinite_reason(16u); }
  if (!finite_scalar(particle.volume) || particle.volume <= 0.0) {
    flag_nonfinite_reason(32u);
  }
}

fn record_diffuse_finiteness(marker: DiffuseParticle) {
  if (!finite_scalar(marker.position.x) || !finite_scalar(marker.position.y)
      || !finite_scalar(marker.velocity.x) || !finite_scalar(marker.velocity.y)
      || !finite_scalar(marker.radius) || !finite_scalar(marker.age)
      || !finite_scalar(marker.lifetime)) {
    flag_nonfinite_reason(64u);
  }
}

const TRANSPORT_F32_EPSILON: f32 = 0.0000019073486328125;
const MAX_TRANSPORT_CONTACTS: u32 = 8u;

fn transport_scale(start_point: vec2<f32>, end_point: vec2<f32>, radius: f32) -> f32 {
  return max(1.1754943508222875e-38, max(radius,
    max(max(abs(start_point.x), abs(start_point.y)),
      max(abs(end_point.x), abs(end_point.y)))));
}

fn surface_segment_normal(segment: u32) -> vec2<f32> {
  let edge = surface_point(segment + 1u) - surface_point(segment);
  let edge_length = max(length(edge), 1.1754943508222875e-38);
  return vec2<f32>(-edge.y, edge.x) / edge_length;
}

fn surface_vertex_fallback_normal(vertex: u32) -> vec2<f32> {
  let segment_count = params.limits.y;
  var summed = vec2<f32>(0.0);
  if (vertex > 0u) { summed = summed + surface_segment_normal(vertex - 1u); }
  if (vertex < segment_count) { summed = summed + surface_segment_normal(vertex); }
  let summed_length = length(summed);
  if (summed_length > TRANSPORT_F32_EPSILON) { return summed / summed_length; }
  var fallback_segment = 0u;
  if (vertex > 0u) { fallback_segment = vertex - 1u; }
  return surface_segment_normal(fallback_segment);
}

// Exact finite-capsule sample. Segment bodies own strict interiors and each
// Transport vertex owns one circular endcap; the two open ends are not
// extrapolated into infinite terrain half-planes.
fn sample_terrain(point: vec2<f32>) -> TerrainSample {
  if (params.limits.y == 0u) {
    return TerrainSample(1.0e6, 0u, point + vec2<f32>(0.0, -1.0e6),
      vec2<f32>(0.0, 1.0), point, vec2<f32>(0.0, 1.0));
  }
  let segment_count = params.limits.y;
  let uniform_step = params.stone.y;
  let raw_segment = floor((point.x - params.stone.x) / uniform_step);
  let local_segment = u32(clamp(raw_segment, 0.0, f32(segment_count - 1u)));
  let local_a = surface_point(local_segment);
  let local_b = surface_point(local_segment + 1u);
  let local_edge = local_b - local_a;
  let local_denominator = max(dot(local_edge, local_edge),
    1.1754943508222875e-38);
  let local_projection = clamp(dot(point - local_a, local_edge)
    / local_denominator, 0.0, 1.0);
  let upper_closest = local_a + local_edge * local_projection;
  let upper_squared = dot(point - upper_closest, point - upper_closest);
  let span_float = clamp(ceil(sqrt(max(upper_squared, 0.0)) / uniform_step) + 2.0,
    0.0, f32(segment_count));
  let span = u32(span_float);
  let first_segment = select(0u, local_segment - span, local_segment > span);
  let last_segment = min(segment_count - 1u, local_segment + span);
  let scale = max(transport_scale(point, point, uniform_step),
    max(abs(surface_point(first_segment).y), abs(surface_point(last_segment + 1u).y)));
  let spatial_tolerance = TRANSPORT_F32_EPSILON * scale;
  let squared_tolerance = TRANSPORT_F32_EPSILON * scale * scale;
  var minimum_squared = 3.4028234663852886e+38;
  var closest = upper_closest;
  var normal0 = surface_segment_normal(local_segment);
  var closest1 = closest;
  var normal1 = normal0;
  var active_count = 0u;

  // Feature order matches the canonical packet: vertex i, then segment i.
  for (var segment = first_segment; segment <= last_segment; segment = segment + 1u) {
    let vertex = surface_point(segment);
    let vertex_delta = point - vertex;
    let vertex_squared = dot(vertex_delta, vertex_delta);
    let vertex_distance = sqrt(max(vertex_squared, 0.0));
    var vertex_normal = surface_vertex_fallback_normal(segment);
    if (vertex_distance > spatial_tolerance) {
      vertex_normal = vertex_delta / vertex_distance;
    }
    if (vertex_squared < minimum_squared - squared_tolerance) {
      minimum_squared = vertex_squared; closest = vertex; normal0 = vertex_normal;
      closest1 = vertex; normal1 = vertex_normal; active_count = 1u;
    } else if (abs(vertex_squared - minimum_squared) <= squared_tolerance
        && active_count < 2u && dot(vertex_normal - normal0, vertex_normal - normal0)
          > TRANSPORT_F32_EPSILON * TRANSPORT_F32_EPSILON) {
      closest1 = vertex; normal1 = vertex_normal; active_count = 2u;
    }

    let a = vertex;
    let b = surface_point(segment + 1u);
    let edge = b - a;
    let edge_length = max(length(edge), 1.1754943508222875e-38);
    let tangent = edge / edge_length;
    let along = dot(point - a, tangent);
    if (along > spatial_tolerance && along < edge_length - spatial_tolerance) {
      let body_closest = a + tangent * along;
      let body_delta = point - body_closest;
      let body_squared = dot(body_delta, body_delta);
      let body_distance = sqrt(max(body_squared, 0.0));
      var body_normal = vec2<f32>(-tangent.y, tangent.x);
      if (body_distance > spatial_tolerance) { body_normal = body_delta / body_distance; }
      if (body_squared < minimum_squared - squared_tolerance) {
        minimum_squared = body_squared; closest = body_closest; normal0 = body_normal;
        closest1 = body_closest; normal1 = body_normal; active_count = 1u;
      } else if (abs(body_squared - minimum_squared) <= squared_tolerance
          && active_count < 2u && dot(body_normal - normal0, body_normal - normal0)
            > TRANSPORT_F32_EPSILON * TRANSPORT_F32_EPSILON) {
        closest1 = body_closest; normal1 = body_normal; active_count = 2u;
      }
    }
  }
  let final_vertex = surface_point(last_segment + 1u);
  let final_delta = point - final_vertex;
  let final_squared = dot(final_delta, final_delta);
  let final_distance = sqrt(max(final_squared, 0.0));
  var final_normal = surface_vertex_fallback_normal(last_segment + 1u);
  if (final_distance > spatial_tolerance) { final_normal = final_delta / final_distance; }
  if (final_squared < minimum_squared - squared_tolerance) {
    minimum_squared = final_squared; closest = final_vertex; normal0 = final_normal;
    closest1 = final_vertex; normal1 = final_normal; active_count = 1u;
  } else if (abs(final_squared - minimum_squared) <= squared_tolerance
      && active_count < 2u && dot(final_normal - normal0, final_normal - normal0)
        > TRANSPORT_F32_EPSILON * TRANSPORT_F32_EPSILON) {
    closest1 = final_vertex; normal1 = final_normal; active_count = 2u;
  }
  return TerrainSample(sqrt(max(minimum_squared, 0.0)), active_count,
    closest, normal0, closest1, normal1);
}

fn overlap_correction(point: vec2<f32>, sample: TerrainSample,
  radius: f32, tolerance: f32) -> vec2<f32> {
  let depth0 = max(0.0, radius - length(point - sample.closest));
  let candidate0 = sample.normal0 * depth0;
  if (sample.active_count < 2u) { return candidate0; }
  let depth1 = max(0.0, radius - length(point - sample.closest1));
  let candidate1 = sample.normal1 * depth1;
  let candidate0_feasible = dot(candidate0, sample.normal1) >= depth1 - tolerance;
  let candidate1_feasible = dot(candidate1, sample.normal0) >= depth0 - tolerance;
  var correction = vec2<f32>(0.0);
  var error = 3.4028234663852886e+38;
  if (candidate0_feasible) { correction = candidate0; error = dot(candidate0, candidate0); }
  if (candidate1_feasible && dot(candidate1, candidate1) < error) {
    correction = candidate1; error = dot(candidate1, candidate1);
  }
  let determinant = sample.normal0.x * sample.normal1.y
    - sample.normal0.y * sample.normal1.x;
  if (abs(determinant) > TRANSPORT_F32_EPSILON) {
    let intersection = vec2<f32>(
      (depth0 * sample.normal1.y - sample.normal0.y * depth1) / determinant,
      (sample.normal0.x * depth1 - depth0 * sample.normal1.x) / determinant);
    let intersection_feasible = dot(intersection, sample.normal0) >= depth0 - tolerance
      && dot(intersection, sample.normal1) >= depth1 - tolerance;
    if (intersection_feasible && dot(intersection, intersection) < error) {
      correction = intersection; error = dot(intersection, intersection);
    }
  }
  return select(candidate0, correction, error < 3.4028234663852886e+38);
}

fn merge_sweep_candidate(best_input: SweepHit, raw_fraction: f32,
  origin: vec2<f32>, displacement: vec2<f32>, outward_normal: vec2<f32>,
  fraction_tolerance: f32, direction_tolerance: f32) -> SweepHit {
  if (raw_fraction < -fraction_tolerance || raw_fraction > 1.0 + fraction_tolerance
      || dot(displacement, outward_normal) >= -direction_tolerance) {
    return best_input;
  }
  let fraction = clamp(raw_fraction, 0.0, 1.0);
  if (best_input.hit == 0u || fraction < best_input.fraction - fraction_tolerance) {
    return SweepHit(fraction, 1u, 0u, 1u, origin + displacement * fraction,
      outward_normal, outward_normal);
  }
  var best = best_input;
  if (abs(fraction - best.fraction) <= fraction_tolerance
      && best.normal_count < 2u
      && dot(outward_normal - best.normal0, outward_normal - best.normal0)
        > TRANSPORT_F32_EPSILON * TRANSPORT_F32_EPSILON) {
    best.normal1 = outward_normal; best.normal_count = 2u;
    if (fraction < best.fraction) {
      best.fraction = fraction; best.point = origin + displacement * fraction;
    }
  }
  return best;
}

fn sweep_terrain_disk(origin: vec2<f32>, velocity: vec2<f32>,
  dt: f32, radius: f32, start_is_exterior: u32) -> SweepHit {
  let displacement = velocity * dt;
  let destination = origin + displacement;
  let travel = length(displacement);
  let scale = transport_scale(origin, destination, radius);
  let spatial_tolerance = TRANSPORT_F32_EPSILON * scale;
  let direction_tolerance = spatial_tolerance + TRANSPORT_F32_EPSILON * travel;
  let fraction_tolerance = TRANSPORT_F32_EPSILON
    * max(1.0, scale / max(travel, spatial_tolerance));
  if (start_is_exterior == 0u) {
    let initial = sample_terrain(origin);
    if (initial.signed_distance < radius - spatial_tolerance) {
      let correction = overlap_correction(origin, initial, radius, spatial_tolerance);
      return SweepHit(0.0, 1u, 1u, initial.active_count, origin + correction,
        initial.normal0, initial.normal1);
    }
  }
  var best = SweepHit(1.0, 0u, 0u, 0u, destination,
    vec2<f32>(0.0), vec2<f32>(0.0));
  if (travel <= spatial_tolerance) { return best; }

  let segment_count = params.limits.y;
  let minimum_x = params.stone.x;
  let maximum_x = surface_point(segment_count).x;
  let uniform_step = params.stone.y;
  let uniform_error = params.stone.w + spatial_tolerance;
  let low_x = min(origin.x, destination.x) - radius - spatial_tolerance;
  let high_x = max(origin.x, destination.x) + radius + spatial_tolerance;
  if (high_x < minimum_x - spatial_tolerance
      || low_x > maximum_x + spatial_tolerance) { return best; }
  let raw_first = i32(ceil((low_x - uniform_error - minimum_x) / uniform_step)) - 1;
  let raw_last = i32(floor((high_x + uniform_error - minimum_x) / uniform_step));
  let first_segment = u32(clamp(raw_first, 0, i32(segment_count) - 1));
  let last_segment = u32(clamp(raw_last, 0, i32(segment_count) - 1));
  if (last_segment < first_segment) { return best; }

  // Analytic line-body TOI for both sides of every bounded candidate segment.
  for (var segment = first_segment; segment <= last_segment; segment = segment + 1u) {
    let a = surface_point(segment);
    let b = surface_point(segment + 1u);
    let edge = b - a;
    let edge_length = max(length(edge), 1.1754943508222875e-38);
    let tangent = edge / edge_length;
    let oriented_normal = vec2<f32>(-tangent.y, tangent.x);
    let normal_start = dot(origin - a, oriented_normal);
    let normal_travel = dot(displacement, oriented_normal);
    if (abs(normal_travel) > direction_tolerance) {
      for (var side_index = 0u; side_index < 2u; side_index = side_index + 1u) {
        let side = select(-1.0, 1.0, side_index == 1u);
        let fraction = (side * radius - normal_start) / normal_travel;
        let point = origin + displacement * fraction;
        let along = dot(point - a, tangent);
        if (along > spatial_tolerance && along < edge_length - spatial_tolerance) {
          best = merge_sweep_candidate(best, fraction, origin, displacement,
            oriented_normal * side, fraction_tolerance, direction_tolerance);
        }
      }
    }
  }

  // Analytic circle TOI for every vertex, including both finite open endcaps.
  let quadratic_a = dot(displacement, displacement);
  for (var vertex_index = first_segment; vertex_index <= last_segment + 1u;
    vertex_index = vertex_index + 1u) {
    let vertex = surface_point(vertex_index);
    let relative = origin - vertex;
    let quadratic_b = dot(relative, displacement);
    let quadratic_c = dot(relative, relative) - radius * radius;
    var discriminant = quadratic_b * quadratic_b - quadratic_a * quadratic_c;
    let discriminant_tolerance = TRANSPORT_F32_EPSILON
      * (abs(quadratic_b * quadratic_b) + abs(quadratic_a * quadratic_c));
    if (quadratic_a > 0.0 && discriminant >= -discriminant_tolerance) {
      discriminant = max(0.0, discriminant);
      let root = sqrt(discriminant);
      let signed_root = select(root, -root, quadratic_b < 0.0);
      let q = -quadratic_b - signed_root;
      let q_tolerance = TRANSPORT_F32_EPSILON * (abs(quadratic_a)
        + abs(quadratic_b) + abs(quadratic_c) + root);
      var root0 = -quadratic_b / quadratic_a;
      var root1 = root0;
      if (abs(q) > q_tolerance) { root0 = q / quadratic_a; root1 = quadratic_c / q; }
      for (var root_index = 0u; root_index < 2u; root_index = root_index + 1u) {
        let fraction = select(root0, root1, root_index == 1u);
        let point = origin + displacement * fraction;
        let radial = point - vertex;
        let radial_length = length(radial);
        let outward_normal = select(surface_vertex_fallback_normal(vertex_index),
          radial / max(radial_length, 1.1754943508222875e-38),
          radial_length > spatial_tolerance);
        best = merge_sweep_candidate(best, fraction, origin, displacement,
          outward_normal, fraction_tolerance, direction_tolerance);
      }
    }
  }
  return best;
}

fn project_contact_cone(value: vec2<f32>, hit: SweepHit,
  tolerance: f32) -> vec2<f32> {
  if (hit.normal_count == 0u) { return value; }
  let inward0 = dot(value, hit.normal0);
  if (hit.normal_count == 1u) {
    return select(value - inward0 * hit.normal0, value, inward0 >= -tolerance);
  }
  let inward1 = dot(value, hit.normal1);
  if (inward0 >= -tolerance && inward1 >= -tolerance) { return value; }
  var result = vec2<f32>(0.0);
  var error = dot(value, value);
  let candidate0 = value - inward0 * hit.normal0;
  if (dot(candidate0, hit.normal1) >= -tolerance) {
    result = candidate0; error = dot(candidate0 - value, candidate0 - value);
  }
  let candidate1 = value - inward1 * hit.normal1;
  let candidate1_error = dot(candidate1 - value, candidate1 - value);
  if (dot(candidate1, hit.normal0) >= -tolerance && candidate1_error < error) {
    result = candidate1; error = candidate1_error;
  }
  return result;
}

fn resolve_terrain_motion(origin: vec2<f32>, input_velocity: vec2<f32>,
  dt: f32, radius: f32, failure_receipt: u32) -> ContactResult {
  if (params.limits.y == 0u) { return ContactResult(origin + input_velocity * dt, input_velocity); }
  var position = origin;
  var velocity = input_velocity;
  var remaining = dt;
  var contact_count = 0u;
  let scale = transport_scale(origin, origin + input_velocity * dt, radius);
  let spatial_tolerance = TRANSPORT_F32_EPSILON * scale;
  let direction_tolerance = spatial_tolerance
    + TRANSPORT_F32_EPSILON * length(input_velocity * dt);

  // Depenetration is independent of requested motion. In particular, a
  // stationary disk at an acute joint must clear every active capsule.
  var current_sample = sample_terrain(position);
  for (var depenetration_index = 0u; depenetration_index < MAX_TRANSPORT_CONTACTS;
    depenetration_index = depenetration_index + 1u) {
    if (current_sample.signed_distance >= radius - spatial_tolerance) { break; }
    if (contact_count >= MAX_TRANSPORT_CONTACTS) {
      flag_contact_budget_exhausted(failure_receipt, 1u);
      return ContactResult(position, velocity);
    }
    let correction = overlap_correction(position, current_sample,
      radius, spatial_tolerance);
    if (length(correction) <= spatial_tolerance) {
      flag_contact_budget_exhausted(failure_receipt, 2u);
      return ContactResult(position, velocity);
    }
    position = position + correction; contact_count = contact_count + 1u;
    current_sample = sample_terrain(position);
  }
  if (current_sample.signed_distance < radius - spatial_tolerance) {
    flag_contact_budget_exhausted(failure_receipt, 3u);
    return ContactResult(position, velocity);
  }

  for (var contact_index = 0u; contact_index < MAX_TRANSPORT_CONTACTS;
    contact_index = contact_index + 1u) {
    if (remaining <= dt * TRANSPORT_F32_EPSILON
        || length(velocity * remaining) <= spatial_tolerance) {
      remaining = 0.0; break;
    }
    let hit = sweep_terrain_disk(position, velocity, remaining, radius, 1u);
    if (hit.hit == 0u) {
      position = position + velocity * remaining; remaining = 0.0; break;
    }
    if (contact_count >= MAX_TRANSPORT_CONTACTS) { break; }
    position = hit.point; contact_count = contact_count + 1u;
    if (hit.initial_overlap != 0u) { continue; }
    let remaining_after_hit = remaining * max(0.0, 1.0 - hit.fraction);
    if (remaining_after_hit > dt * TRANSPORT_F32_EPSILON) {
      // Canonical cone projection acts on displacement (L), so its tolerance
      // remains dimensionally correct. Recover velocity only after projection.
      let projected_displacement = project_contact_cone(
        velocity * remaining_after_hit, hit, direction_tolerance);
      velocity = projected_displacement / remaining_after_hit;
    } else {
      velocity = project_contact_cone(velocity, hit,
        direction_tolerance / max(dt, 1.1754943508222875e-38));
    }
    remaining = remaining_after_hit;
  }

  // A fixed GPU loop may end only with a proof that the unconsumed motion is
  // collision-free. Otherwise emit an explicit failed-physics telemetry bit.
  if (remaining > dt * TRANSPORT_F32_EPSILON
      && length(velocity * remaining) > spatial_tolerance) {
    let blocking = sweep_terrain_disk(position, velocity, remaining, radius, 1u);
    if (blocking.hit == 0u) {
      position = position + velocity * remaining; remaining = 0.0;
    } else {
      flag_contact_budget_exhausted(failure_receipt, 4u);
    }
  }
  // Roundoff at a co-temporal joint can leave the accepted point microscopically
  // inside a second capsule. Spend only the remaining explicit contact budget
  // to depenetrate it, then verify rather than silently certifying the state.
  var final_sample = sample_terrain(position);
  for (var final_projection_index = 0u;
    final_projection_index < MAX_TRANSPORT_CONTACTS;
    final_projection_index = final_projection_index + 1u) {
    if (final_sample.signed_distance >= radius - spatial_tolerance) { break; }
    if (contact_count >= MAX_TRANSPORT_CONTACTS) {
      flag_contact_budget_exhausted(failure_receipt, 5u); break;
    }
    let correction = overlap_correction(position, final_sample,
      radius, spatial_tolerance);
    if (length(correction) <= spatial_tolerance) {
      flag_contact_budget_exhausted(failure_receipt, 6u); break;
    }
    position = position + correction; contact_count = contact_count + 1u;
    final_sample = sample_terrain(position);
  }
  if (final_sample.signed_distance < radius - spatial_tolerance) {
    flag_contact_budget_exhausted(failure_receipt, 7u);
  }
  return ContactResult(position, velocity);
}

// Static transport must leave clearance for the stricter preventive mover.
// Exact-touch f32 projection can sit micrometres inside its rounded capsule.
const WHEEL_TRANSPORT_SKIN: f32 = 0.00002;
fn resolve_wheel_motion_at(position_input: vec2<f32>, velocity_input: vec2<f32>,
  particle_radius: f32, angle: f32, angular_velocity: f32) -> ContactResult {
  var position = position_input;
  var velocity = velocity_input;
  let center = params.terrain.zw;
  let contact_radius = particle_radius + WHEEL_BAR_HALF_WIDTH + WHEEL_TRANSPORT_SKIN;

  // Repeated projections resolve simultaneous baffle/rim contacts.
  for (var projection = 0u; projection < 4u; projection = projection + 1u) {
    for (var segment = 0u; segment < RIGID_SEGMENT_COUNT; segment = segment + 1u) {
      let local = baffle(segment);
      let a = center + rotate_local(local.xy, angle);
      let b = center + rotate_local(local.zw, angle);
      let edge = b - a;
      let along = clamp(dot(position - a, edge) / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
      let closest = a + edge * along;
      let separation = position - closest;
      let distance = length(separation);
      if (distance < contact_radius) {
        let tangent = normalize(edge);
        let normal = select(vec2<f32>(-tangent.y, tangent.x),
          separation / max(distance, 1.0e-8), distance > 1.0e-8);
        position = position + normal * (contact_radius - distance);
        let radius_vector = closest - center;
        let surface_velocity = angular_velocity
          * vec2<f32>(-radius_vector.y, radius_vector.x);
        var relative_velocity = velocity - surface_velocity;
        let inward_speed = dot(relative_velocity, normal);
        if (inward_speed < 0.0) {
          relative_velocity = relative_velocity - normal * inward_speed;
        }
        // The baffles are smooth rigid constraints: only their normal motion
        // transfers momentum. Tangential coupling would behave like an
        // adhesive belt and carry liquid around the drum wall.
        velocity = surface_velocity + relative_velocity;
      }
    }
    let radial = position - center;
    let radial_length = length(radial);
    let rim_limit = WHEEL_RADIUS - contact_radius;
    if (radial_length > rim_limit) {
      let outward = radial / max(radial_length, 1.0e-8);
      let normal = -outward;
      position = center + outward * rim_limit;
      let closest = center + outward * WHEEL_RADIUS;
      let radius_vector = closest - center;
      let surface_velocity = angular_velocity
        * vec2<f32>(-radius_vector.y, radius_vector.x);
      var relative_velocity = velocity - surface_velocity;
      let inward_speed = dot(relative_velocity, normal);
      if (inward_speed < 0.0) {
        relative_velocity = relative_velocity - normal * inward_speed;
      }
      // A circular rim has no normal motion when it spins. Keep the contact
      // frictionless so rotation cannot inject a rim-following velocity.
      velocity = surface_velocity + relative_velocity;
    }
  }
  return ContactResult(position, velocity);
}

// Exact swept capsule/rim intersections; bounded work even at high speed.
fn resolve_wheel_motion(origin: vec2<f32>, destination: vec2<f32>, velocity: vec2<f32>, radius: f32) -> ContactResult {
  let start=resolve_wheel_motion_at(origin,velocity,radius,params.terrain.x,0.0);
  let result=swept_wheel_motion(start.position,destination,start.velocity,params.terrain.zw,
    WHEEL_RADIUS-radius-WHEEL_BAR_HALF_WIDTH-WHEEL_TRANSPORT_SKIN,
    radius+WHEEL_BAR_HALF_WIDTH+WHEEL_TRANSPORT_SKIN,params.terrain.x);
  return resolve_wheel_motion_at(result.position,result.velocity,radius,params.terrain.x,0.0);
}

// Newtonian wall shear transfers tangential momentum into the kinematic drum.
// The particle-grid viscosity law carries that loss from the wall into bulk
// water; no frame-dependent global damping is applied.
fn apply_wheel_wall_shear(position: vec2<f32>, velocity: vec2<f32>, radius: f32) -> vec2<f32> {
  let offset = position - params.terrain.zw;
  let distance = length(offset);
  let gap = WHEEL_RADIUS - radius - WHEEL_BAR_HALF_WIDTH - distance;
  let kinematic_viscosity = params.material.w / params.material.x;
  let boundary_layer = params.fluid.y * 0.5;
  let scale = kinematic_viscosity * params.fluid.x
    / max(boundary_layer * boundary_layer, 1.0e-12);
  var result = velocity;
  if (gap < params.fluid.z && distance > 1.0e-8) {
    let tangent = vec2<f32>(-offset.y, offset.x) / distance;
    let weight = clamp(1.0 - max(0.0, gap) / params.fluid.z, 0.0, 1.0);
    let fraction = 1.0 - exp(-scale * weight);
    let wall_tangent_speed = params.terrain.y * distance;
    result = result - tangent * (dot(result, tangent) - wall_tangent_speed) * fraction;
  }
  return result;
}

@compute @workgroup_size(128)
fn sweep_wheel(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.x){return;}
  let radius=params.fluid.y*0.46;
  let swept=swept_wheel_rotation(particles[i].position,particles[i].velocity,params.terrain.zw,
    radius+WHEEL_BAR_HALF_WIDTH,params.motion.x,params.motion.y,params.motion.w);
  let result=resolve_wheel_motion_at(swept.position,swept.velocity,radius,params.terrain.x,0.0);
  let before=length(particles[i].velocity);
  let after=length(result.velocity);
  particles[i].position=result.position;particles[i].velocity=result.velocity;
  atomicMax(&cell_counts[telemetry_base()+9u],bitcast<u32>(dot(result.velocity,result.velocity)));
  atomicMax(&cell_counts[telemetry_base()+11u],bitcast<u32>(max(0.0,after-before)));
  atomicMax(&cell_counts[telemetry_base()+12u],bitcast<u32>(before*before));
}

${createDynamicParticleGridWgsl2D({
  originExpression: 'params.world.xy',
  cellSizeExpression: 'params.fluid.z',
  particleExpression: 'particles[index].position',
})}
${createDynamicParticleGridAuditWgsl2D({
  particleExpression: 'particles[index].position',
  occupiedBucketTelemetryOffset: 4,
  activeCellTelemetryOffset: 5,
})}

fn valid_solid_cell(cell: vec2<i32>) -> bool {
  let dimensions = vec2<i32>(ceil((params.world.zw - params.world.xy) / params.fluid.z));
  return cell.x >= 0 && cell.y >= 0
    && cell.x < dimensions.x && cell.y < dimensions.y;
}

fn solid_cell_index(cell: vec2<i32>) -> u32 {
  let columns = u32(ceil((params.world.z - params.world.x) / params.fluid.z));
  return u32(cell.y) * columns + u32(cell.x);
}

fn measure_constraint(index: u32) -> ConstraintSample {
  let position = particles[index].position;
  let velocity = particles[index].velocity;
  let inverse_density = 1.0 / params.material.x;
  let inverse_mass = 1.0 / params.fluid.w;
  var density = params.fluid.w * sample_kernel(vec2<f32>(0.0)).weight;
  var center_gradient = vec2<f32>(0.0);
  var gradient_norm_sum = 0.0;
  var neighbor_velocity_term = 0.0;
  var neighbor_count = 0u;
  let center_cell = cell_coordinate(position);
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other >= params.counts.x || other == index) { continue; }
        if (!particle_occupies_cell(other, cell)) { continue; }
        let displacement = position - particles[other].position;
        let kernel = sample_particle_kernel(displacement, index, other);
        if (kernel.supported == 0u) { continue; }
        let gradient = params.fluid.w * inverse_density * kernel.gradient;
        density = density + params.fluid.w * kernel.weight;
        center_gradient = center_gradient + gradient;
        gradient_norm_sum = gradient_norm_sum + inverse_mass * dot(gradient, gradient);
        neighbor_velocity_term = neighbor_velocity_term
          + dot(gradient, particles[other].velocity);
        neighbor_count = neighbor_count + 1u;
      }
    }
  }
  let material_density = density;
  // Each static bucket is pre-expanded to the 3x3 support neighborhood, so
  // the canonical Akinci candidates require one sorted traversal per particle.
  // A solid sample contributes rho0*volume*W to density and volume*grad(W)
  // only to the fluid particle's center gradient. Its static velocity makes
  // the corresponding density-rate subtraction exactly zero.
  if (valid_solid_cell(center_cell)) {
    let solid_bucket = solid_cell_index(center_cell);
    let solid_count = boundary_cell_count(solid_bucket);
    for (var solid_slot = 0u; solid_slot < solid_count;
      solid_slot = solid_slot + 1u) {
      let solid_index = boundary_cell_item(solid_bucket, solid_slot);
      if (solid_index >= params.limits.z) { continue; }
      let solid = boundary_particle(solid_index);
      let displacement = position - solid.xy;
      let kernel = sample_kernel(displacement);
      if (kernel.supported == 0u) { continue; }
      density = density + params.material.x * solid.z * kernel.weight;
      center_gradient = center_gradient + solid.z * kernel.gradient;
    }
  }
  let wheel_boundary = wheel_boundary_support(position);
  density = density + params.material.x * wheel_boundary.density_factor;
  center_gradient = center_gradient + wheel_boundary.gradient;
  let offset=position-params.terrain.zw;
  let wall_velocity=params.terrain.y*vec2<f32>(-offset.y,offset.x);
  let density_rate = dot(center_gradient, velocity) - neighbor_velocity_term
    - dot(wheel_boundary.gradient,wall_velocity);
  let denominator = inverse_mass * dot(center_gradient, center_gradient)
    + gradient_norm_sum;
  return ConstraintSample(density, material_density, density_rate, max(denominator, 1.0e-8),
    neighbor_count, center_gradient);
}

@compute @workgroup_size(128)
fn clear_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let cell_count = params.counts.y * params.counts.z;
  let cell=invocation.x;if(cell>=cell_count){return;}
  let previous=atomicLoad(&cell_counts[cell]);atomicStore(&cell_counts[cell],0u);
  if(previous>0u){for(var slot=0u;slot<min(previous,params.counts.w);slot++){
    atomicStore(&cell_items[cell*params.counts.w+slot],0xffffffffu);
  }
  }
}

@compute @workgroup_size(128)
fn fill_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.counts.x) { return; }
  record_primary_telemetry(index);
  let cell = cell_coordinate(particles[index].position);
  if (!valid_cell(cell)) { return; }
  let bucket = cell_index(cell);
  let occupancy = atomicAdd(&cell_counts[bucket], 1u) + 1u;
  atomicMax(&cell_counts[telemetry_base() + 2u], occupancy);
  // Concurrent atomic-min insertion is a deterministic bounded top-K: if a
  // cell overflows, the lowest source IDs are retained regardless of dispatch
  // order. The following pass sorts those IDs before any floating-point sum.
  var candidate = index;
  for (var slot = 0u; slot < params.counts.w; slot = slot + 1u) {
    let address = bucket * params.counts.w + slot;
    let previous = atomicMin(&cell_items[address], candidate);
    if (previous == 0xffffffffu) { break; }
    candidate = max(candidate, previous);
  }
}

@compute @workgroup_size(128)
fn sort_cells(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let bucket = invocation.x;
  let cell_count = params.counts.y * params.counts.z;
  if (bucket >= cell_count) { return; }
  let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
  let base = bucket * params.counts.w;
  for (var index = 1u; index < count; index = index + 1u) {
    let key = atomicLoad(&cell_items[base + index]);
    var cursor = index;
    while (cursor > 0u) {
      let previous = atomicLoad(&cell_items[base + cursor - 1u]);
      if (dynamic_grid_item_precedes(previous, key)) { break; }
      atomicStore(&cell_items[base + cursor], previous);
      cursor = cursor - 1u;
    }
    atomicStore(&cell_items[base + cursor], key);
  }
  audit_sorted_bucket(bucket, count);
}

fn predict_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let dt = params.fluid.x;
  particles[index].velocity = particles[index].velocity + params.force.xy * dt;
  particles[index].foam = particles[index].foam * exp(-dt / 0.85);
}

fn divergence_lambda_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let sample = measure_constraint(index);
  let residual = max(0.0, sample.density_rate - params.material.y);
  particles[index].lambda = -residual / sample.denominator;
  particles[index].density = sample.material_density;
  particles[index].neighbor_count = sample.neighbor_count;
}

fn density_lambda_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let sample = measure_constraint(index);
  let density_error = sample.density / params.material.x - 1.0;
  let residual = max(0.0, density_error + params.fluid.x * sample.density_rate
    - params.material.z) / params.fluid.x;
  particles[index].lambda = -residual / sample.denominator;
  particles[index].scratch_velocity.x = particles[index].scratch_velocity.x
    + particles[index].lambda * params.force.z;
  particles[index].density = sample.material_density;
  particles[index].neighbor_count = sample.neighbor_count;
}

fn apply_pressure_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let position = particles[index].position;
  let own_lambda = particles[index].lambda;
  let inverse_mass = 1.0 / params.fluid.w;
  let inverse_density = 1.0 / params.material.x;
  var correction = vec2<f32>(0.0);
  let center_cell = cell_coordinate(position);
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other >= params.counts.x || other == index) { continue; }
        if (!particle_occupies_cell(other, cell)) { continue; }
        let displacement = position - particles[other].position;
        let kernel = sample_particle_kernel(displacement, index, other);
        if (kernel.supported == 0u) { continue; }
        let gradient = params.fluid.w * inverse_density * kernel.gradient;
        correction = correction + inverse_mass
          * (own_lambda + particles[other].lambda) * gradient;
      }
    }
  }
  // This specialization treats the solid as an infinite-mass boundary, so its
  // pressure support contributes only the fluid center's correction.
  if (valid_solid_cell(center_cell)) {
    let solid_bucket = solid_cell_index(center_cell);
    let solid_count = boundary_cell_count(solid_bucket);
    for (var solid_slot = 0u; solid_slot < solid_count;
      solid_slot = solid_slot + 1u) {
      let solid_index = boundary_cell_item(solid_bucket, solid_slot);
      if (solid_index >= params.limits.z) { continue; }
      let solid = boundary_particle(solid_index);
      let displacement = position - solid.xy;
      let kernel = sample_kernel(displacement);
      if (kernel.supported == 0u) { continue; }
      let gradient = solid.z * kernel.gradient;
      correction = correction + inverse_mass * own_lambda * gradient;
    }
  }
  let wheel_boundary = wheel_boundary_support(position);
  correction = correction + inverse_mass * own_lambda * wheel_boundary.gradient;
  // Retain the GPU Jacobi relaxation already used by this specialization; it
  // applies uniformly after both canonical fluid and Akinci solid terms.
  let change = correction * params.force.z;
  // The density-rate linearization is only local. One Jacobi update may not
  // extrapolate farther than a quarter of a parcel spacing in one law step.
  let trust_radius = params.fluid.y * 0.25 / params.fluid.x;
  particles[index].velocity = particles[index].velocity
    + change * min(1.0, trust_radius / max(length(change), 1.0e-8));
}

fn advect_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let radius = params.fluid.y * 0.46;
  record_motion_telemetry(particles[index].position, particles[index].velocity);
  atomicMax(&cell_counts[telemetry_base()+10u],
    bitcast<u32>(dot(particles[index].velocity,particles[index].velocity)));
  let contact = resolve_terrain_motion(particles[index].position,
    particles[index].velocity, params.fluid.x, radius, index + 1u);
  let wheel_contact = resolve_wheel_motion(particles[index].position, contact.position, contact.velocity, radius);
  particles[index].position = wheel_contact.position;
  particles[index].velocity = apply_wheel_wall_shear(
    wheel_contact.position, wheel_contact.velocity, radius);
  record_motion_telemetry(wheel_contact.position, particles[index].velocity);
  atomicMax(&cell_counts[telemetry_base() + 6u],
    bitcast<u32>(dot(particles[index].velocity, particles[index].velocity)));
}

fn classify_and_filter_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let position = particles[index].position;
  let velocity = particles[index].velocity;
  let center_cell = cell_coordinate(position);
  var weighted_velocity = vec2<f32>(0.0);
  var weight_sum = 0.0;
  var trapped_air = 0.0;
  var normal_accumulator = vec2<f32>(0.0);
  var vorticity = 0.0;
  var neighbor_count = 0u;
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other >= params.counts.x || other == index) { continue; }
        if (!particle_occupies_cell(other, cell)) { continue; }
        let displacement = position - particles[other].position;
        let distance = length(displacement);
        if (distance <= 1.0e-7 || distance >= params.fluid.z) { continue; }
        let weight = 1.0 - distance / params.fluid.z;
        let relative_velocity = velocity - particles[other].velocity;
        let relative_speed = length(relative_velocity);
        weighted_velocity = weighted_velocity + particles[other].velocity * weight;
        weight_sum = weight_sum + weight;
        normal_accumulator = normal_accumulator + displacement / distance * weight;
        if (relative_speed > 1.0e-6) {
          let alignment = dot(relative_velocity / relative_speed, displacement / distance);
          trapped_air = trapped_air + relative_speed * (1.0 - alignment) * weight;
          vorticity = vorticity + abs(displacement.x * relative_velocity.y
            - displacement.y * relative_velocity.x) / (distance * distance) * weight;
        }
        neighbor_count = neighbor_count + 1u;
      }
    }
  }
  let normal = select(vec2<f32>(0.0, 1.0), normalize(normal_accumulator),
    length(normal_accumulator) > 1.0e-6);
  let exposure = 1.0 - clamp(f32(neighbor_count) / 18.0, 0.0, 1.0);
  let speed_scale = sqrt(max(0.01, length(params.force.xy) * params.fluid.z));
  let air_potential = clamp(trapped_air / max(speed_scale * 10.0, 1.0e-6), 0.0, 1.0);
  let crest_potential = select(0.0,
    clamp(dot(velocity, normal) / max(speed_scale * 2.0, 1.0e-6), 0.0, 1.0),
    dot(velocity, normal) > 0.6 * max(length(velocity), 1.0e-6));
  let vortex_potential = clamp(vorticity * params.fluid.z / 18.0, 0.0, 1.0);
  let kinetic = clamp(dot(velocity, velocity)
    / max(4.0 * length(params.force.xy) * params.fluid.z, 1.0e-6), 0.0, 1.0);
  let generated_foam = clamp((0.45 * air_potential + 0.30 * crest_potential
    + 0.15 * vortex_potential + 0.10 * kinetic) * exposure * 2.4, 0.0, 1.0);
  particles[index].foam = max(particles[index].foam, generated_foam);
  particles[index].neighbor_count = neighbor_count;
  particles[index].surface_class = select(0u, 1u, neighbor_count <= 2u);
  if (weight_sum > 1.0e-6) {
    let average_velocity = weighted_velocity / weight_sum;
    particles[index].scratch_velocity = mix(velocity, average_velocity,
      clamp(params.material.w * params.fluid.x, 0.0, 0.35));
  } else {
    particles[index].scratch_velocity = velocity;
  }
}

fn apply_filter_body(index:u32) {
  if (index < params.counts.x) {
    particles[index].velocity = particles[index].scratch_velocity;
    particles[index].scratch_velocity = vec2<f32>(0.0);
  }
}

fn update_compressibility_body(index:u32) {
  if (index >= params.counts.x) { return; }
  let pressure_impulse = max(0.0, -particles[index].scratch_velocity.x);
  let pressure_scale = params.material.x
    / max(params.fluid.x * params.fluid.w, 1.0e-20);
  let maximum_pressure = params.force.w * 0.01;
  let pressure = min(pressure_impulse, maximum_pressure / pressure_scale)
    * pressure_scale;
  let rest_volume = params.fluid.w / params.material.x;
  particles[index].volume = rest_volume / (1.0 + pressure / params.force.w);
  particles[index].scratch_velocity = vec2<f32>(0.0);
}

@compute @workgroup_size(128) fn predict(@builtin(global_invocation_id) g:vec3<u32>){predict_body(g.x);}
@compute @workgroup_size(128) fn divergence_lambda(@builtin(global_invocation_id) g:vec3<u32>){divergence_lambda_body(g.x);}
@compute @workgroup_size(128) fn density_lambda(@builtin(global_invocation_id) g:vec3<u32>){density_lambda_body(g.x);}
@compute @workgroup_size(128) fn apply_pressure(@builtin(global_invocation_id) g:vec3<u32>){apply_pressure_body(g.x);}
@compute @workgroup_size(128) fn classify_and_filter(@builtin(global_invocation_id) g:vec3<u32>){classify_and_filter_body(g.x);}
@compute @workgroup_size(128) fn apply_filter(@builtin(global_invocation_id) g:vec3<u32>){apply_filter_body(g.x);}
@compute @workgroup_size(128) fn update_compressibility(@builtin(global_invocation_id) g:vec3<u32>){update_compressibility_body(g.x);}
@compute @workgroup_size(128) fn update_compressibility_and_advect(@builtin(global_invocation_id) g:vec3<u32>){update_compressibility_body(g.x);advect_body(g.x);}
@compute @workgroup_size(128) fn preventive_predict(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){predict_body(g.x);}}
@compute @workgroup_size(128) fn preventive_divergence_lambda(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){divergence_lambda_body(g.x);}}
@compute @workgroup_size(128) fn preventive_density_lambda(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){density_lambda_body(g.x);}}
@compute @workgroup_size(128) fn preventive_apply_pressure(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){apply_pressure_body(g.x);}}
@compute @workgroup_size(128) fn preventive_classify_and_filter(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){classify_and_filter_body(g.x);}}
@compute @workgroup_size(128) fn preventive_apply_filter(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){apply_filter_body(g.x);}}
@compute @workgroup_size(128) fn preventive_update_compressibility(@builtin(global_invocation_id) g:vec3<u32>){if(motion_force_ready()){update_compressibility_body(g.x);}}
@compute @workgroup_size(128) fn preventive_audit_predicted(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<params.counts.x&&motion_force_ready()){let v=particles[g.x].velocity;let speed2=dot(v,v);atomicMax(&cell_counts[telemetry_base()+6u],bitcast<u32>(speed2));if(speed2>625.0){let sample=measure_constraint(g.x);atomicMax(&cell_counts[telemetry_base()+7u],bitcast<u32>(1.0/sample.denominator));atomicMax(&cell_counts[telemetry_base()+8u],bitcast<u32>(max(0.0,sample.density/params.material.x-1.0)));}}}

fn try_emit_owned_diffuse(index: u32) {
  if (index >= params.counts.x || particles[index].foam < 0.12) { return; }
  // Derive stochastic emission from the evolving GPU state. Multiple fixed
  // substeps may share one command-buffer submission, so a CPU-written frame
  // uniform would alias all of those substeps to the final value.
  let position_bits = bitcast<vec2<u32>>(particles[index].position);
  let serial = hash_u32((index * 1013904223u) ^ position_bits.x
    ^ (position_bits.y * 1664525u));
  let probability = clamp(particles[index].foam * params.stone.z * params.fluid.x, 0.0, 0.8);
  if (random_unit(serial) >= probability) { return; }
  // Slot ownership is stable: source i owns diffuse slot i. No contended
  // winner election means reset/replay produces the same secondary phase.
  let slot = index;
  if (slot >= params.limits.x || atomicLoad(&diffuse_active[slot]) != 0u) { return; }
  atomicStore(&diffuse_active[slot], 1u);
  let jitter = (random_unit(serial ^ 17u) - 0.5) * params.fluid.y;
  let source = particles[index];
  let spray_source = source.surface_class > 0u;
  let source_lift = select(0.22, 0.72, spray_source);
  diffuse[slot].position = source.position
    + vec2<f32>(jitter, params.fluid.y * source_lift);
  diffuse[slot].velocity = source.velocity + vec2<f32>(jitter * 2.0,
    random_unit(serial ^ 0x6c8e9cf5u) * length(source.velocity)
      * select(0.16, 0.52, spray_source));
  diffuse[slot].radius = params.fluid.y
    * mix(0.055, 0.14, random_unit(serial ^ 0x27d4eb2fu));
  diffuse[slot].age = 0.0;
  diffuse[slot].lifetime = mix(1.2, 3.4, random_unit(serial ^ 0x165667b1u));
  diffuse[slot].kind = select(FOAM, SPRAY, spray_source);
}

@compute @workgroup_size(128)
fn update_diffuse_and_render(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.limits.x) { return; }
  // Frozen or unresolved motion has no representation-time credit. Preserve
  // phase, velocity, emission and age as well as position; do not animate foam
  // independently of the World's accepted clock.
  if (params.fluid.x <= 0.0) { return; }
  // Source i exclusively owns slot i, so emission, phase evolution, and render
  // packing are one race-free dispatch. The prior three-dispatch ordering had
  // no cross-slot dependency and is therefore exactly equivalent.
  try_emit_owned_diffuse(index);
  if (atomicLoad(&diffuse_active[index]) == 0u) {
    diffuse_render[index] = vec4<f32>(0.0);
    return;
  }
  var marker = diffuse[index];
  let center_cell = cell_coordinate(marker.position);
  var velocity_sum = vec2<f32>(0.0);
  var weight_sum = 0.0;
  var neighbor_count = 0u;
  for (var offset_y = -1; offset_y <= 1; offset_y = offset_y + 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x = offset_x + 1) {
      let cell = center_cell + vec2<i32>(offset_x, offset_y);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other >= params.counts.x) { continue; }
        if (!particle_occupies_cell(other, cell)) { continue; }
        let distance = length(marker.position - particles[other].position);
        if (distance >= params.fluid.z) { continue; }
        let weight = 1.0 - distance / params.fluid.z;
        velocity_sum = velocity_sum + particles[other].velocity * weight;
        weight_sum = weight_sum + weight;
        neighbor_count = neighbor_count + 1u;
      }
    }
  }
  let fluid_velocity = select(marker.velocity, velocity_sum / max(weight_sum, 1.0e-6),
    weight_sum > 1.0e-6);
  if (neighbor_count <= 1u) {
    marker.kind = SPRAY;
    marker.velocity = marker.velocity + params.force.xy * params.fluid.x;
  } else if (weight_sum < 3.1) {
    marker.kind = FOAM;
    marker.velocity = mix(marker.velocity, fluid_velocity,
      1.0 - exp(-273.46 * params.fluid.x));
  } else {
    marker.kind = BUBBLE;
    marker.velocity = mix(marker.velocity, fluid_velocity,
      1.0 - exp(-147.87 * params.fluid.x));
    let gravity_magnitude = length(params.force.xy);
    if (gravity_magnitude > 1.0e-8) {
      let buoyancy_direction = -params.force.xy / gravity_magnitude;
      marker.velocity = marker.velocity + buoyancy_direction
        * 0.71 * sqrt(gravity_magnitude * 2.0 * marker.radius) * params.fluid.x;
    }
  }
  let contact = resolve_terrain_motion(marker.position, marker.velocity,
    params.fluid.x, marker.radius, 0x80000000u | (index + 1u));
  let wheel_contact = resolve_wheel_motion(marker.position, contact.position, contact.velocity, marker.radius);
  marker.position = wheel_contact.position;
  marker.velocity = wheel_contact.velocity;
  if (sample_terrain(marker.position).signed_distance
      <= marker.radius + 0.00003) {
    marker.kind = FOAM;
  }
  marker.age = marker.age + params.fluid.x;
  let outside = marker.position.x < params.world.x || marker.position.x > params.world.z
    || marker.position.y < params.world.y || marker.position.y > params.world.w;
  if (marker.age >= marker.lifetime || outside) {
    atomicStore(&diffuse_active[index], 0u);
    marker.radius = 0.0;
  }
  diffuse[index] = marker;
  record_diffuse_finiteness(marker);
  if (atomicLoad(&diffuse_active[index]) == 0u) {
    diffuse_render[index] = vec4<f32>(0.0);
  } else {
    diffuse_render[index] = vec4<f32>(marker.position,
      marker.radius, f32(marker.kind));
  }
}
`;

const requireDevice = (device) => {
  if (!device || typeof device.createBuffer !== 'function'
      || typeof device.createComputePipelineAsync !== 'function') {
    throw new TypeError('WebGPU device required for Liquid particle world');
  }
  return device;
};

const ceilDiv = (value, divisor) => Math.ceil(value / divisor);
const MAXIMUM_LIQUID_PARTICLE_AXIS_COUNT = 4096;
const MAXIMUM_LIQUID_PRIMARY_PARTICLE_COUNT = 262144;
const MAXIMUM_LIQUID_GRID_AXIS_COUNT = 4096;
const MAXIMUM_LIQUID_GRID_CELL_COUNT = 1048576;

export const LIQUID_PARTICLE_WORLD_GPU_POLICY = Object.freeze({
  dimension: 2,
  columns: 36,
  rows: 14,
  particleSpacing: 0.018,
  supportScale: 2.35,
  restDensity: 998.207,
  divergenceTolerance: 0.4,
  densityTolerance: 0.001,
  viscosity: 1.8,
  friction: 0,
  jacobiRelaxation: 1,
  bulkModulus: 2.2e9,
  gravity: Object.freeze([0, -9.82]),
  timeStep: 1 / 240,
  maximumCourantNumber: 0.1,
  divergenceIterations: 2,
  densityIterations: 6,
  worldMinimum: Object.freeze([-0.70, -0.45]),
  worldMaximum: Object.freeze([0.70, 0.86]),
  viewMinimum: Object.freeze([-0.62, -0.20]),
  viewMaximum: Object.freeze([0.62, 0.84]),
  bedSlope: 0,
  bedAtStone: -0.35,
  stoneCenterX: 0,
  stoneHalfWidth: 0.23,
  stoneHeight: 0.0001,
  seedMaximumX: 0.315,
  seedMinimumY: -0.10,
  initialSpeed: 0,
  // Compression reserve. Kernels still traverse only occupied entries, so
  // this raises safety capacity without turning pressure into an all-pairs solve.
  maximumParticlesPerCell: 64,
  boundaryProfileSegments: 160,
  diffuseCapacity: 1024,
  diffuseEmissionRate: 18,
});

// Optional dry-contact Law for the fluid-transport comparison. Pressure
// projection supplies a normal impulse; only that impulse may support
// tangential friction. In free flight lambda is zero, so no artificial
// cohesion, clumping or velocity drag is introduced.
const pressureFrictionWgsl = coefficient => /* wgsl */`
const PRESSURE_FRICTION: f32 = ${Math.fround(coefficient).toExponential(8)};

fn pressure_friction_candidate_body(index: u32) {
  if (index >= params.counts.x) { return; }
  let position = particles[index].position;
  let velocity = particles[index].velocity;
  let center_cell = cell_coordinate(position);
  var change = vec2<f32>(0.0);
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let cell = center_cell + vec2<i32>(dx, dy);
      if (!valid_cell(cell)) { continue; }
      let bucket = cell_index(cell);
      let count = min(atomicLoad(&cell_counts[bucket]), params.counts.w);
      for (var slot = 0u; slot < count; slot = slot + 1u) {
        let other = atomicLoad(&cell_items[bucket * params.counts.w + slot]);
        if (other >= params.counts.x || other == index) { continue; }
        if (!particle_occupies_cell(other, cell)) { continue; }
        let separation = position - particles[other].position;
        let kernel = sample_particle_kernel(separation, index, other);
        if (kernel.supported == 0u) { continue; }
        let distance = length(separation);
        if (distance < 1.0e-6) { continue; }
        let normal = separation / distance;
        let relative = velocity - particles[other].velocity;
        let tangential = relative - normal * dot(relative, normal);
        let speed = length(tangential);
        if (speed < 1.0e-7) { continue; }
        // The DFSPH lambda is the normal pressure impulse in this step.
        // Symmetric pair impulses cancel in total linear momentum.
        let normal_speed = max(0.0, -particles[index].lambda
          - particles[other].lambda) * length(kernel.gradient)
          * params.force.z / max(params.material.x * params.fluid.w, 1.0e-12);
        let slip = min(0.5 * speed, PRESSURE_FRICTION * normal_speed);
        change = change - tangential * (slip / speed);
      }
    }
  }
  // A rigid wall can supply normal load even if no particle pair is nearby.
  // Its Coulomb limit is based on gravity into the wall, not a global drag.
  let offset = position - params.terrain.zw;
  let radius = length(offset);
  let gap = WHEEL_RADIUS - params.fluid.y * 0.46 - WHEEL_BAR_HALF_WIDTH - radius;
  if (radius > 1.0e-6 && gap < params.fluid.z * 0.45) {
    let outward = offset / radius;
    let tangent = vec2<f32>(-outward.y, outward.x);
    let wall_velocity = params.terrain.y * radius;
    let relative_tangent = dot(velocity + change, tangent) - wall_velocity;
    let normal_support = max(0.0, dot(params.force.xy, outward))
      * params.fluid.x * clamp(1.0 - max(0.0, gap)
        / (params.fluid.z * 0.45), 0.0, 1.0);
    change = change - tangent * sign(relative_tangent)
      * min(abs(relative_tangent), PRESSURE_FRICTION * normal_support);
  }
  particles[index].scratch_velocity = velocity + change;
}

fn pressure_friction_apply_body(index: u32) {
  if (index >= params.counts.x) { return; }
  particles[index].velocity = particles[index].scratch_velocity;
  particles[index].scratch_velocity = vec2<f32>(0.0);
}

@compute @workgroup_size(128) fn pressure_friction_candidate(
  @builtin(global_invocation_id) gid: vec3<u32>) {
  pressure_friction_candidate_body(gid.x);
}
@compute @workgroup_size(128) fn pressure_friction_apply(
  @builtin(global_invocation_id) gid: vec3<u32>) {
  pressure_friction_apply_body(gid.x);
}
@compute @workgroup_size(128) fn preventive_pressure_friction_candidate(
  @builtin(global_invocation_id) gid: vec3<u32>) {
  if (motion_force_ready()) { pressure_friction_candidate_body(gid.x); }
}
@compute @workgroup_size(128) fn preventive_pressure_friction_apply(
  @builtin(global_invocation_id) gid: vec3<u32>) {
  if (motion_force_ready()) { pressure_friction_apply_body(gid.x); }
}
`;

function normalizePolicy(overrides = {}) {
  if (overrides === null || typeof overrides !== 'object' || Array.isArray(overrides)) {
    throw new TypeError('Liquid particle world policy overrides must be an object');
  }
  for (const key of Reflect.ownKeys(overrides)) {
    if (!Object.hasOwn(LIQUID_PARTICLE_WORLD_GPU_POLICY, key)) {
      throw new TypeError(`Unknown policy key for Liquid particle world: ${String(key)}`);
    }
  }
  const policy = { ...LIQUID_PARTICLE_WORLD_GPU_POLICY, ...overrides };
  const finiteF32 = (value) => Number.isFinite(value) && Number.isFinite(Math.fround(value));
  const finitePair = (name, value) => {
    if (!Array.isArray(value) || value.length !== 2 || !value.every(finiteF32)) {
      throw new TypeError(`Liquid particle world ${name} must be a finite two-vector`);
    }
    return Object.freeze([...value]);
  };
  policy.gravity = finitePair('gravity',
    overrides.gravity ?? LIQUID_PARTICLE_WORLD_GPU_POLICY.gravity);
  policy.worldMinimum = finitePair('worldMinimum', overrides.worldMinimum
    ?? LIQUID_PARTICLE_WORLD_GPU_POLICY.worldMinimum);
  policy.worldMaximum = finitePair('worldMaximum', overrides.worldMaximum
    ?? LIQUID_PARTICLE_WORLD_GPU_POLICY.worldMaximum);
  policy.viewMinimum = finitePair('viewMinimum', overrides.viewMinimum
    ?? LIQUID_PARTICLE_WORLD_GPU_POLICY.viewMinimum);
  policy.viewMaximum = finitePair('viewMaximum', overrides.viewMaximum
    ?? LIQUID_PARTICLE_WORLD_GPU_POLICY.viewMaximum);
  if (policy.dimension !== 2) {
    throw new RangeError('This GPU kernel specialization is two-dimensional; '
      + 'a 3D shader specialization must be generated from the shared rules');
  }
  for (const name of ['columns', 'rows', 'divergenceIterations', 'densityIterations',
    'maximumParticlesPerCell', 'boundaryProfileSegments', 'diffuseCapacity']) {
    if (!Number.isSafeInteger(policy[name]) || policy[name] < 1) {
      throw new RangeError(`Liquid particle world ${name} must be a positive integer`);
    }
  }
  for (const name of ['columns', 'rows']) {
    if (policy[name] > MAXIMUM_LIQUID_PARTICLE_AXIS_COUNT) {
      throw new RangeError(`Liquid particle world ${name} must be at most `
        + MAXIMUM_LIQUID_PARTICLE_AXIS_COUNT);
    }
  }
  for (const name of ['particleSpacing', 'supportScale', 'restDensity', 'timeStep',
    'maximumCourantNumber', 'diffuseEmissionRate', 'stoneHalfWidth',
    'jacobiRelaxation', 'bulkModulus']) {
    if (!finiteF32(policy[name]) || Math.fround(policy[name]) <= 0) {
      throw new RangeError(`Liquid particle world ${name} must be positive`);
    }
  }
  if (policy.supportScale < 2 || policy.supportScale > 4) {
    throw new RangeError('Liquid particle world supportScale must be from two through four');
  }
  for (const name of ['divergenceTolerance', 'densityTolerance', 'viscosity', 'friction',
    'stoneHeight', 'initialSpeed']) {
    if (!finiteF32(policy[name]) || Math.fround(policy[name]) < 0) {
      throw new RangeError(`Liquid particle world ${name} must be nonnegative`);
    }
  }
  for (const name of ['bedSlope', 'bedAtStone', 'stoneCenterX', 'seedMaximumX', 'seedMinimumY']) {
    if (!finiteF32(policy[name])) {
      throw new TypeError(`Liquid particle world ${name} must be finite`);
    }
  }
  for (let axis = 0; axis < 2; axis += 1) {
    if (!(policy.worldMinimum[axis] < policy.worldMaximum[axis])) {
      throw new RangeError('Liquid particle world bounds must increase on every axis');
    }
    if (!(policy.viewMinimum[axis] < policy.viewMaximum[axis])
        || policy.viewMinimum[axis] < policy.worldMinimum[axis]
        || policy.viewMaximum[axis] > policy.worldMaximum[axis]) {
      throw new RangeError('Liquid particle view must increase and stay inside the world');
    }
  }
  const supportRadius = policy.particleSpacing * policy.supportScale;
  if (!finiteF32(supportRadius) || Math.fround(supportRadius) <= 0) {
    throw new RangeError('Liquid derived support radius must be a finite positive f32 value');
  }
  const gridDimensions = policy.worldMaximum.map((maximum, axis) =>
    Math.ceil((maximum - policy.worldMinimum[axis]) / supportRadius));
  for (const [axis, name] of ['columns', 'rows'].entries()) {
    if (!Number.isSafeInteger(gridDimensions[axis]) || gridDimensions[axis] < 1) {
      throw new RangeError(`Liquid spatial grid ${name} exceed safe indexing`);
    }
    if (gridDimensions[axis] > MAXIMUM_LIQUID_GRID_AXIS_COUNT) {
      throw new RangeError(`Liquid spatial grid ${name} must be at most `
        + MAXIMUM_LIQUID_GRID_AXIS_COUNT);
    }
  }
  const gridCellCount = gridDimensions[0] * gridDimensions[1];
  if (!Number.isSafeInteger(gridCellCount)
      || gridCellCount > MAXIMUM_LIQUID_GRID_CELL_COUNT) {
    throw new RangeError('Liquid spatial grid cell count must be at most '
      + MAXIMUM_LIQUID_GRID_CELL_COUNT);
  }
  const primaryParticleCount = policy.columns * policy.rows;
  if (!Number.isSafeInteger(primaryParticleCount)) {
    throw new RangeError('Liquid primary particle count exceeds safe indexing');
  }
  if (primaryParticleCount > MAXIMUM_LIQUID_PRIMARY_PARTICLE_COUNT) {
    throw new RangeError('Liquid primary particle count must be at most '
      + MAXIMUM_LIQUID_PRIMARY_PARTICLE_COUNT);
  }
  if (policy.diffuseCapacity < primaryParticleCount) {
    throw new RangeError('Liquid diffuse capacity must cover deterministic source slots');
  }
  if (policy.maximumCourantNumber > 1) {
    throw new RangeError('Liquid maximum Courant number must be above zero through one');
  }
  if (policy.jacobiRelaxation > 1.5) {
    throw new RangeError('Liquid Jacobi relaxation must be above zero through 1.5');
  }
  if (policy.maximumParticlesPerCell > 256) {
    throw new RangeError('Liquid maximumParticlesPerCell must be at most 256');
  }
  if (policy.friction > 2) {
    throw new RangeError('Liquid particle world friction must be at most two');
  }
  const minimumSeedX = policy.seedMaximumX
    - (policy.columns - 1) * policy.particleSpacing;
  if (minimumSeedX < policy.worldMinimum[0]
      || policy.seedMaximumX > policy.worldMaximum[0]) {
    throw new RangeError('Liquid seed x extent must stay inside the world');
  }
  if (policy.boundaryProfileSegments < 32 || policy.boundaryProfileSegments > 4096) {
    throw new RangeError('Liquid boundary profile segments must be from 32 through 4096');
  }
  if (policy.stoneCenterX - policy.stoneHalfWidth <= policy.worldMinimum[0]
      || policy.stoneCenterX + policy.stoneHalfWidth >= policy.worldMaximum[0]) {
    throw new RangeError('Liquid stone must stay strictly inside the world');
  }
  return Object.freeze(policy);
}

export function createLiquidParticleWorldGpuSeedReference(options = {}) {
  const policy = normalizePolicy(options);
  const channel = createSlopedStoneChannelGeometryReference({
    minimumX: policy.worldMinimum[0],
    maximumX: policy.worldMaximum[0],
    minimumY: policy.worldMinimum[1],
    bedAtStone: policy.bedAtStone,
    bedSlope: policy.bedSlope,
    stoneCenterX: policy.stoneCenterX,
    stoneHalfWidth: policy.stoneHalfWidth,
    stoneHeight: policy.stoneHeight,
    profileSegments: policy.boundaryProfileSegments,
    boundarySpacing: policy.particleSpacing * 0.5,
  });
  let approximationTolerance = 0;
  for (let segment = 0; segment < channel.surface.length / 2 - 1; segment += 1) {
    const ax = channel.surface[segment * 2];
    const ay = channel.surface[segment * 2 + 1];
    const bx = channel.surface[(segment + 1) * 2];
    const by = channel.surface[(segment + 1) * 2 + 1];
    const middleX = (ax + bx) * 0.5;
    approximationTolerance = Math.max(approximationTolerance,
      Math.abs(channel.surfaceHeightAt(middleX) - (ay + by) * 0.5));
  }
  const boundaryPacket = createTransportPolylineBoundaryPacketReference({
    points: channel.surface,
    approximationTolerance,
  });
  if (!boundaryPacket.uniformX) {
    throw new Error('GPU liquid boundary lowering requires uniform-x Transport vertices');
  }
  const initialWheelAngle = 0.22;
  const wheelCenter = [0, 0.32];
  const wheelRadius = 0.50;
  const barHalfWidth = 0.012;
  const physicalRadius = policy.particleSpacing * 0.46;
  const contactRadius = physicalRadius + barHalfWidth;
  const baffles = [
    [[0.500, 0.000], [0.350, 0.000]], [[0.000, 0.500], [0.000, 0.350]],
    [[-0.500, 0.000], [-0.350, 0.000]], [[0.000, -0.500], [0.000, -0.350]],
    [[-0.220, 0.175], [-0.075, 0.135]], [[0.075, -0.055], [0.215, -0.115]],
    [[-0.105, -0.250], [0.020, -0.155]],
  ];
  const rotate = ([x, y]) => [Math.cos(initialWheelAngle) * x - Math.sin(initialWheelAngle) * y,
    Math.sin(initialWheelAngle) * x + Math.cos(initialWheelAngle) * y];
  const distanceToSegment = ([x, y], a, b) => {
    const dx = b[0] - a[0]; const dy = b[1] - a[1];
    const along = Math.max(0, Math.min(1,
      ((x - a[0]) * dx + (y - a[1]) * dy) / Math.max(dx * dx + dy * dy, 1e-12)));
    return Math.hypot(x - (a[0] + dx * along), y - (a[1] + dy * along));
  };
  const candidates = [];
  const minimumX = policy.seedMaximumX - (policy.columns - 1) * policy.particleSpacing;
  for (let row = 0; row < policy.rows; row += 1) {
    for (let column = 0; column < policy.columns; column += 1) {
      const point = [minimumX + column * policy.particleSpacing,
        policy.seedMinimumY + row * policy.particleSpacing];
      if (Math.hypot(point[0] - wheelCenter[0], point[1] - wheelCenter[1])
          > wheelRadius - contactRadius) continue;
      const overlapsBaffle = baffles.some(([localA, localB]) => {
        const rotatedA = rotate(localA); const rotatedB = rotate(localB);
        const a = [wheelCenter[0] + rotatedA[0], wheelCenter[1] + rotatedA[1]];
        const b = [wheelCenter[0] + rotatedB[0], wheelCenter[1] + rotatedB[1]];
        return distanceToSegment(point, a, b) < contactRadius;
      });
      if (!overlapsBaffle) candidates.push(point);
    }
  }
  const count = candidates.length;
  if (count < 1) throw new RangeError('Liquid drum seed contains no valid particles');
  const stride = 12;
  const bytes = new ArrayBuffer(count * stride * 4);
  const floats = new Float32Array(bytes);
  const integers = new Uint32Array(bytes);
  const supportRadius = policy.particleSpacing * policy.supportScale;
  const particleMass = calibrateUniformLocalLiquidParticleMassReference({
    dimension: 2,
    spacing: policy.particleSpacing,
    supportRadius,
    restDensity: policy.restDensity,
  }).particleMass;
  const tangentNorm = Math.hypot(1, policy.bedSlope);
  const vx = policy.initialSpeed / tangentNorm;
  const vy = policy.initialSpeed * policy.bedSlope / tangentNorm;
  for (let particle = 0; particle < candidates.length; particle += 1) {
      const offset = particle * stride;
      const [x, y] = candidates[particle];
      floats[offset] = x; floats[offset + 1] = y;
      floats[offset + 2] = vx; floats[offset + 3] = vy;
      floats[offset + 4] = x; floats[offset + 5] = y;
      floats[offset + 6] = 0; floats[offset + 7] = policy.restDensity;
      floats[offset + 8] = 0; integers[offset + 9] = 0;
      integers[offset + 10] = PRIMARY_LIQUID;
      floats[offset + 11] = particleMass / policy.restDensity;
  }
  const solidBoundary = createAkinciLiquidSolidBoundaryReference({
    dimension: 2,
    positions: channel.boundarySamples,
    supportRadius,
  });
  if (!Number.isFinite(Math.fround(supportRadius)) || Math.fround(supportRadius) <= 0
      || !Number.isFinite(Math.fround(particleMass)) || Math.fround(particleMass) <= 0) {
    throw new RangeError('Liquid derived support radius and mass must be finite f32 values');
  }
  for (let particle = 0; particle < count; particle += 1) {
    const offset = particle * stride;
    if (![floats[offset], floats[offset + 1], floats[offset + 2], floats[offset + 3]]
      .every(Number.isFinite)) {
      throw new RangeError('Liquid seed must lower to finite f32 state');
    }
    if (floats[offset + 1] > policy.worldMaximum[1]) {
      throw new RangeError('Liquid seed y extent must stay inside the world');
    }
  }
  return Object.freeze({ kind: 'liquid-particle-world-gpu-seed-reference:v1',
    policy, count, stride, bytes, floats, supportRadius, particleMass,
    minimumX, maximumX: policy.seedMaximumX, channel, boundaryPacket, solidBoundary });
}

function createBuffer(device, label, size, usage, data = null) {
  const buffer = device.createBuffer({ label, size: Math.max(4, size), usage,
    mappedAtCreation: data !== null });
  if (data !== null) {
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(
      data.buffer ?? data, data.byteOffset ?? 0, data.byteLength ?? data.byteLength));
    buffer.unmap();
  }
  return buffer;
}

const hashSolidGpuPacket = (bytes, metadata) => {
  let hash = 0x811c9dc5;
  const signature = `liquid-solid-gpu-packet:v1|${metadata.join('|')}|`;
  for (let index = 0; index < signature.length; index += 1) {
    hash ^= signature.charCodeAt(index); hash = Math.imul(hash, 0x01000193);
  }
  for (const value of new Uint8Array(bytes)) {
    hash ^= value; hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
};

export function createLiquidParticleWorldGpuSolidPacketReference(seed, {
  gridColumns = Math.ceil((seed?.policy?.worldMaximum?.[0]
    - seed?.policy?.worldMinimum?.[0]) / seed?.supportRadius),
  gridRows = Math.ceil((seed?.policy?.worldMaximum?.[1]
    - seed?.policy?.worldMinimum?.[1]) / seed?.supportRadius),
  maximumBufferBytes = Number.MAX_SAFE_INTEGER,
} = {}) {
  if (seed?.kind !== 'liquid-particle-world-gpu-seed-reference:v1') {
    throw new TypeError('Liquid particle GPU seed required for solid packet lowering');
  }
  if (!Number.isSafeInteger(gridColumns) || gridColumns < 1
      || !Number.isSafeInteger(gridRows) || gridRows < 1
      || !Number.isSafeInteger(gridColumns * gridRows)) {
    throw new RangeError('Liquid solid packet grid dimensions must be safe positive integers');
  }
  if (!Number.isSafeInteger(maximumBufferBytes) || maximumBufferBytes < 16) {
    throw new RangeError('Liquid solid packet buffer limit must be a safe positive byte count');
  }
  const surfaceVertices = seed.boundaryPacket.copyVertices();
  const surfaceVertexCount = seed.boundaryPacket.segmentCount + 1;
  const surfaceEntryCount = Math.ceil(surfaceVertexCount / 2);
  const boundary = seed.solidBoundary;
  const gridCellCount = gridColumns * gridRows;
  const cellCounts = new Uint32Array(gridCellCount);
  const boundaryCellIndex = (boundaryIndex) => {
    const x = boundary.positions[boundaryIndex * 3];
    const y = boundary.positions[boundaryIndex * 3 + 2];
    if (x < seed.policy.worldMinimum[0] - seed.supportRadius
        || x > seed.policy.worldMaximum[0] + seed.supportRadius
        || y < seed.policy.worldMinimum[1] - seed.supportRadius
        || y > seed.policy.worldMaximum[1] + seed.supportRadius) {
      throw new RangeError('Canonical Akinci boundary sample lies outside the GPU grid halo');
    }
    // A layered sample at an open profile endpoint may sit a fraction of a
    // radius into the halo. Store it in the edge cell; the exact distance test
    // still decides whether it contributes.
    const cellX = Math.max(0, Math.min(gridColumns - 1,
      Math.floor((x - seed.policy.worldMinimum[0]) / seed.supportRadius)));
    const cellY = Math.max(0, Math.min(gridRows - 1,
      Math.floor((y - seed.policy.worldMinimum[1]) / seed.supportRadius)));
    return cellY * gridColumns + cellX;
  };
  const forEachSupportCell = (boundaryIndex, callback) => {
    const sourceCell = boundaryCellIndex(boundaryIndex);
    const sourceX = sourceCell % gridColumns;
    const sourceY = Math.floor(sourceCell / gridColumns);
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const cellY = sourceY + offsetY;
      if (cellY < 0 || cellY >= gridRows) continue;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const cellX = sourceX + offsetX;
        if (cellX < 0 || cellX >= gridColumns) continue;
        callback(cellY * gridColumns + cellX);
      }
    }
  };
  for (let boundaryIndex = 0; boundaryIndex < boundary.count; boundaryIndex += 1) {
    forEachSupportCell(boundaryIndex, (cell) => {
      if (cellCounts[cell] === 0xffffffff) {
        throw new RangeError('Canonical Akinci boundary cell count exceeds u32');
      }
      cellCounts[cell] += 1;
    });
  }
  let maximumCellOccupancy = 1;
  for (const count of cellCounts) maximumCellOccupancy = Math.max(maximumCellOccupancy, count);
  const boundaryEntryOffset = surfaceEntryCount;
  const gridWordOffset = (boundaryEntryOffset + boundary.count) * 4;
  const gridStrideWords = maximumCellOccupancy + 1;
  const requiredWords = gridWordOffset + gridCellCount * gridStrideWords;
  const byteLength = Math.ceil(requiredWords / 4) * 16;
  if (!Number.isSafeInteger(requiredWords) || !Number.isSafeInteger(byteLength)
      || byteLength > maximumBufferBytes) {
    throw new RangeError('Canonical Akinci solid packet exceeds this GPU device limit');
  }
  const bytes = new ArrayBuffer(byteLength);
  const floats = new Float32Array(bytes);
  const integers = new Uint32Array(bytes);
  floats.set(surfaceVertices);
  for (let boundaryIndex = 0; boundaryIndex < boundary.count; boundaryIndex += 1) {
    const offset = (boundaryEntryOffset + boundaryIndex) * 4;
    const values = [boundary.positions[boundaryIndex * 3],
      boundary.positions[boundaryIndex * 3 + 2], boundary.volumes[boundaryIndex]];
    if (!values.every((value) => Number.isFinite(Math.fround(value)))
        || Math.fround(values[2]) <= 0) {
      throw new RangeError('Canonical Akinci boundary must lower to finite positive f32 data');
    }
    floats[offset] = values[0]; floats[offset + 1] = values[1];
    floats[offset + 2] = values[2]; integers[offset + 3] = 0;
  }
  integers.fill(0xffffffff, gridWordOffset, requiredWords);
  for (let cell = 0; cell < gridCellCount; cell += 1) {
    const offset = gridWordOffset + cell * gridStrideWords;
    integers[offset] = cellCounts[cell];
  }
  const cellCursors = new Uint32Array(gridCellCount);
  for (let boundaryIndex = 0; boundaryIndex < boundary.count; boundaryIndex += 1) {
    forEachSupportCell(boundaryIndex, (cell) => {
      const slot = cellCursors[cell]; cellCursors[cell] += 1;
      integers[gridWordOffset + cell * gridStrideWords + slot + 1] = boundaryIndex;
    });
  }
  const stateHash = hashSolidGpuPacket(bytes, [surfaceVertexCount, boundaryEntryOffset,
    boundary.count, gridWordOffset, gridStrideWords, gridColumns, gridRows]);
  return Object.freeze({ kind: 'liquid-solid-gpu-packet-reference:v1', bytes,
    stateHash, surfaceVertexCount, boundaryEntryOffset,
    boundaryParticleCount: boundary.count, gridWordOffset, gridStrideWords,
    gridColumns, gridRows, maximumCellOccupancy });
}

export async function createLiquidParticleWorldGpuRuntime(deviceArgument, options = {}) {
  const device = requireDevice(deviceArgument);
  const seed = options.initialState ?? createLiquidParticleWorldGpuSeedReference(options);
  // Reset owns its immutable byte snapshot; public diagnostic seed views can
  // never mutate the reset state after construction.
  const resetBytes = seed.bytes.slice(0);
  const { policy } = seed;
  const supportRadius = seed.supportRadius;
  const gridColumns = Math.ceil(
    (policy.worldMaximum[0] - policy.worldMinimum[0]) / supportRadius);
  const gridRows = Math.ceil(
    (policy.worldMaximum[1] - policy.worldMinimum[1]) / supportRadius);
  const telemetryAtomicCount = LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.atomicCount;
  const spatialGrid = createDynamicParticleGridGpuPlan({
    particleCapacity: seed.count,
    maximumParticlesPerCell: policy.maximumParticlesPerCell,
    expectedParticlesPerCell: 4,
    telemetryAtomicCount,
  });
  const gridCellCount = spatialGrid.bucketCapacity;
  const cellCountsAtomicCount = spatialGrid.countAtomicCount;
  const cellItemCount = spatialGrid.itemCapacity;
  if (!Number.isSafeInteger(gridCellCount) || !Number.isSafeInteger(cellCountsAtomicCount)
      || !Number.isSafeInteger(cellItemCount)) {
    throw new RangeError('Liquid particle spatial grid exceeds safe buffer indexing');
  }
  const maximumDispatch = device.limits?.maxComputeWorkgroupsPerDimension
    ?? Number.MAX_SAFE_INTEGER;
  for (const [name, count] of [['primary', seed.count], ['grid', gridCellCount],
    ['grid storage', cellItemCount], ['diffuse', policy.diffuseCapacity]]) {
    if (ceilDiv(count, 128) > maximumDispatch) {
      throw new RangeError(`Liquid ${name} dispatch exceeds this GPU device limit`);
    }
  }
  const maximumBufferBytes = Math.min(device.limits?.maxBufferSize ?? Number.MAX_SAFE_INTEGER,
    device.limits?.maxStorageBufferBindingSize ?? Number.MAX_SAFE_INTEGER);
  // Validate the simple grid allocations before constructing any CPU-side
  // boundary packet proportional to the grid size.
  for (const [name, size] of [
    ['cell counts and telemetry', cellCountsAtomicCount * 4],
    ['cell items', cellItemCount * 4],
  ]) {
    if (!Number.isSafeInteger(size) || size > maximumBufferBytes) {
      throw new RangeError(`Liquid ${name} buffer exceeds this GPU device limit`);
    }
  }
  const solidGpuPacket = options.solidPacket ?? createLiquidParticleWorldGpuSolidPacketReference(seed, {
    gridColumns, gridRows, maximumBufferBytes,
  });
  for (const [name, size] of [
    ['primary particles', seed.bytes.byteLength],
    ['cell counts and telemetry', cellCountsAtomicCount * 4],
    ['cell items', cellItemCount * 4],
    ['diffuse particles', policy.diffuseCapacity * LIQUID_PARTICLE_WORLD_GPU_ABI.diffuseStrideBytes],
    ['diffuse activity', policy.diffuseCapacity * 4],
    ['diffuse render', policy.diffuseCapacity
      * LIQUID_PARTICLE_WORLD_GPU_ABI.diffuseRenderStrideBytes],
    ['Transport/Akinci solid packet', solidGpuPacket.bytes.byteLength],
  ]) {
    if (!Number.isSafeInteger(size) || size > maximumBufferBytes) {
      throw new RangeError(`Liquid ${name} buffer exceeds this GPU device limit`);
    }
  }
  const storage = GPUBufferUsage.STORAGE;
  const copyDst = GPUBufferUsage.COPY_DST;
  const copySrc = GPUBufferUsage.COPY_SRC;
  const particleBuffer = createBuffer(device, 'VKF Liquid primary particles',
    seed.bytes.byteLength, storage | copyDst | copySrc, new Uint8Array(seed.bytes));
  const cellCountsBuffer = createBuffer(device, 'VKF Liquid cell counts + telemetry',
    cellCountsAtomicCount * 4, storage | copyDst | copySrc);
  const cellItemsBuffer = createBuffer(device, 'VKF Liquid cell items',
    spatialGrid.bytes.items, storage | copyDst);
  device.queue.writeBuffer(cellItemsBuffer,0,new Uint32Array(cellItemCount).fill(0xffffffff));
  const diffuseStrideBytes = 32;
  const diffuseBuffer = createBuffer(device, 'VKF Liquid diffuse particles',
    policy.diffuseCapacity * diffuseStrideBytes, storage | copyDst);
  const diffuseActiveBuffer = createBuffer(device, 'VKF Liquid diffuse activity',
    policy.diffuseCapacity * 4, storage | copyDst);
  const diffuseRenderBuffer = createBuffer(device, 'VKF Liquid diffuse render',
    policy.diffuseCapacity * 16, storage | copyDst | copySrc);
  const surfaceBuffer = createBuffer(device, 'VKF Transport/Akinci solid packet',
    solidGpuPacket.bytes.byteLength, storage, new Uint8Array(solidGpuPacket.bytes));
  const paramsBuffer = createBuffer(device, 'VKF Liquid world parameters',
    LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes, GPUBufferUsage.UNIFORM | copyDst);
  const diffuseParamsBuffer = createBuffer(device, 'VKF Liquid diffuse batch parameters',
    LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes, GPUBufferUsage.UNIFORM | copyDst);
  const shader = device.createShaderModule({
    label: 'VKF 2D Liquid particle world GPU specialization',
    code: (options.shaderSource ?? (LIQUID_PARTICLE_WORLD_GPU_WGSL + SWEPT_WHEEL_CONTACT_WGSL + PREVENTIVE_PARTICLE_CONTACT_WGSL + LIQUID_PRESSURE_GEOMETRY_CACHE_WGSL))
      + (policy.friction > 0 ? pressureFrictionWgsl(policy.friction) : ''),
  });
  if (typeof shader.getCompilationInfo === 'function') {
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === 'error');
    if (errors.length) {
      throw new Error(errors.map((message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
    }
  }
  const computeVisibility = GPUShaderStage.COMPUTE;
  const layout = device.createBindGroupLayout({
    label: 'VKF Liquid particle world compute bindings',
    entries: [
      { binding: 0, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 1, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 2, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 3, visibility: computeVisibility, buffer: { type: 'uniform' } },
      { binding: 4, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 5, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 6, visibility: computeVisibility, buffer: { type: 'storage' } },
      { binding: 7, visibility: computeVisibility, buffer: { type: 'read-only-storage' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const entryPoints = ['predict', 'clear_cells', 'fill_cells', 'divergence_lambda',
    'density_lambda', 'apply_pressure', 'classify_and_filter',
    'apply_filter', 'update_compressibility', 'update_compressibility_and_advect',
    'update_diffuse_and_render', 'sweep_wheel'];
  if (policy.friction > 0) entryPoints.push('pressure_friction_candidate', 'pressure_friction_apply');
  if (typeof device.pushErrorScope === 'function') device.pushErrorScope('validation');
  let pipelineEntries;
  try {
    pipelineEntries = await Promise.all(entryPoints.map(async (entryPoint) => [entryPoint,
      await createCheckedGpuPipeline(device,'compute',{ label: `VKF Liquid ${entryPoint}`,
        layout: pipelineLayout, compute: { module: shader, entryPoint } })]));
  } finally {
    if (typeof device.popErrorScope === 'function') {
      const validationError = await device.popErrorScope();
      if (validationError) throw new Error(validationError.message);
    }
  }
  const pipelines = Object.fromEntries(pipelineEntries);
  const preventivePipelines={};
  let pressureGeometryBuffer=null,pressureGeometryGroup=null,pressureGeometryBindGroup=null;
  const pressureGeometryPipelines={};
  if(options.preventiveContact){
    const r=options.preventiveContact;
    const guardedLayout=device.createPipelineLayout({bindGroupLayouts:[layout,r.emptyLayout,r.forceLayout]});
    const guardedNames=['predict','divergence_lambda','density_lambda','apply_pressure','classify_and_filter','apply_filter','update_compressibility','audit_predicted'];
    if(policy.friction>0)guardedNames.push('pressure_friction_candidate','pressure_friction_apply');
    await Promise.all(guardedNames.map(async name=>{
      preventivePipelines[name]=await createCheckedGpuPipeline(device,'compute',{label:'VKF liquid guarded forces',layout:guardedLayout,compute:{module:shader,entryPoint:`preventive_${name}`}});
    }));
    if(options.forceGeometryCaching){
      const bytes=seed.count*(3+9*policy.maximumParticlesPerCell)*16;
      if(!Number.isSafeInteger(bytes)||bytes>Math.min(device.limits?.maxBufferSize??Infinity,device.limits?.maxStorageBufferBindingSize??Infinity))throw new Error('Pressure geometry cache exceeds this GPU binding; compact batched caching is required');
      pressureGeometryBuffer=createBuffer(device,'VKF one-law-step pressure geometry',bytes,GPUBufferUsage.STORAGE);
      const cacheLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:computeVisibility,buffer:{type:'uniform'}},{binding:4,visibility:computeVisibility,buffer:{type:'storage'}},{binding:11,visibility:computeVisibility,buffer:{type:'storage'}}]});
      const reducedLayout=device.createBindGroupLayout({entries:[0,1,2,3,7].map(binding=>({binding,visibility:computeVisibility,buffer:{type:binding===3?'uniform':binding===7?'read-only-storage':'storage'}}))});
      pressureGeometryGroup=device.createBindGroup({layout:cacheLayout,entries:[[0,r.uniform],[4,r.control],[11,pressureGeometryBuffer]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
      pressureGeometryBindGroup=device.createBindGroup({layout:reducedLayout,entries:[[0,particleBuffer],[1,cellCountsBuffer],[2,cellItemsBuffer],[3,paramsBuffer],[7,surfaceBuffer]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
      const cachePipelineLayout=device.createPipelineLayout({bindGroupLayouts:[reducedLayout,r.emptyLayout,cacheLayout]});
      for(const name of ['cache_pressure_geometry','cached_divergence_lambda','cached_density_lambda','cached_apply_pressure',
        'preventive_cache_pressure_geometry','preventive_cached_divergence_lambda','preventive_cached_density_lambda','preventive_cached_apply_pressure']){
        const pipeline=await createOptionalGpuPipeline(device,'compute',{label:'VKF optional pressure geometry cache',layout:cachePipelineLayout,compute:{module:shader,entryPoint:name}});
        if(!pipeline){pressureGeometryBuffer.destroy();pressureGeometryBuffer=null;pressureGeometryGroup=null;pressureGeometryBindGroup=null;break;}
        pressureGeometryPipelines[name]=pipeline;
      }
    }
  }
  const bindGroup = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: particleBuffer } },
    { binding: 1, resource: { buffer: cellCountsBuffer } },
    { binding: 2, resource: { buffer: cellItemsBuffer } },
    { binding: 3, resource: { buffer: paramsBuffer } },
    { binding: 4, resource: { buffer: diffuseBuffer } },
    { binding: 5, resource: { buffer: diffuseActiveBuffer } },
    { binding: 6, resource: { buffer: diffuseRenderBuffer } },
    { binding: 7, resource: { buffer: surfaceBuffer } },
  ] });
  const diffuseBindGroup = device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: particleBuffer } },
    { binding: 1, resource: { buffer: cellCountsBuffer } },
    { binding: 2, resource: { buffer: cellItemsBuffer } },
    { binding: 3, resource: { buffer: diffuseParamsBuffer } },
    { binding: 4, resource: { buffer: diffuseBuffer } },
    { binding: 5, resource: { buffer: diffuseActiveBuffer } },
    { binding: 6, resource: { buffer: diffuseRenderBuffer } },
    { binding: 7, resource: { buffer: surfaceBuffer } },
  ] });
  let frameIndex = 0;
  let wheelAngle = 0;
  let wheelAngularVelocity = 0;
  let encodedWheelAngle = null;
  const wheelCenter = Object.freeze(options.geometry?.center ?? [0, 0.32]);
  const paramsBytes = new ArrayBuffer(LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes);
  // Queue writes and submissions are ordered, so one immutable-for-each-
  // submission staging buffer can be reused without allocating every frame.
  const wheelSweepBuffer = createBuffer(device, 'VKF reusable wheel sweep parameters',
    LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes * 2,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const maximumMovingSteps = 16;
  const wheelStepBuffer = createBuffer(device, 'VKF moving wheel substep parameters',
    LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes * maximumMovingSteps,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const diffuseParamsBytes = new ArrayBuffer(LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes);
  const paramsU32 = new Uint32Array(paramsBytes);
  const paramsF32 = new Float32Array(paramsBytes);
  const updateParams = (timeStep) => {
    paramsU32.set([seed.count, spatialGrid.bucketCapacity, 1,
      policy.maximumParticlesPerCell], 0);
    paramsU32.set([policy.diffuseCapacity, seed.boundaryPacket.segmentCount,
      solidGpuPacket.boundaryParticleCount, solidGpuPacket.maximumCellOccupancy], 4);
    paramsF32.set([...policy.worldMinimum, ...policy.worldMaximum], 8);
    paramsF32.set([timeStep, policy.particleSpacing, supportRadius,
      seed.particleMass], 12);
    paramsF32.set([policy.restDensity, policy.divergenceTolerance,
      policy.densityTolerance, policy.viscosity], 16);
    paramsF32.set([...policy.gravity, policy.jacobiRelaxation, policy.bulkModulus], 20);
    paramsF32.set([wheelAngle, wheelAngularVelocity,
      wheelCenter[0], wheelCenter[1]], 24);
    paramsF32.set([seed.boundaryPacket.minimumX,
      seed.boundaryPacket.uniformStepX, policy.diffuseEmissionRate,
      seed.boundaryPacket.maximumUniformXError], 28);
    device.queue.writeBuffer(paramsBuffer, 0, paramsBytes);
  };
  const updateDiffuseParams = (timeStep) => {
    new Uint8Array(diffuseParamsBytes).set(new Uint8Array(paramsBytes));
    new Float32Array(diffuseParamsBytes)[12] = timeStep;
    device.queue.writeBuffer(diffuseParamsBuffer, 0, diffuseParamsBytes);
  };
  const dispatch = (pass, name, count) => {
    pass.setPipeline(pipelines[name]); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(ceilDiv(count, 128));
  };
  const buildGrid = (pass) => {
    dispatch(pass, 'clear_cells', gridCellCount);
    dispatch(pass, 'fill_cells', seed.count);
  };
  let gridReady = false;
  const encodeFixedStep = (pass) => {
    // DFSPH ordering: project the existing velocity field, apply non-pressure
    // forces, project the predicted density, then advance positions. A density
    // correction after advection would only affect the following step.
    const cached=name=>{pass.setPipeline(pressureGeometryPipelines[name]);pass.setBindGroup(0,pressureGeometryBindGroup);pass.setBindGroup(2,pressureGeometryGroup);pass.dispatchWorkgroups(ceilDiv(seed.count,128));};
    if(pressureGeometryBuffer)cached('cache_pressure_geometry');
    const pressure=name=>pressureGeometryBuffer?cached(`cached_${name}`):dispatch(pass,name,seed.count);
    for (let iteration = 0; iteration < policy.divergenceIterations; iteration += 1) {
      pressure('divergence_lambda');
      pressure('apply_pressure');
    }
    dispatch(pass, 'predict', seed.count);
    dispatch(pass, 'classify_and_filter', seed.count);
    dispatch(pass, 'apply_filter', seed.count);
    for (let iteration = 0; iteration < policy.densityIterations; iteration += 1) {
      pressure('density_lambda');
      pressure('apply_pressure');
    }
    dispatch(pass, 'update_compressibility_and_advect', seed.count);
    buildGrid(pass);
    if (policy.friction > 0) {
      dispatch(pass, 'pressure_friction_candidate', seed.count);
      dispatch(pass, 'pressure_friction_apply', seed.count);
    }
  };
  const stepMany = (encoder, count, timeStep = policy.timeStep) => {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new RangeError('Liquid GPU step count must be a nonnegative safe integer');
    }
    if (timeStep !== policy.timeStep) {
      throw new RangeError('Liquid GPU steps use the fixed policy time step');
    }
    if (count === 0) return;
    updateParams(timeStep);
    updateDiffuseParams(timeStep * count);
    const pass = encoder.beginComputePass({ label: `VKF Liquid particle world ${count}-step batch` });
    // Every substep leaves a grid for its advected positions. Preserve that
    // invariant across render batches instead of rebuilding identical cells.
    if (!gridReady) buildGrid(pass);
    for (let index = 0; index < count; index += 1) encodeFixedStep(pass);
    pass.setPipeline(pipelines.update_diffuse_and_render);
    pass.setBindGroup(0,diffuseBindGroup);
    pass.dispatchWorkgroups(ceilDiv(policy.diffuseCapacity,128));
    pass.end(); gridReady = true; frameIndex += count;
  };
  const step = (encoder, timeStep = policy.timeStep) => stepMany(encoder, 1, timeStep);
  const stepMovingWheel = (encoder, count) => {
    if (!Number.isInteger(count) || count < 1 || count > maximumMovingSteps) {
      throw new RangeError('Moving wheel batch must contain one through 16 fixed steps');
    }
    if (encodedWheelAngle === null) encodedWheelAngle = wheelAngle;
    const fromAngle = encodedWheelAngle;
    const delta = wheelAngle - fromAngle;
    const perStep = delta / count;
    const angularVelocity = perStep / policy.timeStep;
    updateParams(policy.timeStep);
    updateDiffuseParams(policy.timeStep * count);
    const stride = LIQUID_PARTICLE_WORLD_GPU_ABI.parameterBytes;
    const substepBytes = new ArrayBuffer(stride * count);
    const bytes = new Uint8Array(substepBytes);
    for (let index = 0; index < count; index += 1) {
      const offset = index * stride;
      bytes.set(new Uint8Array(paramsBytes), offset);
      const fields = new Float32Array(substepBytes, offset, stride / 4);
      const oldAngle = fromAngle + perStep * index;
      fields[24] = oldAngle + perStep;
      fields[25] = angularVelocity;
      fields.set([oldAngle, perStep, angularVelocity, policy.timeStep], 32);
    }
    device.queue.writeBuffer(wheelStepBuffer, 0, substepBytes);
    for (let index = 0; index < count; index += 1) {
      encoder.copyBufferToBuffer(wheelStepBuffer, index * stride, paramsBuffer, 0, stride);
      const pass = encoder.beginComputePass({ label: 'VKF continuously moving wheel and liquid substep' });
      dispatch(pass, 'sweep_wheel', seed.count);
      buildGrid(pass);
      encodeFixedStep(pass);
      pass.end();
    }
    const diffuse = encoder.beginComputePass({ label: 'VKF moving liquid embedding state' });
    diffuse.setPipeline(pipelines.update_diffuse_and_render);
    diffuse.setBindGroup(0, diffuseBindGroup);
    diffuse.dispatchWorkgroups(ceilDiv(policy.diffuseCapacity, 128));
    diffuse.end();
    gridReady = true;
    encodedWheelAngle = wheelAngle;
    frameIndex += count;
  };
  const predictForces=(encoder)=>{
    const r=options.preventiveContact;if(!r)throw new Error('Preventive contact resources are required');
    // A publication uses its accepted elapsed interval. Restore this law's
    // fixed force step through an ordered GPU copy, not an aliased host write.
    encoder.copyBufferToBuffer(r.uniform,80,paramsBuffer,48,4);
    encoder.copyBufferToBuffer(r.control,4,paramsBuffer,96,4);
    encoder.copyBufferToBuffer(r.control,12,paramsBuffer,100,4);
    const pass=encoder.beginComputePass({label:'VKF liquid force prediction, positions unchanged'});
    if(r.forceIndirect){for(const name of ['clear_cells','fill_cells']){pass.setPipeline(pipelines[name]);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,name==='fill_cells'?36:48);}}
    else buildGrid(pass);
    gridReady = true;
    pass.setBindGroup(0,bindGroup);pass.setBindGroup(1,r.emptyGroup);pass.setBindGroup(2,r.forceGroup);
    const run=name=>{pass.setPipeline(preventivePipelines[name]);if(r.forceIndirect)pass.dispatchWorkgroupsIndirect(r.dispatchArgs,36);else pass.dispatchWorkgroups(ceilDiv(seed.count,128));};
    const cached=name=>{pass.setPipeline(pressureGeometryPipelines[name]);pass.setBindGroup(0,pressureGeometryBindGroup);pass.setBindGroup(2,pressureGeometryGroup);if(r.forceIndirect)pass.dispatchWorkgroupsIndirect(r.dispatchArgs,36);else pass.dispatchWorkgroups(ceilDiv(seed.count,128));};
    if(pressureGeometryBuffer)cached('preventive_cache_pressure_geometry');
    const pressure=name=>{if(pressureGeometryBuffer)cached(`preventive_cached_${name}`);else run(name);};
    for(let i=0;i<policy.divergenceIterations;i++){pressure('divergence_lambda');pressure('apply_pressure');}
    pass.setBindGroup(0,bindGroup);pass.setBindGroup(2,r.forceGroup);
    run('predict');run('classify_and_filter');run('apply_filter');
    for(let i=0;i<policy.densityIterations;i++){pressure('density_lambda');pressure('apply_pressure');}
    pass.setBindGroup(0,bindGroup);pass.setBindGroup(2,r.forceGroup);
    run('update_compressibility');
    if(policy.friction>0){run('pressure_friction_candidate');run('pressure_friction_apply');}
    run('audit_predicted');
    pass.end();
  };
  const publishParticles=(encoder)=>{
    const r=options.preventiveContact;
    if(r?.hasRepresentationClock){encoder.copyBufferToBuffer(r.control,224,paramsBuffer,48,4);encoder.copyBufferToBuffer(r.control,4,paramsBuffer,96,4);encoder.copyBufferToBuffer(r.control,12,paramsBuffer,100,4);}
    const pass=encoder.beginComputePass({label:'VKF accepted liquid representation'});
    buildGrid(pass);dispatch(pass,'update_diffuse_and_render',policy.diffuseCapacity);pass.end();gridReady=true;
  };
  const sweepWheel = (encoder, elapsed) => {
    if(encodedWheelAngle===null){encodedWheelAngle=wheelAngle;return;}
    const delta=wheelAngle-encodedWheelAngle;if(Math.abs(delta)<1e-10)return;
    updateParams(policy.timeStep);
    const base=paramsBytes.slice(0),motion=paramsBytes.slice(0);
    new Float32Array(motion).set([encodedWheelAngle,delta,delta/elapsed,elapsed],32);
    device.queue.writeBuffer(wheelSweepBuffer,0,motion);device.queue.writeBuffer(wheelSweepBuffer,base.byteLength,base);
    encoder.copyBufferToBuffer(wheelSweepBuffer,0,paramsBuffer,0,base.byteLength);
    const pass=encoder.beginComputePass({label:'Swept rigid baffles'});dispatch(pass,'sweep_wheel',seed.count);pass.end();
    gridReady=false;
    encoder.copyBufferToBuffer(wheelSweepBuffer,base.byteLength,paramsBuffer,0,base.byteLength);
    encodedWheelAngle=wheelAngle;
  };
  const telemetryByteOffset = spatialGrid.telemetryByteOffset;
  const telemetryByteLength = telemetryAtomicCount * 4;
  const telemetryZero = new Uint32Array(telemetryAtomicCount);
  const resetTelemetry = () => {
    device.queue.writeBuffer(cellCountsBuffer, telemetryByteOffset, telemetryZero);
  };
  const readTelemetry = async () => {
    const readback = device.createBuffer({
      label: 'VKF Liquid telemetry readback',
      size: telemetryByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder({ label: 'VKF Liquid telemetry copy' });
      encoder.copyBufferToBuffer(cellCountsBuffer, telemetryByteOffset,
        readback, 0, telemetryByteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange().slice(0));
      const speedBits = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakSpeedSquaredBits];
      const peakSpeedSquared = new Float32Array(new Uint32Array([speedBits]).buffer)[0];
      const postContactSpeedBits = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakPredictedOrPostContactSpeedSquaredBits];
      const peakPredictedOrPostContactSpeedSquared = new Float32Array(
        new Uint32Array([postContactSpeedBits]).buffer)[0];
      const decodePositiveFloat = (index) => new Float32Array(
        new Uint32Array([words[index]]).buffer)[0];
      const peakFastInverseDenominator = decodePositiveFloat(
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakFastInverseDenominatorBits);
      const peakFastDensityError = decodePositiveFloat(
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakFastDensityErrorBits);
      const nonFiniteFlag = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.nonFiniteFlag];
      const peakCellOccupancy = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakCellOccupancy];
      const gridOverflow = peakCellOccupancy > policy.maximumParticlesPerCell;
      const occupiedBucketEvents = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.occupiedBucketEvents];
      const activeCellEvents = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.activeCellEvents];
      const gridHashCollisionEvents = activeCellEvents - occupiedBucketEvents;
      const contactBudgetExhaustedReceipt = words[
        LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.contactBudgetExhaustedReceipt];
      const contactBudgetExhausted = contactBudgetExhaustedReceipt !== 0;
      const contactBudgetExhaustedPhase = !contactBudgetExhausted ? null
        : ((contactBudgetExhaustedReceipt & 0x80000000) === 0 ? 'primary' : 'diffuse');
      const contactBudgetExhaustedReason = !contactBudgetExhausted ? null
        : ((contactBudgetExhaustedReceipt >>> 28) & 0x7);
      const contactBudgetExhaustedIndex = !contactBudgetExhausted ? null
        : ((contactBudgetExhaustedReceipt & 0x0fffffff) - 1);
      return Object.freeze({ kind: 'liquid-particle-world-gpu-telemetry:v1',
        frameIndex, peakSpeedSquared, peakSpeed: Math.sqrt(peakSpeedSquared),
        peakPredictedOrPostContactSpeedSquared,
        peakPredictedOrPostContactSpeed: Math.sqrt(peakPredictedOrPostContactSpeedSquared),
        peakSweepSpeed: Math.sqrt(decodePositiveFloat(
          LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakSweepSpeedSquaredBits)),
        peakPressureSpeed: Math.sqrt(decodePositiveFloat(
          LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakPressureSpeedSquaredBits)),
        peakSweepAddedSpeed: decodePositiveFloat(
          LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakSweepAddedSpeedBits),
        peakPreSweepSpeed: Math.sqrt(decodePositiveFloat(
          LIQUID_PARTICLE_WORLD_GPU_TELEMETRY.fields.peakPreSweepSpeedSquaredBits)),
        minimumFastDenominator: peakFastInverseDenominator > 0
          ? 1 / peakFastInverseDenominator : null,
        peakFastDensityError,
        nonFinite: nonFiniteFlag !== 0, nonFiniteFlag, peakCellOccupancy,
        gridOverflow, occupiedBucketEvents, activeCellEvents,
        gridHashCollisionEvents, gridAuditComplete: !gridOverflow,
        contactBudgetExhausted, contactBudgetExhaustedReceipt,
        contactBudgetExhaustedPhase, contactBudgetExhaustedReason,
        contactBudgetExhaustedIndex,
        raw: Object.freeze(Array.from(words)) });
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      readback.destroy();
    }
  };
  const reset = () => {
    encodedWheelAngle=null;gridReady=false;
    device.queue.writeBuffer(particleBuffer, 0, resetBytes);
    device.queue.writeBuffer(diffuseActiveBuffer, 0,
      new Uint8Array(policy.diffuseCapacity * 4));
    device.queue.writeBuffer(diffuseRenderBuffer, 0,
      new Uint8Array(policy.diffuseCapacity * 16));
    resetTelemetry();
    frameIndex = 0; updateParams(policy.timeStep);
  };
  const setWheel = ({ angle = wheelAngle,
    angularVelocity = wheelAngularVelocity } = {}) => {
    if (!Number.isFinite(angle) || !Number.isFinite(angularVelocity)) {
      throw new TypeError('Liquid wheel state requires finite angle and angularVelocity');
    }
    // Preserve the pose preceding the first swept update. Otherwise the
    // first moving frame silently skips its entire baffle sweep.
    if (encodedWheelAngle === null) encodedWheelAngle = wheelAngle;
    wheelAngle = angle;
    wheelAngularVelocity = angularVelocity;
    updateParams(policy.timeStep);
  };
  const destroy = () => {
    for (const buffer of [particleBuffer, cellCountsBuffer, cellItemsBuffer,
      diffuseBuffer, diffuseActiveBuffer, diffuseRenderBuffer, surfaceBuffer,
      paramsBuffer, diffuseParamsBuffer, wheelSweepBuffer, wheelStepBuffer]) {
      buffer.destroy();
    }
    pressureGeometryBuffer?.destroy();
  };
  reset();
  return Object.freeze({ kind: 'liquid-particle-world-gpu-runtime:v1', device,
    policy, seed, abi: LIQUID_PARTICLE_WORLD_GPU_ABI,
    particleBuffer, diffuseRenderBuffer, surfaceBuffer, paramsBuffer,
    boundaryPacket: seed.boundaryPacket, solidBoundary: seed.solidBoundary,
    solidGpuPacket, contactBoundaryStateHash: seed.boundaryPacket.stateHash,
    solidPhysicsStateHash: solidGpuPacket.stateHash,
    primaryCount: seed.count, diffuseCapacity: policy.diffuseCapacity,
    gridCellCount, spatialGrid, pipelines: Object.freeze(pipelines), step, stepMany, stepMovingWheel, sweepWheel, predictForces, publishParticles,
    telemetry: Object.freeze({ buffer: cellCountsBuffer,
      byteOffset: telemetryByteOffset, byteLength: telemetryByteLength,
      ...LIQUID_PARTICLE_WORLD_GPU_TELEMETRY }),
    readTelemetry, resetTelemetry, reset, setWheel,
    wheel: Object.freeze({ center: wheelCenter, radius: options.geometry?.radius ?? 0.50,
      barHalfWidth: options.geometry?.half_width ?? 0.012 }),
    destroy,
    get frameIndex() { return frameIndex; } });
}
export { normalizePolicy as normalizeLiquidContainedWorldPolicy };
