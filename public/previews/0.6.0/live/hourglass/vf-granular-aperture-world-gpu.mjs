// Granular World transport for a narrow aperture. Stream parcels carry finite
// area, world position, and world velocity; the Embedding only reads them.
import {GRANULAR_DISCHARGE_LAW_WGSL} from '../../compiled/runtime-wheel-27/vf-granular-discharge-law.mjs';
export const STREAM_CELLS = 64;
export const PILE_CELLS = 128;
export const STATE_FLOATS = 8 + STREAM_CELLS * 5 + PILE_CELLS;
export const INITIAL_AREA = 0.08;
export const CHAMBER_WIDTH = 0.5;
export const FALL_HEIGHT = 0.44;

const LAW_WGSL = `
const STREAM_CELLS: u32 = 64u;
const PILE_CELLS: u32 = 128u;
struct MaterialState {
  chamber_a: f32,
  settling: f32,
  chamber_b: f32,
  time: f32,
  direction: f32,
  release_remainder: f32,
  flow_rate: f32,
  pose_angle: f32,
  stream: array<f32, 64>,
  motion: array<vec4<f32>, 64>,
  pile: array<f32, 128>,
};
struct LawParameters {
  dt: f32,
  opening: f32,
  grain_diameter: f32,
  repose_radians: f32,
  gravity: f32,
  discharge_coefficient: f32,
  empty_annulus: f32,
  settling_seconds: f32,
  pose_angle: f32,
  release_response_seconds: f32,
};
@group(0) @binding(0) var<storage, read> previous: MaterialState;
@group(0) @binding(1) var<storage, read_write> next: MaterialState;
@group(0) @binding(2) var<uniform> law: LawParameters;
${GRANULAR_DISCHARGE_LAW_WGSL}
fn pile_peak_for_area(area: f32, slope: f32, opening: f32) -> f32 {
  // The wall is a sloping capacity constraint, not permission to hide sand
  // behind glass. Integrate the repose line clipped by the vessel wall.
  let radius = ${CHAMBER_WIDTH} * 0.5;
  let half_opening = opening * 0.5;
  let wall_slope = 0.41 / (radius - half_opening);
  var low = 0.0;
  var high = 0.41;
  for (var iteration = 0u; iteration < 19u; iteration = iteration + 1u) {
    let peak = (low + high) * 0.5;
    var capacity = peak * peak / slope;
    if (peak > slope * radius) {
      let crossing = clamp((0.41 + wall_slope * half_opening - peak)
        / (wall_slope - slope), half_opening, radius);
      capacity = 2.0 * (peak * crossing - 0.5 * slope * crossing * crossing
        + 0.5 * wall_slope * (radius - crossing) * (radius - crossing));
    }
    if (capacity < area) { low = peak; } else { high = peak; }
  }
  return (low + high) * 0.5;
}
fn pile_wall_capacity(x: f32, opening: f32) -> f32 {
  let radius = ${CHAMBER_WIDTH} * 0.5;
  return 0.41 * clamp((radius - abs(x)) / (radius - opening * 0.5),
    0.0, 1.0);
}
fn local_position(world: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec2<f32>(c * world.x + s * world.y,
    -s * world.x + c * world.y + 0.485);
}
fn world_position(local: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle);
  let s = sin(angle);
  let y = local.y - 0.485;
  return vec2<f32>(c * local.x - s * y, s * local.x + c * y);
}
fn chamber_wall(y: f32, opening: f32) -> f32 {
  if (y < 0.47) { return 0.004 + 0.246 * clamp((0.47 - y) / 0.41, 0.0, 1.0); }
  if (y > 0.50) { return 0.004 + 0.246 * clamp((y - 0.50) / 0.41, 0.0, 1.0); }
  return opening * 0.5;
}

@compute @workgroup_size(1)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) { return; }
  let axial_gravity = cos(law.pose_angle);
  let direction = select(-1.0, 1.0, axial_gravity >= 0.0);
  let reversed = direction != previous.direction;
  let discharge = granular_discharge_area_rate(law.opening, law.grain_diameter,
    law.gravity, law.discharge_coefficient, law.empty_annulus,
    abs(axial_gravity));
  var chamber_a = previous.chamber_a;
  var chamber_b = previous.chamber_b;
  // Settling is retained for ledger compatibility; airborne material is never
  // reversed or relabelled when the vessel turns.
  if (previous.direction > 0.0) { chamber_b = chamber_b + previous.settling; }
  else { chamber_a = chamber_a + previous.settling; }
  let source_area = select(chamber_b, chamber_a, direction > 0.0);
  let prior_rate = select(previous.flow_rate, 0.0, reversed);
  let target_rate = select(0.0, discharge, source_area > 0.0);
  let flow_rate = relax_granular_discharge_rate(prior_rate, target_rate,
    law.dt, law.release_response_seconds);
  next.flow_rate = flow_rate;
  // Accumulate sub-ULP discharge before subtracting from a coarse parcel.
  // Otherwise a grain-width throat can emit tiny stream masses while its
  // source remains numerically unchanged in f32.
  let desired = select(0.0,
    select(previous.release_remainder, 0.0, reversed) + flow_rate * law.dt,
    source_area > 0.0);
  let quantum = ${INITIAL_AREA} * 2.0e-7;
  var free_slot = STREAM_CELLS;
  for (var i = 0u; i < STREAM_CELLS; i = i + 1u) {
    if (previous.stream[i] <= 0.0 && free_slot == STREAM_CELLS) { free_slot = i; }
  }
  let released = select(0.0, min(source_area, desired),
    free_slot < STREAM_CELLS && (desired >= quantum || desired >= source_area));
  next.release_remainder = desired - released;
  if (direction > 0.0) { chamber_a = chamber_a - released; }
  else { chamber_b = chamber_b - released; }
  next.time = previous.time + law.dt;
  next.direction = direction;
  next.pose_angle = law.pose_angle;
  next.settling = 0.0;
  var deposited_a = 0.0;
  var deposited_b = 0.0;
  let tangent = max(tan(law.repose_radians), 0.001);
  let target_area = select(chamber_a, chamber_b, direction > 0.0);
  let target_peak = pile_peak_for_area(target_area, tangent, law.opening);
  for (var i = 0u; i < STREAM_CELLS; i = i + 1u) {
    let area = previous.stream[i];
    next.stream[i] = area;
    next.motion[i] = previous.motion[i];
    if (area <= 0.0) { continue; }
    let old = previous.motion[i];
    let old_world = old.xy;
    let velocity = vec2<f32>(old.z, old.w + law.gravity * law.dt);
    let world = old_world + velocity * law.dt;
    // World-space momentum survives arbitrary vessel rotation. Rotation only
    // changes contact geometry; it never reverses an existing trajectory.
    let before = local_position(old_world, law.pose_angle);
    let after = local_position(world, law.pose_angle);
    var hit = false;
    var hit_a = after.y < 0.485;
    // A rotating wall can cross a nearly stationary parcel between frames.
    // Check the angular sweep, not only the final vessel orientation.
    let angular_delta = atan2(sin(law.pose_angle - previous.pose_angle),
      cos(law.pose_angle - previous.pose_angle));
    let sweep_count = min(32u, max(1u, u32(ceil(abs(angular_delta) / 0.08))));
    for (var sweep = 1u; sweep <= sweep_count; sweep = sweep + 1u) {
      let angle = previous.pose_angle
        + angular_delta * f32(sweep) / f32(sweep_count);
      let sampled = local_position(old_world, angle);
      if (sampled.y < 0.06 || sampled.y > 0.91
          || abs(sampled.x) > chamber_wall(sampled.y, law.opening)) {
        hit = true;
        hit_a = sampled.y < 0.485;
        break;
      }
    }
    if (!hit && (before.y - 0.485) * (after.y - 0.485) <= 0.0
        && abs(after.y - before.y) > 1.0e-7) {
      let fraction = clamp((0.485 - before.y) / (after.y - before.y), 0.0, 1.0);
      let crossing_x = mix(before.x, after.x, fraction);
      if (abs(crossing_x) > law.opening * 0.5) {
        hit = true;
        hit_a = before.y < 0.485;
      }
    }
    if (!hit && (after.y < 0.06 || after.y > 0.91
        || abs(after.x) > chamber_wall(after.y, law.opening))) {
      hit = true;
      hit_a = after.y < 0.485;
    }
    if (!hit) {
      let pile_height = min(max(0.0, target_peak - tangent * abs(after.x)),
        pile_wall_capacity(after.x, law.opening));
      if ((direction > 0.0 && after.y >= 0.50
          && after.y >= 0.91 - pile_height)
          || (direction < 0.0 && after.y <= 0.47
          && after.y <= 0.06 + pile_height)) {
        hit = true;
        hit_a = direction < 0.0;
      }
    }
    if (hit) {
      if (hit_a) { deposited_a = deposited_a + area; }
      else { deposited_b = deposited_b + area; }
      next.stream[i] = 0.0;
      next.motion[i] = vec4<f32>(0.0);
    } else {
      next.motion[i] = vec4<f32>(world, velocity);
    }
  }
  chamber_a = chamber_a + deposited_a;
  chamber_b = chamber_b + deposited_b;
  if (released > 0.0) {
    // Aperture center is a local source, not a permanent stream anchor.
    // It emits one finite-area parcel per step with world-space velocity.
    next.stream[free_slot] = released;
    let nozzle = world_position(vec2<f32>(0.0, 0.485), law.pose_angle);
    next.motion[free_slot] = vec4<f32>(nozzle, 0.0, 0.0);
  }
  let half_chamber = ${CHAMBER_WIDTH} * 0.5;
  // Finite-volume ledger: the deposited reservoir owns exactly the material
  // absent from the upper, airborne, and settling reservoirs. This avoids
  // accumulating an independent f32 rounding error in every transfer.
  var stream_area = 0.0;
  for (var i = 0u; i < STREAM_CELLS; i = i + 1u) {
    stream_area = stream_area + next.stream[i];
  }
  if (direction > 0.0) {
    next.chamber_a = chamber_a;
    next.chamber_b = max(0.0,
      ${INITIAL_AREA} - next.chamber_a - stream_area);
  } else {
    next.chamber_b = chamber_b;
    next.chamber_a = max(0.0,
      ${INITIAL_AREA} - next.chamber_b - stream_area);
  }
  let next_target = select(next.chamber_a, next.chamber_b, direction > 0.0);
  let peak_height = pile_peak_for_area(next_target, tangent, law.opening);
  let dx = ${CHAMBER_WIDTH} / f32(PILE_CELLS);
  var discretized_area = 0.0;
  for (var i = 0u; i < PILE_CELLS; i = i + 1u) {
    let x = (f32(i) + 0.5) * dx - half_chamber;
    let height = min(max(0.0, peak_height - tangent * abs(x)),
      pile_wall_capacity(x, law.opening));
    next.pile[i] = height;
    discretized_area = discretized_area + height * dx;
  }
  if (discretized_area == 0.0 && next_target > 0.0) {
    next.pile[PILE_CELLS / 2u - 1u] = next_target / (2.0 * dx);
    next.pile[PILE_CELLS / 2u] = next_target / (2.0 * dx);
    discretized_area = next_target;
  }
  // Exact discrete-area normalization: changing repose cannot add or lose sand.
  let area_scale = next_target / max(discretized_area, 1.0e-20);
  for (var i = 0u; i < PILE_CELLS; i = i + 1u) {
    next.pile[i] = next.pile[i] * area_scale;
  }
}
`;

export async function createGranularApertureWorldGpu(device, {
  opening = 0.008,
  grainDiameter = 0.0015,
  reposeDegrees = 32,
  gravity = 9.82,
  dischargeCoefficient = 0.5,
  emptyAnnulus = 1,
  settlingSeconds = 0.35,
  releaseResponseSeconds = 0.25,
} = {}) {
  const stateBytes = STATE_FLOATS * 4;
  const initial = new Float32Array(STATE_FLOATS);
  initial[0] = INITIAL_AREA;
  initial[4] = 1;
  const buffers = Array.from({length: 2}, (_, index) => {
    const buffer = device.createBuffer({
      label: `Granular aperture material state ${index}`,
      size: stateBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(buffer, 0, initial);
    return buffer;
  });
  const uniforms = device.createBuffer({
    label: 'Granular aperture Law parameters', size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const shader = device.createShaderModule({label: 'Granular aperture Law', code: LAW_WGSL});
  const compilation = await shader.getCompilationInfo();
  const errors = compilation.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
  const pipeline = device.createComputePipeline({
    label: 'Granular aperture transport + repose Law', layout: 'auto',
    compute: {module: shader, entryPoint: 'step'},
  });
  const bindings = [0, 1].map(index => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: {buffer: buffers[index]}},
      {binding: 1, resource: {buffer: buffers[1 - index]}},
      {binding: 2, resource: {buffer: uniforms}},
    ],
  }));
  let current = 0;
  let simulatedSteps = 0;
  let repose = reposeDegrees;
  let aperture = opening;
  let poseAngle = 0;
  const parameters = new Float32Array(12);
  const writeParameters = () => {
    parameters.set([1 / 120, aperture, grainDiameter, repose * Math.PI / 180,
      gravity, dischargeCoefficient, emptyAnnulus, settlingSeconds, poseAngle,
      releaseResponseSeconds]);
    device.queue.writeBuffer(uniforms, 0, parameters);
  };
  writeParameters();
  return Object.freeze({
    get stateBuffer() { return buffers[current]; },
    get simulatedSeconds() { return simulatedSteps / 120; },
    get opening() { return aperture; },
    get grainDiameter() { return grainDiameter; },
    get reposeDegrees() { return repose; },
    get poseAngle() { return poseAngle; },
    get initialArea() { return INITIAL_AREA; },
    setReposeDegrees(value) {
      if (!Number.isFinite(value) || value < 20 || value > 45)
        throw new RangeError('Repose angle must be 20–45 degrees');
      repose = value;
      writeParameters();
    },
    setOpening(value) {
      if (!Number.isFinite(value) || value < grainDiameter || value > 0.02)
        throw new RangeError('Opening must be between one grain and 20 mm');
      aperture = value;
      writeParameters();
    },
    setPoseAngle(value) {
      if (!Number.isFinite(value)) throw new RangeError('Pose angle must be finite');
      poseAngle = value;
      writeParameters();
    },
    step(encoder, count = 1) {
      if (count < 1) return;
      const pass = encoder.beginComputePass({label: 'Granular aperture Law'});
      pass.setPipeline(pipeline);
      for (let iteration = 0; iteration < count; iteration++) {
        pass.setBindGroup(0, bindings[current]);
        pass.dispatchWorkgroups(1);
        current = 1 - current;
        simulatedSteps++;
      }
      pass.end();
    },
    reset() {
      device.queue.writeBuffer(buffers[0], 0, initial);
      device.queue.writeBuffer(buffers[1], 0, initial);
      current = 0;
      simulatedSteps = 0;
    },
    async readAudit() {
      const staging = device.createBuffer({size: stateBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      const encoder = device.createCommandEncoder({label: 'Granular conservation audit'});
      encoder.copyBufferToBuffer(buffers[current], 0, staging, 0, stateBytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const floats = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      const streamArea = floats.slice(8, 8 + STREAM_CELLS)
        .reduce((sum, area) => sum + area, 0);
      const pileProfileArea = floats.slice(8 + STREAM_CELLS * 5)
        .reduce((sum, height) => sum + height * CHAMBER_WIDTH / PILE_CELLS, 0);
      const totalArea = floats[0] + floats[1] + floats[2] + streamArea;
      return {upperArea: floats[0], settlingArea: floats[1],
        pileArea: floats[2], chamberAArea: floats[0], chamberBArea: floats[2],
        direction: floats[4], pileProfileArea, streamArea, totalArea,
        releaseRemainder: floats[5], flowRate: floats[6],
        retainedFraction: totalArea / INITIAL_AREA, simulatedSeconds: floats[3],
        stream: [...floats.slice(8, 8 + STREAM_CELLS)],
        streamMotion: Array.from({length: STREAM_CELLS}, (_, index) =>
          [...floats.slice(8 + STREAM_CELLS + index * 4,
            8 + STREAM_CELLS + (index + 1) * 4)]),
        pile: [...floats.slice(8 + STREAM_CELLS * 5)]};
    },
    destroy() { for (const buffer of buffers) buffer.destroy(); uniforms.destroy(); },
  });
}
