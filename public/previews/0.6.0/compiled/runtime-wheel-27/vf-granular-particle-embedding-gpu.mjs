import {createCheckedGpuPipeline} from './vf-gpu-pipeline-errors.mjs';
const PARTICLE_STRIDE_BYTES = 32;
const GRANULAR_FIELD_FORMAT_GPU = 'rgba16float';

const DEFAULT_COLORS = Object.freeze({
  air: Object.freeze([0.035, 0.043, 0.048, 1]),
  wall: Object.freeze([0.18, 0.165, 0.135, 1]),
  floor: Object.freeze([0.12, 0.105, 0.082, 1]),
  sand: Object.freeze([0.78, 0.64, 0.43, 1]),
  particles: Object.freeze([0.24, 0.76, 0.93, 1]),
});

export const GRANULAR_PARTICLE_EMBEDDING_GPU_LOD = Object.freeze({
  minimumRasterRadiusPixels: 0.65,
  glintFadeStartPixels: 0.9,
  glintFullPixels: 2.25,
  looseGrainRadiusPixels: 0.65,
  packedDensityStart: 1.05,
  packedDensityEnd: 1.65,
});

export const GRANULAR_PARTICLE_EMBEDDING_GPU_WGSL = /* wgsl */`
struct Grain {
  position: vec2<f32>,
  velocity: vec2<f32>,
  previous_position: vec2<f32>,
  id: u32,
  contact_count: u32,
};

struct RenderParams {
  view: vec4<f32>,
  canvas: vec4<f32>,
  material: vec4<f32>,
  chamber: vec4<f32>,
  air_color: vec4<f32>,
  wall_color: vec4<f32>,
  floor_color: vec4<f32>,
  sand_color: vec4<f32>,
  particle_color: vec4<f32>,
  light: vec4<f32>,
  visual: vec4<f32>,
  boundary: vec4<f32>,
  boundary_pose: vec4<f32>,
};

struct FullscreenOut {
  @builtin(position) position: vec4<f32>,
};

struct GrainOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) area_scale: f32,
  @interpolate(flat) @location(2) id: u32,
  @interpolate(flat) @location(3) contact_count: u32,
  @location(4) world_position: vec2<f32>,
  @location(5) velocity: vec2<f32>,
};

struct DensitySplatOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) velocity: vec2<f32>,
  @interpolate(flat) @location(2) airborne: f32,
  @location(3) age_fraction: f32,
  @location(4) lane_weight: f32,
  @interpolate(flat) @location(5) seed: u32,
};

@group(0) @binding(0) var<storage, read> grains: array<Grain>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var field_texture: texture_2d<f32>;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(5) var<storage, read> boundary_segments: array<vec4<f32>>;

fn world_to_clip(world: vec2<f32>) -> vec2<f32> {
  return (world - params.view.xy) / (params.view.zw - params.view.xy) * 2.0 - 1.0;
}

fn sand_solid_at(world: vec2<f32>) -> bool {
  let relative = world - params.boundary.xy;
  if (length(relative) >= params.boundary.z - params.boundary.w) { return true; }
  let c = cos(params.boundary_pose.x);
  let s = sin(params.boundary_pose.x);
  let local = vec2<f32>(c * relative.x + s * relative.y,
    -s * relative.x + c * relative.y);
  for (var segment = 0u; segment < u32(params.boundary_pose.y); segment++) {
    let bar = boundary_segments[segment];
    let edge = bar.zw - bar.xy;
    let along = clamp(dot(local - bar.xy, edge)
      / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
    if (length(local - (bar.xy + edge * along)) <= params.boundary.w) {
      return true;
    }
  }
  return false;
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
  let x = select(-1.0, 1.0, vertex_index == 1u || vertex_index == 3u);
  let y = select(-1.0, 1.0, vertex_index >= 2u);
  return vec2<f32>(x, y);
}

fn mix_u32(value: u32) -> u32 {
  var mixed = value;
  mixed ^= mixed >> 16u;
  mixed *= 0x7feb352du;
  mixed ^= mixed >> 15u;
  mixed *= 0x846ca68bu;
  mixed ^= mixed >> 16u;
  return mixed;
}

fn stable_unit(value: u32) -> f32 {
  return f32(mix_u32(value) >> 8u) * (1.0 / 16777216.0);
}

fn mineral_color(id: u32) -> vec3<f32> {
  let mineral = stable_unit(id ^ 0x51a7d39bu);
  let tone = 0.80 + 0.30 * stable_unit(id ^ 0x9e3779b9u);
  let quartz = 1.0 - smoothstep(0.08, 0.18, mineral);
  let iron = smoothstep(0.82, 0.97, mineral);
  var color = params.sand_color.rgb;
  color = mix(color, vec3<f32>(0.95, 0.91, 0.82), quartz * 0.58);
  color = mix(color, vec3<f32>(0.47, 0.24, 0.10), iron * 0.42);
  return clamp(color * tone, vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn fullscreen_vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var position = vec2<f32>(-1.0, -1.0);
  if (vertex_index == 1u) { position = vec2<f32>(3.0, -1.0); }
  if (vertex_index == 2u) { position = vec2<f32>(-1.0, 3.0); }
  var output: FullscreenOut;
  output.position = vec4<f32>(position, 0.0, 1.0);
  return output;
}

fn chamber_checker(world: vec2<f32>) -> f32 {
  let chamber_size = max(params.chamber.zw - params.chamber.xy, vec2<f32>(1.0e-5));
  let cell_size = max(params.canvas.w * 7.0, min(chamber_size.x, chamber_size.y) / 28.0);
  let cell = vec2<i32>(floor((world - params.chamber.xy) / cell_size));
  return f32((cell.x + cell.y) & 1);
}

@fragment
fn background_fragment(input: FullscreenOut) -> @location(0) vec4<f32> {
  let screen_uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(
    mix(params.view.x, params.view.z, screen_uv.x),
    mix(params.view.w, params.view.y, screen_uv.y));
  let view_height = max(params.view.w - params.view.y, 1.0e-6);
  let vertical = clamp((world.y - params.view.y) / view_height, 0.0, 1.0);
  var color = params.air_color.rgb * mix(0.68, 1.16, vertical);

  return vec4<f32>(color, 1.0);
}

@vertex
fn grain_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> GrainOut {
  let grain = grains[instance_index];
  let local = quad_corner(vertex_index);
  let pixel_world = (params.view.z - params.view.x) / max(params.canvas.x, 1.0);
  let visible_radius = max(params.canvas.w, pixel_world * 0.72);
  let world_position = grain.position + local * visible_radius;
  var output: GrainOut;
  output.position = vec4<f32>(world_to_clip(world_position),
    0.0, 1.0);
  output.local = local;
  output.area_scale = 1.0;
  output.id = grain.id;
  output.contact_count = grain.contact_count;
  output.world_position = world_position;
  output.velocity = grain.velocity;
  return output;
}

@vertex
fn density_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> DensitySplatOut {
  let grain = grains[instance_index];
  let local = quad_corner(vertex_index);
  let speed = length(grain.velocity);
  let motion_direction = select(vec2<f32>(0.0, 1.0),
    grain.velocity / max(speed, 1.0e-6), speed > 1.0e-6);
  let motion_normal = vec2<f32>(-motion_direction.y, motion_direction.x);
  let airborne = (grain.contact_count == 0u && grain.velocity.y < -0.04)
    || grain.velocity.y < -0.20;
  // Packed material needs a wider reconstruction kernel to hide the guide
  // lattice. Freefall deposits its mass along a short upstream trajectory,
  // so nearby guides form a flow field rather than isolated round blobs.
  let radius = params.canvas.w * select(3.2, 1.8, airborne);
  let stretch = select(1.0 + clamp(speed * 0.055, 0.0, 0.34),
    1.0, airborne);
  let trail_length = clamp(speed * 0.055, 2.0 * radius, 4.0 * radius);
  let packed_offset = (motion_normal * local.x
    + motion_direction * local.y * stretch) * radius;
  let falling_offset = motion_normal * local.x * radius
    + motion_direction * (local.y - 1.0) * trail_length * 0.5;
  let offset = select(packed_offset, falling_offset, airborne);
  var output: DensitySplatOut;
  // Freefall is represented by ballistic lanes below, not a round splat.
  output.position = vec4<f32>(select(world_to_clip(grain.position + offset),
    vec2<f32>(2.0), airborne), 0.0, 1.0);
  output.local = local;
  output.velocity = grain.velocity;
  // Neighbor contact alone does not make a descending cluster supported.
  // Fast downward flow remains discrete even when its guides touch.
  // Integral of (1-r^2)^2 over a unit disk is pi/3. Compensate for the
  // swept ellipse area so field alpha still represents physical grain area.
  output.airborne = select(0.0,
    6.0 * params.canvas.w * params.canvas.w
      / max(radius * trail_length, 1.0e-8), airborne);
  output.age_fraction = 0.0;
  output.lane_weight = 0.0;
  output.seed = grain.id;
  return output;
}

@fragment
fn density_fragment(input: DensitySplatOut) -> @location(0) vec4<f32> {
  let radius_squared = dot(input.local, input.local);
  if (radius_squared >= 1.0) { discard; }
  let weight = pow(1.0 - radius_squared, 2.0);
  // The fourth channel records unsupported falling material, not color.
  // A fast-moving supported heap must not be cut into smoky holes.
  let mass_weight = weight * select(1.0, input.airborne,
    input.airborne > 0.0);
  return vec4<f32>(mass_weight, input.velocity * mass_weight,
    input.airborne * weight);
}

fn gaussian_lane(seed: u32) -> f32 {
  // Stable approximate normal distribution; no evenly spaced fan or trigonometry.
  let sum = stable_unit(seed ^ 0x1b873593u)
    + stable_unit(seed ^ 0x85ebca6bu)
    + stable_unit(seed ^ 0xc2b2ae35u)
    + stable_unit(seed ^ 0x27d4eb2fu);
  return clamp((sum - 2.0) * 1.7320508, -1.8, 1.8);
}

@vertex
fn lane_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> DensitySplatOut {
  let particle_index = instance_index / 9u;
  let lane_index = (instance_index / 3u) % 3u;
  let segment_index = instance_index % 3u;
  let grain = grains[particle_index];
  let speed = length(grain.velocity);
  let airborne = (grain.contact_count == 0u && grain.velocity.y < -0.04)
    || grain.velocity.y < -0.20;
  let lane_count = u32(params.boundary_pose.w);
  let lane_enabled = airborne && lane_index < lane_count;
  let seed = grain.id ^ (lane_index * 0x9e3779b9u);
  let lateral = vec2<f32>(-grain.velocity.y, grain.velocity.x)
    / max(speed, 1.0e-6);
  let deviation = gaussian_lane(seed) * tan(params.visual.z)
    * max(speed, 0.65);
  let lane_velocity = vec2<f32>(grain.velocity.x + lateral.x * deviation,
    min(grain.velocity.y + lateral.y * deviation, -0.02));
  let duration = min(0.18, 0.12 / max(speed, 0.65));
  let age0 = duration * f32(segment_index) / 3.0;
  let age1 = duration * f32(segment_index + 1u) / 3.0;
  let gravity = vec2<f32>(0.0, params.boundary_pose.z);
  let start = grain.position - lane_velocity * age0
    + gravity * (0.5 * age0 * age0);
  let end = grain.position - lane_velocity * age1
    + gravity * (0.5 * age1 * age1);
  let axis = end - start;
  let normal = vec2<f32>(-axis.y, axis.x) / max(length(axis), 1.0e-6);
  let pixel_world = (params.view.z - params.view.x) / max(params.canvas.x, 1.0);
  let half_width = max(pixel_world * 1.8, params.canvas.w * 0.3);
  let local = quad_corner(vertex_index);
  let along = (local.y + 1.0) * 0.5;
  let position = mix(start, end, along) + normal * local.x * half_width;
  var output: DensitySplatOut;
  output.position = vec4<f32>(select(vec2<f32>(2.0),
    world_to_clip(position), lane_enabled), 0.0, 1.0);
  output.local = local;
  output.velocity = grain.velocity;
  output.airborne = select(0.0, 1.0, lane_enabled);
  output.age_fraction = mix(age0, age1, along) / duration;
  output.lane_weight = 1.0 / max(f32(lane_count), 1.0);
  output.seed = seed;
  return output;
}

@fragment
fn lane_fragment(input: DensitySplatOut) -> @location(0) vec4<f32> {
  if (input.airborne < 0.5) { discard; }
  let age = input.age_fraction;
  // Soft Gaussian-like front; exponential decay leaves a long trailing tail.
  let onset = 1.0 - exp(-pow(age / 0.13, 2.0));
  let tail = exp(-2.7 * age);
  let cross = exp(-0.5 * pow(input.local.x / 0.42, 2.0));
  let time_cell = u32(floor(age * 42.0));
  let granularity = 0.74 + 0.52 * stable_unit(input.seed ^ time_cell);
  let weight = 0.46 * input.lane_weight * onset * tail * cross * granularity;
  // The fourth channel separates airborne visual mass from settled density.
  return vec4<f32>(weight, input.velocity * weight, weight);
}

fn fresnel_schlick(cosine: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let alpha_squared = alpha * alpha;
  let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
  return alpha_squared / max(3.14159265 * denominator * denominator, 1.0e-5);
}

fn smith_visibility_term(n_dot_direction: f32, roughness: f32) -> f32 {
  let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
  return n_dot_direction / max(n_dot_direction * (1.0 - k) + k, 1.0e-5);
}

fn stable_microfacet(normal: vec3<f32>, id: u32) -> vec3<f32> {
  let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    abs(normal.y) > 0.85);
  let tangent = normalize(cross(helper, normal));
  let bitangent = cross(normal, tangent);
  let azimuth = stable_unit(id ^ 0xa511e9b3u) * 6.28318531;
  let tilt = 0.08 + 0.46 * sqrt(stable_unit(id ^ 0x63d83595u));
  return normalize(normal * cos(tilt)
    + (tangent * cos(azimuth) + bitangent * sin(azimuth)) * sin(tilt));
}

fn stable_glint(normal: vec3<f32>, half_vector: vec3<f32>, id: u32,
  n_dot_l: f32) -> f32 {
  let selector = stable_unit(id ^ 0xc2b2ae35u);
  let density = params.material.z;
  let safe_density = max(density, 1.0e-4);
  let gate = smoothstep(1.0 - safe_density,
    1.0 - safe_density * 0.18, selector)
    * select(0.0, 1.0, density > 0.0);
  let microfacet = stable_microfacet(normal, id);
  let alignment_error = 1.0 - max(dot(microfacet, half_vector), 0.0);
  let physical_pixel_radius = params.canvas.w * params.canvas.x
    / max(params.view.z - params.view.x, 1.0e-6);
  let footprint_width = clamp(0.035 + 0.18 / max(physical_pixel_radius, 0.5),
    0.035, 0.22);
  let lobe = exp(-alignment_error
    / max(footprint_width * footprint_width, 1.0e-4));
  let resolved = smoothstep(0.9, 2.25, physical_pixel_radius);
  let strength = params.material.w * mix(0.12, 1.0, resolved);
  let variation = 0.45 + 0.55 * stable_unit(id ^ 0x27d4eb2fu);
  return gate * lobe * n_dot_l * strength * variation;
}

fn sand_shading(local: vec2<f32>, id: u32, contact_count: u32) -> vec3<f32> {
  let radius_squared = min(dot(local, local), 1.0);
  let normal = normalize(vec3<f32>(local, sqrt(max(1.0 - radius_squared, 0.0))));
  let view_direction = vec3<f32>(0.0, 0.0, 1.0);
  let light_direction = normalize(params.light.xyz);
  let half_sum = light_direction + view_direction;
  let half_vector = normalize(select(vec3<f32>(1.0, 0.0, 0.0), half_sum,
    dot(half_sum, half_sum) > 1.0e-8));
  let n_dot_l = max(dot(normal, light_direction), 0.0);
  let n_dot_v = max(dot(normal, view_direction), 0.0);
  let n_dot_h = max(dot(normal, half_vector), 0.0);
  let v_dot_h = max(dot(view_direction, half_vector), 0.0);
  let roughness_variation = stable_unit(id ^ 0x165667b1u) - 0.5;
  let roughness = clamp(params.material.x + roughness_variation * 0.18, 0.24, 0.96);
  let fresnel = fresnel_schlick(v_dot_h, params.material.y);
  let distribution = ggx_distribution(n_dot_h, roughness);
  let geometry = smith_visibility_term(n_dot_l, roughness)
    * smith_visibility_term(n_dot_v, roughness);
  let specular = distribution * geometry * fresnel
    / max(4.0 * n_dot_l * n_dot_v, 1.0e-4);
  let albedo = mineral_color(id);
  let contact_ao = mix(1.0, 0.86, clamp(f32(contact_count) / 8.0, 0.0, 1.0));
  let ambient = albedo * (0.16 + 0.10 * max(normal.y, 0.0));
  let diffuse = albedo * (1.0 - fresnel) * n_dot_l * (1.0 / 3.14159265);
  let glint = stable_glint(normal, half_vector, id, n_dot_l);
  return (ambient + diffuse * params.light.w) * contact_ao
    + vec3<f32>((specular * 0.62 + glint) * params.light.w);
}

fn pixel_noise(pixel: vec2<f32>, phase: u32) -> f32 {
  // World positions may be negative. Clamping them to zero makes every
  // negative x share a row seed and every negative y share a column seed.
  let x = bitcast<u32>(i32(floor(pixel.x)));
  let y = bitcast<u32>(i32(floor(pixel.y)));
  return stable_unit((x * 0x9e3779b9u) ^ (y * 0x85ebca6bu)
    ^ (phase * 0xc2b2ae35u));
}

fn field_at(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(field_texture, field_sampler,
    clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

fn closed_density_at(uv: vec2<f32>) -> f32 {
  // Integrate over roughly one represented grain footprint. Point sampling
  // exposes the seed lattice as horizontal/vertical stripes in settled sand.
  let offset = params.canvas.w * params.canvas.x
    / max(params.view.z - params.view.x, 1.0e-6)
    / params.canvas.xy;
  let center = field_at(uv);
  let left = field_at(uv - vec2<f32>(offset.x, 0.0));
  let right = field_at(uv + vec2<f32>(offset.x, 0.0));
  let below = field_at(uv - vec2<f32>(0.0, offset.y));
  let above = field_at(uv + vec2<f32>(0.0, offset.y));
  return (max(center.x - center.w, 0.0)
    + max(left.x - left.w, 0.0)
    + max(right.x - right.w, 0.0)
    + max(below.x - below.w, 0.0)
    + max(above.x - above.w, 0.0)) * 0.2;
}

@fragment
fn material_composite_fragment(input: FullscreenOut) -> @location(0) vec4<f32> {
  let uv = input.position.xy / params.canvas.xy;
  let field = field_at(uv);
  let density = closed_density_at(uv);
  let edge_width = clamp(fwidth(density) * 1.15, 0.018, 0.06);
  let field_velocity = field.yz / max(field.x, 1.0e-5);
  // Low coverage alone does not mean motion: the edge of a resting pile is
  // sparse too. Only a measured moving field may become a diffuse fall.
  // One isolated compact splat peaks at one. Only overlapping neighborhoods
  // reconstruct a packed surface: isolated grains must not become mud blobs.
  let packed_coverage = smoothstep(1.35 - edge_width,
    1.35 + edge_width, density);
  // The packed surface excludes airborne mass. Freefall remains a continuous
  // projected-density field; physical guides never become visible disks.
  var coverage = packed_coverage;
  let airborne_density = max(field.w, 0.0);

  let screen_uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(
    mix(params.view.x, params.view.z, screen_uv.x),
    mix(params.view.w, params.view.y, screen_uv.y));
  // Reconstruct the contact skin up to the inner rim. Sparse guide centres
  // stay one radius away from steel; their visible surface must not.
  let rim_gap = params.boundary.z - params.boundary.w
    - length(world - params.boundary.xy);
  var solid_gap = rim_gap;
  if (rim_gap < 0.0) { coverage = 0.0; }
  else {
    let near_rim = 1.0 - smoothstep(0.0,
      params.canvas.w * 1.35, rim_gap);
    coverage = max(coverage,
      smoothstep(0.18, 0.72, density) * near_rim);
  }
  if (density > 0.12 || airborne_density > 0.0001) {
    let relative = world - params.boundary.xy;
    let c = cos(params.boundary_pose.x);
    let s = sin(params.boundary_pose.x);
    let local = vec2<f32>(c * relative.x + s * relative.y,
      -s * relative.x + c * relative.y);
    var bar_gap = 1.0e6;
    var bar_outward = vec2<f32>(0.0);
    for (var segment = 0u; segment < u32(params.boundary_pose.y); segment++) {
      let bar = boundary_segments[segment];
      let edge = bar.zw - bar.xy;
      let along = clamp(dot(local - bar.xy, edge)
        / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
      let separation = local - (bar.xy + edge * along);
      let gap = length(separation) - params.boundary.w;
      if (gap < bar_gap) {
        bar_gap = gap;
        let normal = separation / max(length(separation), 1.0e-6);
        bar_outward = vec2<f32>(c * normal.x - s * normal.y,
          s * normal.x + c * normal.y);
      }
    }
    solid_gap = min(solid_gap, bar_gap);
    if (bar_gap < 0.0) { coverage = 0.0; }
    else {
      // Close only sub-grain contact gaps on the material side of a solid.
      // Sample outward into the same pile; never paint through the boundary.
      let near_bar = 1.0 - smoothstep(0.0,
        params.canvas.w * 2.4, bar_gap);
      let support_world = world + bar_outward * params.canvas.w * 1.5;
      let support_uv = vec2<f32>(
        (support_world.x - params.view.x) / (params.view.z - params.view.x),
        (params.view.w - support_world.y) / (params.view.w - params.view.y));
      let support_density = max(density,
        closed_density_at(support_uv) * 0.82);
      coverage = max(coverage,
        smoothstep(0.18, 0.72, support_density) * near_bar);
    }
  }
  // Airborne splats already carry area-normalized mass. Alpha stays
  // continuous; subpixel noise changes tone, not geometry or mass.
  let visual_pixel_world = (params.view.z - params.view.x)
    / max(params.canvas.x, 1.0);
  let visual_seed = pixel_noise(floor((world
    - field_velocity * params.floor_color.w) / visual_pixel_world),
    0xa511e9b3u);
  let airborne_alpha = clamp(airborne_density, 0.0, 1.0)
    * (0.94 + 0.12 * visual_seed);
  let visual_coverage = select(airborne_alpha, 0.0, solid_gap < 0.0);
  coverage = 1.0 - (1.0 - coverage) * (1.0 - visual_coverage);
  let view_height = max(params.view.w - params.view.y, 1.0e-6);
  let vertical = clamp((world.y - params.view.y) / view_height, 0.0, 1.0);
  let background = params.air_color.rgb * mix(0.68, 1.16, vertical);
  if (coverage <= 0.001) { return vec4<f32>(background, 1.0); }

  // Density is coverage, not a liquid height field. Turning its gradient into
  // normals invents rounded menisci and glossy dents in a cohesionless heap.
  let normal = vec3<f32>(0.0, 0.0, 1.0);
  let light_direction = normalize(params.light.xyz);
  let diffuse = 0.42 + 0.58 * max(dot(normal, light_direction), 0.0);

  let velocity = field_velocity;
  let speed = length(velocity);
  let motion = smoothstep(0.08, 0.24, speed);
  let direction = select(vec2<f32>(0.0, 1.0), velocity / max(speed, 1.0e-6),
    speed > 1.0e-6);
  // A resting pile has a world-locked mineral pattern. Density-weighted guide
  // IDs can change with tiny contact jitter even when no sand visibly moves.
  let locked_seed = pixel_noise(floor(world * 950.0), 0u);
  let flowing_seed = pixel_noise(floor((world
    - field_velocity * params.floor_color.w) * 950.0), 0u);
  let material_seed = mix(locked_seed, flowing_seed, motion);
  let material_phase = u32(clamp(material_seed, 0.0, 1.0) * 16777215.0);
  let fine = stable_unit(material_phase ^ 0x9e3779b9u);
  let pixel_world = (params.view.z - params.view.x)
    / max(params.canvas.x, 1.0);
  let next_seed = pixel_noise(floor((world + direction * pixel_world
    - field_velocity * params.floor_color.w) * 950.0), 0u);
  let fine_next = stable_unit(u32(clamp(next_seed, 0.0, 1.0)
    * 16777215.0) ^ 0x9e3779b9u);
  let mineral_noise = stable_unit(material_phase ^ 0x85ebca6bu);
  let stream = stable_unit(material_phase ^ 0xc2b2ae35u);

  var sand = params.sand_color.rgb * (0.79 + fine * 0.35);
  sand = mix(sand, vec3<f32>(0.95, 0.91, 0.82),
    smoothstep(0.93, 0.995, mineral_noise) * 0.60);
  // Reconstructed density has lattice-row variation even in a static pile;
  // using it as a lighting term prints those rows into the material surface.
  sand *= diffuse;
  sand *= 0.96 + 0.08 * mix(fine, stream, motion);
  sand *= mix(1.0, 0.56, params.visual.w);
  // Contact shadow is tied to authored solid geometry, not clock noise.
  let contact_band = 1.0 - smoothstep(0.0,
    params.canvas.w * 2.4, max(solid_gap, 0.0));
  sand *= 1.0 - contact_band * (0.12 + 0.06 * (1.0 - motion));

  let sparkle = smoothstep(0.982, 0.9995, fine);
  let motion_shimmer = abs(fine - fine_next) * motion;
  let glimmer = sparkle * params.material.w * 0.28
    + motion * (sparkle * params.material.w * 0.08
      + motion_shimmer * 0.06);
  sand += vec3<f32>(0.86, 0.84, 0.76) * glimmer * params.light.w;
  return vec4<f32>(mix(background, sand, coverage), 1.0);
}

fn sand_material_shading(input: GrainOut) -> vec3<f32> {
  let speed = length(input.velocity);
  let motion = smoothstep(0.04, 0.85, speed);
  let direction = select(vec2<f32>(0.0, 1.0), input.velocity / max(speed, 1.0e-6),
    speed > 1.0e-6);
  let cross_direction = vec2<f32>(-direction.y, direction.x);
  let time = params.floor_color.w;
  let settled_phase = 0u;
  let moving_phase = u32(floor(time * (5.0 + speed * 7.0)));
  let phase = select(settled_phase, moving_phase, motion > 0.05);
  let fine = pixel_noise(floor(input.position.xy), phase);
  let neighbor_fine = pixel_noise(floor(input.position.xy) + vec2<f32>(1.0, 0.0), phase);
  let stream_phase = dot(input.position.xy, cross_direction) * 2.35
    - time * speed * 18.0 + stable_unit(input.id ^ 0xa511e9b3u) * 6.28318531;
  let stream_detail = sin(stream_phase) * 0.5 + 0.5;
  let mineral_variation = pixel_noise(floor(input.position.xy * 0.5), 0u);
  var mineral = params.sand_color.rgb * (0.94 + fine * 0.10);
  mineral = mix(mineral, vec3<f32>(0.78, 0.69, 0.52),
    smoothstep(0.965, 0.998, mineral_variation) * 0.42);
  let contact_ao = mix(1.0, 0.95,
    clamp(f32(input.contact_count) / 12.0, 0.0, 1.0));
  let light_direction = normalize(params.light.xyz);
  let surface_normal = normalize(vec3<f32>(input.local * 0.025, 1.0));
  let diffuse = 0.38 + 0.62 * max(dot(surface_normal, light_direction), 0.0);
  var color = mineral * diffuse * contact_ao;
  color = color * (0.94 + 0.10 * mix(fine, stream_detail, motion));
  color *= mix(1.0, 0.56, params.visual.w);
  // Sparse sub-grain highlights form a high-frequency mineral glimmer. During
  // pouring the neighboring-pixel difference becomes a directional shimmer,
  // preserving motion below the physical grain footprint.
  let sparkle_gate = smoothstep(0.975, 0.999, fine);
  let motion_shimmer = abs(fine - neighbor_fine) * motion;
  let glimmer = motion * (sparkle_gate * params.material.w * 0.24
    + motion_shimmer * 0.06) * params.light.w;
  return color + vec3<f32>(0.86, 0.84, 0.76) * glimmer;
}

@fragment
fn grain_fragment(input: GrainOut) -> @location(0) vec4<f32> {
  let radial = length(input.local);
  let aa = max(fwidth(radial) * 1.15, 1.0e-4);
  let coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radial);
  if (coverage <= 0.001) { discard; }
  let radius_squared = min(dot(input.local, input.local), 1.0);
  let dome = sqrt(max(1.0 - radius_squared, 0.0));
  let contact = clamp(f32(input.contact_count) / 8.0, 0.0, 1.0);
  let color = mix(params.particle_color.rgb,
    vec3<f32>(1.0, 0.46, 0.16), contact * 0.58) * (0.66 + 0.34 * dome);
  return vec4<f32>(color * coverage, coverage);
}
`;

const requireDevice = (device) => {
  if (!device || typeof device.createRenderPipelineAsync !== 'function') {
    throw new TypeError('WebGPU device required for granular particle embedding');
  }
  return device;
};

const requireCanvas = (canvas) => {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('Canvas required for granular particle embedding');
  }
  return canvas;
};

const finitePair = (value) => value != null && value.length === 2
  && Number.isFinite(value[0]) && Number.isFinite(value[1]);

const runtimeParticleCount = (runtime) => {
  const count = Number.isSafeInteger(runtime?.particleCount)
    ? runtime.particleCount : runtime?.primaryCount;
  return Number.isSafeInteger(count) && count > 0 ? count : null;
};

const hasGrainAbi = (runtime) => runtime?.abi?.particleStrideBytes === PARTICLE_STRIDE_BYTES
  && (!runtime.abi.particleFields || (
    runtime.abi.particleFields.length === 5
    && runtime.abi.particleFields[0] === 'position'
    && runtime.abi.particleFields[1] === 'velocity'
    && runtime.abi.particleFields[2] === 'previous_position'
    && runtime.abi.particleFields[3] === 'id'
    && runtime.abi.particleFields[4] === 'contact_count'
  ));

const requireWorldRuntime = (runtime, device) => {
  const count = runtimeParticleCount(runtime);
  const policy = runtime?.policy;
  if (!runtime?.particleBuffer || count === null || !hasGrainAbi(runtime)
      || !finitePair(policy?.viewMinimum) || !finitePair(policy?.viewMaximum)
      || !finitePair(policy?.worldMinimum) || !finitePair(policy?.worldMaximum)
      || policy.viewMaximum[0] <= policy.viewMinimum[0]
      || policy.viewMaximum[1] <= policy.viewMinimum[1]
      || policy.worldMaximum[0] <= policy.worldMinimum[0]
      || policy.worldMaximum[1] <= policy.worldMinimum[1]
      || !Number.isFinite(policy.grainRadius) || policy.grainRadius <= 0) {
    throw new TypeError('Granular particle GPU world runtime required for embedding');
  }
  if (runtime.device !== device) {
    throw new TypeError('Granular particle embedding and world must share one GPUDevice');
  }
  return count;
};

const normalizedColor = (value, fallback) => {
  const source = value ?? fallback;
  if ((!Array.isArray(source) && !ArrayBuffer.isView(source)) || source.length < 3
      || ![source[0], source[1], source[2]].every(Number.isFinite)) {
    throw new TypeError('Granular particle embedding colors require finite RGB arrays');
  }
  const alpha = source.length > 3 ? source[3] : 1;
  if (!Number.isFinite(alpha)) {
    throw new TypeError('Granular particle embedding colors require finite RGB arrays');
  }
  return [source[0], source[1], source[2], alpha];
};

const finiteRange = (value, fallback, minimum, maximum, name) => {
  const result = value ?? fallback;
  if (!Number.isFinite(result) || result < minimum || result > maximum) {
    throw new RangeError(`${name} must be from ${minimum} through ${maximum}`);
  }
  return result;
};

const normalizedDirection = (value) => {
  const source = value ?? [-0.42, 0.68, 0.60];
  if ((!Array.isArray(source) && !ArrayBuffer.isView(source)) || source.length !== 3
      || ![source[0], source[1], source[2]].every(Number.isFinite)) {
    throw new TypeError('Granular particle embedding lightDirection must be finite xyz');
  }
  const length = Math.hypot(source[0], source[1], source[2]);
  if (!(length > 0)) {
    throw new RangeError('Granular particle embedding lightDirection must be nonzero');
  }
  return [source[0] / length, source[1] / length, source[2] / length];
};

const createBuffer = (device, label, size, usage) => device.createBuffer({
  label,
  size: Math.max(4, Math.ceil(size / 4) * 4),
  usage,
});

const preferredFormat = () => {
  const format = globalThis.navigator?.gpu?.getPreferredCanvasFormat?.();
  if (!format) {
    throw new Error('WebGPU preferred canvas format unavailable');
  }
  return format;
};

export async function createGranularParticleEmbeddingGpu(deviceArgument, canvasArgument,
  worldRuntime, options = {}) {
  const device = requireDevice(deviceArgument);
  const canvas = requireCanvas(canvasArgument);
  const particleCount = requireWorldRuntime(worldRuntime, device);
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Canvas WebGPU context unavailable');

  const format = options.format ?? preferredFormat();
  const maximumPixelRatio = finiteRange(options.maximumPixelRatio, 2, 1, 4,
    'maximumPixelRatio');
  const roughness = finiteRange(options.roughness, 0.72, 0.05, 1, 'roughness');
  const fresnelF0 = finiteRange(options.fresnelF0, 0.04, 0, 1, 'fresnelF0');
  const glintDensity = finiteRange(options.glintDensity, 0.055, 0, 0.25,
    'glintDensity');
  const glintStrength = finiteRange(options.glintStrength, 0.30, 0, 2,
    'glintStrength');
  const lightIntensity = finiteRange(options.lightIntensity, 1.35, 0, 8,
    'lightIntensity');
  const lightDirection = normalizedDirection(options.lightDirection);
  const colors = Object.fromEntries(Object.entries(DEFAULT_COLORS).map(([name, fallback]) => [
    name, normalizedColor(options.colors?.[name], fallback),
  ]));

  const shader = device.createShaderModule({
    label: 'VKF Granular particle embedding',
    code: GRANULAR_PARTICLE_EMBEDDING_GPU_WGSL,
  });
  if (typeof shader.getCompilationInfo === 'function') {
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === 'error');
    if (errors.length) {
      throw new Error(errors.map((message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
    }
  }

  const backgroundPipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Granular static chamber background',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'fullscreen_vertex' },
    fragment: { module: shader, entryPoint: 'background_fragment', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const particlePipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Granular instanced analytic grains',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'grain_vertex' },
    fragment: {
      module: shader,
      entryPoint: 'grain_fragment',
      targets: [{
        format,
        blend: {
          color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });
  const densityPipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Granular full-resolution density field',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'density_vertex' },
    fragment: {
      module: shader,
      entryPoint: 'density_fragment',
      targets: [{
        format: GRANULAR_FIELD_FORMAT_GPU,
        blend: {
          color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
          alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });
  const lanePipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Granular ballistic visual lanes',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'lane_vertex' },
    fragment: {
      module: shader,
      entryPoint: 'lane_fragment',
      targets: [{
        format: GRANULAR_FIELD_FORMAT_GPU,
        blend: {
          color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
          alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
        },
      }],
    },
    primitive: { topology: 'triangle-strip' },
  });
  const compositePipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Granular continuous material composite',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'fullscreen_vertex' },
    fragment: {
      module: shader,
      entryPoint: 'material_composite_fragment',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const paramsBuffer = createBuffer(device, 'VKF Granular embedding params', 320,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const boundarySegments = worldRuntime.wheel.segments ?? [];
  const boundaryWords = new Float32Array(Math.max(4, boundarySegments.length * 4));
  for (let index = 0; index < boundarySegments.length; index++)
    boundaryWords.set(boundarySegments[index], index * 4);
  const boundaryBuffer = createBuffer(device, 'VKF authored granular boundary segments',
    boundaryWords.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    boundaryWords);
  const storageBinding = {
    buffer: worldRuntime.particleBuffer,
    offset: 0,
    size: particleCount * PARTICLE_STRIDE_BYTES,
  };
  const backgroundBindGroup = device.createBindGroup({
    label: 'VKF Granular background bindings',
    layout: backgroundPipeline.getBindGroupLayout(0),
    entries: [{ binding: 1, resource: { buffer: paramsBuffer } }],
  });
  const fieldSampler = device.createSampler({
    label: 'VKF Granular density field sampler',
    magFilter: 'linear',
    minFilter: 'linear',
  });

  let width = 0;
  let height = 0;
  let configured = false;
  let densityTexture = null;
  let densityView = null;
  let densityBindGroup = null;
  let laneBindGroup = null;
  let compositeBindGroup = null;
  let particleBindGroup = null;
  let lastVisualTime = null;

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const pixelRatio = Math.min(maximumPixelRatio,
      Math.max(1, globalThis.devicePixelRatio || 1));
    const nextWidth = Math.max(1, Math.round((rect.width || canvas.width || 640) * pixelRatio));
    const nextHeight = Math.max(1,
      Math.round((rect.height || canvas.height || 360) * pixelRatio));
    if (configured && nextWidth === width && nextHeight === height) return false;
    width = nextWidth;
    height = nextHeight;
    canvas.width = width;
    canvas.height = height;
    context.configure({ device, format, alphaMode: 'opaque' });
    densityTexture?.destroy();
    densityTexture = device.createTexture({
      label: 'VKF Granular density and motion field',
      size: { width, height, depthOrArrayLayers: 1 },
      format: GRANULAR_FIELD_FORMAT_GPU,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    densityView = densityTexture.createView();
    particleBindGroup = device.createBindGroup({
      label: 'VKF Granular neighbor-adaptive grain bindings',
      layout: particlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageBinding },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    densityBindGroup = device.createBindGroup({
      label: 'VKF Granular density field bindings',
      layout: densityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageBinding },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    laneBindGroup = device.createBindGroup({
      label: 'VKF Granular lane bindings',
      layout: lanePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageBinding },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    compositeBindGroup = device.createBindGroup({
      label: 'VKF Granular material composite bindings',
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: densityView },
        { binding: 3, resource: fieldSampler },
        { binding: 5, resource: { buffer: boundaryBuffer } },
      ],
    });
    configured = true;
    return true;
  };

  let wetness = 0;
  let laneCount = 2;
  let laneSpreadDegrees = 12;
  const setWetness = value => {
    if (!Number.isFinite(value) || value < 0 || value > 1)
      throw new RangeError('Sand wetness must be between zero and one');
    wetness = value;
  };
  const setLaneCount = value => {
    if (!Number.isInteger(value) || value < 1 || value > 3)
      throw new RangeError('Sand lane count must be one through three');
    laneCount = value;
  };
  const setLaneSpread = value => {
    if (!Number.isFinite(value) || value < 0 || value > 30)
      throw new RangeError('Sand lane spread must be zero through thirty degrees');
    laneSpreadDegrees = value;
  };
  const updateParams = (mode, time, deltaTime, wheelAngle) => {
    const { policy } = worldRuntime;
    const centerX = (policy.viewMinimum[0] + policy.viewMaximum[0]) * 0.5;
    const centerY = (policy.viewMinimum[1] + policy.viewMaximum[1]) * 0.5;
    let viewWidth = policy.viewMaximum[0] - policy.viewMinimum[0];
    let viewHeight = policy.viewMaximum[1] - policy.viewMinimum[1];
    const canvasAspect = width / Math.max(height, 1);
    const worldAspect = viewWidth / viewHeight;
    if (canvasAspect < worldAspect) viewHeight = viewWidth / canvasAspect;
    else viewWidth = viewHeight * canvasAspect;

    const values = new Float32Array(80);
    values.set([centerX - viewWidth * 0.5, centerY - viewHeight * 0.5,
      centerX + viewWidth * 0.5, centerY + viewHeight * 0.5], 0);
    values.set([width, height, mode === 'particles' ? 1 : 0, policy.grainRadius], 4);
    values.set([roughness, fresnelF0, glintDensity, glintStrength], 8);
    values.set([policy.worldMinimum[0], policy.worldMinimum[1],
      policy.worldMaximum[0], policy.worldMaximum[1]], 12);
    values.set(colors.air, 16);
    values.set(colors.wall, 20);
    values.set([colors.floor[0], colors.floor[1], colors.floor[2], time], 24);
    values.set(colors.sand, 28);
    values.set(colors.particles, 32);
    values.set([...lightDirection, lightIntensity], 36);
    values.set([deltaTime, policy.grainRadius,
      laneSpreadDegrees * Math.PI / 180, wetness], 40);
    const wheel = worldRuntime.wheel;
    values.set([wheel.center[0], wheel.center[1], wheel.radius,
      wheel.barHalfWidth], 44);
    const segments = wheel.segments ?? [];
    values.set([wheelAngle, segments.length, policy.gravity[1], laneCount], 48);
    device.queue.writeBuffer(paramsBuffer, 0, values);
  };

  const setColors = (nextColors = {}) => {
    for (const [name, fallback] of Object.entries(DEFAULT_COLORS)) {
      if (name in nextColors) colors[name] = normalizedColor(nextColors[name], fallback);
    }
  };

  const render = (encoder, { mode = 'sand', time = 0, wheelAngle = 0 } = {}) => {
    if (!encoder || typeof encoder.beginRenderPass !== 'function') {
      throw new TypeError('WebGPU command encoder required for granular embedding render');
    }
    if (mode !== 'sand' && mode !== 'particles') {
      throw new RangeError('Granular embedding mode must be sand or particles');
    }
    resize();
    const renderTime = Number.isFinite(time) ? time : 0;
    const visualDeltaTime = lastVisualTime === null ? 0
      : Math.max(0, Math.min(0.05, renderTime - lastVisualTime));
    updateParams(mode, renderTime, visualDeltaTime, wheelAngle);
    if (mode === 'sand') {
      const densityPass = encoder.beginRenderPass({
        label: 'VKF Granular density accumulation pass',
        colorAttachments: [{
          view: densityView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      densityPass.setPipeline(densityPipeline);
      densityPass.setBindGroup(0, densityBindGroup);
      densityPass.draw(4, particleCount);
      densityPass.setPipeline(lanePipeline);
      densityPass.setBindGroup(0, laneBindGroup);
      densityPass.draw(4, particleCount * 9);
      densityPass.end();
      lastVisualTime = renderTime;
    }
    const pass = encoder.beginRenderPass({
      label: 'VKF Granular particle embedding pass',
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    if (mode === 'sand') {
      pass.setPipeline(compositePipeline);
      pass.setBindGroup(0, compositeBindGroup);
      pass.draw(3);
    } else {
      pass.setPipeline(backgroundPipeline);
      pass.setBindGroup(0, backgroundBindGroup);
      pass.draw(3);
      pass.setPipeline(particlePipeline);
      pass.setBindGroup(0, particleBindGroup);
      pass.draw(4, particleCount);
    }
    pass.end();
  };

  const destroy = () => {
    densityTexture?.destroy();
    paramsBuffer.destroy();
    boundaryBuffer.destroy();
  };
  const reset = () => {
    lastVisualTime = null;
  };

  resize();
  return Object.freeze({
    kind: 'granular-particle-embedding-gpu:v1',
    particleCount,
    render,
    reset,
    resize,
    setColors,
    setWetness,
    setLaneCount,
    setLaneSpread,
    destroy,
    get laneCount() { return laneCount; },
    get laneSpreadDegrees() { return laneSpreadDegrees; },
    get width() { return width; },
    get height() { return height; },
  });
}
