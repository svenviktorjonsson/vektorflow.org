const PARTICLE_STRIDE_BYTES = 32;
const GRANULAR_FIELD_FORMAT_GPU = 'rgba16float';

const DEFAULT_COLORS = Object.freeze({
  air: Object.freeze([0.035, 0.043, 0.048, 1]),
  wall: Object.freeze([0.18, 0.165, 0.135, 1]),
  floor: Object.freeze([0.12, 0.105, 0.082, 1]),
  sand: Object.freeze([0.68, 0.49, 0.27, 1]),
  particles: Object.freeze([0.24, 0.76, 0.93, 1]),
});

export const GRANULAR_PARTICLE_EMBEDDING_GPU_LOD = Object.freeze({
  minimumRasterRadiusPixels: 0.65,
  glintFadeStartPixels: 0.9,
  glintFullPixels: 2.25,
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
  gravity: vec4<f32>,
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
  @location(2) contact_count: f32,
  @location(3) freefall: f32,
  @location(4) age_fraction: f32,
  @interpolate(flat) @location(5) seed: u32,
};

@group(0) @binding(0) var<storage, read> grains: array<Grain>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var field_texture: texture_2d<f32>;
@group(0) @binding(3) var field_sampler: sampler;

fn world_to_clip(world: vec2<f32>) -> vec2<f32> {
  return (world - params.view.xy) / (params.view.zw - params.view.xy) * 2.0 - 1.0;
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
  color = mix(color, vec3<f32>(0.80, 0.74, 0.62), quartz * 0.46);
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
  let physical_radius = params.canvas.w;
  let particle_mode = params.canvas.z > 0.5;
  // Material mode embeds the discrete state as an overlapping density field;
  // particle mode exposes the actual contact disks. These are two views of
  // the same solver state, not two different grain sizes.
  let visible_radius = select(max(physical_radius * 1.62, pixel_world * 1.35),
    max(physical_radius, pixel_world * 0.72), particle_mode);
  let speed = length(grain.velocity);
  let motion_direction = select(vec2<f32>(0.0, 1.0), grain.velocity / max(speed, 1.0e-6),
    speed > 1.0e-6);
  let motion_normal = vec2<f32>(-motion_direction.y, motion_direction.x);
  let stretch = select(1.0 + clamp(speed * 0.10, 0.0, 0.65), 1.0, particle_mode);
  let offset = select((motion_normal * local.x + motion_direction * local.y * stretch)
    * visible_radius, local * visible_radius, particle_mode);
  let world_position = grain.position + offset;
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
  // A granular reconstruction closes sub-grain gaps without the broad kernel
  // that makes a heap resemble a cohesive liquid.
  let freefall = grain.contact_count == 0u && speed > 0.25;
  let pixel_world = (params.view.z - params.view.x) / max(params.canvas.x, 1.0);
  let radius = select(params.canvas.w * 4.15,
    max(pixel_world * 0.8, params.canvas.w * 0.5), freefall);
  let stretch = 1.0 + clamp(speed * 0.055, 0.0, 0.34);
  let offset = (motion_normal * local.x + motion_direction * local.y * stretch) * radius;
  var output: DensitySplatOut;
  output.position = vec4<f32>(world_to_clip(grain.position + offset), 0.0, 1.0);
  output.local = local;
  output.velocity = grain.velocity;
  output.contact_count = f32(grain.contact_count);
  output.freefall = select(0.0, 1.0, freefall);
  output.age_fraction = 0.0;
  output.seed = grain.id;
  return output;
}

@fragment
fn density_fragment(input: DensitySplatOut) -> @location(0) vec4<f32> {
  let radius_squared = dot(input.local, input.local);
  if (radius_squared >= 1.0) { discard; }
  let weight = pow(1.0 - radius_squared, 3.0)
    * mix(1.0, 0.12, input.freefall);
  return vec4<f32>(weight, input.velocity * weight,
    input.contact_count * weight);
}

fn gaussian_lane(seed: u32) -> f32 {
  // Four independent uniform draws approximate a normal variate without
  // trigonometry in every vertex. Lane positions are stable, not evenly spaced.
  let sum = stable_unit(seed ^ 0x1b873593u)
    + stable_unit(seed ^ 0x85ebca6bu)
    + stable_unit(seed ^ 0xc2b2ae35u)
    + stable_unit(seed ^ 0x27d4eb2fu);
  return clamp((sum - 2.0) * 1.7320508, -1.8, 1.8);
}

@vertex
fn lane_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> DensitySplatOut {
  let particle_index = instance_index / 12u;
  let lane_index = (instance_index / 3u) % 4u;
  let segment_index = instance_index % 3u;
  let grain = grains[particle_index];
  let lane_count = 2u + mix_u32(grain.id ^ 0x51a7d39bu) % 3u;
  let speed = length(grain.velocity);
  let lane_enabled = lane_index < lane_count
    && grain.contact_count == 0u && speed > 0.25;
  let seed = grain.id ^ (lane_index * 0x9e3779b9u);
  let lateral = vec2<f32>(-grain.velocity.y, grain.velocity.x)
    / max(speed, 1.0e-6);
  let lane_velocity = grain.velocity
    + lateral * gaussian_lane(seed) * max(0.11, speed * 0.065);
  let duration = min(0.18, 0.13 / max(speed, 0.25));
  let age0 = duration * f32(segment_index) / 3.0;
  let age1 = duration * f32(segment_index + 1u) / 3.0;
  let gravity = params.gravity.xy;
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
  // Inactive lanes clip before rasterization, so resting beds only pay the
  // vertex cost rather than shading thousands of invisible trail fragments.
  output.position = vec4<f32>(select(vec2<f32>(2.0),
    world_to_clip(position), lane_enabled), 0.0, 1.0);
  output.local = local;
  output.velocity = grain.velocity;
  output.contact_count = 0.0;
  output.freefall = select(0.0, 1.0, lane_enabled);
  output.age_fraction = mix(age0, age1, along) / duration;
  output.seed = seed;
  return output;
}

@fragment
fn lane_fragment(input: DensitySplatOut) -> @location(0) vec4<f32> {
  if (input.freefall < 0.5) { discard; }
  let age = input.age_fraction;
  // A soft Gaussian-like onset and slow exponential tail make a visibly
  // skewed front, with density distributed through time rather than a disk.
  let onset = 1.0 - exp(-pow(age / 0.13, 2.0));
  let tail = exp(-2.7 * age);
  let cross = exp(-0.5 * pow(input.local.x / 0.42, 2.0));
  let time_cell = u32(floor(age * 42.0));
  let granularity = 0.74 + 0.52 * stable_unit(input.seed ^ time_cell);
  let weight = 0.18 * onset * tail * cross * granularity;
  return vec4<f32>(weight, input.velocity * weight, 0.0);
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
  let x = u32(max(pixel.x, 0.0));
  let y = u32(max(pixel.y, 0.0));
  return stable_unit((x * 0x9e3779b9u) ^ (y * 0x85ebca6bu)
    ^ (phase * 0xc2b2ae35u));
}

fn field_at(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(field_texture, field_sampler,
    clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

fn closed_density_at(uv: vec2<f32>) -> f32 {
  let offset = 2.0 / params.canvas.xy;
  let center = field_at(uv).x;
  let enclosed = min(min(field_at(uv - vec2<f32>(offset.x, 0.0)).x,
    field_at(uv + vec2<f32>(offset.x, 0.0)).x),
    min(field_at(uv - vec2<f32>(0.0, offset.y)).x,
      field_at(uv + vec2<f32>(0.0, offset.y)).x));
  return max(center, enclosed * 0.97);
}

@fragment
fn material_composite_fragment(input: FullscreenOut) -> @location(0) vec4<f32> {
  let uv = input.position.xy / params.canvas.xy;
  let field = field_at(uv);
  let density = closed_density_at(uv);
  // Beer-Lambert coverage preserves faint, overlapping falling lanes while
  // dense settled material still becomes opaque. No hard density cutoff.
  let coverage = 1.0 - exp(-1.35 * density);

  let screen_uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(
    mix(params.view.x, params.view.z, screen_uv.x),
    mix(params.view.w, params.view.y, screen_uv.y));
  let view_height = max(params.view.w - params.view.y, 1.0e-6);
  let vertical = clamp((world.y - params.view.y) / view_height, 0.0, 1.0);
  let background = params.air_color.rgb * mix(0.68, 1.16, vertical);
  if (coverage <= 0.001) { return vec4<f32>(background, 1.0); }

  let texel = 1.0 / params.canvas.xy;
  let gradient_step = texel * 2.0;
  let density_left = closed_density_at(uv - vec2<f32>(gradient_step.x, 0.0));
  let density_right = closed_density_at(uv + vec2<f32>(gradient_step.x, 0.0));
  let density_up = closed_density_at(uv - vec2<f32>(0.0, gradient_step.y));
  let density_down = closed_density_at(uv + vec2<f32>(0.0, gradient_step.y));
  let gradient = vec2<f32>(density_right - density_left,
    density_down - density_up);
  let normal = normalize(vec3<f32>(-gradient.x * 2.4,
    gradient.y * 2.4, 1.0));
  let light_direction = normalize(params.light.xyz);
  let diffuse = 0.42 + 0.58 * max(dot(normal, light_direction), 0.0);

  let velocity = field.yz / max(density, 1.0e-5);
  let speed = length(velocity);
  let motion = smoothstep(0.035, 0.82, speed);
  let direction = select(vec2<f32>(0.0, 1.0), velocity / max(speed, 1.0e-6),
    speed > 1.0e-6);
  let across = vec2<f32>(-direction.y, direction.x);
  let time = params.floor_color.w;
  let settled_phase = 0u;
  let moving_phase = u32(floor(time * (7.0 + speed * 10.0)));
  let phase = select(settled_phase, moving_phase, motion > 0.04);
  let pixel = floor(input.position.xy);
  let fine = pixel_noise(pixel, phase);
  let fine_next = pixel_noise(pixel + direction, phase + 1u);
  let mineral_noise = pixel_noise(floor(pixel * 0.37), 0u);
  let stream = 0.5 + 0.5 * sin(dot(pixel, across) * 2.9
    - time * speed * 27.0 + fine * 3.4);

  var sand = params.sand_color.rgb * (0.91 + fine * 0.12);
  sand = mix(sand, vec3<f32>(0.80, 0.71, 0.55),
    smoothstep(0.965, 0.998, mineral_noise) * 0.38);
  let contacts = field.w / max(density, 1.0e-5);
  let contact_ao = mix(1.0, 0.88, clamp(contacts / 12.0, 0.0, 1.0));
  sand *= diffuse * contact_ao;
  sand *= 0.96 + 0.08 * mix(fine, stream, motion);

  let sparkle = smoothstep(0.982, 0.9995, fine);
  let motion_shimmer = abs(fine - fine_next) * motion;
  let glimmer = sparkle * (0.16 + params.material.w * 0.72)
    + motion_shimmer * 0.15;
  sand += vec3<f32>(1.0, 0.84, 0.56) * glimmer * params.light.w;
  let fine_coverage = clamp(coverage * (0.88 + 0.24 * fine), 0.0, 1.0);
  return vec4<f32>(mix(background, sand, fine_coverage), 1.0);
}

fn sand_material_shading(input: GrainOut) -> vec3<f32> {
  let speed = length(input.velocity);
  let motion = smoothstep(0.04, 0.85, speed);
  let direction = select(vec2<f32>(0.0, 1.0), input.velocity / max(speed, 1.0e-6),
    speed > 1.0e-6);
  let cross_direction = vec2<f32>(-direction.y, direction.x);
  let time = params.floor_color.w;
  let settled_phase = u32(floor(time * 1.5));
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
  // Sparse sub-grain highlights form a high-frequency mineral glimmer. During
  // pouring the neighboring-pixel difference becomes a directional shimmer,
  // preserving motion below the physical grain footprint.
  let sparkle_gate = smoothstep(0.975, 0.999, fine);
  let motion_shimmer = abs(fine - neighbor_fine) * motion;
  let glimmer = (sparkle_gate * (0.22 + params.material.w)
    + motion_shimmer * 0.20) * params.light.w;
  return color + vec3<f32>(1.0, 0.84, 0.55) * glimmer;
}

@fragment
fn grain_fragment(input: GrainOut) -> @location(0) vec4<f32> {
  let radial = length(input.local);
  let aa = max(fwidth(radial) * 1.15, 1.0e-4);
  let particle_mode = params.canvas.z > 0.5;
  let material_coverage = (1.0 - smoothstep(0.80 - aa, 1.03 + aa, radial))
    * input.area_scale;
  let particle_coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radial);
  let coverage = select(material_coverage, particle_coverage, particle_mode);
  if (coverage <= 0.001) { discard; }

  var color: vec3<f32>;
  if (particle_mode) {
    let radius_squared = min(dot(input.local, input.local), 1.0);
    let dome = sqrt(max(1.0 - radius_squared, 0.0));
    let contact = clamp(f32(input.contact_count) / 8.0, 0.0, 1.0);
    color = mix(params.particle_color.rgb,
      vec3<f32>(1.0, 0.46, 0.16), contact * 0.58) * (0.66 + 0.34 * dome);
  } else {
    color = sand_material_shading(input);
  }
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

  const backgroundPipeline = await device.createRenderPipelineAsync({
    label: 'VKF Granular static chamber background',
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'fullscreen_vertex' },
    fragment: { module: shader, entryPoint: 'background_fragment', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const particlePipeline = await device.createRenderPipelineAsync({
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
  const densityPipeline = await device.createRenderPipelineAsync({
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
  const lanePipeline = await device.createRenderPipelineAsync({
    label: 'VKF Granular ballistic falling lanes',
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
  const compositePipeline = await device.createRenderPipelineAsync({
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

  const paramsBuffer = createBuffer(device, 'VKF Granular embedding params', 176,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
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
  const particleBindGroup = device.createBindGroup({
    label: 'VKF Granular particle bindings',
    layout: particlePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: storageBinding },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
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
      size: [width, height],
      format: GRANULAR_FIELD_FORMAT_GPU,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    densityView = densityTexture.createView();
    densityBindGroup = device.createBindGroup({
      label: 'VKF Granular density field bindings',
      layout: densityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: storageBinding },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    laneBindGroup = device.createBindGroup({
      label: 'VKF Granular ballistic lane bindings',
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
      ],
    });
    configured = true;
    return true;
  };

  const updateParams = (mode, time) => {
    const { policy } = worldRuntime;
    const centerX = (policy.viewMinimum[0] + policy.viewMaximum[0]) * 0.5;
    const centerY = (policy.viewMinimum[1] + policy.viewMaximum[1]) * 0.5;
    let viewWidth = policy.viewMaximum[0] - policy.viewMinimum[0];
    let viewHeight = policy.viewMaximum[1] - policy.viewMinimum[1];
    const canvasAspect = width / Math.max(height, 1);
    const worldAspect = viewWidth / viewHeight;
    if (canvasAspect < worldAspect) viewHeight = viewWidth / canvasAspect;
    else viewWidth = viewHeight * canvasAspect;

    const values = new Float32Array(44);
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
    values.set([policy.gravity[0], policy.gravity[1], 0, 0], 40);
    device.queue.writeBuffer(paramsBuffer, 0, values);
  };

  const setColors = (nextColors = {}) => {
    for (const [name, fallback] of Object.entries(DEFAULT_COLORS)) {
      if (name in nextColors) colors[name] = normalizedColor(nextColors[name], fallback);
    }
  };

  const render = (encoder, { mode = 'sand', time = 0 } = {}) => {
    if (!encoder || typeof encoder.beginRenderPass !== 'function') {
      throw new TypeError('WebGPU command encoder required for granular embedding render');
    }
    if (mode !== 'sand' && mode !== 'particles') {
      throw new RangeError('Granular embedding mode must be sand or particles');
    }
    resize();
    updateParams(mode, Number.isFinite(time) ? time : 0);
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
      densityPass.draw(4, particleCount * 12);
      densityPass.end();
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
  };

  resize();
  return Object.freeze({
    kind: 'granular-particle-embedding-gpu:v1',
    particleCount,
    render,
    resize,
    setColors,
    destroy,
    get width() { return width; },
    get height() { return height; },
  });
}
