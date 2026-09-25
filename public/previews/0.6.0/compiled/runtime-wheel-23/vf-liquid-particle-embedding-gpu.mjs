import {createCheckedGpuPipeline} from './vf-gpu-pipeline-errors.mjs';
const DEFAULT_COLORS = Object.freeze({
  air: Object.freeze([0.035, 0.052, 0.058, 1]),
  ground: Object.freeze([0.15, 0.135, 0.105, 1]),
  rock: Object.freeze([0.25, 0.235, 0.205, 1]),
  water: Object.freeze([0.88, 0.94, 0.96, 1]),
  foam: Object.freeze([0.93, 0.97, 0.95, 1]),
  accent: Object.freeze([0.92, 0.97, 1.0, 1]),
});

// The continuous field consumes only density and foam, but rg16float is gated
// by WebGPU's optional texture-formats-tier1 feature. This API receives an
// already-created GPUDevice and therefore keeps the baseline rgba16float
// format instead of silently requiring a feature the caller did not request.
export const LIQUID_PARTICLE_FIELD_FORMAT_GPU = 'rgba16float';

export const LIQUID_PARTICLE_EMBEDDING_GPU_WGSL = /* wgsl */`
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

struct RenderParams {
  view: vec4<f32>,
  canvas: vec4<f32>,
  fluid: vec4<f32>,
  terrain: vec4<f32>,
  stone: vec4<f32>,
  air_color: vec4<f32>,
  ground_color: vec4<f32>,
  rock_color: vec4<f32>,
  water_color: vec4<f32>,
  foam_color: vec4<f32>,
  accent_color: vec4<f32>,
};

struct DensitySplatOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) foam: f32,
};

struct PrimaryOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) foam: f32,
  @location(2) phase: f32,
};

struct FullscreenOut {
  @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(5) var<storage, read> surface_points: array<vec2<f32>>;

fn quad_corner(vertex_index: u32) -> vec2<f32> {
  let x = select(-1.0, 1.0, vertex_index == 1u || vertex_index == 3u);
  let y = select(-1.0, 1.0, vertex_index >= 2u);
  return vec2<f32>(x, y);
}

fn world_to_clip(world: vec2<f32>) -> vec2<f32> {
  return (world - params.view.xy) / (params.view.zw - params.view.xy) * 2.0 - 1.0;
}

@vertex
fn density_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> DensitySplatOut {
  let particle = particles[instance_index];
  let local = quad_corner(vertex_index);
  let radius = params.fluid.y;
  var output: DensitySplatOut;
  output.position = vec4<f32>(world_to_clip(particle.position + local * radius), 0.0, 1.0);
  output.local = local;
  output.foam = particle.foam;
  return output;
}

@fragment
fn density_fragment(input: DensitySplatOut) -> @location(0) vec4<f32> {
  let radius_squared = dot(input.local, input.local);
  if (radius_squared >= 1.0) { discard; }
  let weight = pow(1.0 - radius_squared, 3.0);
  return vec4<f32>(weight, input.foam * weight, 0.0, 0.0);
}

fn terrain_height(x: f32) -> f32 {
  let segment_count = u32(params.stone.w + 0.5);
  let raw_segment = floor((x - params.stone.x) / params.stone.y);
  let segment = u32(clamp(raw_segment, 0.0, f32(segment_count - 1u)));
  let a = surface_points[segment];
  let b = surface_points[segment + 1u];
  return mix(a.y, b.y, (x - a.x) / max(b.x - a.x, 1.0e-12));
}

fn hash21(point: vec2<f32>) -> f32 {
  let value = dot(point, vec2<f32>(127.1, 311.7));
  return fract(sin(value) * 43758.5453123);
}

fn value_noise(point: vec2<f32>) -> f32 {
  let cell = floor(point);
  let local = fract(point);
  let curve = local * local * (3.0 - 2.0 * local);
  let a = mix(hash21(cell), hash21(cell + vec2<f32>(1.0, 0.0)), curve.x);
  let b = mix(hash21(cell + vec2<f32>(0.0, 1.0)),
    hash21(cell + vec2<f32>(1.0, 1.0)), curve.x);
  return mix(a, b, curve.y);
}

fn scene_background(world: vec2<f32>) -> vec3<f32> {
  if (params.stone.w < 1.0) {
    return params.air_color.rgb;
  }
  let bed = terrain_height(world.x);
  if (world.y >= bed) {
    let vertical = clamp((world.y - params.view.y) /
      max(params.view.w - params.view.y, 1.0e-6), 0.0, 1.0);
    return params.air_color.rgb * mix(0.74, 1.18, vertical);
  }
  let q = abs((world.x - params.terrain.z) / params.terrain.w);
  let base_bed = params.terrain.y + params.terrain.x
    * (world.x - params.terrain.z);
  let is_stone = select(0.0, 1.0, q < 1.0
    && terrain_height(world.x) - base_bed > 0.002
    && world.y > params.terrain.y + params.terrain.x
      * (world.x - params.terrain.z) - 0.035);
  let coarse = value_noise(world * vec2<f32>(8.0, 15.0));
  let grain = value_noise(world * vec2<f32>(51.0, 43.0) + vec2<f32>(7.1, 3.7));
  let material = mix(params.ground_color.rgb, params.rock_color.rgb, is_stone);
  let shallow = exp(-max(0.0, bed - world.y) * 11.0);
  return material * (0.72 + 0.22 * coarse + 0.09 * grain + 0.12 * shallow);
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

@group(0) @binding(2) var field_texture: texture_2d<f32>;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(6) var background_texture: texture_2d<f32>;

fn field_at(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(field_texture, field_sampler,
    clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

@fragment
fn background_fragment(input: FullscreenOut) -> @location(0) vec4<f32> {
  let screen_uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(
    mix(params.view.x, params.view.z, screen_uv.x),
    mix(params.view.w, params.view.y, screen_uv.y));
  return vec4<f32>(scene_background(world), 1.0);
}

fn cached_background_at(uv: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(background_texture, field_sampler,
    clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

@fragment
fn composite_fragment(input: FullscreenOut) -> @location(0) vec4<f32> {
  let screen_uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(
    mix(params.view.x, params.view.z, screen_uv.x),
    mix(params.view.w, params.view.y, screen_uv.y));
  let background_pixel = vec2<i32>(input.position.xy);
  var color = textureLoad(background_texture, background_pixel, 0).rgb;
  if (params.canvas.w < 0.5) { return vec4<f32>(color, 1.0); }

  let texel = 1.0 / params.canvas.xy;
  let field = field_at(screen_uv);
  let edge_width = max(fwidth(field.x) * 1.4, 0.018);
  let coverage = smoothstep(params.fluid.z - edge_width,
    params.fluid.z + edge_width, field.x);
  if (coverage <= 0.001) { return vec4<f32>(color, 1.0); }

  let gradient = vec2<f32>(
    field_at(screen_uv + vec2<f32>(texel.x, 0.0)).x
      - field_at(screen_uv - vec2<f32>(texel.x, 0.0)).x,
    field_at(screen_uv + vec2<f32>(0.0, texel.y)).x
      - field_at(screen_uv - vec2<f32>(0.0, texel.y)).x);
  let normal = normalize(vec2<f32>(-gradient.x,
    max(abs(gradient.y), 1.0e-4)));
  let refract_uv = screen_uv + vec2<f32>(gradient.x, -gradient.y)
    * params.fluid.w / max(field.x, 0.35);
  let refracted = cached_background_at(refract_uv);
  let surface_band = 1.0 - smoothstep(params.fluid.z,
    params.fluid.z + 1.35, field.x);
  let fresnel = 0.0204 + 0.9796 * pow(1.0 - clamp(normal.y, 0.0, 1.0), 5.0);
  let light_direction = normalize(vec2<f32>(-0.42, 0.91));
  let specular = pow(max(dot(normal, light_direction), 0.0), 72.0)
    * surface_band;
  // Clear water transmits the refracted scene. Keep only subtle wavelength
  // attenuation at depth; surface Fresnel/specular terms provide visibility.
  let absorption = 1.0 - exp(-0.34 * max(field.x - params.fluid.z, 0.0));
  var water = refracted * vec3<f32>(1.0 - 0.018 * absorption,
    1.0 - 0.007 * absorption, 1.0 - 0.003 * absorption);
  let reflected_air = mix(vec3<f32>(0.88, 0.92, 0.94),
    params.accent_color.rgb, 0.32 + 0.12 * sin(world.x * 4.3));
  water = mix(water, reflected_air, clamp(fresnel * surface_band * 0.58, 0.0, 0.7));
  water = water + params.accent_color.rgb
    * (specular * 0.74 + surface_band * 0.11);
  let foam = (1.0 - exp(-4.0 * max(field.y, 0.0)))
    * smoothstep(0.35, 0.78, field.x) * surface_band;
  water = mix(water, params.foam_color.rgb,
    clamp(foam * 0.86, 0.0, 0.9));
  color = mix(color, water, coverage * (0.68 + 0.26 * absorption));
  return vec4<f32>(color, 1.0);
}

@vertex
fn primary_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> PrimaryOut {
  let particle = particles[instance_index];
  let local = quad_corner(vertex_index);
  let pixel_world = (params.view.z - params.view.x) / max(params.canvas.x, 1.0);
  let diagnostic_radius = params.fluid.x * 0.46;
  let spray_radius = max(params.fluid.x * 0.13, pixel_world * 0.72);
  let radius = select(diagnostic_radius, spray_radius,
    particle.surface_class > 0u);
  var output: PrimaryOut;
  output.position = vec4<f32>(world_to_clip(particle.position + local * radius), 0.0, 1.0);
  output.local = local;
  output.foam = particle.foam;
  output.phase = f32(particle.surface_class);
  return output;
}

@fragment
fn primary_fragment(input: PrimaryOut) -> @location(0) vec4<f32> {
  if (params.canvas.w > 0.5 && input.phase < 0.5) { discard; }
  let radial = length(input.local);
  let aa = max(fwidth(radial) * 1.35, 0.012);
  let coverage = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radial);
  if (coverage <= 0.001) { discard; }
  if (params.canvas.w < 0.5) {
    return vec4<f32>(params.water_color.rgb * coverage, coverage);
  }
  let rim = 1.0 - smoothstep(0.76 - aa, 0.89 + aa, radial);
  let isolated = select(0.0, 1.0, input.phase > 0.5);
  let fill = mix(params.water_color.rgb, params.foam_color.rgb,
    clamp(input.foam * 0.55 + isolated * 0.38, 0.0, 0.72));
  let alpha = coverage * mix(0.70 + 0.20 * rim, 0.88, isolated);
  return vec4<f32>(fill * alpha, alpha);
}

struct DiffuseOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) kind: f32,
  @location(2) is_active: f32,
};

@group(0) @binding(4) var<storage, read> diffuse_render: array<vec4<f32>>;

@vertex
fn diffuse_vertex(@builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32) -> DiffuseOut {
  let marker = diffuse_render[instance_index];
  let local = quad_corner(vertex_index);
  let pixel_world = (params.view.z - params.view.x)
    / max(params.canvas.x, 1.0);
  let radius = max(marker.z, pixel_world * 0.78);
  var output: DiffuseOut;
  output.position = vec4<f32>(world_to_clip(marker.xy + local * radius), 0.0, 1.0);
  output.local = local;
  output.kind = marker.w;
  output.is_active = select(0.0, 1.0, marker.z > 0.0);
  return output;
}

@fragment
fn diffuse_fragment(input: DiffuseOut) -> @location(0) vec4<f32> {
  if (input.is_active < 0.5) { discard; }
  let radial = length(input.local);
  let aa = max(fwidth(radial) * 1.45, 0.015);
  let disc = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radial);
  let ring = 1.0 - smoothstep(0.10, 0.10 + aa,
    abs(radial - 0.70));
  let bubble = select(0.0, 1.0, input.kind > 2.5);
  if (bubble > 0.5) {
    let screen_uv = input.position.xy / params.canvas.xy;
    if (field_at(screen_uv).x < params.fluid.z * 0.72) { discard; }
  }
  let coverage = mix(disc, ring, bubble);
  if (coverage <= 0.001) { discard; }
  let tint = mix(params.foam_color.rgb,
    params.accent_color.rgb, bubble * 0.34);
  let alpha = coverage * mix(0.92, 0.68, bubble);
  return vec4<f32>(tint * alpha, alpha);
}
`;

const requireDevice = (device) => {
  if (!device || typeof device.createRenderPipelineAsync !== 'function') {
    throw new TypeError('WebGPU device required for Liquid particle embedding');
  }
  return device;
};

const requireCanvas = (canvas) => {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('Canvas required for Liquid particle embedding');
  }
  return canvas;
};

const normalizedColor = (value, fallback) => {
  const source = value ?? fallback;
  if (!Array.isArray(source) || source.length < 3
      || source.slice(0, 3).some((channel) => !Number.isFinite(channel))) {
    throw new TypeError('Liquid particle embedding colors require finite RGB arrays');
  }
  return [source[0], source[1], source[2], Number.isFinite(source[3]) ? source[3] : 1];
};

const createBuffer = (device, label, size, usage) => device.createBuffer({
  label,
  size: Math.max(4, Math.ceil(size / 4) * 4),
  usage,
});

export async function createLiquidParticleEmbeddingGpu(deviceArgument, canvasArgument,
  worldRuntime, options = {}) {
  const device = requireDevice(deviceArgument);
  const canvas = requireCanvas(canvasArgument);
  if (!worldRuntime?.particleBuffer || !worldRuntime?.diffuseRenderBuffer
      || !worldRuntime?.surfaceBuffer
      || !Number.isSafeInteger(worldRuntime.primaryCount)
      || worldRuntime.abi?.primaryStrideBytes !== 48
      || worldRuntime.abi?.diffuseRenderStrideBytes !== 16
      || worldRuntime.abi?.surfaceVertexStrideBytes !== 8
      || !Number.isSafeInteger(worldRuntime.boundaryPacket?.segmentCount)) {
    throw new TypeError('Liquid particle GPU world runtime required for embedding');
  }
  if (worldRuntime.device !== device) {
    throw new TypeError('Liquid particle embedding and world must share one GPUDevice');
  }
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Canvas WebGPU context unavailable');
  const format = options.format ?? navigator.gpu.getPreferredCanvasFormat();
  const colors = Object.fromEntries(Object.entries(DEFAULT_COLORS).map(([name, fallback]) => [
    name, normalizedColor(options.colors?.[name], fallback),
  ]));
  const maximumPixelRatio = Number.isFinite(options.maximumPixelRatio)
    ? Math.max(1, options.maximumPixelRatio) : 2;
  const shader = device.createShaderModule({
    label: 'VKF Liquid particle embedding',
    code: LIQUID_PARTICLE_EMBEDDING_GPU_WGSL,
  });
  if (typeof shader.getCompilationInfo === 'function') {
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === 'error');
    if (errors.length) {
      throw new Error(errors.map((message) =>
        `line ${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
    }
  }
  const premultipliedBlend = {
    color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const densityPipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Liquid continuous field splats', layout: 'auto',
    vertex: { module: shader, entryPoint: 'density_vertex' },
    fragment: { module: shader, entryPoint: 'density_fragment',
      targets: [{ format: LIQUID_PARTICLE_FIELD_FORMAT_GPU, blend: {
        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
      } }] },
    primitive: { topology: 'triangle-strip' },
  });
  const compositePipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Liquid transparent continuous embedding', layout: 'auto',
    vertex: { module: shader, entryPoint: 'fullscreen_vertex' },
    fragment: { module: shader, entryPoint: 'composite_fragment',
      targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const backgroundPipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Liquid cached static scene background', layout: 'auto',
    vertex: { module: shader, entryPoint: 'fullscreen_vertex' },
    fragment: { module: shader, entryPoint: 'background_fragment',
      targets: [{ format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list' },
  });
  const primaryPipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Liquid analytic primary particles', layout: 'auto',
    vertex: { module: shader, entryPoint: 'primary_vertex' },
    fragment: { module: shader, entryPoint: 'primary_fragment',
      targets: [{ format, blend: premultipliedBlend }] },
    primitive: { topology: 'triangle-strip' },
  });
  const diffusePipeline = await createCheckedGpuPipeline(device,'render',{
    label: 'VKF Liquid analytic spray foam bubbles', layout: 'auto',
    vertex: { module: shader, entryPoint: 'diffuse_vertex' },
    fragment: { module: shader, entryPoint: 'diffuse_fragment',
      targets: [{ format, blend: premultipliedBlend }] },
    primitive: { topology: 'triangle-strip' },
  });

  const paramsBuffer = createBuffer(device, 'VKF Liquid embedding params', 176,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const sampler = device.createSampler({
    label: 'VKF Liquid continuous field sampler',
    magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  let densityTexture = null;
  let densityView = null;
  let backgroundTexture = null;
  let backgroundView = null;
  let width = 0;
  let height = 0;
  let configured = false;
  let backgroundDirty = true;
  let densityBindGroup = null;
  let backgroundBindGroup = null;
  let compositeBindGroup = null;
  let primaryBindGroup = null;
  let diffuseBindGroup = null;

  const makeStorageBinding = (buffer, size) => ({ buffer, offset: 0, size });
  const rebuildBindGroups = () => {
    densityBindGroup = device.createBindGroup({
      label: 'VKF Liquid density bindings',
      layout: densityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: makeStorageBinding(worldRuntime.particleBuffer,
          worldRuntime.primaryCount * worldRuntime.abi.primaryStrideBytes) },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    backgroundBindGroup = device.createBindGroup({
      label: 'VKF Liquid cached background bindings',
      layout: backgroundPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 5, resource: makeStorageBinding(worldRuntime.surfaceBuffer,
          (worldRuntime.boundaryPacket.segmentCount + 1)
            * worldRuntime.abi.surfaceVertexStrideBytes) },
      ],
    });
    compositeBindGroup = device.createBindGroup({
      label: 'VKF Liquid composite bindings',
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: densityView },
        { binding: 3, resource: sampler },
        { binding: 6, resource: backgroundView },
      ],
    });
    primaryBindGroup = device.createBindGroup({
      label: 'VKF Liquid primary render bindings',
      layout: primaryPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: makeStorageBinding(worldRuntime.particleBuffer,
          worldRuntime.primaryCount * worldRuntime.abi.primaryStrideBytes) },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    diffuseBindGroup = device.createBindGroup({
      label: 'VKF Liquid diffuse render bindings',
      layout: diffusePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: densityView },
        { binding: 3, resource: sampler },
        { binding: 4, resource: makeStorageBinding(worldRuntime.diffuseRenderBuffer,
          worldRuntime.diffuseCapacity * worldRuntime.abi.diffuseRenderStrideBytes) },
      ],
    });
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const pixelRatio = Math.min(maximumPixelRatio,
      Math.max(1, globalThis.devicePixelRatio || 1));
    const nextWidth = Math.max(1, Math.round((rect.width || canvas.width || 640) * pixelRatio));
    const nextHeight = Math.max(1, Math.round((rect.height || canvas.height || 360) * pixelRatio));
    if (nextWidth === width && nextHeight === height && configured) return false;
    width = nextWidth; height = nextHeight;
    canvas.width = width; canvas.height = height;
    context.configure({ device, format, alphaMode: 'opaque' });
    if (densityTexture) densityTexture.destroy();
    if (backgroundTexture) backgroundTexture.destroy();
    densityTexture = device.createTexture({
      label: 'VKF Liquid continuous field',
      size: { width, height, depthOrArrayLayers: 1 }, format: LIQUID_PARTICLE_FIELD_FORMAT_GPU,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    densityView = densityTexture.createView();
    backgroundTexture = device.createTexture({
      label: 'VKF Liquid cached static scene background',
      size: { width, height, depthOrArrayLayers: 1 }, format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    backgroundView = backgroundTexture.createView();
    configured = true;
    backgroundDirty = true;
    rebuildBindGroups();
    return true;
  };

  const updateParams = (time, mode) => {
    const policy = worldRuntime.policy;
    const values = new Float32Array(44);
    const centerX = (policy.viewMinimum[0] + policy.viewMaximum[0]) * 0.5;
    const centerY = (policy.viewMinimum[1] + policy.viewMaximum[1]) * 0.5;
    let viewWidth = policy.viewMaximum[0] - policy.viewMinimum[0];
    let viewHeight = policy.viewMaximum[1] - policy.viewMinimum[1];
    const canvasAspect = width / Math.max(height, 1);
    const worldAspect = viewWidth / viewHeight;
    if (canvasAspect < worldAspect) viewHeight = viewWidth / canvasAspect;
    else viewWidth = viewHeight * canvasAspect;
    values.set([centerX - viewWidth * 0.5, centerY - viewHeight * 0.5,
      centerX + viewWidth * 0.5, centerY + viewHeight * 0.5], 0);
    values.set([width, height, time, mode === 'fluid' ? 1 : 0], 4);
    values.set([policy.particleSpacing, worldRuntime.seed.supportRadius,
      Number.isFinite(options.isoThreshold) ? options.isoThreshold : 1.32,
      Number.isFinite(options.refractionScale) ? options.refractionScale : 0.0055], 8);
    values.set([policy.bedSlope, policy.bedAtStone,
      policy.stoneCenterX, policy.stoneHalfWidth], 12);
    values.set([worldRuntime.boundaryPacket.minimumX,
      worldRuntime.boundaryPacket.uniformStepX, 0,
      worldRuntime.boundaryPacket.segmentCount], 16);
    values.set(colors.air, 20);
    values.set(colors.ground, 24);
    values.set(colors.rock, 28);
    values.set(colors.water, 32);
    values.set(colors.foam, 36);
    values.set(colors.accent, 40);
    device.queue.writeBuffer(paramsBuffer, 0, values);
  };

  const setColors = (nextColors = {}) => {
    for (const [name, fallback] of Object.entries(DEFAULT_COLORS)) {
      if (!(name in nextColors)) continue;
      const normalized = normalizedColor(nextColors[name], fallback);
      if ((name === 'air' || name === 'ground' || name === 'rock')
          && normalized.some((channel, index) => channel !== colors[name][index])) {
        backgroundDirty = true;
      }
      colors[name] = normalized;
    }
  };

  const render = (encoder, { time = 0, mode = 'fluid' } = {}) => {
    if (!encoder || typeof encoder.beginRenderPass !== 'function') {
      throw new TypeError('WebGPU command encoder required for Liquid embedding render');
    }
    if (mode !== 'fluid' && mode !== 'particles') {
      throw new RangeError('Liquid embedding mode must be fluid or particles');
    }
    resize();
    updateParams(time, mode);
    if (backgroundDirty) {
      const backgroundPass = encoder.beginRenderPass({
        label: 'VKF Liquid cached static scene background pass',
        colorAttachments: [{ view: backgroundView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store' }],
      });
      backgroundPass.setPipeline(backgroundPipeline);
      backgroundPass.setBindGroup(0, backgroundBindGroup);
      backgroundPass.draw(3);
      backgroundPass.end();
      backgroundDirty = false;
    }
    const densityPass = encoder.beginRenderPass({
      label: 'VKF Liquid continuous field pass',
      colorAttachments: [{ view: densityView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store' }],
    });
    densityPass.setPipeline(densityPipeline);
    densityPass.setBindGroup(0, densityBindGroup);
    densityPass.draw(4, worldRuntime.primaryCount);
    densityPass.end();

    const target = context.getCurrentTexture().createView();
    const scenePass = encoder.beginRenderPass({
      label: 'VKF Liquid embedding pass',
      colorAttachments: [{ view: target,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store' }],
    });
    scenePass.setPipeline(compositePipeline);
    scenePass.setBindGroup(0, compositeBindGroup);
    scenePass.draw(3);
    if (mode === 'particles') {
      scenePass.setPipeline(primaryPipeline);
      scenePass.setBindGroup(0, primaryBindGroup);
      scenePass.draw(4, worldRuntime.primaryCount);
    } else {
      scenePass.setPipeline(diffusePipeline);
      scenePass.setBindGroup(0, diffuseBindGroup);
      scenePass.draw(4, worldRuntime.diffuseCapacity);
    }
    scenePass.end();
  };

  const destroy = () => {
    if (densityTexture) densityTexture.destroy();
    if (backgroundTexture) backgroundTexture.destroy();
    paramsBuffer.destroy();
  };
  resize();
  return Object.freeze({
    kind: 'liquid-particle-embedding-gpu:v1',
    render,
    resize,
    setColors,
    destroy,
    get width() { return width; },
    get height() { return height; },
  });
}
