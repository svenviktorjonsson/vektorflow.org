// Visual Embedding only. Physical area lives in the granular GPU World buffer.
// The falling stream uses alpha = k * area density; brightness noise does not
// alter alpha or the material ledger. Settled sparkle is position-seeded.
const EMBEDDING_WGSL = `
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
struct View {
  canvas: vec2<f32>,
  flow_width_pixels: f32,
  opening: f32,
  pose_angle: f32,
};
@group(0) @binding(0) var<storage, read> material: MaterialState;
@group(0) @binding(1) var<uniform> view: View;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};
@vertex fn vertex(@builtin(vertex_index) index: u32) -> VertexOut {
  var vertices = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var output: VertexOut;
  output.position = vec4<f32>(vertices[index], 0.0, 1.0);
  return output;
}
fn hash(pixel: vec2<f32>) -> f32 {
  // A direct f32→u32 cast clamps negative coordinates to zero in WGSL.
  // Bitcast signed cell indices instead, preserving unique seeds on both
  // sides of the glass and avoiding repeated horizontal stripes.
  let px = bitcast<u32>(i32(floor(pixel.x)));
  let py = bitcast<u32>(i32(floor(pixel.y)));
  var n = px * 1664525u + py * 1013904223u + 0x9e3779b9u;
  n = (n ^ (n >> 16u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n ^ (n >> 16u)) / 4294967295.0;
}
fn upper_wall(y: f32) -> f32 {
  return 0.004 + 0.246 * clamp((0.47 - y) / 0.41, 0.0, 1.0);
}
fn lower_wall(y: f32) -> f32 {
  return 0.004 + 0.246 * clamp((y - 0.50) / 0.41, 0.0, 1.0);
}
@fragment fn fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
  let screen = vec2<f32>((pixel.x / view.canvas.x - 0.5) * 0.6,
    pixel.y / view.canvas.y - 0.485);
  let c = cos(view.pose_angle);
  let s = sin(view.pose_angle);
  let x = c * screen.x + s * screen.y;
  let y = -s * screen.x + c * screen.y + 0.485;
  let px = 0.6 / view.canvas.x;
  let py = 1.0 / view.canvas.y;
  let upper_half = upper_wall(y);
  let lower_half = lower_wall(y);
  let in_upper = y >= 0.06 && y <= 0.47 && abs(x) <= upper_half;
  let in_lower = y >= 0.50 && y <= 0.91 && abs(x) <= lower_half;
  let shade = mix(0.72, 1.06, 1.0 - y);
  var color = vec3<f32>(0.025, 0.045, 0.056) * shade;
  // Grain texture is locked to the material coordinates, so a compact pile
  // neither exposes simulation cells nor shimmers while the scene is still.
  let grain_pixel = floor(vec2<f32>(x / px, y / py));
  let noise = hash(grain_pixel);
  let sparkle = pow(noise, 36.0) * 0.38;
  let sand = vec3<f32>(0.76, 0.62, 0.43) * (0.84 + 0.28 * noise)
    + vec3<f32>(sparkle);
  var compact = false;

  // Fill from the throat upward. Integrating the wedge width from the throat
  // yields A = opening*h + slope*h²; this is the inverse for h.
  let slope = 0.246 / 0.41;
  let a_height = (sqrt(view.opening * view.opening
    + 4.0 * slope * material.chamber_a) - view.opening) / (2.0 * slope);
  let b_height = (sqrt(view.opening * view.opening
    + 4.0 * slope * material.chamber_b) - view.opening) / (2.0 * slope);
  if (material.direction > 0.0 && in_upper && y >= 0.47 - a_height) {
    color = sand; compact = true;
  }
  if (material.direction < 0.0 && in_lower && y <= 0.50 + b_height) {
    color = sand; compact = true;
  }

  // The repose-limited height profile is computed by the GPU Law, never by
  // the Embedding; this pass only samples its conserved area bins.
  let pile_coordinate = clamp((x + 0.25) / 0.5 * 128.0 - 0.5, 0.0, 127.0);
  let pile_left = u32(floor(pile_coordinate));
  let pile_right = min(127u, pile_left + 1u);
  let pile_height = mix(material.pile[pile_left], material.pile[pile_right],
    fract(pile_coordinate));
  let pile_top = select(0.06 + pile_height, 0.91 - pile_height,
    material.direction > 0.0);
  if (material.direction > 0.0 && in_lower && y >= pile_top) {
    color = sand; compact = true;
  }
  if (material.direction < 0.0 && in_upper && y <= pile_top) {
    color = sand; compact = true;
  }

  if (!compact && (in_upper || in_lower || (y >= 0.47 && y <= 0.50
      && abs(x) <= view.opening * 0.5))) {
    let width_world = max(view.flow_width_pixels * px, px);
    let sigma_cross = max(width_world * 0.42, px * 0.6);
    var coverage = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
      let area = material.stream[i];
      if (area <= 0.0) { continue; }
      let point = material.motion[i];
      let origin = vec2<f32>(c * point.x + s * point.y,
        -s * point.x + c * point.y + 0.485);
      let velocity = vec2<f32>(c * point.z + s * point.w,
        -s * point.z + c * point.w);
      let speed = length(velocity);
      let axis = select(vec2<f32>(0.0, material.direction),
        velocity / max(speed, 1.0e-6), speed > 0.05);
      // Each physical parcel contributes area to a continuous density field.
      // Kernel length follows inter-parcel travel (v*dt), so faster flow
      // remains connected rather than exposing individual release events.
      let step_distance = speed / 120.0;
      let sigma_back = max(0.02, step_distance * 5.0);
      let sigma_front = max(0.01, step_distance * 2.5);
      let offset = vec2<f32>(x, y) - origin;
      let longitudinal = dot(offset, axis);
      let transverse = abs(offset.x * axis.y - offset.y * axis.x);
      let sigma_long = select(sigma_front, sigma_back, longitudinal < 0.0);
      if (abs(longitudinal) > 3.0 * sigma_long
          || transverse > 3.0 * sigma_cross) { continue; }
      let along = longitudinal / sigma_long;
      let across = transverse / sigma_cross;
      // Integral over the unbounded plane equals parcel area. Brightness
      // variation below is optical only; it cannot create physical material.
      let normalization = 3.14159265 * (sigma_back + sigma_front) * sigma_cross;
      coverage = coverage + area / normalization
        * exp(-0.5 * (along * along + across * across));
    }
    let falling_noise = hash(floor(grain_pixel + vec2<f32>(0.0, material.time * 400.0)));
    let falling_sand = vec3<f32>(0.82, 0.69, 0.49)
      * (0.88 + 0.2 * falling_noise);
    color = mix(color, falling_sand, clamp(coverage * 0.7, 0.0, 1.0));
  }

  // Glass is a geometric boundary. The aperture stays genuinely open.
  var glass = false;
  if (y >= 0.06 - py && y <= 0.47 + py
      && abs(abs(x) - upper_half) < 1.5 * px) { glass = true; }
  if (y >= 0.50 - py && y <= 0.91 + py
      && abs(abs(x) - lower_half) < 1.5 * px) { glass = true; }
  if (abs(y - 0.06) < 1.5 * py && abs(x) <= 0.25) { glass = true; }
  if (abs(y - 0.91) < 1.5 * py && abs(x) <= 0.25) { glass = true; }
  if (glass) { color = vec3<f32>(0.52, 0.65, 0.66); }
  return vec4<f32>(color, 1.0);
}
`;

export async function createGranularApertureEmbeddingGpu(device, canvas, context, format,
  world, {flowWidthPixels = 2} = {}) {
  const shader = device.createShaderModule({
    label: 'Granular aperture material Embedding', code: EMBEDDING_WGSL});
  const compilation = await shader.getCompilationInfo();
  const errors = compilation.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
  const pipeline = device.createRenderPipeline({
    label: 'Granular aperture Embedding', layout: 'auto',
    vertex: {module: shader, entryPoint: 'vertex'},
    fragment: {module: shader, entryPoint: 'fragment',
      targets: [{format}]},
    primitive: {topology: 'triangle-list'},
  });
  const uniforms = device.createBuffer({label: 'Granular Embedding view',
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const bindings = new Map();
  const bindingFor = buffer => {
    if (!bindings.has(buffer)) bindings.set(buffer, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{binding: 0, resource: {buffer}},
        {binding: 1, resource: {buffer: uniforms}}],
    }));
    return bindings.get(buffer);
  };
  let width = flowWidthPixels;
  return Object.freeze({
    setFlowWidthPixels(value) {
      if (!Number.isFinite(value) || value < 2 || value > 4)
        throw new RangeError('Flow width must be 2–4 pixels');
      width = value;
    },
    render(encoder) {
      device.queue.writeBuffer(uniforms, 0, new Float32Array([
        canvas.width, canvas.height, width, world.opening,
        world.poseAngle, 0, 0, 0]));
      const pass = encoder.beginRenderPass({
        colorAttachments: [{view: context.getCurrentTexture().createView(),
          clearValue: {r: 0.025, g: 0.045, b: 0.056, a: 1},
          loadOp: 'clear', storeOp: 'store'}],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindingFor(world.stateBuffer));
      pass.draw(3);
      pass.end();
    },
    destroy() { uniforms.destroy(); },
  });
}
