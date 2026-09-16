const WHEEL_WGSL = /* wgsl */`
struct Params {
  view: vec4<f32>,
  canvas: vec4<f32>,
  wheel: vec4<f32>,
};

struct Out { @builtin(position) position: vec4<f32>, };
@group(0) @binding(0) var<uniform> params: Params;

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> Out {
  var point = vec2<f32>(-1.0, -1.0);
  if (index == 1u) { point = vec2<f32>(3.0, -1.0); }
  if (index == 2u) { point = vec2<f32>(-1.0, 3.0); }
  var output: Out;
  output.position = vec4<f32>(point, 0.0, 1.0);
  return output;
}

fn segment_distance(point: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let edge = b - a;
  let t = clamp(dot(point - a, edge) / max(dot(edge, edge), 1.0e-8), 0.0, 1.0);
  return length(point - (a + edge * t));
}

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

@fragment
fn fragment_main(input: Out) -> @location(0) vec4<f32> {
  let uv = input.position.xy / params.canvas.xy;
  let world = vec2<f32>(mix(params.view.x, params.view.z, uv.x),
    mix(params.view.w, params.view.y, uv.y));
  let center = params.wheel.xy;
  let radius = params.wheel.z;
  let angle = params.wheel.w;
  let radial = length(world - center);
  var distance = abs(radial - radius);
  for (var segment = 0u; segment < 7u; segment = segment + 1u) {
    let local = baffle(segment);
    distance = min(distance, segment_distance(world,
      center + rotate_local(local.xy, angle), center + rotate_local(local.zw, angle)));
  }
  let pixel_world = (params.view.z - params.view.x) / max(params.canvas.x, 1.0);
  let half_width = max(0.012, pixel_world * 1.35);
  let aa = max(fwidth(distance), pixel_world);
  let alpha = 1.0 - smoothstep(half_width - aa, half_width + aa, distance);
  if (alpha <= 0.001) { discard; }
  let highlight = 1.0 - smoothstep(0.0, half_width * 0.85,
    abs(distance - half_width * 0.35));
  let steel = mix(vec3<f32>(0.16, 0.19, 0.20), vec3<f32>(0.62, 0.69, 0.69),
    0.32 + 0.36 * highlight);
  return vec4<f32>(steel * alpha, alpha);
}
`;

const viewFor = (policy, width, height) => {
  const centerX = (policy.viewMinimum[0] + policy.viewMaximum[0]) * 0.5;
  const centerY = (policy.viewMinimum[1] + policy.viewMaximum[1]) * 0.5;
  let viewWidth = policy.viewMaximum[0] - policy.viewMinimum[0];
  let viewHeight = policy.viewMaximum[1] - policy.viewMinimum[1];
  const canvasAspect = width / Math.max(height, 1);
  const worldAspect = viewWidth / viewHeight;
  if (canvasAspect < worldAspect) viewHeight = viewWidth / canvasAspect;
  else viewWidth = viewHeight * canvasAspect;
  return [centerX - viewWidth * 0.5, centerY - viewHeight * 0.5,
    centerX + viewWidth * 0.5, centerY + viewHeight * 0.5];
};

export async function createWheelEmbeddingGpu(device, canvas, format) {
  const context = canvas.getContext('webgpu');
  const module = device.createShaderModule({ label: 'VKF rigid wheel embedding', code: WHEEL_WGSL });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'VKF rigid wheel overlay', layout: 'auto',
    vertex: { module, entryPoint: 'vertex_main' },
    fragment: { module, entryPoint: 'fragment_main', targets: [{ format, blend: {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    } }] },
    primitive: { topology: 'triangle-list' },
  });
  const paramsBuffer = device.createBuffer({ label: 'VKF rigid wheel params', size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: paramsBuffer } }] });

  const render = (encoder, target, policy, wheel, angle) => {
    const view = viewFor(policy, canvas.width, canvas.height);
    const values = new Float32Array(12);
    values.set(view, 0);
    values.set([canvas.width, canvas.height, 0, 0], 4);
    values.set([wheel.center[0], wheel.center[1], wheel.radius, angle], 8);
    device.queue.writeBuffer(paramsBuffer, 0, values);
    const pass = encoder.beginRenderPass({ label: 'VKF rigid wheel pass',
      colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store' }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  };

  const screenToWorld = (event, policy) => {
    const rect = canvas.getBoundingClientRect();
    const view = viewFor(policy, canvas.width, canvas.height);
    const u = (event.clientX - rect.left) / Math.max(rect.width, 1);
    const v = (event.clientY - rect.top) / Math.max(rect.height, 1);
    return [view[0] + u * (view[2] - view[0]), view[3] - v * (view[3] - view[1])];
  };
  return Object.freeze({ render, screenToWorld, destroy: () => paramsBuffer.destroy() });
}
