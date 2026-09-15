const SHADERS = {
  'stones-water': `
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn circle(p: vec2f, c: vec2f, r: f32) -> f32 { return length(p - c) - r; }
fn scene(uv: vec2f, t: f32, pointer: vec2f, seed: f32, params: vec4f) -> vec3f {
  let wave = (sin(uv.x * 17.0 + t * 1.7) * 0.018 + sin(uv.x * 31.0 - t * 1.2) * 0.009) * params.x;
  let ripple = sin(distance(uv, pointer) * 55.0 - t * 5.0) * exp(-distance(uv, pointer) * 7.0) * 0.018 * params.x;
  let surface = 0.58 + wave + ripple;
  var color = mix(vec3f(0.025, 0.10, 0.16), vec3f(0.04, 0.31, 0.43), smoothstep(0.0, surface, uv.y));
  color += vec3f(0.08, 0.22, 0.25) * exp(-abs(uv.y - surface) * 80.0);
  for (var i = 0; i < 8; i++) {
    let fi = f32(i);
    let x = 0.12 + fi * 0.105 + (hash(vec2f(fi, seed)) - 0.5) * 0.035;
    let drop = fract(t * (0.055 + fi * 0.004) * params.z + hash(vec2f(seed, fi)));
    let y = 0.13 + drop * 0.63;
    let r = (0.028 + hash(vec2f(fi + 4.0, seed)) * 0.026) * params.y;
    let d = circle(uv, vec2f(x, y), r);
    let stone = mix(vec3f(0.16, 0.17, 0.17), vec3f(0.54, 0.48, 0.37), hash(vec2f(fi, 9.0)));
    color = mix(color, stone + vec3f(0.18) * smoothstep(r, -r, d + r * 0.35), smoothstep(0.008, -0.008, d));
    color += vec3f(0.30, 0.55, 0.60) * exp(-abs(d) * 90.0) * step(surface - 0.05, y);
  }
  let grabbed = circle(uv, pointer, 0.055 * params.y);
  color = mix(color, vec3f(0.72, 0.56, 0.29), smoothstep(0.008, -0.008, grabbed) * params.w);
  let foam = smoothstep(0.77, 0.98, hash(floor(uv * vec2f(115.0, 70.0)) + floor(t * 4.0))) * exp(-abs(uv.y - surface) * 55.0);
  return color + vec3f(0.45, 0.72, 0.75) * foam * 0.42;
}`,
  'tree-wind': `
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn segment(p: vec2f, a: vec2f, b: vec2f) -> f32 { let pa=p-a; let ba=b-a; return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0)); }
fn scene(uv: vec2f, t: f32, pointer: vec2f, seed: f32, params: vec4f) -> vec3f {
  let wind = sin(t * 1.45 + uv.y * 8.0) * (0.012 + params.x * 0.045) + (pointer.x - 0.5) * 0.11;
  var color = mix(vec3f(0.055, 0.10, 0.13), vec3f(0.24, 0.42, 0.43), uv.y);
  color = mix(color, vec3f(0.07, 0.13, 0.08), smoothstep(0.24, 0.20, uv.y));
  let base = vec2f(0.50, 0.18);
  let crown = vec2f(0.50 + wind, 0.78);
  let trunk = segment(uv, base, crown);
  color = mix(color, vec3f(0.27, 0.16, 0.075), smoothstep(0.022, 0.006, trunk));
  for (var i = 0; i < 34; i++) {
    let fi = f32(i);
    let h = 0.25 + fract(fi * 0.618) * 0.50;
    let side = select(-1.0, 1.0, (i % 2) == 0);
    let spread = (0.11 + hash(vec2f(fi, seed)) * 0.18) * (0.65 + params.y * 0.55);
    let a = vec2f(0.50 + wind * h, h);
    let b = a + vec2f(side * spread, 0.09 + hash(vec2f(seed, fi)) * 0.15 + wind * side);
    let branch = segment(uv, a, b);
    color = mix(color, vec3f(0.31, 0.19, 0.08), smoothstep(0.010, 0.003, branch));
    let leafD = length(uv - b) - (0.035 + params.z * 0.018 + hash(vec2f(fi, 4.0)) * 0.018);
    let leaf = mix(vec3f(0.13, 0.31, 0.10), vec3f(0.48, 0.62, 0.16), hash(vec2f(fi, seed + 8.0)));
    color = mix(color, leaf, smoothstep(0.01, -0.01, leafD));
  }
  let sun = exp(-length(uv - vec2f(0.80, 0.80)) * 12.0);
  return color + vec3f(0.75, 0.58, 0.24) * sun * 0.35;
}`,
  'sand-drum': `
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn scene(uv: vec2f, t: f32, pointer: vec2f, seed: f32, params: vec4f) -> vec3f {
  let p = (uv - 0.5) * vec2f(1.65, 1.0);
  let r = length(p);
  let angle = t * 0.22 * params.x + (pointer.x - 0.5) * 2.8;
  let ca = cos(angle); let sa = sin(angle);
  let q = vec2f(ca*p.x-sa*p.y, sa*p.x+ca*p.y);
  var color = mix(vec3f(0.025, 0.022, 0.020), vec3f(0.11, 0.075, 0.045), uv.y);
  let shell = abs(r - 0.39);
  color = mix(color, vec3f(0.34, 0.22, 0.10), smoothstep(0.022, 0.005, shell));
  let vane = min(abs(q.x), abs(q.y));
  color = mix(color, vec3f(0.30, 0.18, 0.07), smoothstep(0.012, 0.003, vane) * step(r, 0.36));
  let bed = -0.12 + params.y * 0.10 + 0.045*sin(p.x*8.0 + t*1.1) + 0.025*sin(p.x*19.0 - t*0.7);
  let insideSand = step(r, 0.365) * step(p.y, bed);
  let cell = floor((p + vec2f(0.8,0.5)) * vec2f(170.0, 170.0));
  let grain = hash(cell + floor(t * 0.7));
  let sparkle = pow(max(0.0, sin(grain*120.0 + t*2.2)), 38.0);
  var sand = mix(vec3f(0.42, 0.20, 0.055), vec3f(0.92, 0.63, 0.18), grain);
  sand += vec3f(1.0, 0.83, 0.48) * sparkle * params.z;
  if (params.w > 0.5) { sand = mix(vec3f(0.06,0.11,0.13), vec3f(0.23,0.78,0.72), grain); }
  color = mix(color, sand, insideSand);
  color += vec3f(0.48,0.25,0.07) * exp(-abs(p.y-bed)*80.0) * step(r,0.36);
  return color;
}`,
};

const VERTEX = `
@vertex fn vsMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[i], 0.0, 1.0);
}`;

function fragment(scene) {
  return `struct U { viewport: vec2f, time: f32, seed: f32, pointer: vec2f, params: vec4f, pad: vec2f };
@group(0) @binding(0) var<uniform> u: U;
${scene}
@fragment fn fsMain(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let uv = vec2f(p.x / u.viewport.x, 1.0 - p.y / u.viewport.y);
  let vignette = 1.0 - 0.28 * smoothstep(0.25, 0.75, distance(uv, vec2f(0.5)));
  return vec4f(pow(max(scene(uv, u.time, u.pointer, u.seed, u.params) * vignette, vec3f(0.0)), vec3f(0.92)), 1.0);
}`;
}

export async function createGpuShowcase(kind, { canvas, onStatus }) {
  if (!SHADERS[kind]) throw new Error(`Unknown GPU preview ${kind}`);
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('The canvas cannot create a WebGPU context');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const module = device.createShaderModule({ code: `${VERTEX}\n${fragment(SHADERS[kind])}` });
  const compilation = await module.getCompilationInfo();
  const shaderErrors = compilation.messages.filter(({ type }) => type === 'error');
  if (shaderErrors.length) {
    throw new Error(`WebGPU shader rejected: ${shaderErrors.map(({ message, lineNum }) => `line ${lineNum}: ${message}`).join('; ')}`);
  }
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vsMain' },
    fragment: { module, entryPoint: 'fsMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const uniforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniforms } }] });
  const defaults = kind === 'tree-wind' ? [0.7, 0.55, 1, 0] : kind === 'sand-drum' ? [1, 1, 0.75, 0] : [1, 1, 1, 0];
  const state = { kind, time: 0, seed: 208287, pointer: [0.5, 0.5], params: defaults, running: false };
  let frame = 0, previous = performance.now(), disposed = false;

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
  }
  function draw(now) {
    frame = 0;
    if (!state.running || disposed) return;
    resize();
    state.time += Math.min((now - previous) / 1000, 0.05); previous = now;
    const data = new Float32Array([canvas.width, canvas.height, state.time, state.seed, ...state.pointer, 0, 0, ...state.params, 0, 0, 0, 0]);
    device.queue.writeBuffer(uniforms, 0, data);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0.02, g: 0.02, b: 0.02, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end();
    device.queue.submit([encoder.finish()]);
    frame = requestAnimationFrame(draw);
  }
  function setActive(active) {
    state.running = Boolean(active);
    previous = performance.now();
    if (state.running && !frame) frame = requestAnimationFrame(draw);
    if (!state.running && frame) { cancelAnimationFrame(frame); frame = 0; }
  }
  function reset() { state.time = 0; state.pointer = [0.5, 0.5]; }
  function pointer(sample) {
    state.pointer = [sample.normalizedX, 1 - sample.normalizedY];
    if (kind === 'stones-water') state.params[3] = sample.phase === 'end' || sample.phase === 'cancel' ? 0 : 1;
  }
  function dispose() { disposed = true; setActive(false); uniforms.destroy(); context.unconfigure(); }
  onStatus?.('Live WebGPU preview · drag to interact');
  return { state, setActive, reset, pointer, dispose };
}
