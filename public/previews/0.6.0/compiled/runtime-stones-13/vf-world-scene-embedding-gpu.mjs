// View embedding only. Positions come directly from the World GPU buffers.
import {ELASTIC_BLADE_POSE_WGSL} from './vf-elastic-blade-pose-wgsl.mjs';
export const WORLD_SCENE_WGSL = /* wgsl */`
struct Scene { vp:mat4x4<f32>, light_vp:mat4x4<f32>, eye:vec4<f32>, sun:vec4<f32>, domain_min:vec4<f32>, domain_span:vec4<f32>, grid:vec4<u32>, flags:vec4<f32>, right:vec4<f32>, up:vec4<f32> };
struct Body { p:vec4<f32>, v:vec4<f32>, q:vec4<f32>, w:vec4<f32>, info:vec4<f32> };
struct Parcel { p:vec4<f32>, v:vec4<f32> };
struct Node { d:vec4<f32>, v:vec4<f32> };
@group(0) @binding(0) var<uniform> scene:Scene;
@group(0) @binding(1) var<storage,read> bodies:array<Body>;
@group(0) @binding(2) var<storage,read> parcels:array<Parcel>;
@group(0) @binding(3) var<storage,read> nodes:array<Node>;
@group(0) @binding(4) var shadow_texture:texture_depth_2d;
@group(0) @binding(5) var shadow_sampler:sampler_comparison;
struct Vertex { @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) color:vec4<f32>, @location(3) tag:f32, @location(4) compliance:f32, @location(5) emission:vec3<f32>, @location(6) anchor:vec3<f32>, @location(7) leaf:f32, @location(8) hinge:vec3<f32>, @location(9) length:f32, @location(10) category:f32 };
struct Out { @builtin(position) clip:vec4<f32>, @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) color:vec4<f32>, @location(3) emission:vec3<f32>, @location(4) local:vec3<f32>, @location(5) @interpolate(flat) stone:f32, @location(6) uv:vec2<f32>, @location(7) @interpolate(flat) category:f32 };
fn rotate(q:vec4<f32>,p:vec3<f32>)->vec3<f32>{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
fn mode_displacement(p:vec3<f32>,mode:u32)->vec3<f32>{
  let c=clamp((p-scene.domain_min.xyz)/scene.domain_span.xyz*vec3<f32>(scene.grid.xyz)-0.5,vec3<f32>(0.0),vec3<f32>(scene.grid.xyz)-1.001);
  let low=vec3<u32>(floor(c));let t=fract(c);var d=vec3<f32>(0.0);
  for(var z=0u;z<2u;z++){for(var y=0u;y<2u;y++){for(var x=0u;x<2u;x++){
    let at=low+vec3<u32>(x,y,z);let w=select(1.0-t,t,vec3<bool>(x==1u,y==1u,z==1u));
    d+=nodes[mode*scene.grid.x*scene.grid.y*scene.grid.z+at.x+scene.grid.x*(at.y+scene.grid.y*at.z)].d.xyz*w.x*w.y*w.z;
  }}}return d;
}
fn displacement(p:vec3<f32>)->vec3<f32>{return mode_displacement(p,0u);}
fn hinge_rotate(p:vec3<f32>,axis:vec3<f32>,angle:f32)->vec3<f32>{return p*cos(angle)+cross(axis,p)*sin(angle)+axis*dot(axis,p)*(1.0-cos(angle));}
fn posed(v:Vertex)->Out {
  var p=v.p;var n=v.n;
  if(scene.flags.x<0.5&&v.tag>=0.0){let b=bodies[u32(v.tag)];p=b.p.xyz+rotate(b.q,p);n=rotate(b.q,n);}
  if(scene.flags.x>0.5&&v.compliance>0.0){
    if(v.leaf>0.5){let offset=p-v.anchor;let angle=dot(mode_displacement(v.anchor,1u),v.n);
      let blade=elastic_blade_pose(v.p,v.n,v.anchor,v.hinge,v.length,angle,displacement(v.anchor)*v.compliance);p=blade.p;n=blade.n;
    }else{p+=displacement(p)*v.compliance;}
  }
  if(scene.flags.x<0.5&&v.category<0.5&&length(v.emission)>0.0){p+=scene.sun.xyz-scene.domain_min.xyz;}
  return Out(scene.vp*vec4<f32>(p,1.0),p,n,v.color,select(vec3<f32>(0.0),v.emission,v.category<0.5),v.p,select(-1.0,v.tag,scene.flags.x<0.5),v.emission.xy,v.category);
}
@vertex fn mesh_vertex(v:Vertex)->Out {return posed(v);}
@vertex fn shadow_vertex(v:Vertex)->@builtin(position) vec4<f32>{return scene.light_vp*vec4<f32>(posed(v).p,1.0);}
fn hash(x:u32)->u32{var v=x;v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn unit(x:u32)->f32{return f32(hash(x)&0xffffffu)/16777216.0;}
fn mineral_hash(p:vec3<i32>,seed:u32)->u32{return hash((bitcast<u32>(p.x)*73856093u)^(bitcast<u32>(p.y)*19349663u)^(bitcast<u32>(p.z)*83492791u)^seed);}
fn mineral_noise(p:vec3<f32>,seed:u32)->f32{
  let cell=vec3<i32>(floor(p));let t=fract(p);let w=t*t*(3.0-2.0*t);var value=0.0;
  for(var z=0;z<2;z++){for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    let k=vec3<i32>(x,y,z);let weight=select(1.0-w,w,k==vec3<i32>(1));
    value+=unit(mineral_hash(cell+k,seed))*weight.x*weight.y*weight.z;
  }}}return value;
}
// Crystal colour lives in object-space, not latitude/UV or screen coordinates.
// Identical coordinates on either side of any mesh seam sample the same field.
fn granite(p:vec3<f32>,tint:vec3<f32>,seed:u32,kind:u32,footprint:f32)->vec4<f32>{
  // Continuous, object-space mineral fields preserve the old specimen's
  // mottling without hard Voronoi patches or screen-space animated noise.
  let coarse=mineral_noise(p*9.0,seed+43u);
  let weather=mineral_noise(p*31.0,seed+59u);
  let mineral=mineral_noise(p*93.0,seed+71u);
  let fine=mineral_noise(p*720.0,seed+91u);
  let quartz=smoothstep(0.57,0.76,mineral);
  let mica=1.0-smoothstep(0.22,0.41,mineral);
  var color=tint*(0.74+coarse*0.35+weather*0.19);
  color=mix(color,vec3<f32>(0.67,0.66,0.62),quartz*0.15);
  color=mix(color,vec3<f32>(0.16,0.16,0.15),mica*0.09);
  var roughness=0.82+0.09*(1.0-weather);
  if(kind==1u){
    let pore=1.0-smoothstep(0.18,0.34,weather);
    color=mix(vec3<f32>(0.11,0.125,0.14)*(0.78+coarse*0.4),vec3<f32>(0.38,0.39,0.38),quartz*0.17);
    color*=1.0-pore*0.23;roughness=0.89;
  }else if(kind==2u){
    let feldspar=vec3<f32>(0.53,0.31,0.27)*(0.75+coarse*0.44+weather*0.12);
    color=mix(feldspar,vec3<f32>(0.69,0.63,0.57),quartz*0.27);
    color=mix(color,vec3<f32>(0.15,0.15,0.15),mica*0.25);roughness=0.85;
  }else if(kind==3u){
    let vein=1.0-smoothstep(0.015,0.09,abs(coarse-0.51));
    color=mix(vec3<f32>(0.62,0.61,0.56)*(0.75+weather*0.36),vec3<f32>(0.75,0.73,0.66),vein*0.25);roughness=0.79;
  }else if(kind==4u){
    let band=0.5+0.5*sin(dot(p,vec3<f32>(7.5,3.2,5.1))+(coarse-0.5)*3.2);
    let light=smoothstep(0.37,0.70,band);
    color=mix(vec3<f32>(0.23,0.25,0.22),vec3<f32>(0.49,0.50,0.46),light)*(0.8+weather*0.3);roughness=0.87;
  }
  let micro=1.0-smoothstep(0.25,0.56,fine);
  let grainContrast=1.0-smoothstep(0.002,0.015,footprint);
  color=mix(tint*(0.84+coarse*0.25),color*(0.89+fine*0.22)*(1.0-micro*0.11),grainContrast);
  let darkDots=1.0-smoothstep(0.23,0.34,fine);
  let lightDots=smoothstep(0.69,0.81,fine);
  color*=1.0-grainContrast*darkDots*0.15+grainContrast*lightDots*0.04;
  return vec4<f32>(clamp(color,vec3<f32>(0.025),vec3<f32>(0.84)),roughness);
}
fn grass_pose(vertex:u32,id:u32)->Out {
  let u=unit(id*11u);let v=unit(id*29u);let angle=unit(id*31u)*6.283185307;
  let root=scene.domain_min.xyz+vec3<f32>(u*scene.domain_span.x,v*scene.domain_span.y,0.0);
  let height=0.11+unit(id*37u)*0.09;let width=0.008+unit(id*43u)*0.009;
  let corners=array<vec2<f32>,6>(vec2<f32>(-1.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(-1.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(0.0,0.65));
  let local=corners[vertex];let direction=vec3<f32>(cos(angle),sin(angle),0.0);
  let bend=displacement(root+vec3<f32>(0.0,0.0,0.55));
  let offset=vec3<f32>(0.0,0.0,height*local.y)+bend*local.y*local.y*0.32;
  let p=root+direction*local.x*width+offset/max(length(offset),1.0e-8)*height*local.y;
  let n=normalize(cross(direction,vec3<f32>(bend.xy*0.32,height)));
  let color=mix(vec3<f32>(0.075,0.19,0.025),vec3<f32>(0.24,0.43,0.075),unit(id*53u))*mix(0.6,1.15,local.y);
  return Out(scene.vp*vec4<f32>(p,1.0),p,n,vec4<f32>(color,1.0),vec3<f32>(0.0),p,-1.0,vec2<f32>(0.0),0.0);
}
@vertex fn grass_vertex(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->Out{return grass_pose(v,i);}
@vertex fn grass_shadow(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->@builtin(position) vec4<f32>{return scene.light_vp*vec4<f32>(grass_pose(v,i).p,1.0);}
@fragment fn material_fragment(v:Out)->@location(0) vec4<f32>{
  var n=normalize(v.n);var albedo=v.color.rgb;var roughness=0.90;
  let l=normalize(scene.sun.xyz-v.p);var microOcclusion=1.0;var cavityAmbient=1.0;
  let vein_width=max(0.008,fwidth(v.uv.x));
  if(v.category>1.5){
    // One tree-space field spans the whole woody network. No primitive ID,
    // bottom-trunk cutoff or texture-resolution boundary changes the material.
    let coarse=mineral_noise(v.local*vec3<f32>(18.0,18.0,2.8),731u);
    let fine=mineral_noise(v.local*vec3<f32>(120.0,120.0,24.0),947u);
    let micro=mineral_noise(v.local*vec3<f32>(310.0,310.0,75.0),173u);
    let cracks=1.0-smoothstep(0.018,0.095,abs(fine-0.50));
    let age=smoothstep(0.0,8.0,v.local.z);
    albedo=mix(vec3<f32>(0.27,0.17,0.09),vec3<f32>(0.32,0.22,0.13),age)*(0.72+coarse*0.45+micro*0.12-cracks*0.18);
    roughness=0.93;
  }
  if(v.category>0.5&&v.category<1.5){
    let lateral=abs(v.uv.x-0.5)*2.0;let t=v.uv.y;
    let midrib=1.0-smoothstep(vein_width,vein_width+0.028,abs(v.uv.x-0.5));
    let phase=fract((t-0.16)*7.0-lateral*0.22-lateral*lateral*0.13);
    let secondary=(1.0-smoothstep(0.025,0.085,min(phase,1.0-phase)))*smoothstep(0.08,0.25,lateral)*(1.0-smoothstep(0.82,1.0,lateral))*smoothstep(0.16,0.23,t);
    albedo*=1.0-midrib*0.15-secondary*0.10;
    roughness=0.78;
  }
  if(scene.flags.x<0.5){
    let stone=v.stone>=0.0;let seed=hash(u32(max(v.stone,0.0))+8187u);
    let footprint=max(length(dpdx(v.local)),length(dpdy(v.local)));
    let material=granite(v.local,albedo,seed,u32(max(v.stone,0.0)),footprint);albedo=select(albedo,material.rgb,stone);roughness=select(roughness,material.a,stone);
    // Fine, filtered indent relief; larger stone markings remain albedo only.
    // The relief is object-space, so it follows the rock under the light.
    let pitVisibility=1.0-smoothstep(0.004,0.016,footprint);
    let fineVisibility=1.0-smoothstep(0.002,0.007,footprint);
    let coarseRelief=mineral_noise(v.local*185.0,seed);
    let reliefField=mix(coarseRelief,coarseRelief*0.72
      +mineral_noise(v.local*370.0,seed+29u)*0.28,fineVisibility);
    let pit=1.0-smoothstep(0.25,0.40,reliefField);
    let relief=(reliefField-0.5)*mix(0.0018,0.0028,unit(seed+131u))*pitVisibility;
    let dx=dpdx(v.p);let dy=dpdy(v.p);let determinant=dot(dx,cross(dy,n));
    let gradient=(cross(dy,n)*dpdx(relief)+cross(n,dx)*dpdy(relief))/select(1.0,determinant,abs(determinant)>1.0e-10);
    n=select(n,normalize(n-clamp(gradient,vec3<f32>(-0.3),vec3<f32>(0.3))),stone);
    // Recesses have modest ambient occlusion; their light-facing rim casts
    // the much stronger directional shade into the pit.
    cavityAmbient=select(1.0,1.0-0.06*pit*pitVisibility,stone);
    microOcclusion=select(1.0,1.0-0.62*pit*pitVisibility
      *smoothstep(-0.02,0.04,dot(gradient,l)),stone);
  }
  let projected=scene.light_vp*vec4<f32>(v.p+n*0.002,1.0);
  let ndc=projected.xyz/projected.w;let uv=ndc.xy*vec2<f32>(0.5,-0.5)+0.5;
  var shadow=1.0;
  if(all(uv>=vec2<f32>(0.0))&&all(uv<=vec2<f32>(1.0))&&ndc.z>=0.0&&ndc.z<=1.0){
    let texel=1.0/vec2<f32>(textureDimensions(shadow_texture));shadow=0.0;
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){shadow+=textureSampleCompareLevel(shadow_texture,shadow_sampler,uv+vec2<f32>(f32(x),f32(y))*texel,ndc.z-0.00008)/9.0;}}
  }
  // The shadow map gives the light-cast shape. A short-range ground contact
  // term anchors any rigid body whose collision envelope touches the floor.
  var contactOcclusion=0.0;
  if(scene.flags.x<0.5&&v.stone<0.0&&v.category<0.5&&v.n.z>0.9&&length(v.emission)<0.001){
    for(var body=0u;body<arrayLength(&bodies);body++){
      let b=bodies[body];let radius=max(b.info.z,0.01);
      let planar=v.p.xy-b.p.xy;
      let footprint=1.0-smoothstep(0.72,1.35,dot(planar,planar)/(radius*radius));
      let gap=max(0.0,b.p.z-radius-v.p.z);
      let grounded=1.0-smoothstep(0.015,0.22,gap/radius);
      contactOcclusion=max(contactOcclusion,footprint*grounded);
    }
  }
  let leaf=v.category>0.5&&v.category<1.5;
  // A thin blade receives transmitted light on its back as well as reflection.
  let diffuse=select(max(0.0,dot(n,l)),max(0.0,dot(n,l))+max(0.0,-dot(n,l))*0.35,leaf);let view=normalize(scene.eye.xyz-v.p);let sheen=pow(max(0.0,dot(n,normalize(l+view))),mix(90.0,16.0,roughness))*mix(0.13,0.025,roughness);
  let irradiance=scene.sun.w/max(dot(scene.sun.xyz-v.p,scene.sun.xyz-v.p),0.01);
  let linear=max(vec3<f32>(0.0),(v.emission+albedo*(vec3<f32>(0.09)*cavityAmbient+scene.flags.yzw*irradiance*diffuse*shadow*microOcclusion/3.141592654)+vec3<f32>(sheen*shadow))*(1.0-0.58*contactOcclusion));
  let display=select(linear*12.92,1.055*pow(linear,vec3<f32>(1.0/2.4))-0.055,linear>vec3<f32>(0.0031308));
  return vec4<f32>(display,1.0);
}
struct ParcelOut {@builtin(position) clip:vec4<f32>,@location(0) local:vec2<f32>,@location(1) alpha:f32};
@vertex fn parcel_vertex(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->ParcelOut{
  let corners=array<vec2<f32>,4>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),vec2<f32>(1.0,1.0));let q=corners[v];let p=parcels[i];
  let location=p.p.xyz+(scene.right.xyz*q.x+scene.up.xyz*q.y)*0.0035;
  var clip=scene.vp*vec4<f32>(location,1.0);if(p.p.w<=0.0){clip=vec4<f32>(2.0,2.0,2.0,1.0);}
  return ParcelOut(clip,q,0.5*clamp(p.p.w/0.25,0.0,1.0));
}
@fragment fn parcel_fragment(v:ParcelOut)->@location(0) vec4<f32>{if(dot(v.local,v.local)>1.0){discard;}return vec4<f32>(0.78,0.91,0.96,v.alpha);}
` + ELASTIC_BLADE_POSE_WGSL;

export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const normalize=a=>a.map(v=>v/Math.max(1e-12,Math.hypot(...a)));
const multiply=(a,b)=>Array.from({length:16},(_,i)=>{const r=i%4,c=Math.floor(i/4);return [0,1,2,3].reduce((s,k)=>s+a[k*4+r]*b[c*4+k],0);});
function view(eye,target){const z=normalize(sub(eye,target)),x=normalize(cross([0,0,1],z)),y=cross(z,x);return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];}
function perspective(fov,aspect){const f=1/Math.tan(fov*Math.PI/360),near=.03,far=2000;return [f/aspect,0,0,0,0,f,0,0,0,0,far/(near-far),-1,0,0,far*near/(near-far),0];}
export function shadowProjection(distance,extent=12){const near=Math.max(0.03,distance-extent*2),far=distance+extent*2;return [1/extent,0,0,0,0,1/extent,0,0,0,0,1/(near-far),0,0,0,near/(near-far),1];}
export function projectPoint(p,camera,rect){const f=normalize(sub(camera.target,camera.pos)),r=normalize(cross(f,[0,0,1])),u=cross(r,f),d=sub(p,camera.pos),depth=dot(d,f),half=Math.tan(camera.fov*Math.PI/360);return {x:rect.left+(dot(d,r)/(depth*half*rect.width/rect.height)+1)*rect.width/2,y:rect.top+(1-dot(d,u)/(depth*half))*rect.height/2,depth};}
export async function createWorldSceneEmbeddingGpu(device,canvas,world,physics,meshes,shaderSource){
  const format=navigator.gpu.getPreferredCanvasFormat(),context=canvas.getContext('webgpu');context.configure({device,format,alphaMode:'opaque'});
  const uniform=device.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const shadow=device.createTexture({size:{width:2048,height:2048,depthOrArrayLayers:1},format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},...[1,2,3].map(binding=>({binding,visibility:binding===1?GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}})),{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'depth'}},{binding:5,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'comparison'}}]});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniform}},...[1,2,3].map(binding=>({binding,resource:{buffer:physics.buffers[binding-1]}})),{binding:4,resource:shadow.createView()},{binding:5,resource:device.createSampler({compare:'less-equal',magFilter:'linear',minFilter:'linear'})}]});
  // Shadow pipelines must not bind the texture being written as a sampled resource.
  const shadowLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}},...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}}))]});
  const shadowGroup=device.createBindGroup({layout:shadowLayout,entries:[{binding:0,resource:{buffer:uniform}},...[1,2,3].map(binding=>({binding,resource:{buffer:physics.buffers[binding-1]}}))]});
  const module=device.createShaderModule({label:'Compiled World scene embedding',code:shaderSource});
  const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
  if(errors.length)throw new Error(errors.map(error=>`embedding WGSL ${error.lineNum}:${error.linePos}: ${error.message}`).join('\n'));
  const attributes=[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'},{shaderLocation:2,offset:24,format:'float32x4'},{shaderLocation:3,offset:40,format:'float32'},{shaderLocation:4,offset:44,format:'float32'},{shaderLocation:5,offset:48,format:'float32x3'},{shaderLocation:6,offset:64,format:'float32x3'},{shaderLocation:7,offset:76,format:'float32'},{shaderLocation:8,offset:80,format:'float32x3'},{shaderLocation:9,offset:92,format:'float32'},{shaderLocation:10,offset:60,format:'float32'}];
  const make=(entry,shadowPass=false,particles=false)=>device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[shadowPass?shadowLayout:layout]}),vertex:{module,entryPoint:entry,buffers:entry.startsWith('mesh')||entry==='shadow_vertex'?[{arrayStride:96,attributes}]:[]},...(shadowPass?{}:{fragment:{module,entryPoint:particles?'parcel_fragment':'material_fragment',targets:[{format,...(particles?{blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}:{})}]}}),primitive:{topology:particles?'triangle-strip':'triangle-list',cullMode:'none'},depthStencil:{format:'depth32float',depthWriteEnabled:!particles,depthCompare:'less-equal',...(shadowPass?{depthBias:2,depthBiasSlopeScale:2}: {})}});
  const [meshPipeline,grassPipeline,particlePipeline,meshShadow,grassShadow]=await Promise.all([make('mesh_vertex'),make('grass_vertex'),make('parcel_vertex',false,true),make('shadow_vertex',true),make('grass_shadow',true)]);
  const packets=meshes.map(mesh=>{const vb=device.createBuffer({size:mesh.vertices.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),ib=device.createBuffer({size:mesh.indices.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(vb,0,mesh.vertices);device.queue.writeBuffer(ib,0,mesh.indices);return {vb,ib,count:mesh.indices.length,shadow:mesh.shadow!==false};});
  let depth=null,width=0,height=0;const bytes=new ArrayBuffer(256),f=new Float32Array(bytes),u=new Uint32Array(bytes);const p=world.properties;
  return {render(encoder,camera,{grass=true,particles=false}={}){
    const rect=canvas.getBoundingClientRect(),w=Math.max(1,Math.round(rect.width*Math.min(1.5,devicePixelRatio||1))),h=Math.max(1,Math.round(rect.height*Math.min(1.5,devicePixelRatio||1)));
    if(w!==width||h!==height){width=w;height=h;canvas.width=w;canvas.height=h;depth?.destroy();depth=device.createTexture({size:{width:w,height:h,depthOrArrayLayers:1},format:'depth32float',usage:GPUTextureUsage.RENDER_ATTACHMENT});}
    const light=p.lights?.[0]??{position:[-4,-5,2.8],radius:1,radiance:38,color:[1,1,1]};
    const pivot=light.orbitTarget??[0,0,0],angle=(light.orbitSpeed??0)*physics.time,c=Math.cos(angle),s=Math.sin(angle),dx=light.position[0]-pivot[0],dy=light.position[1]-pivot[1];
    const sun=[pivot[0]+c*dx-s*dy,pivot[1]+s*dx+c*dy,light.position[2]],target=world.kind==='wind'?[0,0,3]:[0,0,.7];
    f.set(multiply(perspective(camera.fov,w/h),view(camera.pos,camera.target)),0);f.set(multiply(shadowProjection(Math.hypot(...sub(sun,target))),view(sun,target)),16);
    f.set([...camera.pos,physics.time],32);f.set([...sun,Math.PI*light.radius**2*light.radiance],36);f.set([...(world.kind==='wind'?p.domain_min??[-6,-6,0]:light.position),0],40);f.set([...(p.domain_span??[12,12,9]),0],44);u.set([...(p.grid??[12,12,18]),0],48);f.set([world.kind==='wind'?1:0,...light.color],52);
    const forward=normalize(sub(camera.target,camera.pos)),right=normalize(cross(forward,[0,0,1]));f.set([...right,0],56);f.set([...cross(right,forward),0],60);device.queue.writeBuffer(uniform,0,bytes);
    const shadowPass=encoder.beginRenderPass({colorAttachments:[],depthStencilAttachment:{view:shadow.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}});shadowPass.setBindGroup(0,shadowGroup);shadowPass.setPipeline(meshShadow);
    for(const packet of packets)if(packet.shadow){shadowPass.setVertexBuffer(0,packet.vb);shadowPass.setIndexBuffer(packet.ib,'uint32');shadowPass.drawIndexed(packet.count);}if(grass&&world.kind==='wind'){shadowPass.setPipeline(grassShadow);shadowPass.draw(6,p.grass_count??81920);}shadowPass.end();
    const pass=encoder.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),clearValue:world.kind==='wind'?{r:.12,g:.22,b:.30,a:1}:{r:.09,g:.11,b:.12,a:1},loadOp:'clear',storeOp:'store'}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}});pass.setBindGroup(0,group);pass.setPipeline(meshPipeline);for(const packet of packets){pass.setVertexBuffer(0,packet.vb);pass.setIndexBuffer(packet.ib,'uint32');pass.drawIndexed(packet.count);}if(grass&&world.kind==='wind'){pass.setPipeline(grassPipeline);pass.draw(6,p.grass_count??81920);}if(particles&&world.kind==='wind'){pass.setPipeline(particlePipeline);pass.draw(4,physics.initial.parcelCount);}pass.end();
  },destroy(){depth?.destroy();shadow.destroy();uniform.destroy();for(const packet of packets){packet.vb.destroy();packet.ib.destroy();}}};
}
