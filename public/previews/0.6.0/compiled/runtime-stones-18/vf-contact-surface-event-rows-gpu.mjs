import {createContactPrefixScanGpu} from './vf-contact-prefix-scan-gpu.mjs';

// Swept events become signed four-vertex rows for any indexed surface.
export const CONTACT_SURFACE_EVENT_ROWS_WGSL=/* wgsl */`
struct Params { vertices:u32, capacity:u32, dt:f32, pad:f32 };
struct Motion { start:vec4<f32>, finish:vec4<f32> };
struct Pair { ids:vec4<u32>, kind:u32, thickness:f32, pad:vec2<f32> };
struct Verdict { code:u32, first:f32, gap:f32, evaluations:u32 };
struct Row { ids:vec4<u32>, weights:vec4<f32>, normal:vec4<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> motion:array<Motion>;
@group(0) @binding(2) var<storage,read> pairs:array<Pair>;
@group(0) @binding(3) var<storage,read> verdicts:array<Verdict>;
@group(0) @binding(4) var<storage,read> input_count:array<u32>;
@group(0) @binding(5) var<storage,read_write> raw_rows:array<Row>;
@group(0) @binding(6) var<storage,read_write> raw_rhs:array<f32>;
@group(0) @binding(7) var<storage,read_write> flags:array<u32>;
@group(0) @binding(8) var<storage,read> offsets:array<u32>;
@group(0) @binding(9) var<storage,read_write> rows:array<Row>;
@group(0) @binding(10) var<storage,read_write> rhs:array<f32>;
@group(0) @binding(11) var<storage,read_write> output_count:array<u32>;
@group(0) @binding(12) var<storage,read_write> status:array<atomic<u32>>;
fn position(i:u32,t:f32)->vec3<f32>{let m=motion[i];return mix(m.start.xyz,m.finish.xyz,t);}
fn velocity(i:u32)->vec3<f32>{let m=motion[i];return (m.finish.xyz-m.start.xyz)/params.dt;}
fn finite(x:f32)->bool{return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn segment_t(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->f32{
 let d=b-a;return clamp(dot(p-a,d)/max(dot(d,d),1e-30),0.0,1.0);
}
fn face_weights(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->vec3<f32>{
 let ab=b-a;let ac=c-a;let ap=p-a;
 let aa=dot(ab,ab);let bb=dot(ab,ac);let cc=dot(ac,ac);
 let d=aa*cc-bb*bb;
 if(d>1e-24){
  let u=(dot(ap,ab)*cc-dot(ap,ac)*bb)/d;
  let v=(dot(ap,ac)*aa-dot(ap,ab)*bb)/d;
  if(u>=0.0&&v>=0.0&&u+v<=1.0){return vec3<f32>(1.0-u-v,u,v);}
 }
 let ab_t=segment_t(p,a,b);let bc_t=segment_t(p,b,c);let ca_t=segment_t(p,c,a);
 let ab_q=mix(a,b,ab_t);let bc_q=mix(b,c,bc_t);let ca_q=mix(c,a,ca_t);
 let ab_d=dot(p-ab_q,p-ab_q);let bc_d=dot(p-bc_q,p-bc_q);let ca_d=dot(p-ca_q,p-ca_q);
 if(ab_d<=bc_d&&ab_d<=ca_d){return vec3<f32>(1.0-ab_t,ab_t,0.0);}
 if(bc_d<=ca_d){return vec3<f32>(0.0,1.0-bc_t,bc_t);}
 return vec3<f32>(ca_t,0.0,1.0-ca_t);
}
fn edge_parameters(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>)->vec2<f32>{
 let u=b-a;let v=d-c;let w=a-c;
 let uu=dot(u,u);let uv=dot(u,v);let vv=dot(v,v);
 let uw=dot(u,w);let vw=dot(v,w);let determinant=uu*vv-uv*uv;
 var s=0.0;
 if(determinant>max(1e-24,uu*vv*1e-7)){
  s=clamp((uv*vw-vv*uw)/determinant,0.0,1.0);
 }
 var t=clamp((uv*s+vw)/max(vv,1e-30),0.0,1.0);
 s=clamp((uv*t-uw)/max(uu,1e-30),0.0,1.0);
 t=clamp((uv*s+vw)/max(vv,1e-30),0.0,1.0);
 return vec2<f32>(s,t);
}
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.capacity){return;}flags[i]=0u;
 if(i==0u&&input_count[0]>params.capacity){atomicOr(&status[0],32u);}
 if(i>=input_count[0]||input_count[0]>params.capacity){return;}
 let verdict=verdicts[i];if(verdict.code!=1u){return;}
 let pair=pairs[i];if(pair.kind>1u){atomicOr(&status[0],2u);return;}
 if(any(pair.ids>=vec4<u32>(params.vertices))||!finite(verdict.first)){
  atomicOr(&status[0],1u);return;
 }
 let t=clamp(verdict.first,0.0,1.0);
 let p=position(pair.ids.x,t);let a=position(pair.ids.y,t);
 let b=position(pair.ids.z,t);let c=position(pair.ids.w,t);
 var weights=vec4<f32>(0.0);var delta=vec3<f32>(0.0);
 var fallback=vec3<f32>(0.0);
 if(pair.kind==0u){
  let face=cross(b-a,c-a);let area=length(face);
  if(!finite(area)||area<1e-12){atomicOr(&status[0],4u);return;}
  let bary=face_weights(p,a,b,c);
  weights=vec4<f32>(1.0,-bary);
  delta=p-(a*bary.x+b*bary.y+c*bary.z);
  fallback=face/area;
 }else{
  let edge0=a-p;let edge1=c-b;
  if(length(edge0)<1e-12||length(edge1)<1e-12){atomicOr(&status[0],4u);return;}
  let parameters=edge_parameters(p,a,b,c);
  weights=vec4<f32>(1.0-parameters.x,parameters.x,
    -(1.0-parameters.y),-parameters.y);
  delta=mix(p,a,parameters.x)-mix(b,c,parameters.y);
  let axis=cross(edge0,edge1);let magnitude=length(axis);
  if(magnitude>1e-10){fallback=axis/magnitude;}
 }
 let relative=weights.x*velocity(pair.ids.x)+weights.y*velocity(pair.ids.y)+
  weights.z*velocity(pair.ids.z)+weights.w*velocity(pair.ids.w);
 let gap=length(delta);var normal=fallback;
 if(gap>1e-8){normal=delta/gap;}
 else if(length(normal)>0.0){if(dot(normal,relative)>0.0){normal=-normal;}}
 else if(length(relative)>1e-8){normal=-normalize(relative);}
 else{atomicOr(&status[0],4u);return;}
 let closing=-dot(normal,relative);
 if(!finite(closing)||!finite(gap)){atomicOr(&status[0],8u);return;}
 if(closing<=1e-6){return;}
 raw_rows[i]=Row(pair.ids,weights,vec4<f32>(normal,0.0));
 raw_rhs[i]=closing;flags[i]=1u;
}
@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i==0u){output_count[0]=offsets[params.capacity];}
 if(i>=params.capacity||flags[i]==0u){return;}
 let at=offsets[i];if(at>=params.capacity){atomicOr(&status[0],16u);return;}
 rows[at]=raw_rows[i];rhs[at]=raw_rhs[i];
}
`;

export async function createContactSurfaceEventRowsGpu(device,{vertexCount,pairCapacity,
  motion,pairs,verdicts,activeCount,timeStep}){
  if(!Number.isSafeInteger(vertexCount)||vertexCount<4||
    !Number.isSafeInteger(pairCapacity)||pairCapacity<1||
    !Number.isFinite(timeStep)||timeStep<=0||
    !motion||motion.size<vertexCount*32||!pairs||pairs.size<pairCapacity*32||
    !verdicts||verdicts.size<pairCapacity*16||!activeCount||activeCount.size<4)
    throw new RangeError('Surface event rows require bounded swept motion and pairs');
  const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
  if(pairCapacity*48>limit||Math.ceil(pairCapacity/64)>device.limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Surface event rows require GPU batching');
  const owned=[],make=(size,usage)=>{const buffer=device.createBuffer({size,usage});owned.push(buffer);return buffer;};
  const storage=GPUBufferUsage.STORAGE;
  const rawRows=make(pairCapacity*48,storage),rawRhs=make(pairCapacity*4,storage);
  const flags=make(pairCapacity*4,storage),offsets=make((pairCapacity+1)*4,storage);
  const rows=make(pairCapacity*48,storage|GPUBufferUsage.COPY_SRC);
  const rhs=make(pairCapacity*4,storage|GPUBufferUsage.COPY_SRC);
  const outputCount=make(4,storage|GPUBufferUsage.COPY_SRC);
  const status=make(4,storage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  const uniform=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  const data=new ArrayBuffer(16);new Uint32Array(data).set([vertexCount,pairCapacity]);
  new Float32Array(data)[2]=timeStep;device.queue.writeBuffer(uniform,0,data);
  const scan=await createContactPrefixScanGpu(device,{count:pairCapacity,input:flags,output:offsets});
  const module=device.createShaderModule({label:'VKF generic swept surface event rows',code:CONTACT_SURFACE_EVENT_ROWS_WGSL});
  const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
  if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
  const buffers=[uniform,motion,pairs,verdicts,activeCount,rawRows,rawRhs,flags,offsets,rows,rhs,outputCount,status];
  const build=async(entryPoint,bindings)=>{
    const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,
      visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':[1,2,3,4,8].includes(binding)?'read-only-storage':'storage'}}))});
    const group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
    const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
    return {pipeline,group};
  };
  const [prepare,scatter]=await Promise.all([
    build('prepare',[0,1,2,3,4,5,6,7,12]),
    build('scatter',[0,5,6,7,8,9,10,11,12]),
  ]);
  return {rows,rhs,activeCount:outputCount,status,encode(encoder){
    encoder.clearBuffer(status);
    const first=encoder.beginComputePass({label:'VKF swept event row preparation'});
    first.setPipeline(prepare.pipeline);first.setBindGroup(0,prepare.group);
    first.dispatchWorkgroups(Math.ceil(pairCapacity/64));first.end();
    scan.encode(encoder);
    const last=encoder.beginComputePass({label:'VKF compact swept event rows'});
    last.setPipeline(scatter.pipeline);last.setBindGroup(0,scatter.group);
    last.dispatchWorkgroups(Math.ceil(pairCapacity/64));last.end();
  },destroy(){scan.destroy();for(const buffer of owned)buffer.destroy();}};
}
