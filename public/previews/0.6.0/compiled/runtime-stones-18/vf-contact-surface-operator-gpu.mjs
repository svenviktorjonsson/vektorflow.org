// Matrix-free J M^-1 J^T for four-vertex surface contacts. A row can encode
// vertex-face or edge-edge contact with signed barycentric weights summing to 0.
// The same law applies to any indexed deformable surfaces, independent of name.
export const CONTACT_SURFACE_OPERATOR_WGSL=/* wgsl */`
struct Params { bodies:u32, contacts:u32 };
struct Body { position:vec4<f32>, inverse_diagonal:vec4<f32>, inverse_off_diagonal:vec4<f32> };
struct Row { ids:vec4<u32>, weights:vec4<f32>, normal:vec4<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> bodies:array<Body>;
@group(0) @binding(2) var<storage,read> offsets:array<u32>;
@group(0) @binding(3) var<storage,read> incidents:array<u32>;
@group(0) @binding(4) var<storage,read> rows:array<Row>;
@group(0) @binding(5) var<storage,read> direction:array<f32>;
@group(0) @binding(6) var<storage,read_write> changes:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> product:array<f32>;
@group(0) @binding(9) var<storage,read> active_count:array<u32>;
@group(0) @binding(10) var<storage,read_write> dispatch_args:array<u32>;
fn finite(x:f32)->bool{return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn finite3(x:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(x)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn surface_active()->u32{return min(params.contacts,active_count[0]);}
fn slot_for(word:u32)->u32{let tag=word>>30u;return select(select(0u,2u,tag==1u),select(1u,3u,tag==3u),tag>=2u);}
fn write_dispatch(skip_independent:bool){
 if(active_count[0]>params.contacts){atomicOr(&status[0],128u);}
 let enabled=select(1u,0u,skip_independent&&atomicLoad(&status[1])<=1u);
 dispatch_args[0]=enabled*((params.bodies+127u)/128u);dispatch_args[1]=1u;dispatch_args[2]=1u;
 dispatch_args[3]=enabled*((surface_active()+127u)/128u);dispatch_args[4]=1u;dispatch_args[5]=1u;
}
@compute @workgroup_size(1) fn surface_prepare(){write_dispatch(false);}
@compute @workgroup_size(1) fn surface_prepare_iter(){write_dispatch(true);}
@compute @workgroup_size(128) fn surface_gather(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.bodies){return;}
 let inverse_mass=bodies[i].inverse_diagonal.x;
 if(!finite(inverse_mass)||inverse_mass<0.0){atomicOr(&status[0],64u);changes[i*2u]=vec4<f32>(0.0);changes[i*2u+1u]=vec4<f32>(0.0);return;}
 if(inverse_mass==0.0){changes[i*2u]=vec4<f32>(0.0);changes[i*2u+1u]=vec4<f32>(0.0);return;}
 var impulse=vec3<f32>(0.0);
 for(var index=offsets[i];index<offsets[i+1u];index++){
  let word=incidents[index];let e=word&0x3fffffffu;let slot=slot_for(word);
  if(e>=surface_active()){atomicOr(&status[0],4u);continue;}
  let row=rows[e];if(row.ids[slot]!=i){atomicOr(&status[0],4u);continue;}
  let weight=row.weights[slot];if(!finite(weight)||!finite3(row.normal.xyz)||!finite(direction[e])){atomicOr(&status[0],8u);continue;}
  impulse+=row.normal.xyz*(weight*direction[e]);
 }
 let delta=impulse*inverse_mass;if(!finite3(delta)){atomicOr(&status[0],8u);}
 changes[i*2u]=vec4<f32>(delta,0.0);changes[i*2u+1u]=vec4<f32>(0.0);
}
@compute @workgroup_size(128) fn surface_project(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=surface_active()){return;}
 let row=rows[e];let n=row.normal.xyz;
 if(!finite3(n)||abs(dot(n,n)-1.0)>1.0e-4){atomicOr(&status[0],16u);return;}
 var weight_sum=0.0;var value=0.0;
 for(var slot=0u;slot<4u;slot++){
   let i=row.ids[slot];let weight=row.weights[slot];
  if(i>=params.bodies||!finite(weight)){atomicOr(&status[0],1u);return;}
  for(var previous=0u;previous<slot;previous++){
   if(i==row.ids[previous]){atomicOr(&status[0],1u);return;}
  }
  weight_sum+=weight;value+=weight*dot(n,changes[i*2u].xyz);
 }
 if(abs(weight_sum)>1.0e-4||!finite(value)){atomicOr(&status[0],32u);return;}
 product[e]=value;
}
`;

export async function createContactSurfaceOperatorGpu(device,{graph,bodies,contacts,direction,product}){
  const {bodyCount,edgeCount}=graph;
  if(graph.participantsPerRow!==4||graph.edgeStrideWords!==12)
    throw new RangeError('Surface operator requires four-participant 48-byte rows');
  for(const [buffer,required] of [[bodies,bodyCount*48],[contacts,Math.max(48,edgeCount*48)],
      [direction,Math.max(4,edgeCount*4)],[product,Math.max(4,edgeCount*4)]])
    if(buffer.size<required)throw new RangeError('Surface operator buffer capacity is too small');
  const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
  if(bodyCount*48>limit||Math.ceil(bodyCount/128)>device.limits.maxComputeWorkgroupsPerDimension)
    throw new RangeError('Surface operator requires body batching for this GPU');
  const changes=device.createBuffer({size:bodyCount*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const dispatchArgs=device.createBuffer({size:24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
  device.queue.writeBuffer(uniform,0,new Uint32Array([bodyCount,edgeCount,0,0]));
  const module=device.createShaderModule({label:'VKF four-participant surface contact operator',code:CONTACT_SURFACE_OPERATOR_WGSL});
  const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
  if(errors.length)throw Error(errors.map(error=>`${error.lineNum}:${error.linePos} ${error.message}`).join('\n'));
  const buffers=[uniform,bodies,graph.offsets,graph.incidents,contacts,direction,changes,graph.status,product,graph.activeCount,dispatchArgs];
  const types={0:'uniform',1:'read-only-storage',2:'read-only-storage',3:'read-only-storage',4:'read-only-storage',5:'read-only-storage',6:'storage',7:'storage',8:'storage',9:'read-only-storage',10:'storage'};
  const create=(entryPoint,bindings)=>{
    const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:types[binding]}}))});
    const group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
    return device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}}).then(pipeline=>({pipeline,group}));
  };
  const [prepare,prepareIter,gather,project]=await Promise.all([
    create('surface_prepare',[0,7,9,10]),
    create('surface_prepare_iter',[0,7,9,10]),
    create('surface_gather',[0,1,2,3,4,5,6,7,9]),
    create('surface_project',[0,4,6,7,8,9]),
  ]);
  return {changes,encode(encoder,{skipIndependent=false}={}){
    const mode=skipIndependent?prepareIter:prepare;
    const setup=encoder.beginComputePass();setup.setPipeline(mode.pipeline);setup.setBindGroup(0,mode.group);setup.dispatchWorkgroups(1);setup.end();
    const pass=encoder.beginComputePass({label:'VKF four-participant J inverse-mass J-transpose'});
    pass.setPipeline(gather.pipeline);pass.setBindGroup(0,gather.group);pass.dispatchWorkgroupsIndirect(dispatchArgs,0);
    if(edgeCount){pass.setPipeline(project.pipeline);pass.setBindGroup(0,project.group);pass.dispatchWorkgroupsIndirect(dispatchArgs,12);}
    pass.end();
  },destroy(){uniform.destroy();changes.destroy();dispatchArgs.destroy();}};
}
