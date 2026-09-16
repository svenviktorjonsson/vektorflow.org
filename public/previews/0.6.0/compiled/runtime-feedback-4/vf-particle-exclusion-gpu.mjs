// Shared World acceptance gate. Solvers propose state; only certified state is
// committed. Stored IDs are copied as integers, not float/denormal payloads.
export const PARTICLE_EXCLUSION_WGSL = /* wgsl */`
struct ExclusionParams { counts:vec4<u32>, grid:vec4<u32>, minimum_radius:vec4<f32>, wheel:vec4<f32> };
@group(1) @binding(0) var<uniform> exclusion:ExclusionParams;
@group(1) @binding(1) var<storage,read_write> exclusion_candidate:array<vec4<u32>>;
@group(1) @binding(2) var<storage,read> exclusion_snapshot:array<vec4<u32>>;
@group(1) @binding(3) var<storage,read_write> exclusion_cells:array<atomic<u32>>;
@group(1) @binding(4) var<storage,read_write> exclusion_items:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read_write> exclusion_receipt:array<atomic<u32>>;
@group(1) @binding(6) var<storage,read> exclusion_frame_snapshot:array<vec4<u32>>;
fn exclusion_point(i:u32,prior:bool)->vec2<f32>{let index=i*exclusion.counts.y;if(prior){return bitcast<vec4<f32>>(exclusion_snapshot[index]).xy;}return bitcast<vec4<f32>>(exclusion_candidate[index]).xy;}
fn exclusion_bounds(i:u32)->vec4<i32>{
  let a=exclusion_point(i,true);let b=exclusion_point(i,false);let r=exclusion.minimum_radius.z;
  return vec4<i32>(vec2<i32>(floor((min(a,b)-r-exclusion.minimum_radius.xy)/(2.0*r))),vec2<i32>(floor((max(a,b)+r-exclusion.minimum_radius.xy)/(2.0*r))));
}
fn reject_exclusion(reason:u32){atomicStore(&exclusion_receipt[0],1u);atomicAdd(&exclusion_receipt[reason],1u);}
@compute @workgroup_size(128)
fn exclusion_clear(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<exclusion.grid.x*exclusion.grid.y){atomicStore(&exclusion_cells[i],0u);}}
@compute @workgroup_size(128)
fn exclusion_fill(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=exclusion.counts.x){return;}
  let bits=exclusion_candidate[i*exclusion.counts.y];if(any((bits&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))){reject_exclusion(3u);return;}
  let b=exclusion_bounds(i);let size=b.zw-b.xy+vec2<i32>(1);
  // Refuse an uncertifiable large proposal rather than truncate its broadphase.
  if(any(b.xy<vec2<i32>(0))||any(b.zw>=vec2<i32>(exclusion.grid.xy))||size.x*size.y>256){reject_exclusion(3u);return;}
  for(var y=b.y;y<=b.w;y++){for(var x=b.x;x<=b.z;x++){let cell=u32(x)+exclusion.grid.x*u32(y);let slot=atomicAdd(&exclusion_cells[cell],1u);
    if(slot>=exclusion.grid.z){reject_exclusion(3u);}else{atomicStore(&exclusion_items[cell*exclusion.grid.z+slot],i);}
  }}
}
@compute @workgroup_size(128)
fn exclusion_audit(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=exclusion.counts.x||atomicLoad(&exclusion_receipt[3])>0u){return;}let p=exclusion_point(i,false);let r=exclusion.minimum_radius.z;let contact=r+WHEEL_BAR_HALF_WIDTH;
  if(length(p-exclusion.wheel.xy)>WHEEL_RADIUS-contact+1.0e-7){reject_exclusion(2u);}
  for(var segment=0u;segment < 7u;segment++){let local=baffle(segment);let a=exclusion.wheel.xy+rotate_local(local.xy,exclusion.wheel.z);let b=exclusion.wheel.xy+rotate_local(local.zw,exclusion.wheel.z);let edge=b-a;
    let closest=a+edge*clamp(dot(p-a,edge)/max(dot(edge,edge),1.0e-12),0.0,1.0);if(length(p-closest)<contact-1.0e-7){reject_exclusion(2u);}
  }
  let bounds=exclusion_bounds(i);let before=exclusion_point(i,true);let delta=p-before;
  for(var y=bounds.y;y<=bounds.w;y++){for(var x=bounds.x;x<=bounds.z;x++){let cell=u32(x)+exclusion.grid.x*u32(y);let count=min(atomicLoad(&exclusion_cells[cell]),exclusion.grid.z);
    for(var slot=0u;slot<count;slot++){let j=atomicLoad(&exclusion_items[cell*exclusion.grid.z+slot]);if(j<=i){continue;}let other_before=exclusion_point(j,true);let separation=before-other_before;let relative=delta-(exclusion_point(j,false)-other_before);
      let t=clamp(-dot(separation,relative)/max(dot(relative,relative),1.0e-20),0.0,1.0);let closest=separation+relative*t;
      if(dot(closest,closest)<4.0*r*r){reject_exclusion(1u);}
    }
  }}
}
@compute @workgroup_size(128)
fn exclusion_restore(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(atomicLoad(&exclusion_receipt[0])!=0u&&i<exclusion.counts.x*exclusion.counts.y){exclusion_candidate[i]=exclusion_snapshot[i];}}
@compute @workgroup_size(128)
fn exclusion_restore_frame(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(atomicLoad(&exclusion_receipt[0])!=0u&&i<exclusion.counts.x*exclusion.counts.y){exclusion_candidate[i]=exclusion_frame_snapshot[i];}}
`;

export async function createParticleExclusionGpu(device,world,physics,shaderSource){
  const radius=world.kind==='granular'?physics.policy.grainRadius:physics.policy.particleSpacing*0.46;
  const count=physics.seed?.count??physics.count??physics.initialState?.count;
  if(!Number.isSafeInteger(count)||count<1)throw new Error('Particle exclusion requires the authored entity count');
  const stride=world.kind==='granular'?2:3,size=count*stride*16,minimum=world.geometry.center.map(x=>x-world.geometry.radius-radius*4),side=Math.ceil((world.geometry.radius*2+radius*8)/(radius*2));
  const make=(size,usage)=>device.createBuffer({size,usage});
  const snapshot=make(size,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),cells=make(side*side*4,GPUBufferUsage.STORAGE),items=make(side*side*64*4,GPUBufferUsage.STORAGE),receipt=make(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),uniform=make(64,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),readback=make(16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const frameSnapshot=make(size,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2||binding===6?'read-only-storage':'storage'}}))]});
  const empty=device.createBindGroupLayout({entries:[]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[empty,layout]}),module=device.createShaderModule({code:shaderSource});
  const pipelines=Object.fromEntries(await Promise.all(['exclusion_clear','exclusion_fill','exclusion_audit','exclusion_restore','exclusion_restore_frame'].map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
  const group=device.createBindGroup({layout,entries:[uniform,physics.particleBuffer,snapshot,cells,items,receipt,frameSnapshot].map((buffer,binding)=>({binding,resource:{buffer}}))}),emptyGroup=device.createBindGroup({layout:empty,entries:[]});
  const params=new ArrayBuffer(64),u=new Uint32Array(params),f=new Float32Array(params);u.set([count,stride,0,0,side,side,64,0]);f.set([...minimum,radius,0],8);
  return {
    beginFrame(encoder){device.queue.writeBuffer(receipt,0,new Uint32Array(4));encoder.copyBufferToBuffer(physics.particleBuffer,0,frameSnapshot,0,size);},
    begin(encoder){encoder.copyBufferToBuffer(physics.particleBuffer,0,snapshot,0,size);},
    audit(encoder,angle){f.set([...world.geometry.center,angle,0],12);device.queue.writeBuffer(uniform,0,params);const pass=encoder.beginComputePass({label:'VKF hard particle exclusion acceptance'});pass.setBindGroup(0,emptyGroup);pass.setBindGroup(1,group);
      for(const [name,n] of [['exclusion_clear',side*side],['exclusion_fill',count],['exclusion_audit',count],['exclusion_restore',count*stride]]){pass.setPipeline(pipelines[name]);pass.dispatchWorkgroups(Math.ceil(n/128));}pass.end();encoder.copyBufferToBuffer(receipt,0,readback,0,16);
    },
    finishFrame(encoder){const pass=encoder.beginComputePass({label:'VKF transactional frame rollback'});pass.setBindGroup(0,emptyGroup);pass.setBindGroup(1,group);pass.setPipeline(pipelines.exclusion_restore_frame);pass.dispatchWorkgroups(Math.ceil(count*stride/128));pass.end();},
    async accepted(){await readback.mapAsync(GPUMapMode.READ);const words=new Uint32Array(readback.getMappedRange()).slice();readback.unmap();if(words[0])throw new Error(`Non-overlap invariant rejected this step; last valid state retained. Swept particle contacts: ${words[1]}, boundary contacts: ${words[2]}, uncertifiable broadphase: ${words[3]}.`);},
    destroy(){for(const buffer of [snapshot,frameSnapshot,cells,items,receipt,uniform,readback])buffer.destroy();},
  };
}
