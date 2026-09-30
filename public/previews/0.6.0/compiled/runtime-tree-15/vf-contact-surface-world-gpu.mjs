import {createContactSurfaceSweepGpu} from './vf-contact-surface-sweep-gpu.mjs';
import {createContactSurfaceBroadphaseGpu} from './vf-contact-surface-broadphase-gpu.mjs';
import {createContactSurfaceResponseGpu} from './vf-contact-surface-response-gpu.mjs';

// Headless, geometry-agnostic World motion ledger. It commits only swept-safe
// surfaces. Its bounded sparse broadphase rejects capacity faults; contact
// impulses remain a separate Law. This is not yet the live renderer's state owner.
export const CONTACT_SURFACE_WORLD_WGSL=/* wgsl */`
struct Params { vertices:u32,capacity:u32,dt:f32,pad:f32 };
struct Motion { start:vec4<f32>,finish:vec4<f32> };
struct Pair { ids:vec4<u32>,kind:u32,thickness:f32,pad:vec2<f32> };
struct Verdict { code:u32,first:f32,gap:f32,evaluations:u32 };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> positions:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> proposed:array<Motion>;
@group(0) @binding(4) var<storage,read> pairs:array<Pair>;
@group(0) @binding(5) var<storage,read> verdicts:array<Verdict>;
@group(0) @binding(6) var<storage,read_write> blocked:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> sweep_status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> active_count:array<u32>;
@group(0) @binding(9) var<storage,read_write> state_status:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read> islands:array<u32>;
@group(0) @binding(11) var<storage,read_write> broadphase_status:array<atomic<u32>>;
@group(0) @binding(12) var<storage,read> targets:array<vec4<f32>>;
@group(0) @binding(13) var<storage,read_write> contact_stop:array<atomic<u32>>;
@group(0) @binding(15) var<storage,read_write> dispatch_args:array<u32>;
fn finite3(x:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(x)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn finite(x:f32)->bool{return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
@compute @workgroup_size(64) fn surface_propose(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.vertices){return;}
 let p=positions[i];let v=velocities[i];
 if(!finite3(p.xyz)||!finite3(v.xyz)||!finite(p.w)||p.w<0.0){atomicOr(&state_status[0],1u);proposed[i]=Motion(vec4<f32>(p.xyz,0.0),vec4<f32>(p.xyz,0.0));return;}
 var next=p.xyz+select(vec3<f32>(0.0),v.xyz*params.dt,p.w>0.0);
 if(params.pad>0.5){next=select(p.xyz,targets[i].xyz,p.w>0.0);}
 if(!finite3(next)){atomicOr(&state_status[0],2u);next=p.xyz;}
 proposed[i]=Motion(vec4<f32>(p.xyz,p.w),vec4<f32>(next,0.0));
}
@compute @workgroup_size(1) fn surface_dispatch(){
 // Keep one workgroup so row zero can report an over-capacity candidate set.
 dispatch_args[0]=max(1u,min((active_count[0]+63u)/64u,(params.capacity+63u)/64u));
 dispatch_args[1]=1u;dispatch_args[2]=1u;
}
@compute @workgroup_size(64) fn surface_mark(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.capacity||i>=active_count[0]){return;}
 if(verdicts[i].code==0u){return;}
 for(var slot=0u;slot<4u;slot++){
  let vertex=pairs[i].ids[slot];if(vertex>=params.vertices){continue;}
  let island=islands[vertex];
  // Both real impact and work-limited prefix stop motion; only impact emits
  // a contact impulse in the separate response Law.
  if(verdicts[i].code==1u||verdicts[i].code==4u){
   // Positive f32 bit patterns preserve ordering. Zero means full-step travel;
   // the maximum stopped fraction gives the earliest certified contact.
   let stopped=1.0-clamp(verdicts[i].first,0.0,1.0);
   atomicMax(&contact_stop[island],bitcast<u32>(stopped));
  }else{atomicStore(&blocked[island],1u);}
 }
}
@compute @workgroup_size(64) fn surface_commit(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.vertices){return;}
 if(i==0u){
  atomicOr(&state_status[1],atomicLoad(&broadphase_status[0]));
  atomicOr(&state_status[2],atomicLoad(&sweep_status[0]));
  atomicOr(&state_status[3],atomicLoad(&state_status[0]));
 }
 // Malformed candidate sets abort the entire proposal. Unresolved motion is
 // rolled back; a certified collision advances only to its earliest event.
 // Remaining motion awaits the common impulse Law and another swept check.
 if(atomicLoad(&state_status[0])!=0u||atomicLoad(&broadphase_status[0])!=0u||
  (atomicLoad(&sweep_status[0])&3u)!=0u||atomicLoad(&blocked[islands[i]])!=0u){return;}
 let travel=1.0-bitcast<f32>(atomicLoad(&contact_stop[islands[i]]));
 positions[i]=vec4<f32>(mix(proposed[i].start.xyz,proposed[i].finish.xyz,travel),positions[i].w);
}
`;

export async function createContactSurfaceWorldGpu(device,{positions,velocities,islandIds,islandCount,pairs,activeCount,
 pairCapacity,dt,maxDepth=18,budget=2048,margin=1e-5,surfaceTopology=null,surfaceThickness=null,
 broadphaseCellSize=0,broadphaseCellSlots=64,broadphaseNodeCapacity=null,
 broadphaseBucketCount=0,targetPositions=null,contactResponse=false,
 responseIterations=32,responseTolerance=1e-4}){
 if(!(positions instanceof Float32Array)||!(velocities instanceof Float32Array)||positions.length!==velocities.length||
    positions.length%4!==0||positions.length<16||!(islandIds instanceof Uint32Array)||islandIds.length!==positions.length/4||
    !Number.isSafeInteger(islandCount)||islandCount<1||!Number.isFinite(dt)||dt<=0)
  throw new RangeError('Surface World requires aligned physical positions, velocities and positive dt');
 if(islandIds.some(id=>id>=islandCount))throw new RangeError('Surface World island identity is outside its declared range');
 for(let i=0;i<positions.length;i++)if(!Number.isFinite(positions[i])||!Number.isFinite(velocities[i])||i%4===3&&positions[i]<0)
  throw new RangeError('Surface World physical state must be finite with nonnegative inverse mass');
 if(surfaceTopology!==null&&(!(surfaceTopology instanceof Uint32Array)||surfaceTopology.length%4!==0||
   surfaceTopology.length<8||!(surfaceThickness instanceof Float32Array)||
   surfaceThickness.length!==surfaceTopology.length/4||
   surfaceThickness.some(value=>!Number.isFinite(value)||value<0)))
  throw new RangeError('Surface World requires triangle topology and finite nonnegative thickness');
 const count=positions.length/4;
 if(targetPositions!==null&&(!Number.isSafeInteger(targetPositions.size)||targetPositions.size<count*16))
  throw new RangeError('Surface World target positions require one GPU vec4 per vertex');
 if(contactResponse&&targetPositions!==null)
  throw new RangeError('Surface contact response requires unconstrained World motion');
 const storage=GPUBufferUsage.STORAGE,owned=[];
 const make=(size,usage)=>{const buffer=device.createBuffer({size,usage});owned.push(buffer);return buffer;};
 const positionBuffer=make(positions.byteLength,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
 const velocityBuffer=make(velocities.byteLength,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
 const islandBuffer=make(islandIds.byteLength,storage|GPUBufferUsage.COPY_DST);
 const proposed=make(count*32,storage|GPUBufferUsage.COPY_SRC);
 const blocked=make(islandCount*4,storage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
 const contactStop=make(islandCount*4,storage|GPUBufferUsage.COPY_DST);
 const stateStatus=make(16,storage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
 const dispatchArgs=make(12,storage|GPUBufferUsage.INDIRECT);
 const emptyBroadphaseStatus=make(4,storage|GPUBufferUsage.COPY_DST);
 const emptyTargets=targetPositions??make(count*16,storage);
 const uniform=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 device.queue.writeBuffer(positionBuffer,0,positions);device.queue.writeBuffer(velocityBuffer,0,velocities);device.queue.writeBuffer(islandBuffer,0,islandIds);
 const data=new ArrayBuffer(16);new Uint32Array(data).set([count,pairCapacity]);
 new Float32Array(data).set([dt,targetPositions===null?0:1],2);device.queue.writeBuffer(uniform,0,data);
 let broadphase=null;
 if(surfaceTopology!==null){
  const topology=make(surfaceTopology.byteLength,storage|GPUBufferUsage.COPY_DST);
  const thickness=make(surfaceThickness.byteLength,storage|GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(topology,0,surfaceTopology);device.queue.writeBuffer(thickness,0,surfaceThickness);
  broadphase=await createContactSurfaceBroadphaseGpu(device,{vertexCount:count,
   triangleCount:surfaceThickness.length,pairCapacity,motion:proposed,topology,thickness,
   cellSize:broadphaseCellSize,cellSlots:broadphaseCellSlots,
   ...(broadphaseNodeCapacity===null?{}:{nodeCapacity:broadphaseNodeCapacity}),
   bucketCount:broadphaseBucketCount,reserve:margin});
 }else if(!pairs||!activeCount)throw new RangeError('Surface World requires generated topology or externally supplied candidates');
 const candidatePairs=broadphase?.pairs??pairs,candidateCount=broadphase?.activeCount??activeCount;
 const candidateStatus=broadphase?.status??emptyBroadphaseStatus;
 const sweep=await createContactSurfaceSweepGpu(device,{vertexCount:count,pairCapacity,vertices:proposed,
  pairs:candidatePairs,activeCount:candidateCount,indirectDispatch:dispatchArgs,maxDepth,budget,margin});
 const response=contactResponse?await createContactSurfaceResponseGpu(device,{vertexCount:count,
  pairCapacity,positions:positionBuffer,velocities:velocityBuffer,motion:proposed,
  pairs:candidatePairs,verdicts:sweep.results,activeCount:candidateCount,
  sweepStatus:sweep.status,broadphaseStatus:candidateStatus,stateStatus,
  timeStep:dt,iterations:responseIterations,tolerance:responseTolerance}):null;
 const module=device.createShaderModule({label:'VKF physical surface World motion',code:CONTACT_SURFACE_WORLD_WGSL});
 const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
 if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
 const buffers=[uniform,positionBuffer,velocityBuffer,proposed,candidatePairs,sweep.results,blocked,
  sweep.status,candidateCount,stateStatus,islandBuffer,candidateStatus,emptyTargets,contactStop,null,dispatchArgs];
 const type=binding=>binding===0?'uniform':[2,4,5,8,10,12].includes(binding)?'read-only-storage':'storage';
 const build=async(entryPoint,bindings)=>{
  const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:type(binding)}}))});
  const group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
  const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
  return {group,pipeline};
 };
 const [propose,prepareDispatch,mark,commit]=await Promise.all([
  build('surface_propose',[0,1,2,3,9,12]),
  build('surface_dispatch',[0,8,15]),
  build('surface_mark',[0,4,5,6,8,10,13]),
  build('surface_commit',[0,1,3,6,7,9,10,11,13]),
 ]);
 const dispatch=(encoder,stage,work)=>{const pass=encoder.beginComputePass();pass.setPipeline(stage.pipeline);pass.setBindGroup(0,stage.group);pass.dispatchWorkgroups(Math.ceil(work/64));pass.end();};
 return {positionBuffer,velocityBuffer,proposed,blocked,sweep,response,encode(encoder){
   encoder.clearBuffer(blocked);encoder.clearBuffer(contactStop);encoder.clearBuffer(stateStatus,0,4);
   if(!broadphase)encoder.clearBuffer(emptyBroadphaseStatus);
   dispatch(encoder,propose,count);broadphase?.encode(encoder);dispatch(encoder,prepareDispatch,1);sweep.encode(encoder);
   const markPass=encoder.beginComputePass();markPass.setPipeline(mark.pipeline);markPass.setBindGroup(0,mark.group);
   markPass.dispatchWorkgroupsIndirect(dispatchArgs,0);markPass.end();
   dispatch(encoder,commit,count);
   response?.encode(encoder);
  },reset(){device.queue.writeBuffer(positionBuffer,0,positions);device.queue.writeBuffer(velocityBuffer,0,velocities);
   device.queue.writeBuffer(stateStatus,0,new Uint32Array(4));},
  async inspect(){const copy=device.createBuffer({size:positions.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(positionBuffer,0,copy,0,positions.byteLength);device.queue.submit([encoder.finish()]);
   await copy.mapAsync(GPUMapMode.READ);const value=new Float32Array(copy.getMappedRange()).slice();copy.unmap();copy.destroy();return value;},
  async inspectVelocities(){const copy=device.createBuffer({size:velocities.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(velocityBuffer,0,copy,0,velocities.byteLength);device.queue.submit([encoder.finish()]);
   await copy.mapAsync(GPUMapMode.READ);const value=new Float32Array(copy.getMappedRange()).slice();copy.unmap();copy.destroy();return value;},
  async inspectStatus(){
   const copy=device.createBuffer({size:28,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();
   for(const [index,buffer] of [candidateStatus,sweep.status,stateStatus,candidateCount].entries())
    encoder.copyBufferToBuffer(buffer,0,copy,index*4,4);
   encoder.copyBufferToBuffer(stateStatus,4,copy,16,12);
   device.queue.submit([encoder.finish()]);await copy.mapAsync(GPUMapMode.READ);
   const values=new Uint32Array(copy.getMappedRange()).slice();copy.unmap();copy.destroy();
   return {broadphaseFault:values[0],sweepFault:values[1],stateFault:values[2],candidates:values[3],
    faultHistory:{broadphase:values[4],sweep:values[5],state:values[6]}};
  },
  async inspectCandidateKinds(){
   const {candidates}=await this.inspectStatus(),count=Math.min(candidates,pairCapacity);
   if(count===0)return {rows:0,mixed:0,moving:0,static:0};
   const copy=device.createBuffer({size:count*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(candidatePairs,0,copy,0,count*32);
   device.queue.submit([encoder.finish()]);await copy.mapAsync(GPUMapMode.READ);
   const ids=new Uint32Array(copy.getMappedRange()),result={rows:count,mixed:0,moving:0,static:0};
   for(let row=0;row<count;row++){
    let moving=0;for(let corner=0;corner<4;corner++)moving+=Number(positions[ids[row*8+corner]*4+3]>0);
    if(moving===4)result.moving++;else if(moving===0)result.static++;else result.mixed++;
   }
   copy.unmap();copy.destroy();return result;
  },
  destroy(){response?.destroy();sweep.destroy();broadphase?.destroy();for(const buffer of owned)buffer.destroy();}};
}
