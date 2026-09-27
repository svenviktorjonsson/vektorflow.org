import {createContactSurfaceEventRowsGpu} from './vf-contact-surface-event-rows-gpu.mjs';
import {createContactGraphGpu} from './vf-contact-graph-gpu.mjs';
import {createContactDualSolveGpu} from './vf-contact-dual-solve-gpu.mjs';
import {createContactVelocityAuditGpu} from './vf-contact-velocity-audit-gpu.mjs';

// Generic momentum transaction after a swept World proposal. This updates
// velocity only; the already certified position remains at first contact.
export const CONTACT_SURFACE_RESPONSE_WGSL=/* wgsl */`
struct Params { vertices:u32, tolerance:f32, dt:f32, pad:f32 };
struct Motion { start:vec4<f32>, finish:vec4<f32> };
struct Body { position:vec4<f32>, diagonal:vec4<f32>, off_diagonal:vec4<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> motion:array<Motion>;
@group(0) @binding(3) var<storage,read_write> bodies:array<Body>;
@group(0) @binding(4) var<storage,read_write> free_velocity:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read> candidate:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> event_status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> graph_status:array<u32>;
@group(0) @binding(9) var<storage,read> scan_status:array<u32>;
@group(0) @binding(10) var<storage,read> solve_report:array<u32>;
@group(0) @binding(11) var<storage,read> audit_report:array<u32>;
@group(0) @binding(12) var<storage,read> sweep_status:array<u32>;
@group(0) @binding(13) var<storage,read> broadphase_status:array<u32>;
@group(0) @binding(14) var<storage,read> state_status:array<u32>;
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.vertices){return;}
 let p=positions[i];let m=motion[i];
 bodies[i]=Body(p,vec4<f32>(p.w,0.0,0.0,0.0),vec4<f32>(0.0));
 let speed=select(vec3<f32>(0.0),(m.finish.xyz-m.start.xyz)/params.dt,p.w>0.0);
 free_velocity[i*2u]=vec4<f32>(speed,0.0);
 free_velocity[i*2u+1u]=vec4<f32>(0.0);
}
@compute @workgroup_size(1) fn certify(){
 if(graph_status[0]!=0u||scan_status[0]!=0u||audit_report[0]!=0u||
  audit_report[2]!=0u||sweep_status[0]!=0u||broadphase_status[0]!=0u||
  state_status[0]!=0u||bitcast<f32>(solve_report[0])>params.tolerance||
  bitcast<f32>(solve_report[1])>params.tolerance){
  atomicOr(&event_status[0],0x100u);
 }
}
@compute @workgroup_size(64) fn apply(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.vertices||atomicLoad(&event_status[0])!=0u){return;}
 velocities[i]=vec4<f32>(candidate[i*2u].xyz,0.0);
}
`;

export async function createContactSurfaceResponseGpu(device,{vertexCount,pairCapacity,
  positions,velocities,motion,pairs,verdicts,activeCount,sweepStatus,
  broadphaseStatus,stateStatus,timeStep,iterations=32,tolerance=1e-4}){
  if(!Number.isSafeInteger(vertexCount)||vertexCount<4||
    !Number.isSafeInteger(pairCapacity)||pairCapacity<1||
    !Number.isFinite(timeStep)||timeStep<=0||
    !Number.isInteger(iterations)||iterations<1||iterations>512||
    !Number.isFinite(tolerance)||tolerance<=0||
    !positions||positions.size<vertexCount*16||
    !velocities||velocities.size<vertexCount*16||
    !motion||motion.size<vertexCount*32||!pairs||pairs.size<pairCapacity*32||
    !verdicts||verdicts.size<pairCapacity*16||!activeCount||activeCount.size<4||
    !sweepStatus||sweepStatus.size<4||!broadphaseStatus||broadphaseStatus.size<4||
    !stateStatus||stateStatus.size<4)
    throw new RangeError('Surface response requires bounded GPU World state');
  const owned=[],make=(size,usage)=>{const buffer=device.createBuffer({size,usage});owned.push(buffer);return buffer;};
  const storage=GPUBufferUsage.STORAGE;
  const bodies=make(vertexCount*48,storage),freeVelocity=make(vertexCount*32,storage);
  const requiredSpeed=make(pairCapacity*4,storage|GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(requiredSpeed,0,new Float32Array(pairCapacity));
  const uniform=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),data=new ArrayBuffer(16);
  new Uint32Array(data)[0]=vertexCount;
  new Float32Array(data).set([tolerance,timeStep],1);
  device.queue.writeBuffer(uniform,0,data);
  let events,graph,solve,audit;
  try{
    events=await createContactSurfaceEventRowsGpu(device,{vertexCount,pairCapacity,motion,
      pairs,verdicts,activeCount,timeStep});
    graph=await createContactGraphGpu(device,{bodyCount:vertexCount,edgeCount:pairCapacity,
      edgeStrideWords:12,participantsPerRow:4,edges:events.rows,activeCount:events.activeCount});
    solve=await createContactDualSolveGpu(device,{graph,bodies,contacts:events.rows,
      rhs:events.rhs,iterations,tolerance});
    audit=await createContactVelocityAuditGpu(device,{bodyCount:vertexCount,edgeCount:pairCapacity,
      bodies,contacts:events.rows,freeVelocity,changes:solve.changes,
      requiredSpeed,activeCount:events.activeCount,participantsPerRow:4,tolerance});
    const module=device.createShaderModule({label:'VKF generic surface momentum transaction',code:CONTACT_SURFACE_RESPONSE_WGSL});
    const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
    if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
    const buffers=[uniform,positions,motion,bodies,freeVelocity,velocities,audit.candidate,
      events.status,graph.status,graph.scanStatus,solve.report,audit.report,
      sweepStatus,broadphaseStatus,stateStatus];
    const build=async(entryPoint,bindings)=>{
      const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,
        visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':
          [1,2,6,8,9,10,11,12,13,14].includes(binding)?'read-only-storage':'storage'}}))});
      const group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
      const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
      return {group,pipeline};
    };
    const [prepare,certify,apply]=await Promise.all([
      build('prepare',[0,1,2,3,4]),
      build('certify',[0,7,8,9,10,11,12,13,14]),
      build('apply',[0,5,6,7]),
    ]);
    const dispatch=(encoder,stage,count)=>{const pass=encoder.beginComputePass();
      pass.setPipeline(stage.pipeline);pass.setBindGroup(0,stage.group);
      pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();};
    return {status:events.status,encode(encoder){
      dispatch(encoder,prepare,vertexCount);
      events.encode(encoder);graph.encode(encoder);solve.encode(encoder);audit.encode(encoder);
      dispatch(encoder,certify,1);dispatch(encoder,apply,vertexCount);
    },destroy(){audit.destroy();solve.destroy();graph.destroy();events.destroy();
      for(const buffer of owned)buffer.destroy();}};
  }catch(error){audit?.destroy();solve?.destroy();graph?.destroy();events?.destroy();
    for(const buffer of owned)buffer.destroy();throw error;}
}
