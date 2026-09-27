import {createContactOperatorGpu} from './vf-contact-operator-gpu.mjs';
import {createContactSurfaceOperatorGpu} from './vf-contact-surface-operator-gpu.mjs';

// Nonnegative contact impulses solve min .5 lambda^T A lambda - b^T lambda.
// A=J M^-1 J^T. Diagonal-metric accelerated projected gradient uses a GPU
// degree bound, not a guessed material-dependent relaxation constant.
export const CONTACT_DUAL_SOLVE_WGSL = /* wgsl */`
struct DualParams { count:u32, spare:u32, momentum:f32, tolerance:f32 };
struct DualBody { position:vec4<f32>, diagonal:vec4<f32>, off_diagonal:vec4<f32> };
@group(0) @binding(0) var<uniform> params:DualParams;
@group(0) @binding(1) var<storage,read> bodies:array<DualBody>;
@group(0) @binding(2) var<storage,read> contacts:array<u32>;
@group(0) @binding(3) var<storage,read> rhs:array<f32>;
@group(0) @binding(4) var<storage,read> product:array<f32>;
@group(0) @binding(5) var<storage,read_write> direction:array<f32>;
@group(0) @binding(6) var<storage,read_write> state:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> report:array<atomic<u32>>;
@group(0) @binding(9) var<storage,read> active_edges:array<u32>;
@group(0) @binding(10) var<storage,read_write> iteration_dispatch:array<u32>;
fn dual_active()->u32{return min(params.count,active_edges[0]);}
fn dual_rotational(body:DualBody,j:vec3<f32>)->f32{let d=body.diagonal;let o=body.off_diagonal;return dot(j,vec3<f32>(d.y*j.x+o.x*j.y+o.y*j.z,o.x*j.x+d.z*j.y+o.z*j.z,o.y*j.x+o.z*j.y+d.w*j.z));}
@compute @workgroup_size(128) fn dual_initialize(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=dual_active()){return;}
 var diagonal=0.0;
 if(params.spare==4u){
  let at=e*12u;let n=vec3<f32>(bitcast<f32>(contacts[at+8u]),bitcast<f32>(contacts[at+9u]),bitcast<f32>(contacts[at+10u]));
  for(var slot=0u;slot<4u;slot++){
   let body=contacts[at+slot];if(body>=arrayLength(&bodies)){atomicOr(&status[0],1u);return;}
   let weight=bitcast<f32>(contacts[at+4u+slot]);diagonal+=bodies[body].diagonal.x*weight*weight*dot(n,n);
  }
 }else{
  let at=e*8u;let a_id=contacts[at];let b_id=contacts[at+1u];
  if(a_id>=arrayLength(&bodies)||b_id>=arrayLength(&bodies)){atomicOr(&status[0],1u);return;}
  let a=bodies[a_id];let b=bodies[b_id];let n=vec3<f32>(bitcast<f32>(contacts[at+2u]),bitcast<f32>(contacts[at+3u]),bitcast<f32>(contacts[at+4u]));
  let point=vec3<f32>(bitcast<f32>(contacts[at+5u]),bitcast<f32>(contacts[at+6u]),bitcast<f32>(contacts[at+7u]));
  let ja=cross(point-a.position.xyz,n);let jb=cross(point-b.position.xyz,n);
  diagonal=(a.diagonal.x+b.diagonal.x)*dot(n,n)+dual_rotational(a,ja)+dual_rotational(b,jb);
 }
 if(diagonal<0.0||(bitcast<u32>(diagonal)&0x7f800000u)==0x7f800000u||(bitcast<u32>(rhs[e])&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);}
 // Degree one makes each surface row independent: solve its impulse exactly.
 var direct=0.0;
 if(params.spare==4u&&atomicLoad(&status[1])<=1u&&diagonal>0.0){direct=max(0.0,rhs[e]/diagonal);}
 state[e]=vec4<f32>(direct,diagonal,0.0,0.0);direction[e]=direct;
}
@compute @workgroup_size(1) fn dual_iteration_args(){
 let independent=params.spare==4u&&atomicLoad(&status[1])<=1u;
 iteration_dispatch[0]=select((dual_active()+127u)/128u,0u,independent);
 iteration_dispatch[1]=1u;iteration_dispatch[2]=1u;
}
@compute @workgroup_size(128) fn dual_project(@builtin(global_invocation_id) gid:vec3<u32>){
 // Four-vertex surfaces use graph incidence degree in status[1].
 // Two-body contacts retain their operator-computed degree in status[2].
 let e=gid.x;if(e>=dual_active()){return;}let old=state[e];let degree=f32(max(1u,atomicLoad(&status[select(2u,1u,params.spare==4u)])));var impulse=0.0;if(old.y>0.0){impulse=max(0.0,direction[e]+(rhs[e]-product[e])/(old.y*degree));}state[e].x=impulse;direction[e]=impulse+params.momentum*(impulse-old.x);
 if((bitcast<u32>(direction[e])&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);}
}
@compute @workgroup_size(128) fn dual_solution(@builtin(global_invocation_id) gid:vec3<u32>){let e=gid.x;if(e<dual_active()){direction[e]=state[e].x;}}
@compute @workgroup_size(128) fn dual_inspect(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=dual_active()){return;}let residual=rhs[e]-product[e];if((bitcast<u32>(residual)&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);return;}atomicMax(&report[0],bitcast<u32>(max(0.0,residual))&0x7fffffffu);if(state[e].x>0.0){atomicMax(&report[1],bitcast<u32>(abs(residual))&0x7fffffffu);}if(state[e].x<0.0){atomicOr(&status[0],32u);}
}
`;

export async function createContactDualSolveGpu(device,{graph,bodies,contacts,rhs,iterations=128,tolerance=1e-5}){
 if(!Number.isInteger(iterations)||iterations<1||iterations>512||!Number.isFinite(tolerance)||tolerance<=0)throw new RangeError('Contact solver iteration/tolerance is invalid');
 const surface=graph.participantsPerRow===4;
 const expectedStride=surface?12:8;
 if(![2,4].includes(graph.participantsPerRow)||graph.edgeStrideWords!==expectedStride)
  throw new RangeError('Contact solver row layout is unsupported');
 const E=graph.edgeCount;if(rhs.size<Math.max(4,E*4))throw new RangeError('Contact solver RHS capacity is too small');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);if(E*16>limit)throw new RangeError('Contact solver requires edge batching for this GPU');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;},storage=GPUBufferUsage.STORAGE;
 const direction=make(Math.max(4,E*4),storage),product=make(Math.max(4,E*4),storage),state=make(Math.max(16,E*16),storage|GPUBufferUsage.COPY_SRC),report=make(16,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),readback=make(48,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),iterationDispatch=make(12,storage|GPUBufferUsage.INDIRECT);
 const operator=surface?await createContactSurfaceOperatorGpu(device,{graph,bodies,contacts,direction,product}):await createContactOperatorGpu(device,{graph,bodies,contacts,direction,product});
 const alignment=device.limits.minUniformBufferOffsetAlignment,uniform=make((iterations+1)*alignment,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),parameters=new ArrayBuffer((iterations+1)*alignment);let t=1;
 for(let i=0;i<=iterations;i++){const next=(1+Math.sqrt(1+4*t*t))/2,beta=i===iterations?0:(t-1)/next;new Uint32Array(parameters,i*alignment,2).set([E,surface?4:2]);new Float32Array(parameters,i*alignment+8,2).set([beta,tolerance]);t=next;}device.queue.writeBuffer(uniform,0,parameters);
 const module=device.createShaderModule({label:'VKF feasible contact dual solve',code:CONTACT_DUAL_SOLVE_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 const buffers=[uniform,bodies,contacts,rhs,product,direction,state,graph.status,report,graph.activeCount,iterationDispatch],bindingSets={dual_initialize:[0,1,2,3,5,6,7,9],dual_iteration_args:[0,7,9,10],dual_project:[0,3,4,5,6,7,9],dual_solution:[0,5,6,9],dual_inspect:[0,3,4,6,7,8,9]};
 const pipelines=Object.fromEntries(await Promise.all(Object.entries(bindingSets).map(async([entryPoint,bindings])=>{const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:binding===0?{type:'uniform',hasDynamicOffset:true,minBindingSize:16}:{type:binding<=4||binding===9?'read-only-storage':'storage'}}))}),pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}}),group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding],...(binding===0?{size:16}:{})}}))});return [entryPoint,{pipeline,group}];})));
 const dispatch=(encoder,name,offset=0)=>{if(!E)return;const pass=encoder.beginComputePass({label:`VKF ${name}`});pass.setPipeline(pipelines[name].pipeline);pass.setBindGroup(0,pipelines[name].group,[offset]);if(name==='dual_iteration_args')pass.dispatchWorkgroups(1);else pass.dispatchWorkgroupsIndirect(name==='dual_project'?iterationDispatch:graph.dispatchArgs,0);pass.end();};
 return {state,report,changes:operator.changes,iterations,tolerance,
  encode(encoder){encoder.clearBuffer(report);dispatch(encoder,'dual_initialize');dispatch(encoder,'dual_iteration_args');for(let i=0;i<iterations;i++){operator.encode(encoder,{skipEmpty:true,skipIndependent:true});dispatch(encoder,'dual_project',i*alignment);}dispatch(encoder,'dual_solution',iterations*alignment);operator.encode(encoder);dispatch(encoder,'dual_inspect',iterations*alignment);encoder.copyBufferToBuffer(report,0,readback,0,16);encoder.copyBufferToBuffer(graph.status,0,readback,16,16);encoder.copyBufferToBuffer(graph.scanStatus,0,readback,32,16);},
  async inspect(){await readback.mapAsync(GPUMapMode.READ);const bytes=readback.getMappedRange().slice(0);readback.unmap();const f=new Float32Array(bytes),u=new Uint32Array(bytes),fault=u[4]|u[8];return {certificate:'linearized-contact-model',feasible:!fault&&f[0]<=tolerance,converged:!fault&&f[0]<=tolerance&&f[1]<=tolerance,maxClosingResidual:f[0],maxActiveResidual:f[1],fault,iterations,tolerance};},
  destroy(){operator.destroy();for(const b of owned)b.destroy();}};
}
