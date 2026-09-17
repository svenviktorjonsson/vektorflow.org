import {createContactOperatorGpu} from './vf-contact-operator-gpu.mjs';

// Nonnegative contact impulses solve min .5 lambda^T A lambda - b^T lambda.
// A=J M^-1 J^T. Diagonal-metric accelerated projected gradient uses a GPU
// degree bound, not a guessed material-dependent relaxation constant.
export const CONTACT_DUAL_SOLVE_WGSL = /* wgsl */`
struct DualParams { count:u32, spare:u32, momentum:f32, tolerance:f32 };
struct DualBody { position:vec4<f32>, diagonal:vec4<f32>, off_diagonal:vec4<f32> };
struct DualContact { header:vec4<u32>, geometry:vec4<f32> };
@group(0) @binding(0) var<uniform> params:DualParams;
@group(0) @binding(1) var<storage,read> bodies:array<DualBody>;
@group(0) @binding(2) var<storage,read> contacts:array<DualContact>;
@group(0) @binding(3) var<storage,read> rhs:array<f32>;
@group(0) @binding(4) var<storage,read> product:array<f32>;
@group(0) @binding(5) var<storage,read_write> direction:array<f32>;
@group(0) @binding(6) var<storage,read_write> state:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> report:array<atomic<u32>>;
fn dual_rotational(body:DualBody,j:vec3<f32>)->f32{let d=body.diagonal;let o=body.off_diagonal;return dot(j,vec3<f32>(d.y*j.x+o.x*j.y+o.y*j.z,o.x*j.x+d.z*j.y+o.z*j.z,o.y*j.x+o.z*j.y+d.w*j.z));}
@compute @workgroup_size(128) fn dual_initialize(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.count){return;}let row=contacts[e];if(row.header.x>=arrayLength(&bodies)||row.header.y>=arrayLength(&bodies)){atomicOr(&status[0],1u);return;}let a=bodies[row.header.x];let b=bodies[row.header.y];let n=vec3<f32>(bitcast<vec2<f32>>(row.header.zw),row.geometry.x);let ja=cross(row.geometry.yzw-a.position.xyz,n);let jb=cross(row.geometry.yzw-b.position.xyz,n);
 let diagonal=(a.diagonal.x+b.diagonal.x)*dot(n,n)+dual_rotational(a,ja)+dual_rotational(b,jb);if(diagonal<0.0||(bitcast<u32>(diagonal)&0x7f800000u)==0x7f800000u||(bitcast<u32>(rhs[e])&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);}state[e]=vec4<f32>(0.0,diagonal,0.0,0.0);direction[e]=0.0;
}
@compute @workgroup_size(128) fn dual_project(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.count){return;}let old=state[e];let degree=f32(max(1u,atomicLoad(&status[2])));var impulse=0.0;if(old.y>0.0){impulse=max(0.0,direction[e]+(rhs[e]-product[e])/(old.y*degree));}state[e].x=impulse;direction[e]=impulse+params.momentum*(impulse-old.x);
 if((bitcast<u32>(direction[e])&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);}
}
@compute @workgroup_size(128) fn dual_solution(@builtin(global_invocation_id) gid:vec3<u32>){let e=gid.x;if(e<params.count){direction[e]=state[e].x;}}
@compute @workgroup_size(128) fn dual_inspect(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.count){return;}let residual=rhs[e]-product[e];if((bitcast<u32>(residual)&0x7f800000u)==0x7f800000u){atomicOr(&status[0],32u);return;}atomicMax(&report[0],bitcast<u32>(max(0.0,residual))&0x7fffffffu);if(state[e].x>0.0){atomicMax(&report[1],bitcast<u32>(abs(residual))&0x7fffffffu);}if(state[e].x<0.0){atomicOr(&status[0],32u);}
}
`;

export async function createContactDualSolveGpu(device,{graph,bodies,contacts,rhs,iterations=128,tolerance=1e-5}){
 if(!Number.isInteger(iterations)||iterations<1||iterations>512||!Number.isFinite(tolerance)||tolerance<=0)throw new RangeError('Contact solver iteration/tolerance is invalid');
 const E=graph.edgeCount;if(rhs.size<Math.max(4,E*4))throw new RangeError('Contact solver RHS capacity is too small');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);if(E*16>limit)throw new RangeError('Contact solver requires edge batching for this GPU');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;},storage=GPUBufferUsage.STORAGE;
 const direction=make(Math.max(4,E*4),storage),product=make(Math.max(4,E*4),storage),state=make(Math.max(16,E*16),storage|GPUBufferUsage.COPY_SRC),report=make(16,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),readback=make(48,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const operator=await createContactOperatorGpu(device,{graph,bodies,contacts,direction,product});
 const alignment=device.limits.minUniformBufferOffsetAlignment,uniform=make((iterations+1)*alignment,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),parameters=new ArrayBuffer((iterations+1)*alignment);let t=1;
 for(let i=0;i<=iterations;i++){const next=(1+Math.sqrt(1+4*t*t))/2,beta=i===iterations?0:(t-1)/next;new Uint32Array(parameters,i*alignment,2).set([E,0]);new Float32Array(parameters,i*alignment+8,2).set([beta,tolerance]);t=next;}device.queue.writeBuffer(uniform,0,parameters);
 const module=device.createShaderModule({label:'VKF feasible contact dual solve',code:CONTACT_DUAL_SOLVE_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:16}},...[1,2,3,4,5,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding<=4?'read-only-storage':'storage'}}))]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const pipelines=Object.fromEntries(await Promise.all(['dual_initialize','dual_project','dual_solution','dual_inspect'].map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
 const group=device.createBindGroup({layout,entries:[uniform,bodies,contacts,rhs,product,direction,state,graph.status,report].map((buffer,binding)=>({binding,resource:{buffer,...(binding===0?{size:16}:{})}}))});
 const dispatch=(encoder,name,offset=0)=>{if(!E)return;const pass=encoder.beginComputePass({label:`VKF ${name}`});pass.setPipeline(pipelines[name]);pass.setBindGroup(0,group,[offset]);pass.dispatchWorkgroups(Math.ceil(E/128));pass.end();};
 return {state,changes:operator.changes,iterations,tolerance,
  encode(encoder){encoder.clearBuffer(report);dispatch(encoder,'dual_initialize');for(let i=0;i<iterations;i++){operator.encode(encoder);dispatch(encoder,'dual_project',i*alignment);}dispatch(encoder,'dual_solution',iterations*alignment);operator.encode(encoder);dispatch(encoder,'dual_inspect',iterations*alignment);encoder.copyBufferToBuffer(report,0,readback,0,16);encoder.copyBufferToBuffer(graph.status,0,readback,16,16);encoder.copyBufferToBuffer(graph.scanStatus,0,readback,32,16);},
  async inspect(){await readback.mapAsync(GPUMapMode.READ);const bytes=readback.getMappedRange().slice(0);readback.unmap();const f=new Float32Array(bytes),u=new Uint32Array(bytes),fault=u[4]|u[8];return {certificate:'linearized-contact-model',feasible:!fault&&f[0]<=tolerance,converged:!fault&&f[0]<=tolerance&&f[1]<=tolerance,maxClosingResidual:f[0],maxActiveResidual:f[1],fault,iterations,tolerance};},
  destroy(){operator.destroy();for(const b of owned)b.destroy();}};
}
