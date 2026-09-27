// Stage actual f32 velocities, then check J v_actual >= requiredSpeed for
// either two-body rigid contacts or four-vertex weighted surface contacts.
// This is NOT geometric CCD. Never commit positions on this receipt alone.
export const CONTACT_VELOCITY_AUDIT_WGSL = /* wgsl */`
struct AuditParams { bodies:u32, edges:u32, tolerance:f32, spare:u32 };
struct AuditBody { position:vec4<f32>, diagonal:vec4<f32>, off_diagonal:vec4<f32> };
@group(0) @binding(0) var<uniform> params:AuditParams;
@group(0) @binding(1) var<storage,read> bodies:array<AuditBody>;
@group(0) @binding(2) var<storage,read> contacts:array<u32>;
@group(0) @binding(3) var<storage,read> free_velocity:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> changes:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> required_speed:array<f32>;
@group(0) @binding(6) var<storage,read_write> candidate:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> report:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> active_edges:array<u32>;
fn audit_finite(v:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(v)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
@compute @workgroup_size(128) fn audit_stage(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.bodies){return;}
 if(i==0u&&active_edges[0]>params.edges){atomicOr(&report[0],16u);}
 let linear=free_velocity[i*2u].xyz+changes[i*2u].xyz;
 let angular=free_velocity[i*2u+1u].xyz+changes[i*2u+1u].xyz;
 candidate[i*2u]=vec4<f32>(linear,0.0);candidate[i*2u+1u]=vec4<f32>(angular,0.0);
 if(!audit_finite(linear)||!audit_finite(angular)){atomicOr(&report[0],1u);}
}
@compute @workgroup_size(128) fn audit_contacts(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=min(params.edges,active_edges[0])){return;}
 var actual_speed=0.0;
 if(params.spare==4u){
  let at=e*12u;let n=vec3<f32>(bitcast<f32>(contacts[at+8u]),bitcast<f32>(contacts[at+9u]),bitcast<f32>(contacts[at+10u]));
  if(!audit_finite(n)||abs(dot(n,n)-1.0)>1e-4){atomicOr(&report[0],4u);return;}
  var weight_sum=0.0;
  for(var slot=0u;slot<4u;slot++){
   let i=contacts[at+slot];let weight=bitcast<f32>(contacts[at+4u+slot]);
   if(i>=params.bodies||!audit_finite(vec3<f32>(weight))){atomicOr(&report[0],2u);return;}
   for(var previous=0u;previous<slot;previous++){
    if(i==contacts[at+previous]){atomicOr(&report[0],2u);return;}
   }
   weight_sum+=weight;actual_speed+=weight*dot(n,candidate[i*2u].xyz);
  }
  if(abs(weight_sum)>1e-4){atomicOr(&report[0],4u);return;}
 }else{
  let at=e*8u;let a=contacts[at];let b=contacts[at+1u];
  if(a>=params.bodies||b>=params.bodies||a==b){atomicOr(&report[0],2u);return;}
  let n=vec3<f32>(bitcast<f32>(contacts[at+2u]),bitcast<f32>(contacts[at+3u]),bitcast<f32>(contacts[at+4u]));
  let point=vec3<f32>(bitcast<f32>(contacts[at+5u]),bitcast<f32>(contacts[at+6u]),bitcast<f32>(contacts[at+7u]));
  if(!audit_finite(n)||!audit_finite(point)||!audit_finite(bodies[a].position.xyz)||!audit_finite(bodies[b].position.xyz)||abs(dot(n,n)-1.0)>1e-4){atomicOr(&report[0],4u);return;}
  let va=candidate[a*2u].xyz+cross(candidate[a*2u+1u].xyz,point-bodies[a].position.xyz);
  let vb=candidate[b*2u].xyz+cross(candidate[b*2u+1u].xyz,point-bodies[b].position.xyz);
  actual_speed=dot(n,va-vb);
 }
 let required=required_speed[e];let residual=required-actual_speed;
 if(!audit_finite(vec3<f32>(actual_speed,required,residual))){atomicOr(&report[0],8u);return;}
 // Avoid a million pointless atomics for already satisfied contacts.
 if(residual>0.0){atomicMax(&report[1],bitcast<u32>(residual));}
 if(residual>params.tolerance){atomicAdd(&report[2],1u);}
}
`;

export function contactVelocityAuditLayout({bodyCount,edgeCount,participantsPerRow=2,tolerance=0}){
 if(!Number.isInteger(bodyCount)||bodyCount<1||bodyCount>0x7fffffff||!Number.isInteger(edgeCount)||edgeCount<0||edgeCount>0x3fffffff||![2,4].includes(participantsPerRow)||!Number.isFinite(tolerance)||!Number.isFinite(Math.fround(tolerance))||tolerance<0)throw new RangeError('Invalid actual-velocity audit dimensions or tolerance');
 return {bodyCount,edgeCount,participantsPerRow,tolerance,candidateBytes:bodyCount*32,bodyBytes:bodyCount*48,contactBytes:Math.max(participantsPerRow===4?48:32,edgeCount*(participantsPerRow===4?48:32)),requiredBytes:Math.max(4,edgeCount*4)};
}

export async function createContactVelocityAuditGpu(device,{bodyCount,edgeCount,bodies,contacts,freeVelocity,changes,requiredSpeed,activeCount,participantsPerRow=2,tolerance=0}){
 const plan=contactVelocityAuditLayout({bodyCount,edgeCount,participantsPerRow,tolerance}),limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if(Math.max(plan.candidateBytes,plan.bodyBytes,plan.contactBytes,plan.requiredBytes)>limit||Math.ceil(Math.max(bodyCount,edgeCount)/128)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Actual-velocity audit requires batching for this GPU');
 for(const [buffer,size] of [[bodies,plan.bodyBytes],[contacts,plan.contactBytes],[freeVelocity,plan.candidateBytes],[changes,plan.candidateBytes],[requiredSpeed,plan.requiredBytes]])if(buffer.size<size)throw new RangeError('Actual-velocity audit input capacity is too small');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;};
 const activeEdges=activeCount??make(4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);if(activeEdges.size<4)throw new RangeError('Actual-velocity active-count buffer is too small');if(!activeCount)device.queue.writeBuffer(activeEdges,0,new Uint32Array([edgeCount]));
 const uniform=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),data=new ArrayBuffer(16);new Uint32Array(data).set([bodyCount,edgeCount]);new Float32Array(data)[2]=tolerance;new Uint32Array(data)[3]=participantsPerRow;device.queue.writeBuffer(uniform,0,data);
 const candidate=make(plan.candidateBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),report=make(16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),readback=make(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const module=device.createShaderModule({label:'VKF actual rounded contact velocity audit',code:CONTACT_VELOCITY_AUDIT_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 const buffers=[uniform,bodies,contacts,freeVelocity,changes,requiredSpeed,candidate,report,activeEdges];
 const build=async(entryPoint,bindings)=>{const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===6||binding===7?'storage':'read-only-storage'}}))}),pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}}),group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});return {pipeline,group};};
 const [stage,check]=await Promise.all([build('audit_stage',[0,3,4,6,7,8]),build('audit_contacts',[0,1,2,5,6,7,8])]);
 return {candidate,report,encode(encoder){encoder.clearBuffer(report);const pass=encoder.beginComputePass({label:'VKF actual velocity stage and contact audit'});pass.setPipeline(stage.pipeline);pass.setBindGroup(0,stage.group);pass.dispatchWorkgroups(Math.ceil(bodyCount/128));if(edgeCount){pass.setPipeline(check.pipeline);pass.setBindGroup(0,check.group);pass.dispatchWorkgroups(Math.ceil(edgeCount/128));}pass.end();encoder.copyBufferToBuffer(report,0,readback,0,16);},
  async inspect(){await readback.mapAsync(GPUMapMode.READ);const bytes=readback.getMappedRange().slice(0);readback.unmap();const u=new Uint32Array(bytes),f=new Float32Array(bytes);return {certificate:'actual-rounded-contact-velocity',feasible:u[0]===0&&u[2]===0,fault:u[0],maxClosingResidual:f[1],violatingContacts:u[2],tolerance};},
  destroy(){for(const b of owned)b.destroy();}};
}
