// Matrix-free J M^-1 J^T for arbitrary 3D rigid-body contact normals.
// Particles are the zero-inertia special case; no drum geometry is involved.
export const CONTACT_OPERATOR_WGSL = /* wgsl */`
struct OperatorParams { bodies:u32, edges:u32, spare:vec2<u32> };
struct ContactBody { position:vec4<f32>, inverse_diagonal:vec4<f32>, inverse_off_diagonal:vec4<f32> };
// 32-byte contact: body IDs, normal.xy bit patterns, normal.z and contact point.
struct ContactRow { header:vec4<u32>, geometry:vec4<f32> };
@group(0) @binding(0) var<uniform> params:OperatorParams;
@group(0) @binding(1) var<storage,read> bodies:array<ContactBody>;
@group(0) @binding(2) var<storage,read> offsets:array<u32>;
@group(0) @binding(3) var<storage,read> incidents:array<u32>;
@group(0) @binding(4) var<storage,read> contacts:array<ContactRow>;
@group(0) @binding(5) var<storage,read> direction:array<f32>;
@group(0) @binding(6) var<storage,read_write> changes:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> product:array<f32>;
fn operator_normal(row:ContactRow)->vec3<f32>{return vec3<f32>(bitcast<vec2<f32>>(row.header.zw),row.geometry.x);}
fn operator_finite(v:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(v)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn operator_valid_mass(body:ContactBody)->bool{
 let d=body.inverse_diagonal;let o=body.inverse_off_diagonal.xyz;
 if(any((bitcast<vec4<u32>>(d)&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))||!operator_finite(o)||any(d<vec4<f32>(0.0))){return false;}
 let scale=max(max(max(d.y,d.z),d.w),max(max(abs(o.x),abs(o.y)),abs(o.z)));if(scale==0.0){return true;}
 let v=d.yzw/scale;let w=o/scale;
 let determinant=v.x*v.y*v.z+2.0*w.x*w.y*w.z-v.x*w.z*w.z-v.y*w.y*w.y-v.z*w.x*w.x;
 return v.x*v.y>=w.x*w.x&&v.x*v.z>=w.y*w.y&&v.y*v.z>=w.z*w.z&&determinant>=0.0;
}
@compute @workgroup_size(128) fn operator_gather(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.bodies){return;}let body=bodies[i];
 if(!operator_valid_mass(body)||!operator_finite(body.position.xyz)){atomicOr(&status[0],64u);changes[i*2u]=vec4<f32>(0.0);changes[i*2u+1u]=vec4<f32>(0.0);return;}
 // A fixed object's velocity response is identically zero. Summing all its
 // contacts would turn a shared floor into a serial GPU hotspot.
 if(all(body.inverse_diagonal==vec4<f32>(0.0))&&all(body.inverse_off_diagonal.xyz==vec3<f32>(0.0))){changes[i*2u]=vec4<f32>(0.0);changes[i*2u+1u]=vec4<f32>(0.0);return;}
 atomicMax(&status[2],offsets[i+1u]-offsets[i]);var linear=vec3<f32>(0.0);var angular=vec3<f32>(0.0);
 for(var slot=offsets[i];slot<offsets[i+1u];slot++){let incident=incidents[slot];let e=incident&0x7fffffffu;let negative=(incident&0x80000000u)!=0u;if(e>=params.edges){atomicOr(&status[0],4u);continue;}let row=contacts[e];let owner=select(row.header.x,row.header.y,negative);if(owner!=i){atomicOr(&status[0],4u);continue;}let n=operator_normal(row)*select(1.0,-1.0,negative);let impulse=n*direction[e];linear+=impulse;angular+=cross(row.geometry.yzw-body.position.xyz,impulse);}
 let d=body.inverse_diagonal;let o=body.inverse_off_diagonal;linear*=d.x;angular=vec3<f32>(d.y*angular.x+o.x*angular.y+o.y*angular.z,o.x*angular.x+d.z*angular.y+o.z*angular.z,o.y*angular.x+o.z*angular.y+d.w*angular.z);
 if(!operator_finite(linear)||!operator_finite(angular)){atomicOr(&status[0],8u);}changes[i*2u]=vec4<f32>(linear,0.0);changes[i*2u+1u]=vec4<f32>(angular,0.0);
}
@compute @workgroup_size(128) fn operator_project(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.edges){return;}let row=contacts[e];let a=row.header.x;let b=row.header.y;if(a>=params.bodies||b>=params.bodies||a==b){atomicOr(&status[0],1u);return;}let n=operator_normal(row);if(!operator_finite(n)||abs(dot(n,n)-1.0)>1.0e-4){atomicOr(&status[0],16u);return;}
 let ja=cross(row.geometry.yzw-bodies[a].position.xyz,n);let jb=cross(row.geometry.yzw-bodies[b].position.xyz,n);let value=dot(n,changes[a*2u].xyz-changes[b*2u].xyz)+dot(ja,changes[a*2u+1u].xyz)-dot(jb,changes[b*2u+1u].xyz);
 if((bitcast<u32>(value)&0x7f800000u)==0x7f800000u){atomicOr(&status[0],8u);}product[e]=value;
}
`;

export async function createContactOperatorGpu(device,{graph,bodies,contacts,direction,product}){
 const {bodyCount,edgeCount}=graph;if(graph.edgeStrideWords!==8)throw new RangeError('Contact operator requires 32-byte geometry rows');
 for(const [buffer,required] of [[bodies,bodyCount*48],[contacts,Math.max(32,edgeCount*32)],[direction,Math.max(4,edgeCount*4)],[product,Math.max(4,edgeCount*4)]])if(buffer.size<required)throw new RangeError('Contact operator buffer capacity is too small');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);if(bodyCount*48>limit||Math.ceil(bodyCount/128)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Contact operator requires body batching for this GPU');
 const changes=device.createBuffer({size:bodyCount*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(uniform,0,new Uint32Array([bodyCount,edgeCount,0,0]));
 const module=device.createShaderModule({label:'VKF general rigid contact operator',code:CONTACT_OPERATOR_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 // Eight storage bindings is the WebGPU baseline; projection adds binding8,
 // but does not use incidence or direction, so entry-specific layouts are used.
 const types={0:'uniform',1:'read-only-storage',2:'read-only-storage',3:'read-only-storage',4:'read-only-storage',5:'read-only-storage',6:'storage',7:'storage',8:'storage'};
 const makeLayout=bindings=>device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:types[binding]}}))});
 const gatherLayout=makeLayout([0,1,2,3,4,5,6,7]),projectLayout=makeLayout([0,1,4,6,7,8]);
 const buffers=[uniform,bodies,graph.offsets,graph.incidents,contacts,direction,changes,graph.status,product],makeGroup=(layout,bindings)=>device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
 const gatherGroup=makeGroup(gatherLayout,[0,1,2,3,4,5,6,7]),projectGroup=makeGroup(projectLayout,[0,1,4,6,7,8]);
 const [gather,project]=await Promise.all([[gatherLayout,'operator_gather'],[projectLayout,'operator_project']].map(([layout,entryPoint])=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})));
 return {changes,encode(encoder){const pass=encoder.beginComputePass({label:'VKF J inverse-mass J-transpose'});pass.setPipeline(gather);pass.setBindGroup(0,gatherGroup);pass.dispatchWorkgroups(Math.ceil(bodyCount/128));if(edgeCount){pass.setPipeline(project);pass.setBindGroup(0,projectGroup);pass.dispatchWorkgroups(Math.ceil(edgeCount/128));}pass.end();},destroy(){uniform.destroy();changes.destroy();}};
}
