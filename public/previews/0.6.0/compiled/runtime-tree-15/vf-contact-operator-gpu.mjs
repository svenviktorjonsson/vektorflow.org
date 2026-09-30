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
@group(0) @binding(9) var<storage,read> active_edges:array<u32>;
@group(0) @binding(10) var<storage,read_write> dispatch_args:array<u32>;
fn operator_active()->u32{return min(params.edges,active_edges[0]);}
fn operator_dispatch(optional:bool){let count_active=operator_active();if(active_edges[0]>params.edges){atomicOr(&status[0],128u);}dispatch_args[0]=select((params.bodies+127u)/128u,0u,optional&&count_active==0u);dispatch_args[1]=1u;dispatch_args[2]=1u;dispatch_args[3]=(count_active+127u)/128u;dispatch_args[4]=1u;dispatch_args[5]=1u;}
@compute @workgroup_size(1) fn operator_prepare(){operator_dispatch(false);}
@compute @workgroup_size(1) fn operator_prepare_optional(){operator_dispatch(true);}
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
 for(var slot=offsets[i];slot<offsets[i+1u];slot++){let incident=incidents[slot];let e=incident&0x7fffffffu;let negative=(incident&0x80000000u)!=0u;if(e>=operator_active()){atomicOr(&status[0],4u);continue;}let magnitude=direction[e];if(magnitude==0.0){continue;}let row=contacts[e];let owner=select(row.header.x,row.header.y,negative);if(owner!=i){atomicOr(&status[0],4u);continue;}let n=operator_normal(row)*select(1.0,-1.0,negative);let impulse=n*magnitude;linear+=impulse;if(any(body.inverse_diagonal.yzw!=vec3<f32>(0.0))||any(body.inverse_off_diagonal.xyz!=vec3<f32>(0.0))){angular+=cross(row.geometry.yzw-body.position.xyz,impulse);}}
 let d=body.inverse_diagonal;let o=body.inverse_off_diagonal;linear*=d.x;angular=vec3<f32>(d.y*angular.x+o.x*angular.y+o.y*angular.z,o.x*angular.x+d.z*angular.y+o.z*angular.z,o.y*angular.x+o.z*angular.y+d.w*angular.z);
 if(!operator_finite(linear)||!operator_finite(angular)){atomicOr(&status[0],8u);}changes[i*2u]=vec4<f32>(linear,0.0);changes[i*2u+1u]=vec4<f32>(angular,0.0);
}
@compute @workgroup_size(128) fn operator_project(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=operator_active()){return;}let row=contacts[e];let a=row.header.x;let b=row.header.y;if(a>=params.bodies||b>=params.bodies||a==b){atomicOr(&status[0],1u);return;}let n=operator_normal(row);if(!operator_finite(n)||abs(dot(n,n)-1.0)>1.0e-4){atomicOr(&status[0],16u);return;}
 var value=dot(n,changes[a*2u].xyz-changes[b*2u].xyz);let wa=changes[a*2u+1u].xyz;let wb=changes[b*2u+1u].xyz;if(any(wa!=vec3<f32>(0.0))){value+=dot(cross(row.geometry.yzw-bodies[a].position.xyz,n),wa);}if(any(wb!=vec3<f32>(0.0))){value-=dot(cross(row.geometry.yzw-bodies[b].position.xyz,n),wb);}
 if((bitcast<u32>(value)&0x7f800000u)==0x7f800000u){atomicOr(&status[0],8u);}product[e]=value;
}
`;

export async function createContactOperatorGpu(device,{graph,bodies,contacts,direction,product}){
 const {bodyCount,edgeCount}=graph;if(graph.edgeStrideWords!==8||graph.participantsPerRow!==2)throw new RangeError('Rigid contact operator requires two-participant 32-byte geometry rows');
 for(const [buffer,required] of [[bodies,bodyCount*48],[contacts,Math.max(32,edgeCount*32)],[direction,Math.max(4,edgeCount*4)],[product,Math.max(4,edgeCount*4)]])if(buffer.size<required)throw new RangeError('Contact operator buffer capacity is too small');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);if(bodyCount*48>limit||Math.ceil(bodyCount/128)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Contact operator requires body batching for this GPU');
 const changes=device.createBuffer({size:bodyCount*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(uniform,0,new Uint32Array([bodyCount,edgeCount,0,0]));
 const module=device.createShaderModule({label:'VKF general rigid contact operator',code:CONTACT_OPERATOR_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 // Eight storage bindings is the WebGPU baseline; projection adds binding8,
 // but does not use incidence or direction, so entry-specific layouts are used.
 const types={0:'uniform',1:'read-only-storage',2:'read-only-storage',3:'read-only-storage',4:'read-only-storage',5:'read-only-storage',6:'storage',7:'storage',8:'storage',9:'read-only-storage',10:'storage'};
 const makeLayout=bindings=>device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:types[binding]}}))});
 const gatherLayout=makeLayout([0,1,2,3,4,5,6,7,9]),projectLayout=makeLayout([0,1,4,6,7,8,9]),prepareLayout=makeLayout([0,7,9,10]),dispatchArgs=device.createBuffer({size:24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
 const buffers=[uniform,bodies,graph.offsets,graph.incidents,contacts,direction,changes,graph.status,product,graph.activeCount,dispatchArgs],makeGroup=(layout,bindings)=>device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
 const gatherGroup=makeGroup(gatherLayout,[0,1,2,3,4,5,6,7,9]),projectGroup=makeGroup(projectLayout,[0,1,4,6,7,8,9]),prepareGroup=makeGroup(prepareLayout,[0,7,9,10]);
 const [gather,project]=await Promise.all([[gatherLayout,'operator_gather'],[projectLayout,'operator_project']].map(([layout,entryPoint])=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})));
 const [prepare,optional]=await Promise.all(['operator_prepare','operator_prepare_optional'].map(entryPoint=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[prepareLayout]}),compute:{module,entryPoint}})));
 return {changes,encode(encoder,{skipEmpty=false}={}){const controller=encoder.beginComputePass();controller.setPipeline(skipEmpty?optional:prepare);controller.setBindGroup(0,prepareGroup);controller.dispatchWorkgroups(1);controller.end();const pass=encoder.beginComputePass({label:'VKF J inverse-mass J-transpose'});pass.setPipeline(gather);pass.setBindGroup(0,gatherGroup);pass.dispatchWorkgroupsIndirect(dispatchArgs,0);if(edgeCount){pass.setPipeline(project);pass.setBindGroup(0,projectGroup);pass.dispatchWorkgroupsIndirect(dispatchArgs,12);}pass.end();},destroy(){uniform.destroy();changes.destroy();dispatchArgs.destroy();}};
}
