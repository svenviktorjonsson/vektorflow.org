import {createContactPrefixScanGpu,contactScanLayout} from './vf-contact-prefix-scan-gpu.mjs';

export const CONTACT_GRAPH_WGSL = /* wgsl */`
struct GraphParams { bodies:u32, edges:u32, stride:u32, capacity:u32, participants:u32 };
@group(0) @binding(0) var<uniform> params:GraphParams;
@group(0) @binding(1) var<storage,read> edges:array<u32>;
@group(0) @binding(2) var<storage,read_write> degrees:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> cursors:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read> offsets:array<u32>;
@group(0) @binding(5) var<storage,read_write> incidents:array<u32>;
@group(0) @binding(6) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> active_edges:array<u32>;
@group(0) @binding(8) var<storage,read_write> dispatch_args:array<u32>;
fn graph_active()->u32{return min(params.edges,active_edges[0]);}
@compute @workgroup_size(1) fn graph_prepare(){if(active_edges[0]>params.edges){atomicOr(&status[0],128u);}dispatch_args[0]=(graph_active()+127u)/128u;dispatch_args[1]=1u;dispatch_args[2]=1u;}
fn graph_valid(e:u32)->bool{
 for(var slot=0u;slot<params.participants;slot++){
  let body=edges[e*params.stride+slot];if(body>=params.bodies){atomicOr(&status[0],1u);return false;}
  for(var previous=0u;previous<slot;previous++){
   if(body==edges[e*params.stride+previous]){atomicOr(&status[0],1u);return false;}
  }
 }return true;
}
@compute @workgroup_size(128) fn graph_count(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=graph_active()||!graph_valid(e)){return;}
 for(var slot=0u;slot<params.participants;slot++){
  let body=edges[e*params.stride+slot];atomicMax(&status[1],atomicAdd(&degrees[body],1u)+1u);
 }
}
fn graph_emit(body:u32,e:u32,participant:u32){
 let slot=offsets[body]+atomicAdd(&cursors[body],1u);if(slot>=params.capacity){atomicOr(&status[0],2u);return;}
 // Keep the legacy second-body sign bit for two-body rows. The second-high
 // bit distinguishes slots 2 and 3 when a surface contact has four vertices.
 let tag=select(select(0u,0x80000000u,participant==1u),select(0x40000000u,0xc0000000u,participant==3u),participant>=2u);
 incidents[slot]=e|tag;
}
@compute @workgroup_size(128) fn graph_scatter(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=graph_active()||!graph_valid(e)){return;}
 for(var slot=0u;slot<params.participants;slot++){graph_emit(edges[e*params.stride+slot],e,slot);}
}
`;

export function contactGraphLayout({bodyCount,edgeCount,edgeStrideWords=2,participantsPerRow=2}){
 contactScanLayout(bodyCount);if(!Number.isSafeInteger(edgeCount)||edgeCount<0||edgeCount>0x3fffffff||![2,4].includes(participantsPerRow)||!Number.isSafeInteger(edgeStrideWords)||edgeStrideWords<participantsPerRow)throw new RangeError('Contact graph edge count/stride/participants is invalid');
 const edgeBytes=edgeCount*edgeStrideWords*4;if(!Number.isSafeInteger(edgeBytes))throw new RangeError('Contact graph edge capacity overflow');
 return {bodyCount,edgeCount,edgeStrideWords,participantsPerRow,edgeBytes,degreeBytes:bodyCount*4,offsetBytes:(bodyCount+1)*4,incidentBytes:Math.max(4,edgeCount*participantsPerRow*4),scan:contactScanLayout(bodyCount)};
}

export async function createContactGraphGpu(device,options){
 const p=contactGraphLayout(options),limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if([p.edgeBytes,p.degreeBytes,p.offsetBytes,p.incidentBytes].some(n=>n>limit)||Math.ceil(p.edgeCount/128)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Contact graph requires buffer batching for this GPU');
 if(options.edges.size<Math.max(4,p.edgeBytes))throw new RangeError('Contact graph edge buffer is too small');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;},storage=GPUBufferUsage.STORAGE;
 const degrees=make(p.degreeBytes,storage|GPUBufferUsage.COPY_DST),cursors=make(p.degreeBytes,storage|GPUBufferUsage.COPY_DST),offsets=make(p.offsetBytes,storage|GPUBufferUsage.COPY_SRC),incidents=make(p.incidentBytes,storage|GPUBufferUsage.COPY_SRC),status=make(16,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),uniform=make(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 const activeCount=options.activeCount??make(4,storage|GPUBufferUsage.COPY_DST),dispatchArgs=make(12,storage|GPUBufferUsage.INDIRECT);if(activeCount.size<4)throw new RangeError('Contact active-count buffer is too small');if(!options.activeCount)device.queue.writeBuffer(activeCount,0,new Uint32Array([p.edgeCount]));
 device.queue.writeBuffer(uniform,0,new Uint32Array([p.bodyCount,p.edgeCount,p.edgeStrideWords,p.edgeCount*p.participantsPerRow,p.participantsPerRow,0,0,0]));
 const scan=await createContactPrefixScanGpu(device,{count:p.bodyCount,input:degrees,output:offsets});
 const module=device.createShaderModule({label:'VKF sparse contact incidence',code:CONTACT_GRAPH_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:[1,4,7].includes(binding)?'read-only-storage':'storage'}}))]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const [count,scatter]=await Promise.all(['graph_count','graph_scatter'].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
 const group=device.createBindGroup({layout,entries:[uniform,options.edges,degrees,cursors,offsets,incidents,status,activeCount].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const prepareLayout=device.createBindGroupLayout({entries:[0,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===7?'read-only-storage':'storage'}}))}),prepare=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[prepareLayout]}),compute:{module,entryPoint:'graph_prepare'}}),prepareGroup=device.createBindGroup({layout:prepareLayout,entries:[[0,uniform],[6,status],[7,activeCount],[8,dispatchArgs]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
 return {...p,degrees,offsets,incidents,status,activeCount,dispatchArgs,scanStatus:scan.status,
  encode(encoder){encoder.clearBuffer(degrees);encoder.clearBuffer(cursors);encoder.clearBuffer(status);{const pass=encoder.beginComputePass({label:'VKF active contact count'});pass.setPipeline(prepare);pass.setBindGroup(0,prepareGroup);pass.dispatchWorkgroups(1);pass.end();}if(p.edgeCount){const pass=encoder.beginComputePass({label:'VKF contact degrees'});pass.setPipeline(count);pass.setBindGroup(0,group);pass.dispatchWorkgroupsIndirect(dispatchArgs,0);pass.end();}scan.encode(encoder);if(p.edgeCount){const pass=encoder.beginComputePass({label:'VKF contact incidences'});pass.setPipeline(scatter);pass.setBindGroup(0,group);pass.dispatchWorkgroupsIndirect(dispatchArgs,0);pass.end();}},
  destroy(){scan.destroy();for(const b of owned)b.destroy();}};
}
