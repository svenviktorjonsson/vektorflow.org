import {createContactPrefixScanGpu,contactScanLayout} from './vf-contact-prefix-scan-gpu.mjs';

export const CONTACT_GRAPH_WGSL = /* wgsl */`
struct GraphParams { bodies:u32, edges:u32, stride:u32, capacity:u32 };
@group(0) @binding(0) var<uniform> params:GraphParams;
@group(0) @binding(1) var<storage,read> edges:array<u32>;
@group(0) @binding(2) var<storage,read_write> degrees:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> cursors:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read> offsets:array<u32>;
@group(0) @binding(5) var<storage,read_write> incidents:array<u32>;
@group(0) @binding(6) var<storage,read_write> status:array<atomic<u32>>;
fn graph_valid(a:u32,b:u32)->bool{if(a>=params.bodies||b>=params.bodies||a==b){atomicOr(&status[0],1u);return false;}return true;}
@compute @workgroup_size(128) fn graph_count(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.edges){return;}let a=edges[e*params.stride];let b=edges[e*params.stride+1u];if(!graph_valid(a,b)){return;}
 atomicMax(&status[1],atomicAdd(&degrees[a],1u)+1u);atomicMax(&status[1],atomicAdd(&degrees[b],1u)+1u);
}
fn graph_emit(body:u32,e:u32){let slot=offsets[body]+atomicAdd(&cursors[body],1u);if(slot>=params.capacity){atomicOr(&status[0],2u);return;}incidents[slot]=e;}
@compute @workgroup_size(128) fn graph_scatter(@builtin(global_invocation_id) gid:vec3<u32>){
 let e=gid.x;if(e>=params.edges){return;}let a=edges[e*params.stride];let b=edges[e*params.stride+1u];if(!graph_valid(a,b)){return;}graph_emit(a,e);graph_emit(b,e|0x80000000u);
}
`;

export function contactGraphLayout({bodyCount,edgeCount,edgeStrideWords=2}){
 contactScanLayout(bodyCount);if(!Number.isSafeInteger(edgeCount)||edgeCount<0||edgeCount>0x3fffffff||!Number.isSafeInteger(edgeStrideWords)||edgeStrideWords<2)throw new RangeError('Contact graph edge count/stride is invalid');
 const edgeBytes=edgeCount*edgeStrideWords*4;if(!Number.isSafeInteger(edgeBytes))throw new RangeError('Contact graph edge capacity overflow');
 return {bodyCount,edgeCount,edgeStrideWords,edgeBytes,degreeBytes:bodyCount*4,offsetBytes:(bodyCount+1)*4,incidentBytes:Math.max(4,edgeCount*8),scan:contactScanLayout(bodyCount)};
}

export async function createContactGraphGpu(device,options){
 const p=contactGraphLayout(options),limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if([p.edgeBytes,p.degreeBytes,p.offsetBytes,p.incidentBytes].some(n=>n>limit)||Math.ceil(p.edgeCount/128)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Contact graph requires buffer batching for this GPU');
 if(options.edges.size<Math.max(4,p.edgeBytes))throw new RangeError('Contact graph edge buffer is too small');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;},storage=GPUBufferUsage.STORAGE;
 const degrees=make(p.degreeBytes,storage|GPUBufferUsage.COPY_DST),cursors=make(p.degreeBytes,storage|GPUBufferUsage.COPY_DST),offsets=make(p.offsetBytes,storage|GPUBufferUsage.COPY_SRC),incidents=make(p.incidentBytes,storage|GPUBufferUsage.COPY_SRC),status=make(16,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),uniform=make(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 device.queue.writeBuffer(uniform,0,new Uint32Array([p.bodyCount,p.edgeCount,p.edgeStrideWords,p.edgeCount*2]));
 const scan=await createContactPrefixScanGpu(device,{count:p.bodyCount,input:degrees,output:offsets});
 const module=device.createShaderModule({label:'VKF sparse contact incidence',code:CONTACT_GRAPH_WGSL}),messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>m.message).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:[1,4].includes(binding)?'read-only-storage':'storage'}}))]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const [count,scatter]=await Promise.all(['graph_count','graph_scatter'].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
 const group=device.createBindGroup({layout,entries:[uniform,options.edges,degrees,cursors,offsets,incidents,status].map((buffer,binding)=>({binding,resource:{buffer}}))});
 return {...p,degrees,offsets,incidents,status,scanStatus:scan.status,
  encode(encoder){encoder.clearBuffer(degrees);encoder.clearBuffer(cursors);encoder.clearBuffer(status);if(p.edgeCount){const pass=encoder.beginComputePass({label:'VKF contact degrees'});pass.setPipeline(count);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(p.edgeCount/128));pass.end();}scan.encode(encoder);if(p.edgeCount){const pass=encoder.beginComputePass({label:'VKF contact incidences'});pass.setPipeline(scatter);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(p.edgeCount/128));pass.end();}},
  destroy(){scan.destroy();for(const b of owned)b.destroy();}};
}
