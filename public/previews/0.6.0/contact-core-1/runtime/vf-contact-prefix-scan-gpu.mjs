// Exclusive count scan for compact contact edges, incidences and spatial bins.
// All levels execute on the GPU. No fixed neighbor slots or host count scan.
export const CONTACT_PREFIX_SCAN_WGSL = /* wgsl */`
struct ScanParams { count:u32, blocks:u32, spare:vec2<u32> };
@group(0) @binding(0) var<uniform> params:ScanParams;
@group(0) @binding(1) var<storage,read> source:array<u32>;
@group(0) @binding(2) var<storage,read_write> offsets:array<u32>;
@group(0) @binding(3) var<storage,read_write> totals:array<u32>;
@group(0) @binding(4) var<storage,read_write> fault:array<atomic<u32>>;
var<workgroup> scan_values:array<u32,512>;
fn scan_sum(a:u32,b:u32)->u32{let sum=a+b;if(sum<a){atomicOr(&fault[0],1u);}return sum;}
@compute @workgroup_size(256)
fn scan_block(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let base=group.x*512u;let a=base+lane;let b=a+256u;var av=0u;var bv=0u;
 if(a<params.count){av=source[a];}if(b<params.count){bv=source[b];}
 scan_values[lane]=av;scan_values[lane+256u]=bv;workgroupBarrier();
 for(var stride=1u;stride<512u;stride*=2u){let i=(lane+1u)*stride*2u-1u;if(i<512u){scan_values[i]=scan_sum(scan_values[i-stride],scan_values[i]);}workgroupBarrier();}
 if(lane==0u){totals[group.x]=scan_values[511];scan_values[511]=0u;}workgroupBarrier();
 for(var stride=256u;stride>0u;stride/=2u){let i=(lane+1u)*stride*2u-1u;if(i<512u){let left=scan_values[i-stride];scan_values[i-stride]=scan_values[i];scan_values[i]=scan_sum(left,scan_values[i]);}workgroupBarrier();}
 if(a<params.count){offsets[a]=scan_values[lane];}if(b<params.count){offsets[b]=scan_values[lane+256u];}
}
@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i<params.count){offsets[i]=scan_sum(offsets[i],source[i/512u]);}
 if(i==0u){offsets[params.count]=scan_sum(source[params.blocks-1u],totals[params.blocks-1u]);}
}
`;

export function contactScanLayout(count){
 if(!Number.isSafeInteger(count)||count<1||count>0x7ffffffe)throw new RangeError('Contact scan count must be a positive bounded integer');
 const levels=[];for(let n=count;;){const blocks=Math.ceil(n/512);levels.push({count:n,blocks});if(blocks===1)break;n=blocks;}
 return {levels,outputBytes:(count+1)*4,scratchBytes:levels.reduce((sum,l,index)=>sum+l.blocks*4+(index?(l.count+1)*4:0)+16,24)};
}

export async function createContactPrefixScanGpu(device,{count,input,output}){
 const plan=contactScanLayout(count),limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if(plan.outputBytes>limit||Math.ceil(count/256)>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError('Contact scan requires buffer batching for this GPU');
 if(input.size<count*4||output.size<plan.outputBytes)throw new RangeError('Contact scan input/output capacity is too small');
 const owned=[],make=(size,usage)=>{const b=device.createBuffer({size,usage});owned.push(b);return b;};
 const storage=GPUBufferUsage.STORAGE,paramsUsage=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;
 const status=make(16,storage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),zero=make(8,storage|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(zero,0,new Uint32Array(2));
 const module=device.createShaderModule({label:'VKF compact contact scan',code:CONTACT_PREFIX_SCAN_WGSL});
 const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw new Error(messages.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},...[2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const [scan,add]=await Promise.all(['scan_block','scan_add'].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
 const group=buffers=>device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
 const levels=[];let source=input;
 for(const [index,level] of plan.levels.entries()){
  const target=index?make((level.count+1)*4,storage):output,totals=make(level.blocks*4,storage),uniform=make(16,paramsUsage);device.queue.writeBuffer(uniform,0,new Uint32Array([level.count,level.blocks,0,0]));
  levels.push({...level,uniform,target,totals,scanGroup:group([uniform,source,target,totals,status])});source=totals;
 }
 for(let index=0;index<levels.length;index++){const level=levels[index],offsets=levels[index+1]?.target??zero;level.addGroup=group([level.uniform,offsets,level.target,level.totals,status]);}
 return {count,status,plan,encode(encoder){encoder.clearBuffer(status);const pass=encoder.beginComputePass({label:'VKF compact contact offsets'});for(const l of levels){pass.setPipeline(scan);pass.setBindGroup(0,l.scanGroup);pass.dispatchWorkgroups(l.blocks);}for(let i=levels.length-1;i>=0;i--){const l=levels[i];pass.setPipeline(add);pass.setBindGroup(0,l.addGroup);pass.dispatchWorkgroups(Math.ceil(l.count/256));}pass.end();},destroy(){for(const b of owned)b.destroy();}};
}
