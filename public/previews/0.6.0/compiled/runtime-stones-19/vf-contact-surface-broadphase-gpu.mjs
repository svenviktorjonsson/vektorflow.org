// Geometry-neutral swept triangle broadphase. Hash collisions add work, never
// erase candidates: every node retains its exact signed cell coordinate.
// Overflow is a rejected World proposal, not a truncated contact set.
export const CONTACT_SURFACE_BROADPHASE_WGSL=/* wgsl */`
struct Params { vertices:u32,triangles:u32,capacity:u32,slots:u32,buckets:u32,cell_size:f32,reserve:f32,node_capacity:u32 };
struct Motion { start:vec4<f32>,finish:vec4<f32> };
struct Bounds { lo:vec4<f32>,hi:vec4<f32>,cell_lo:vec4<i32>,cell_hi:vec4<i32> };
struct WindowBounds { lo:vec3<f32>,hi:vec3<f32> };
struct Node { triangle:u32,cx:i32,cy:i32,cz:i32,next:u32,pad:vec2<u32> };
struct Pair { ids:vec4<u32>,kind:u32,thickness:f32,pad:vec2<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> motion:array<Motion>;
@group(0) @binding(2) var<storage,read> topology:array<vec4<u32>>;
@group(0) @binding(3) var<storage,read> thickness:array<f32>;
@group(0) @binding(4) var<storage,read_write> bounds:array<Bounds>;
@group(0) @binding(5) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> nodes:array<Node>;
@group(0) @binding(7) var<storage,read_write> pairs:array<Pair>;
@group(0) @binding(8) var<storage,read_write> active_count:array<atomic<u32>>;
@group(0) @binding(9) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read_write> next_node:array<atomic<u32>>;
fn finite(x:f32)->bool{return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn finite3(x:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(x)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn cell_hash(c:vec3<i32>)->u32{
 return ((bitcast<u32>(c.x)*0x8da6b343u)^(bitcast<u32>(c.y)*0xd8163841u)^
  (bitcast<u32>(c.z)*0xcb1ab31fu))&(params.buckets-1u);
}
fn empty_bounds()->Bounds{return Bounds(vec4<f32>(0.0),vec4<f32>(0.0),
 vec4<i32>(1,1,1,0),vec4<i32>(0,0,0,0));}
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.triangles){return;}
 let ids=topology[i];let t=thickness[i];
 var valid=ids.x<params.vertices&&ids.y<params.vertices&&ids.z<params.vertices&&
  ids.x!=ids.y&&ids.x!=ids.z&&ids.y!=ids.z&&finite(t)&&t>=0.0;
 if(!valid){bounds[i]=empty_bounds();atomicOr(&status[0],1u);return;}
 var lo=vec3<f32>(3.402823e38);var hi=-lo;var scale=1.0;var moving=false;
 for(var k=0u;k<3u;k++){
  let v=motion[ids[k]];valid=valid&&finite3(v.start.xyz)&&finite3(v.finish.xyz);
  moving=moving||v.start.w>0.0;
  lo=min(lo,min(v.start.xyz,v.finish.xyz));hi=max(hi,max(v.start.xyz,v.finish.xyz));
  scale=max(scale,max(max(abs(v.start.x),abs(v.start.y)),max(abs(v.start.z),
   max(max(abs(v.finish.x),abs(v.finish.y)),abs(v.finish.z)))));
 }
 if(!valid){bounds[i]=empty_bounds();atomicOr(&status[0],1u);return;}
 let expansion=t+max(params.reserve,2e-6*scale);lo-=vec3<f32>(expansion);hi+=vec3<f32>(expansion);
 let low=floor(lo/params.cell_size);let high=floor(hi/params.cell_size);
 if(!finite3(low)||!finite3(high)||any(low<vec3<f32>(-8000000.0))||
  any(high>vec3<f32>(8000000.0))){bounds[i]=empty_bounds();atomicOr(&status[0],1u);return;}
 let a=vec3<i32>(low);let b=vec3<i32>(high);let span=b-a+vec3<i32>(1);
 if(any(span<=vec3<i32>(0))||f32(span.x)*f32(span.y)*f32(span.z)>f32(params.slots)){
  bounds[i]=empty_bounds();atomicOr(&status[0],2u);return;
 }
 bounds[i]=Bounds(vec4<f32>(lo,t),vec4<f32>(hi,0.0),vec4<i32>(a,select(0,1,moving)),vec4<i32>(b,0));
}
@compute @workgroup_size(64) fn insert(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.triangles){return;}let b=bounds[i];
 if(any(b.cell_lo.xyz>b.cell_hi.xyz)){return;}
 let span=b.cell_hi.xyz-b.cell_lo.xyz+vec3<i32>(1);
 let count=u32(span.x*span.y*span.z);
 let base=atomicAdd(&next_node[0],count);
 if(base>=params.node_capacity||count>params.node_capacity-base){atomicOr(&status[0],2u);return;}
 var slot=0u;
 for(var z=b.cell_lo.z;z<=b.cell_hi.z;z++){
  for(var y=b.cell_lo.y;y<=b.cell_hi.y;y++){
   for(var x=b.cell_lo.x;x<=b.cell_hi.x;x++){
    let c=vec3<i32>(x,y,z);let at=base+slot;slot+=1u;
    nodes[at]=Node(i,x,y,z,0u,vec2<u32>(0u));
    nodes[at].next=atomicExchange(&heads[cell_hash(c)],at+1u);
   }
  }
 }
}
fn shares_edge(a:vec4<u32>,b:vec4<u32>)->bool{
 var matches=0u;
 for(var x=0u;x<3u;x++){for(var y=0u;y<3u;y++){
  matches+=select(0u,1u,a[x]==b[y]);
 }}return matches>=2u;
}
fn overlap(a:Bounds,b:Bounds)->bool{
 return all(a.lo.xyz<=b.hi.xyz)&&all(b.lo.xyz<=a.hi.xyz);
}
fn valid_row(ids:vec4<u32>)->bool{
 for(var x=0u;x<4u;x++){for(var y=x+1u;y<4u;y++){
  if(ids[x]==ids[y]){return false;}
 }}return true;
}
fn feature_bounds(ids:vec4<u32>,begin:u32,count:u32)->WindowBounds{
 var lo=vec3<f32>(3.402823e38);var hi=-lo;
 for(var k=0u;k<count;k++){
  let vertex=motion[ids[begin+k]];
  lo=min(lo,min(vertex.start.xyz,vertex.finish.xyz));
  hi=max(hi,max(vertex.start.xyz,vertex.finish.xyz));
 }
 return WindowBounds(lo,hi);
}
fn feature_sweeps_overlap(ids:vec4<u32>,kind:u32,gap:f32)->bool{
 let first_count=select(1u,2u,kind==1u);
 let first=feature_bounds(ids,0u,first_count);
 let second=feature_bounds(ids,first_count,4u-first_count);
 let expansion=vec3<f32>(gap+params.reserve*2.0);
 return all(first.lo<=second.hi+expansion)&&all(second.lo<=first.hi+expansion);
}
fn write_row(ids:vec4<u32>,kind:u32,gap:f32){
 if(!valid_row(ids)||!feature_sweeps_overlap(ids,kind,gap)){return;}
 let at=atomicAdd(&active_count[0],1u);
 if(at>=params.capacity){atomicOr(&status[0],4u);return;}
 pairs[at]=Pair(ids,kind,gap,vec2<f32>(0.0));
}
fn write_rows(a:vec4<u32>,b:vec4<u32>,gap:f32){
 for(var k=0u;k<3u;k++){
  write_row(vec4<u32>(a[k],b.x,b.y,b.z),0u,gap);
  write_row(vec4<u32>(b[k],a.x,a.y,a.z),0u,gap);
 }
 for(var x=0u;x<3u;x++){for(var y=0u;y<3u;y++){
  write_row(vec4<u32>(a[x],a[(x+1u)%3u],b[y],b[(y+1u)%3u]),1u,gap);
 }}
}
fn projected_range(ids:vec4<u32>,axis:vec3<f32>)->vec2<f32>{
 var lo=3.402823e38;var hi=-lo;
 for(var corner=0u;corner<3u;corner++){
  let sample=motion[ids[corner]];
  let first=dot(sample.start.xyz,axis);let last=dot(sample.finish.xyz,axis);
  lo=min(lo,min(first,last));hi=max(hi,max(first,last));
 }
 return vec2<f32>(lo,hi);
}
fn plane_separates(a:vec4<u32>,b:vec4<u32>,axis:vec3<f32>,gap:f32)->bool{
 let magnitude=length(axis);if(magnitude<1e-10){return false;}
 let unit=axis/magnitude;let first=projected_range(a,unit);let second=projected_range(b,unit);
 return first.y+gap<second.x||second.y+gap<first.x;
}
fn swept_planes_separate(a:vec4<u32>,b:vec4<u32>,gap:f32)->bool{
 let a0=motion[a.x];let a1=motion[a.y];let a2=motion[a.z];
 let b0=motion[b.x];let b1=motion[b.y];let b2=motion[b.z];
 return plane_separates(a,b,cross(a1.start.xyz-a0.start.xyz,a2.start.xyz-a0.start.xyz),gap)||
  plane_separates(a,b,cross(a1.finish.xyz-a0.finish.xyz,a2.finish.xyz-a0.finish.xyz),gap)||
  plane_separates(a,b,cross(b1.start.xyz-b0.start.xyz,b2.start.xyz-b0.start.xyz),gap)||
  plane_separates(a,b,cross(b1.finish.xyz-b0.finish.xyz,b2.finish.xyz-b0.finish.xyz),gap);
}
// A fixed edge-cross-edge axis is a valid separator for every linearly swept
// vertex: each projection stays between its start and finish values. These
// axes reject skewed triangles that all four face-normal tests retain.
fn swept_edges_separate(a:vec4<u32>,b:vec4<u32>,gap:f32)->bool{
 let a0=motion[a.x];let a1=motion[a.y];let a2=motion[a.z];
 let b0=motion[b.x];let b1=motion[b.y];let b2=motion[b.z];
 for(var phase=0u;phase<2u;phase++){
  let finish=phase==1u;
  let pa0=select(a0.start.xyz,a0.finish.xyz,finish);
  let pa1=select(a1.start.xyz,a1.finish.xyz,finish);
  let pa2=select(a2.start.xyz,a2.finish.xyz,finish);
  let pb0=select(b0.start.xyz,b0.finish.xyz,finish);
  let pb1=select(b1.start.xyz,b1.finish.xyz,finish);
  let pb2=select(b2.start.xyz,b2.finish.xyz,finish);
  let ea=array<vec3<f32>,3>(pa1-pa0,pa2-pa1,pa0-pa2);
  let eb=array<vec3<f32>,3>(pb1-pb0,pb2-pb1,pb0-pb2);
  for(var x=0u;x<3u;x++){
   for(var y=0u;y<3u;y++){
    if(plane_separates(a,b,cross(ea[x],eb[y]),gap)){return true;}
   }
  }
 }
 return false;
}
fn window_bounds(ids:vec4<u32>,left:f32,right:f32)->WindowBounds{
 var lo=vec3<f32>(3.402823e38);var hi=-lo;
 for(var corner=0u;corner<3u;corner++){
  let vertex=motion[ids[corner]];
  let first=mix(vertex.start.xyz,vertex.finish.xyz,left);
  let last=mix(vertex.start.xyz,vertex.finish.xyz,right);
  lo=min(lo,min(first,last));hi=max(hi,max(first,last));
 }
 return WindowBounds(lo,hi);
}
// The union of four synchronous slabs covers the whole linear motion. A
// pair separated in every slab cannot touch even when full-step AABBs cross.
fn time_slabs_separate(a:vec4<u32>,b:vec4<u32>,gap:f32)->bool{
 for(var slab=0u;slab<4u;slab++){
  let left=f32(slab)*0.25;let right=left+0.25;
  let first=window_bounds(a,left,right);
  let second=window_bounds(b,left,right);
  if(all(first.lo<=second.hi+vec3<f32>(gap))&&
    all(second.lo<=first.hi+vec3<f32>(gap))){return false;}
 }
 return true;
}
@compute @workgroup_size(64) fn emit(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.triangles||atomicLoad(&status[0])!=0u){return;}let a=bounds[i];
 if(a.cell_lo.w==0){return;}
 for(var z=a.cell_lo.z;z<=a.cell_hi.z;z++){
  for(var y=a.cell_lo.y;y<=a.cell_hi.y;y++){
   for(var x=a.cell_lo.x;x<=a.cell_hi.x;x++){
    let c=vec3<i32>(x,y,z);var link=atomicLoad(&heads[cell_hash(c)]);var steps=0u;
    while(link!=0u&&steps<params.node_capacity){
     if(atomicLoad(&status[0])!=0u){return;}
     let node=nodes[link-1u];link=node.next;steps+=1u;
     let j=node.triangle;if(j==i||node.cx!=x||node.cy!=y||node.cz!=z){continue;}
     let b=bounds[j];if(j<i&&b.cell_lo.w!=0){continue;}
     let first=max(a.cell_lo.xyz,b.cell_lo.xyz);
     if(any(first!=c)||!overlap(a,b)){continue;}
     let ta=topology[i];let tb=topology[j];
     if(ta.w!=0u&&ta.w==tb.w&&shares_edge(ta,tb)){continue;}
     let gap=a.lo.w+b.lo.w;
     if(time_slabs_separate(ta,tb,gap+params.reserve*2.0)){continue;}
     if(swept_planes_separate(ta,tb,gap+params.reserve*2.0)||
       swept_edges_separate(ta,tb,gap+params.reserve*2.0)){continue;}
     write_rows(ta,tb,gap);
    }
    if(link!=0u){atomicOr(&status[0],8u);}
   }
  }
 }
}
`;

export async function createContactSurfaceBroadphaseGpu(device,{vertexCount,triangleCount,pairCapacity,
 motion,topology,thickness,cellSize,cellSlots=64,nodeCapacity=triangleCount*cellSlots,bucketCount=0,reserve=1e-5}){
 if(!Number.isSafeInteger(vertexCount)||vertexCount<4||!Number.isSafeInteger(triangleCount)||triangleCount<2||triangleCount>16384||
  !Number.isSafeInteger(pairCapacity)||pairCapacity<15||pairCapacity%15!==0||
  !Number.isInteger(cellSlots)||cellSlots<1||cellSlots>512||
  !Number.isSafeInteger(nodeCapacity)||nodeCapacity<1||nodeCapacity>0x7fffffff||
  !Number.isFinite(Math.fround(cellSize))||
  Math.fround(cellSize)<=0||!Number.isFinite(Math.fround(reserve))||Math.fround(reserve)<=0)
  throw new RangeError('Invalid swept surface broadphase dimensions');
 if(bucketCount===0)bucketCount=2**Math.ceil(Math.log2(Math.max(64,nodeCapacity*2)));
 if(!Number.isSafeInteger(bucketCount)||bucketCount<2||(bucketCount&(bucketCount-1))!==0)
  throw new RangeError('Swept surface bucket count must be a power of two');
 if(motion.size<vertexCount*32||topology.size<triangleCount*16||thickness.size<triangleCount*4)
  throw new RangeError('Swept surface broadphase input capacity is too small');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 const sizes=[triangleCount*64,bucketCount*4,nodeCapacity*32,pairCapacity*32];
 if(Math.max(...sizes)>limit||Math.ceil(triangleCount/64)>device.limits.maxComputeWorkgroupsPerDimension)
  throw new RangeError('Swept surface broadphase requires GPU batching');
 const owned=[],storage=GPUBufferUsage.STORAGE;
 const make=(size,usage)=>{const buffer=device.createBuffer({size,usage});owned.push(buffer);return buffer;};
 const bounds=make(sizes[0],storage),heads=make(sizes[1],storage|GPUBufferUsage.COPY_DST);
 const nodes=make(sizes[2],storage),pairs=make(sizes[3],storage|GPUBufferUsage.COPY_SRC);
 const activeCount=make(4,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
 const status=make(4,storage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
 const nextNode=make(4,storage|GPUBufferUsage.COPY_DST);
 const uniform=make(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 const data=new ArrayBuffer(32),u=new Uint32Array(data),f=new Float32Array(data);
 u.set([vertexCount,triangleCount,pairCapacity,cellSlots,bucketCount]);f[5]=cellSize;f[6]=reserve;u[7]=nodeCapacity;
 device.queue.writeBuffer(uniform,0,data);
 const shader=device.createShaderModule({label:'VKF swept surface hash broadphase',code:CONTACT_SURFACE_BROADPHASE_WGSL});
 const errors=(await shader.getCompilationInfo()).messages.filter(message=>message.type==='error');
 if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
 const buffers=[uniform,motion,topology,thickness,bounds,heads,nodes,pairs,activeCount,status,nextNode];
 const readOnly=new Set([1,2,3]);
 const build=async(entryPoint,bindings)=>{
  const layout=device.createBindGroupLayout({entries:bindings.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,
   buffer:{type:binding===0?'uniform':readOnly.has(binding)?'read-only-storage':'storage'}}))});
  const group=device.createBindGroup({layout,entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
  const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),
   compute:{module:shader,entryPoint}});return {pipeline,group};
 };
 const stages=await Promise.all([
  build('prepare',[0,1,2,3,4,9]),build('insert',[0,4,5,6,9,10]),
  build('emit',[0,1,2,4,5,6,7,8,9]),
 ]);
 return {pairs,activeCount,status,bounds,encode(encoder){
  encoder.clearBuffer(heads);encoder.clearBuffer(activeCount);encoder.clearBuffer(status);encoder.clearBuffer(nextNode);
  for(const stage of stages){const pass=encoder.beginComputePass();pass.setPipeline(stage.pipeline);
   pass.setBindGroup(0,stage.group);pass.dispatchWorkgroups(Math.ceil(triangleCount/64));pass.end();}
 },destroy(){for(const buffer of owned)buffer.destroy();}};
}
