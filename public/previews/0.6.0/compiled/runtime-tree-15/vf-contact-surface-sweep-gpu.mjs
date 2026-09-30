// Generic four-vertex continuous-contact candidate check. The caller supplies
// broadphase pairs; this module never enumerates all surfaces or commits state.
// Each interval is excluded only when a relative-motion Lipschitz bound proves
// its clearance. At the work limit, a speed-bound prefix may still be safely
// committed; no unchecked remainder is ever advanced.
export const CONTACT_SURFACE_SWEEP_WGSL=/* wgsl */`
struct Params { vertices:u32,capacity:u32,max_depth:u32,budget:u32,margin:f32 };
struct Vertex { start:vec4<f32>, finish:vec4<f32> };
struct Pair { ids:vec4<u32>, kind:u32,thickness:f32,pad:vec2<f32> };
// 0 clear, 1 impact, 2 unresolved, 3 invalid, 4 certified prefix only.
// Prefix-only is not an impulse event.
struct Result { verdict:u32, first_uncertified:f32, minimum_gap:f32, evaluations:u32 };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> vertices:array<Vertex>;
@group(0) @binding(2) var<storage,read> pairs:array<Pair>;
@group(0) @binding(3) var<storage,read> active_count:array<u32>;
@group(0) @binding(4) var<storage,read_write> results:array<Result>;
@group(0) @binding(5) var<storage,read_write> status:array<atomic<u32>>;
fn finite(x:f32)->bool{return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn finite3(x:vec3<f32>)->bool{return all((bitcast<vec3<u32>>(x)&vec3<u32>(0x7f800000u))!=vec3<u32>(0x7f800000u));}
fn position(i:u32,t:f32)->vec3<f32>{let v=vertices[i];return mix(v.start.xyz,v.finish.xyz,t);}
fn velocity(i:u32)->vec3<f32>{return vertices[i].finish.xyz-vertices[i].start.xyz;}
fn point_segment(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->f32{
 let edge=b-a;let denominator=dot(edge,edge);
 let t=select(0.0,clamp(dot(p-a,edge)/max(denominator,1e-30),0.0,1.0),denominator>1e-30);
 return length(p-(a+edge*t));
}
fn vertex_face(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->f32{
 let n=cross(b-a,c-a);let area2=dot(n,n);
 if(area2>1e-24){
  let signed=dot(p-a,n);let q=p-n*(signed/area2);
  let u=dot(cross(b-q,c-q),n)/area2;
  let v=dot(cross(c-q,a-q),n)/area2;
  let w=1.0-u-v;
  // The small outward allowance can underestimate distance near an edge;
  // underestimation is safe for a clearance test.
  if(min(min(u,v),w)>=-1e-5){return abs(signed)/sqrt(area2);}
 }
 return min(point_segment(p,a,b),min(point_segment(p,b,c),point_segment(p,c,a)));
}
fn edge_edge(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>)->f32{
 let u=b-a;let v=d-c;let w=a-c;
 let aa=dot(u,u);let bb=dot(u,v);let cc=dot(v,v);let dd=dot(u,w);let ee=dot(v,w);
 var distance=min(point_segment(a,c,d),min(point_segment(b,c,d),min(point_segment(c,a,b),point_segment(d,a,b))));
 let denominator=aa*cc-bb*bb;
 if(denominator>max(1e-24,aa*cc*1e-7)){
  let s=(bb*ee-cc*dd)/denominator;let t=(aa*ee-bb*dd)/denominator;
  if(s>=0.0&&s<=1.0&&t>=0.0&&t<=1.0){distance=min(distance,length(w+u*s-v*t));}
 }else{
  // Near-parallel arithmetic is ill-conditioned. An AABB gap is a lower
  // bound on the true distance; using it may defer motion but cannot miss it.
  let lo1=min(a,b);let hi1=max(a,b);let lo2=min(c,d);let hi2=max(c,d);
  let gap=max(max(lo1-hi2,lo2-hi1),vec3<f32>(0.0));
  distance=min(distance,length(gap));
 }
 return distance;
}
fn distance_at(pair:Pair,t:f32)->f32{
 let a=position(pair.ids.x,t);let b=position(pair.ids.y,t);
 let c=position(pair.ids.z,t);let d=position(pair.ids.w,t);
 if(pair.kind==0u){return vertex_face(a,b,c,d);}
 return edge_edge(a,b,c,d);
}
fn speed_bound(pair:Pair)->f32{
 let a=velocity(pair.ids.x);let b=velocity(pair.ids.y);
 let c=velocity(pair.ids.z);let d=velocity(pair.ids.w);
 if(pair.kind==0u){return max(length(a-b),max(length(a-c),length(a-d)));}
 return max(max(length(a-c),length(a-d)),max(length(b-c),length(b-d)));
}
fn precision_reserve(pair:Pair)->f32{
 var scale=1.0;
 for(var slot=0u;slot<4u;slot++){
  let vertex=vertices[pair.ids[slot]];
  scale=max(scale,max(max(max(abs(vertex.start.x),abs(vertex.start.y)),abs(vertex.start.z)),
    max(max(abs(vertex.finish.x),abs(vertex.finish.y)),abs(vertex.finish.z))));
 }
 // Reserve at least sixteen f32 ulps at the pair's coordinate scale.
 // Large coordinates become unresolved rather than falsely clear.
 return max(params.margin,2e-6*scale);
}
@compute @workgroup_size(64) fn surface_sweep(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.capacity){return;}
 if(i==0u&&active_count[0]>params.capacity){atomicOr(&status[0],1u);}
 if(i>=active_count[0]){return;}
 let pair=pairs[i];
 var valid=pair.kind<=1u&&finite(pair.thickness)&&pair.thickness>=0.0;
 for(var slot=0u;slot<4u;slot++){
  let id=pair.ids[slot];valid=valid&&id<params.vertices;
  if(id<params.vertices){valid=valid&&finite3(vertices[id].start.xyz)&&finite3(vertices[id].finish.xyz);}
  for(var previous=0u;previous<slot;previous++){valid=valid&&id!=pair.ids[previous];}
 }
 if(!valid){results[i]=Result(3u,0.0,0.0,0u);atomicOr(&status[0],2u);return;}
 let bound=speed_bound(pair);let reserve=precision_reserve(pair);let clearance=pair.thickness+reserve;
 if(!finite(bound)||!finite(clearance)){results[i]=Result(3u,0.0,0.0,0u);atomicOr(&status[0],2u);return;}
 var minimum=distance_at(pair,0.0)-pair.thickness;
 if(!finite(minimum)){results[i]=Result(3u,0.0,0.0,1u);atomicOr(&status[0],2u);return;}
 if(minimum<=reserve){results[i]=Result(1u,0.0,minimum,1u);return;}
 var stack:array<vec4<f32>,24>;stack[0]=vec4<f32>(0.0,1.0,0.0,0.0);
 var top=1u;var evaluations=1u;var verdict=0u;var first=1.0;
 while(top>0u&&evaluations<params.budget){
  top-=1u;let interval=stack[top];let left=interval.x;let right=interval.y;
  let midpoint=(left+right)*0.5;let depth=u32(interval.z);
  let distance=distance_at(pair,midpoint);evaluations+=1u;
  if(!finite(distance)){verdict=3u;first=left;break;}
  minimum=min(minimum,distance-pair.thickness);
  if(distance-bound*(right-left)*0.5-reserve>pair.thickness){continue;}
  if(distance<=clearance){
   // The left endpoint is clear (initial overlap was handled above). The
   // relative-speed bound certifies a prefix even when this midpoint hits.
   // Stay inside that prefix to absorb f32 interpolation roundoff.
   let left_distance=distance_at(pair,left);
   if(!finite(left_distance)){verdict=3u;first=left;break;}
   let left_gap=max(0.0,left_distance-clearance);
   let safe_step=left_gap/max(bound,1e-30)*0.99;
   verdict=1u;first=left+min(midpoint-left,safe_step);break;
  }
  if(depth>=params.max_depth||top+2u>24u||midpoint==left||midpoint==right){
   let left_distance=distance_at(pair,left);
   if(!finite(left_distance)){verdict=3u;first=left;break;}
   let safe_step=max(0.0,left_distance-clearance)/max(bound,1e-30)*0.99;
   verdict=select(4u,1u,left_distance<=clearance);first=left+min(midpoint-left,safe_step);break;
  }
  stack[top]=vec4<f32>(midpoint,right,f32(depth+1u),0.0);
  stack[top+1u]=vec4<f32>(left,midpoint,f32(depth+1u),0.0);
  top+=2u;
 }
 if(verdict==0u&&top>0u){
  let interval=stack[top-1u];let left=interval.x;let right=interval.y;
  let left_distance=distance_at(pair,left);evaluations+=1u;
  if(!finite(left_distance)){verdict=3u;first=left;}
  else{let safe_step=max(0.0,left_distance-clearance)/max(bound,1e-30)*0.99;
   verdict=select(4u,1u,left_distance<=clearance);first=left+min((right-left)*0.5,safe_step);}
 }
 results[i]=Result(verdict,first,minimum,evaluations);
 if(verdict==2u||verdict==3u){atomicOr(&status[0],select(4u,2u,verdict==3u));}
}
`;

export function surfaceSweepLayout({vertexCount,pairCapacity,maxDepth=18,budget=2048,margin=1e-5}){
 if(!Number.isSafeInteger(vertexCount)||vertexCount<4||!Number.isSafeInteger(pairCapacity)||pairCapacity<1||
    !Number.isInteger(maxDepth)||maxDepth<1||maxDepth>20||!Number.isInteger(budget)||budget<2||budget>4096||
    !Number.isFinite(margin)||margin<=0||!Number.isFinite(Math.fround(margin)))
   throw new RangeError('Surface sweep dimensions or precision reserve are invalid');
 return {vertexCount,pairCapacity,maxDepth,budget,margin,vertexBytes:vertexCount*32,pairBytes:pairCapacity*32,resultBytes:pairCapacity*16};
}

export async function createContactSurfaceSweepGpu(device,{vertexCount,pairCapacity,vertices,pairs,activeCount,
 indirectDispatch=null,maxDepth=18,budget=2048,margin=1e-5}){
 const p=surfaceSweepLayout({vertexCount,pairCapacity,maxDepth,budget,margin});
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if(Math.max(p.vertexBytes,p.pairBytes,p.resultBytes)>limit||Math.ceil(pairCapacity/64)>device.limits.maxComputeWorkgroupsPerDimension)
  throw new RangeError('Surface sweep requires GPU batching');
 if(vertices.size<p.vertexBytes||pairs.size<p.pairBytes||activeCount.size<4)throw new RangeError('Surface sweep input capacity is too small');
 if(indirectDispatch!==null&&(!Number.isSafeInteger(indirectDispatch.size)||indirectDispatch.size<12||
   (indirectDispatch.usage&GPUBufferUsage.INDIRECT)===0))
  throw new RangeError('Surface sweep indirect dispatch requires three GPU workgroup counts');
 const uniform=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const data=new ArrayBuffer(32);new Uint32Array(data).set([vertexCount,pairCapacity,maxDepth,budget]);new Float32Array(data)[4]=margin;device.queue.writeBuffer(uniform,0,data);
 const results=device.createBuffer({size:p.resultBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 const status=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
 const module=device.createShaderModule({label:'VKF generic swept surface contact',code:CONTACT_SURFACE_SWEEP_WGSL});
 const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
 if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
 const layout=device.createBindGroupLayout({entries:[
  {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
  ...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}})),
  ...[4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}})),
 ]});
 const group=device.createBindGroup({layout,entries:[uniform,vertices,pairs,activeCount,results,status].map((buffer,binding)=>({binding,resource:{buffer}}))});
 const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'surface_sweep'}});
 return {results,status,layout:p,encode(encoder){encoder.clearBuffer(status);const pass=encoder.beginComputePass({label:'VKF swept surface clearance'});pass.setPipeline(pipeline);pass.setBindGroup(0,group);
   if(indirectDispatch)pass.dispatchWorkgroupsIndirect(indirectDispatch,0);
   else pass.dispatchWorkgroups(Math.ceil(pairCapacity/64));
   pass.end();},destroy(){uniform.destroy();results.destroy();status.destroy();}};
}
