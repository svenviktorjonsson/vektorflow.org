// Contact-core motion adapter. Position writes are preceded by continuous
// collision bounds. A bounded event budget defers time; it never consumes an
// unchecked motion tail or uses depenetration as a physical contact response.
export const PREVENTIVE_PARTICLE_CONTACT_WGSL = /* wgsl */`
struct MotionParams { counts:vec4<u32>,grid:vec4<u32>,minimum:vec4<f32>,wheel:vec4<f32>,material:vec4<f32>,schedule:vec4<f32> };
@group(2) @binding(0) var<uniform> motion:MotionParams;
@group(2) @binding(1) var<storage,read_write> motion_particles:array<vec4<u32>>;
@group(2) @binding(2) var<storage,read_write> motion_cells:array<atomic<u32>>;
@group(2) @binding(3) var<storage,read_write> motion_items:array<atomic<u32>>;
@group(2) @binding(4) var<storage,read_write> motion_control:array<atomic<u32>>;
// control: remainder, wrapped pose, law time, omega, TOI, speed, fault,
// operation-law flag, horizon, event duration, reserved, reserved, frame angle.
fn motion_value(i:u32)->f32{return bitcast<f32>(atomicLoad(&motion_control[i]));}
fn motion_store(i:u32,v:f32){atomicStore(&motion_control[i],bitcast<u32>(v));}
fn motion_force_ready()->bool{return motion.schedule.z>0.5&&motion_value(0)==0.0&&motion_value(2)+motion.schedule.x<=motion.schedule.y+1.0e-7&&atomicLoad(&motion_control[6])==0u;}
fn motion_state(i:u32)->vec4<f32>{return bitcast<vec4<f32>>(motion_particles[i*motion.counts.y]);}
fn motion_write(i:u32,s:vec4<f32>){motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(s);}
fn motion_cell(p:vec2<f32>)->vec2<i32>{return vec2<i32>(floor((p-motion.minimum.xy)/motion.minimum.w));}
fn motion_valid(c:vec2<i32>)->bool{return all(c>=vec2<i32>(0))&&all(c<vec2<i32>(motion.grid.xy));}
fn motion_bucket(c:vec2<i32>)->u32{return u32(c.x)+motion.grid.x*u32(c.y);}
fn motion_distance(p:vec2<f32>,a:vec2<f32>,b:vec2<f32>)->f32{let edge=b-a;return length(p-a-edge*clamp(dot(p-a,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0));}
fn motion_pair_toi(s:vec2<f32>,d:vec2<f32>,radius:f32)->f32{
  let b=dot(s,d);if(b>=0.0){return 1.0;}let length_s=length(s);let c=(length_s-radius)*(length_s+radius);if(c<=0.0){return 0.0;}
  let a=dot(d,d);let discriminant=b*b-a*c;if(a<=1.0e-20||discriminant<=0.0){return 1.0;}
  return clamp(c/(-b+sqrt(discriminant)),0.0,1.0);
}
// Certified intervals for an actual rotating segment, not endpoint sampling or
// a straight chord substituted for its arc. Distance is 1-Lipschitz in point
// motion and the segment's Hausdorff motion. Uncertified intervals are deferred.
fn motion_bar_toi(p:vec2<f32>,d:vec2<f32>,segment:u32,angle:f32,delta:f32)->f32{
  let bar=baffle(segment);let radius=motion.minimum.z+WHEEL_BAR_HALF_WIDTH+motion.material.w;
  let offset=p-motion.wheel.xy;
  let perpendicular=vec2<f32>(-offset.y,offset.x);
  let speed=length(d-delta*perpendicular)+abs(delta)*length(d);
  let acceleration=2.0*abs(delta)*length(d)+delta*delta*(length(offset)+length(d));
  let edge=bar.zw-bar.xy;
  let normal=vec2<f32>(-edge.y,edge.x)/max(length(edge),1.0e-20);
  if(speed==0.0){return 1.0;}
  var stack:array<vec2<f32>,32>;var top=1u;stack[0]=vec2<f32>(0.0,1.0);
  for(var visit=0u;visit<96u;visit++){
    if(top==0u){return 1.0;}top--;let interval=stack[top];let mid=(interval.x+interval.y)*0.5;
    let local=rotate_local(p+d*mid-motion.wheel.xy,-angle-delta*mid);
    let half=(interval.y-interval.x)*0.5;
    let at=offset+d*mid;
    let derivative=rotate_local(d-delta*vec2<f32>(-at.y,at.x),-angle-delta*mid);
    let plane_clearance=abs(dot(local-bar.xy,normal))-abs(dot(derivative,normal))*half-0.5*acceleration*half*half;
    if(plane_clearance>=radius||motion_distance(local,bar.xy,bar.zw)-speed*half>=radius){continue;}
    if(interval.y-interval.x<=1.0e-6||top+2u>32u){return interval.x;}
    stack[top]=vec2<f32>(mid,interval.y);stack[top+1u]=vec2<f32>(interval.x,mid);top+=2u;
  }
  if(top>0u){return stack[top-1u].x;}return 1.0;
}
@compute @workgroup_size(128)
fn motion_pause_velocity(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x>=motion.counts.x||motion.schedule.z>0.5||motion_value(0)>0.0||abs(motion.wheel.z)<1.0e-7){return;}let state=motion_state(gid.x);motion_write(gid.x,vec4<f32>(state.xy,0.0,0.0));}
@compute @workgroup_size(1)
fn motion_begin(){
  if(atomicLoad(&motion_control[6])!=0u||motion_value(0)>0.0){return;}
  if(motion_force_ready()){motion_store(0,motion.schedule.x);atomicStore(&motion_control[7],1u);}
  else if(motion.schedule.z<0.5&&abs(motion.wheel.z-motion_value(12))>1.0e-7){motion_store(0,motion.schedule.x);atomicStore(&motion_control[7],0u);}
  else{return;}
  let credit=select(motion.schedule.x,max(motion.schedule.x,motion.schedule.y-motion_value(2)),motion.schedule.z>0.5);
  let angular_velocity=(motion.wheel.z-motion_value(12))/credit;
  if((bitcast<u32>(angular_velocity)&0x7f800000u)==0x7f800000u){atomicOr(&motion_control[6],1u);return;}
  motion_store(3,angular_velocity);
}
@compute @workgroup_size(1)
fn motion_reset_speed(){atomicStore(&motion_control[5],0u);}
@compute @workgroup_size(128)
fn motion_measure_speed(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&motion_value(0)>0.0){let state=motion_state(gid.x);if(any((bitcast<vec4<u32>>(state)&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))){atomicOr(&motion_control[6],1u);}else{atomicMax(&motion_control[5],bitcast<u32>(length(state.zw)));}}}
@compute @workgroup_size(1)
fn motion_prepare(){motion_store(4,1.0);let horizon=min(motion_value(0),min(motion.minimum.z*1.2/max(motion_value(5),1.0e-12),0.2/max(abs(motion_value(3)),1.0e-12)));motion_store(8,horizon);motion_store(9,0.0);}
@compute @workgroup_size(128)
fn motion_clear(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.grid.x*motion.grid.y&&motion_value(0)>0.0){atomicStore(&motion_cells[gid.x],0u);}}
fn motion_insert(i:u32,swept:bool){
  if(i>=motion.counts.x||motion_value(0)==0.0){return;}let state=motion_state(i);let end=state.xy+state.zw*select(0.0,motion_value(8),swept);
  let padding=select(0.0,motion.minimum.z+motion.material.w,swept);let lo=motion_cell(min(state.xy,end)-padding);let hi=motion_cell(max(state.xy,end)+padding);
  if(!motion_valid(lo)||!motion_valid(hi)){atomicOr(&motion_control[6],2u);return;}
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let bucket=motion_bucket(vec2<i32>(x,y));let slot=atomicAdd(&motion_cells[bucket],1u);if(slot<motion.grid.z){atomicStore(&motion_items[bucket*motion.grid.z+slot],i);}else{atomicOr(&motion_control[6],4u);}}}
}
@compute @workgroup_size(128) fn motion_fill_swept(@builtin(global_invocation_id) gid:vec3<u32>){motion_insert(gid.x,true);}
@compute @workgroup_size(128) fn motion_fill_static(@builtin(global_invocation_id) gid:vec3<u32>){motion_insert(gid.x,false);}
@compute @workgroup_size(128)
fn motion_sort(@builtin(global_invocation_id) gid:vec3<u32>){let bucket=gid.x;if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)==0.0){return;}let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);let base=bucket*motion.grid.z;for(var i=1u;i<count;i++){let key=atomicLoad(&motion_items[base+i]);var j=i;while(j>0u){let previous=atomicLoad(&motion_items[base+j-1u]);if(previous<=key){break;}atomicStore(&motion_items[base+j],previous);j--;}atomicStore(&motion_items[base+j],key);}}
@compute @workgroup_size(128)
fn motion_find_toi(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}let state=motion_state(i);let horizon=motion_value(8);let d=state.zw*horizon;let radius=motion.minimum.z;let padding=radius+motion.material.w;var fraction=1.0;
  let lo=motion_cell(min(state.xy,state.xy+d)-padding);let hi=motion_cell(max(state.xy,state.xy+d)+padding);
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let bucket=motion_bucket(vec2<i32>(x,y));let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);if(j<=i){continue;}let other=motion_state(j);fraction=min(fraction,motion_pair_toi(state.xy-other.xy,d-other.zw*horizon,2.0*radius+motion.material.w));}}}
  let offset=state.xy-motion.wheel.xy;let limit=WHEEL_RADIUS-radius-WHEEL_BAR_HALF_WIDTH-motion.material.w;let a=dot(d,d);let b=dot(offset,d);let c=(length(offset)-limit)*(length(offset)+limit);
  if(c>=0.0&&b>0.0){fraction=0.0;}
  else if(a>1.0e-20&&b*b-a*c>=0.0){let exit=(-b+sqrt(max(0.0,b*b-a*c)))/a;if(exit>=0.0&&exit<1.0){fraction=min(fraction,exit);}}
  for(var segment=0u;segment< motion.counts.w;segment++){fraction=min(fraction,motion_bar_toi(state.xy,d,segment,motion_value(1),motion_value(3)*horizon));}
  // Positive IEEE float bits are ordered; canonicalize negative zero too.
  atomicMin(&motion_control[4],bitcast<u32>(fraction)&0x7fffffffu);
}
@compute @workgroup_size(1)
fn motion_advance_control(){
  if(atomicLoad(&motion_control[6])!=0u){return;}let fraction=motion_value(4);let duration=motion_value(8)*select(fraction*0.999,1.0,fraction>=1.0);motion_store(9,duration);
  motion_store(0,max(0.0,motion_value(0)-duration));let delta=motion_value(3)*duration;let angle=motion_value(1)+delta;motion_store(1,atan2(sin(angle),cos(angle)));motion_store(12,motion_value(12)+delta);
  if(atomicLoad(&motion_control[7])!=0u){motion_store(2,motion_value(2)+duration);}
}
@compute @workgroup_size(128)
fn motion_advance_particles(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&atomicLoad(&motion_control[6])==0u){let state=motion_state(gid.x);motion_write(gid.x,vec4<f32>(state.xy+state.zw*motion_value(9),state.zw));}}
fn motion_wall_impulse(i:u32,normal:vec2<f32>,surface:vec2<f32>,friction:f32){
  let state=motion_state(i);var velocity=state.zw;let relative=velocity-surface;let vn=dot(relative,normal);
  // A tiny separating velocity keeps curved tangential contacts from an exact-
  // touching zero-TOI deadlock in finite precision. No position is repaired.
  let separating_speed=motion.material.w/max(motion.schedule.x,1.0e-6);
  if(vn>=separating_speed){return;}let jn=separating_speed-vn;velocity+=normal*jn;let tangent=relative-normal*vn;let speed=length(tangent);velocity-=tangent/max(speed,1.0e-20)*min(speed,friction*jn);motion_write(i,vec4<f32>(state.xy,velocity));
}
fn motion_solve_cell(bucket:u32,color:u32){
  if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}
  let cell=vec2<i32>(i32(bucket%motion.grid.x),i32(bucket/motion.grid.x));if(u32(cell.x%3)+3u*u32(cell.y%3)!=color){return;}
  let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);let diameter=2.0*motion.minimum.z;let band=diameter+motion.material.w*8.0;
  for(var slot=0u;slot<count;slot++){
    let i=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let neighbor=cell+vec2<i32>(x,y);if(!motion_valid(neighbor)){continue;}let other_bucket=motion_bucket(neighbor);let other_count=min(atomicLoad(&motion_cells[other_bucket]),motion.grid.z);
      for(var other_slot=0u;other_slot<other_count;other_slot++){let j=atomicLoad(&motion_items[other_bucket*motion.grid.z+other_slot]);if(j<=i){continue;}let own=motion_state(i);let other=motion_state(j);let separation=own.xy-other.xy;let distance=length(separation);if(distance>band||distance<1.0e-12){continue;}let normal=separation/distance;let relative=own.zw-other.zw;let vn=dot(relative,normal);if(vn>=0.0){continue;}
        let bounce=select(0.0,motion.material.x,vn < -0.3);let jn=-(1.0+bounce)*vn*0.5;let tangent=relative-vn*normal;let speed=length(tangent);let impulse=normal*jn-tangent/max(speed,1.0e-20)*min(speed*0.5,motion.material.y*jn);
        motion_write(i,vec4<f32>(own.xy,own.zw+impulse));motion_write(j,vec4<f32>(other.xy,other.zw-impulse));
      }
    }}
    let own=motion_state(i);let offset=own.xy-motion.wheel.xy;let radius=length(offset);let clearance=motion.minimum.z+WHEEL_BAR_HALF_WIDTH;
    if(radius>=WHEEL_RADIUS-clearance-motion.material.w*8.0){motion_wall_impulse(i,-offset/max(radius,1.0e-20),vec2<f32>(0.0),motion.material.z);}
    let local=rotate_local(own.xy-motion.wheel.xy,-motion_value(1));
    for(var segment=0u;segment<motion.counts.w;segment++){let bar=baffle(segment);let edge=bar.zw-bar.xy;let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let separation=local-closest;let distance=length(separation);if(distance<=clearance+motion.material.w*8.0&&distance>1.0e-12){let normal=rotate_local(separation/distance,motion_value(1));let point=rotate_local(closest,motion_value(1));let surface=vec2<f32>(-point.y,point.x)*motion_value(3);motion_wall_impulse(i,normal,surface,motion.schedule.w);}}
  }
}
// Nine colors have disjoint one-cell write neighborhoods. Gauss-Seidel pair
// impulses remain equal-and-opposite without atomics or Jacobi impulse summing.
@compute @workgroup_size(128) fn motion_contacts_0(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,0u);}
@compute @workgroup_size(128) fn motion_contacts_1(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,1u);}
@compute @workgroup_size(128) fn motion_contacts_2(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,2u);}
@compute @workgroup_size(128) fn motion_contacts_3(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,3u);}
@compute @workgroup_size(128) fn motion_contacts_4(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,4u);}
@compute @workgroup_size(128) fn motion_contacts_5(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,5u);}
@compute @workgroup_size(128) fn motion_contacts_6(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,6u);}
@compute @workgroup_size(128) fn motion_contacts_7(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,7u);}
@compute @workgroup_size(128) fn motion_contacts_8(@builtin(global_invocation_id) gid:vec3<u32>){motion_solve_cell(gid.x,8u);}
`;

export function createPreventiveContactResources(device,world,count){
  const radius=world.kind==='granular'?world.properties.radius:world.properties.spacing*.46,cellWidth=radius*2*1.1,margin=1e-6;
  if(!(radius>margin*64)||!Number.isSafeInteger(count)||count<1)throw new Error('Hard-contact radius/count is outside this metre-scale specialization');
  const side=Math.ceil((world.geometry.radius*2+radius*8)/cellWidth),minimum=world.geometry.center.map(v=>v-world.geometry.radius-radius*4),make=(size,usage)=>device.createBuffer({size,usage});
  const maximumBytes=Math.min(device.limits?.maxBufferSize??Number.MAX_SAFE_INTEGER,device.limits?.maxStorageBufferBindingSize??Number.MAX_SAFE_INTEGER);
  if(!Number.isSafeInteger(side)||side<1||!Number.isSafeInteger(side*side*64*4)||side*side*64*4>maximumBytes||Math.ceil(side*side/128)>(device.limits?.maxComputeWorkgroupsPerDimension??65535))throw new Error('Preventive contact broadphase exceeds this GPU device limit');
  const uniform=make(96,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),cells=make(side*side*4,GPUBufferUsage.STORAGE),items=make(side*side*64*4,GPUBufferUsage.STORAGE),control=make(64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),readback=make(64,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const emptyLayout=device.createBindGroupLayout({entries:[]}),forceLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),forceGroup=device.createBindGroup({layout:forceLayout,entries:[{binding:0,resource:{buffer:uniform}},{binding:4,resource:{buffer:control}}]});
  const initialControl=new Float32Array(16);initialControl[1]=world.geometry.rotation;const reset=()=>device.queue.writeBuffer(control,0,initialControl);reset();
  const emptyGroup=device.createBindGroup({layout:emptyLayout,entries:[]});
  return {uniform,cells,items,control,readback,emptyLayout,emptyGroup,forceLayout,forceGroup,side,minimum,radius,cellWidth,margin,count,reset,destroy(){for(const b of [uniform,cells,items,control,readback])b.destroy();}};
}

export async function createPreventiveParticleContactGpu(device,world,physics,source,resources){
  const r=resources,layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,layout]}),module=device.createShaderModule({code:source});
  const entries=['motion_pause_velocity','motion_begin','motion_reset_speed','motion_measure_speed','motion_prepare','motion_clear','motion_fill_swept','motion_fill_static','motion_sort','motion_find_toi','motion_advance_control','motion_advance_particles',...Array.from({length:9},(_,i)=>`motion_contacts_${i}`)];
  const pipelines=Object.fromEntries(await Promise.all(entries.map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
  const group=device.createBindGroup({layout,entries:[r.uniform,physics.particleBuffer,r.cells,r.items,r.control].map((buffer,binding)=>({binding,resource:{buffer}}))}),empty=r.emptyGroup;
  const params=new ArrayBuffer(96),u=new Uint32Array(params),f=new Float32Array(params);u.set([r.count,world.kind==='granular'?2:3,0,world.geometry.segments.length,r.side,r.side,64,0]);f.set([...r.minimum,r.radius,r.cellWidth],8);f.set([...world.geometry.center,0,0],12);const sand=world.kind==='granular';f.set([world.properties.restitution??0,sand?(world.properties.friction??.9):0,sand?(world.properties.wall_friction??.78):0,r.margin],16);
  const dispatch=(pass,name,count)=>{pass.setPipeline(pipelines[name]);pass.dispatchWorkgroups(Math.ceil(count/128));};
  return {
    resources:r,
    beginFrame(encoder,{requestedDelta,windowDuration,timeLimit,paused}){f[14]=requestedDelta;f[15]=windowDuration;f.set([paused?windowDuration:world.time_step,timeLimit,paused?0:1,sand?(world.properties.wall_friction??.78):0],20);device.queue.writeBuffer(r.uniform,0,params);encoder.clearBuffer(r.control,48,4);},
    advance(encoder){const pass=encoder.beginComputePass({label:'VKF preventive contact events'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);pass.setBindGroup(2,group);dispatch(pass,'motion_pause_velocity',r.count);pass.setPipeline(pipelines.motion_begin);pass.dispatchWorkgroups(1);
      for(let event=0;event<8;event++){
        pass.setPipeline(pipelines.motion_reset_speed);pass.dispatchWorkgroups(1);dispatch(pass,'motion_measure_speed',r.count);pass.setPipeline(pipelines.motion_prepare);pass.dispatchWorkgroups(1);
        dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_swept',r.count);dispatch(pass,'motion_sort',r.side*r.side);dispatch(pass,'motion_find_toi',r.count);pass.setPipeline(pipelines.motion_advance_control);pass.dispatchWorkgroups(1);dispatch(pass,'motion_advance_particles',r.count);
        dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_static',r.count);dispatch(pass,'motion_sort',r.side*r.side);for(let color=0;color<9;color++)dispatch(pass,`motion_contacts_${color}`,r.side*r.side);
      }pass.end();
    },
    finishFrame(encoder){encoder.copyBufferToBuffer(r.control,0,r.readback,0,64);},
    async inspect(){await r.readback.mapAsync(GPUMapMode.READ);const bytes=r.readback.getMappedRange().slice(0);r.readback.unmap();const f=new Float32Array(bytes),u=new Uint32Array(bytes);if(u[6]||![f[0],f[1],f[2],f[12]].every(Number.isFinite))throw new Error(`Preventive contact state/broadphase failure ${u[6]}; no unchecked motion tail was advanced.`);return {remainingTime:f[0],angle:f[1],time:f[2],angularDelta:f[12]};},
    reset:r.reset,destroy:r.destroy,
  };
}
