// Contact-core motion adapter. Position writes are preceded by continuous
// collision bounds. A bounded event budget defers time; it never consumes an
// unchecked motion tail or uses depenetration as a physical contact response.
import {ContactSchedulerPolicy} from './vf-contact-scheduler-policy.mjs';
export const PREVENTIVE_PARTICLE_CONTACT_WGSL = /* wgsl */`
struct MotionParams { counts:vec4<u32>,grid:vec4<u32>,minimum:vec4<f32>,wheel:vec4<f32>,material:vec4<f32>,schedule:vec4<f32> };
@group(2) @binding(0) var<uniform> motion:MotionParams;
@group(2) @binding(1) var<storage,read_write> motion_particles:array<vec4<u32>>;
@group(2) @binding(2) var<storage,read_write> motion_cells:array<atomic<u32>>;
@group(2) @binding(3) var<storage,read_write> motion_items:array<atomic<u32>>;
@group(2) @binding(4) var<storage,read_write> motion_control:array<atomic<u32>>;
@group(2) @binding(5) var<storage,read_write> motion_actors:array<vec4<u32>>;
struct MotionEvent { time:f32, i:u32, j:u32, kind:u32, ri:u32, rj:u32, spare:vec2<u32> };
@group(2) @binding(6) var<storage,read_write> motion_incoming:array<MotionEvent>;
@group(2) @binding(7) var<storage,read_write> motion_heap:array<MotionEvent>;
// control: remainder, wrapped pose, law time, omega, TOI, speed, fault,
// operation-law flag, horizon, event duration, reserved, reserved, frame angle.
fn motion_value(i:u32)->f32{return bitcast<f32>(atomicLoad(&motion_control[i]));}
fn motion_store(i:u32,v:f32){atomicStore(&motion_control[i],bitcast<u32>(v));}
fn motion_force_ready()->bool{return motion.schedule.z>0.5&&motion_value(0)==0.0&&motion_value(2)+motion.schedule.x<=motion.schedule.y+1.0e-7&&atomicLoad(&motion_control[6])==0u;}
fn motion_ledger()->bool{return motion.counts.z==1u&&atomicLoad(&motion_control[29])==0u;}
fn motion_state(i:u32)->vec4<f32>{
  if(motion_ledger()&&atomicLoad(&motion_control[18])!=0u){let s=bitcast<vec4<f32>>(motion_actors[i*2u]);let t=bitcast<f32>(motion_actors[i*2u+1u].z);return vec4<f32>(s.xy+s.zw*(motion_value(13)-t),s.zw);}
  return bitcast<vec4<f32>>(motion_particles[i*motion.counts.y]);
}
fn motion_mark(i:u32){
  var m=motion_actors[i*2u+1u];if(m.y==0u){atomicAdd(&motion_control[22],1u);atomicAdd(&motion_control[27],1u);}m.x++;m.y=1u;m.z=bitcast<u32>(motion_value(13));motion_actors[i*2u+1u]=m;motion_actors[i*2u]=motion_particles[i*motion.counts.y];
}
fn motion_write(i:u32,s:vec4<f32>){let old=motion_particles[i*motion.counts.y];motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(s);if(any(old.zw!=bitcast<vec2<u32>>(s.zw))){motion_mark(i);}}
fn motion_cell(p:vec2<f32>)->vec2<i32>{return vec2<i32>(floor((p-motion.minimum.xy)/motion.minimum.w));}
fn motion_valid(c:vec2<i32>)->bool{return all(c>=vec2<i32>(0))&&all(c<vec2<i32>(motion.grid.xy));}
fn motion_bucket(c:vec2<i32>)->u32{return u32(c.x)+motion.grid.x*u32(c.y);}
// Metre-scale contact clearances cannot use implementation-dependent native
// trigonometric approximation. Bounded polynomial rotation avoids the measured
// geometry discrepancy on Intel gen-9 and keeps physics/guard coordinates equal.
fn motion_rotate(p:vec2<f32>,angle:f32)->vec2<f32>{
  let a=angle-6.28318530718*floor((angle+3.14159265359)/6.28318530718);let folded=abs(a)>1.57079632679;let x=select(a,sign(a)*3.14159265359-a,folded);let q=x*x;
  let s=x*(1.0+q*(-0.166666666666667+q*(0.00833333333333333+q*(-0.000198412698412698+q*(0.00000275573192239859+q*(-0.0000000250521083854417+q*0.000000000160590438368216))))));
  let c=(1.0+q*(-0.5+q*(0.0416666666666667+q*(-0.00138888888888889+q*(0.0000248015873015873+q*(-0.000000275573192239859+q*(0.00000000208767569878681-q*0.0000000000114707455977297)))))))*select(1.0,-1.0,folded);
  return vec2<f32>(c*p.x-s*p.y,s*p.x+c*p.y);
}
fn motion_distance(p:vec2<f32>,a:vec2<f32>,b:vec2<f32>)->f32{let edge=b-a;return length(p-a-edge*clamp(dot(p-a,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0));}
fn motion_pair_toi(s:vec2<f32>,d:vec2<f32>,radius:f32)->f32{
  let b=dot(s,d);if(b>=0.0){return 1.0;}let length_s=length(s);let c=(length_s-radius)*(length_s+radius);let a=dot(d,d);
  if(c<=0.0){
    // An f32 impulse can leave tiny closing residuals inside the numerical
    // clearance, causing endless zero-time events. Permit the whole segment
    // only when its analytic closest approach retains 75% of that clearance.
    // This never ignores a trajectory that reaches the physical diameter.
    let closest=s+d*clamp(-b/max(a,1.0e-20),0.0,1.0);
    if(length_s>=radius-motion.material.w*.25&&length(closest)>=radius-motion.material.w*.25){return 1.0;}return 0.0;
  }
  let discriminant=b*b-a*c;if(a<=1.0e-20||discriminant<=0.0){return 1.0;}
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
    let local=motion_rotate(p+d*mid-motion.wheel.xy,-angle-delta*mid);
    let half=(interval.y-interval.x)*0.5;
    let at=offset+d*mid;
    let derivative=motion_rotate(d-delta*vec2<f32>(-at.y,at.x),-angle-delta*mid);
    let plane_clearance=abs(dot(local-bar.xy,normal))-abs(dot(derivative,normal))*half-0.5*acceleration*half*half;
    let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let separation=local-closest;let distance=length(separation);let support=separation/max(distance,1.0e-20);
    // A convex capsule's nearest-point supporting plane also certifies its
    // rounded ends. The old face-only/Lipschitz test stalled tangential motion.
    let capsule_clearance=distance-abs(dot(derivative,support))*half-0.5*acceleration*half*half;
    if(plane_clearance>=radius||capsule_clearance>=radius||distance-speed*half>=radius){continue;}
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
  motion_store(13,0.0);motion_store(31,motion_value(1));atomicStore(&motion_control[18],0u);
}
@compute @workgroup_size(1)
fn motion_reset_speed(){atomicStore(&motion_control[5],0u);}
@compute @workgroup_size(128)
fn motion_measure_speed(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&motion_value(0)>0.0){let state=motion_state(gid.x);if(any((bitcast<vec4<u32>>(state)&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))){atomicOr(&motion_control[6],1u);}else{atomicMax(&motion_control[5],bitcast<u32>(length(state.zw)));}}}
@compute @workgroup_size(1)
fn motion_prepare(){
  motion_store(4,1.0);motion_store(9,0.0);atomicStore(&motion_control[11],1u);atomicStore(&motion_control[19],0xffffffffu);atomicStore(&motion_control[20],0xffffffffu);
  var horizon=min(motion_value(0),min(motion.minimum.z*1.2/max(motion_value(5),1.0e-12),0.2/max(abs(motion_value(3)),1.0e-12)));
  if(!motion_ledger()){atomicStore(&motion_control[18],0u);}
  else if(atomicLoad(&motion_control[18])!=0u){let remaining=motion_value(16)-motion_value(13);if(remaining<=1.0e-8){atomicStore(&motion_control[18],0u);}else{horizon=min(horizon,remaining);}}
  if(motion_ledger()&&atomicLoad(&motion_control[18])==0u){motion_store(16,motion_value(13)+horizon);}
  motion_store(8,horizon);atomicStore(&motion_control[15],0u);
}
@compute @workgroup_size(128) fn motion_ledger_initialize(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<motion.counts.x&&motion_value(0)>0.0&&motion_ledger()&&atomicLoad(&motion_control[18])==0u){motion_actors[i*2u]=motion_particles[i*motion.counts.y];motion_actors[i*2u+1u]=vec4<u32>(0u,1u,bitcast<u32>(motion_value(13)),0u);}}
fn motion_emit(time:f32,i:u32,j:u32,kind:u32){
  if(time>=motion_value(16)){return;}let k=atomicAdd(&motion_control[15],1u);if(k<arrayLength(&motion_incoming)){let ri=motion_actors[i*2u+1u].x;var rj=0u;if(j!=0xffffffffu){rj=motion_actors[j*2u+1u].x;}motion_incoming[k]=MotionEvent(time,i,j,kind,ri,rj,vec2<u32>(0u));}
}
@compute @workgroup_size(128)
fn motion_clear(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.grid.x*motion.grid.y&&motion_value(0)>0.0){atomicStore(&motion_cells[gid.x],0u);}}
fn motion_insert(i:u32,swept:bool){
  if(i>=motion.counts.x||motion_value(0)==0.0){return;}let state=motion_state(i);let horizon=select(motion_value(8),max(0.0,motion_value(16)-motion_value(13)),motion_ledger());let end=state.xy+state.zw*select(0.0,horizon,swept);
  let padding=select(0.0,motion.minimum.z+motion.material.w,swept);let lo=motion_cell(min(state.xy,end)-padding);let hi=motion_cell(max(state.xy,end)+padding);
  if(!motion_valid(lo)||!motion_valid(hi)){atomicOr(&motion_control[6],2u);return;}
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let bucket=motion_bucket(vec2<i32>(x,y));let slot=atomicAdd(&motion_cells[bucket],1u);if(slot<motion.grid.z){atomicStore(&motion_items[bucket*motion.grid.z+slot],i);}else{atomicOr(&motion_control[6],4u);}}}
}
@compute @workgroup_size(128) fn motion_fill_swept(@builtin(global_invocation_id) gid:vec3<u32>){motion_insert(gid.x,true);}
@compute @workgroup_size(128) fn motion_fill_static(@builtin(global_invocation_id) gid:vec3<u32>){motion_insert(gid.x,false);}
@compute @workgroup_size(128)
fn motion_sort(@builtin(global_invocation_id) gid:vec3<u32>){let bucket=gid.x;if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)==0.0){return;}let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);let base=bucket*motion.grid.z;for(var i=1u;i<count;i++){let key=atomicLoad(&motion_items[base+i]);var j=i;while(j>0u){let previous=atomicLoad(&motion_items[base+j-1u]);if(previous<=key){break;}atomicStore(&motion_items[base+j],previous);j--;}atomicStore(&motion_items[base+j],key);}}
fn motion_find_body(i:u32){
  if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}
  let ledger=motion_ledger();let rebuilding=atomicLoad(&motion_control[18])==0u;
  let state=motion_state(i);let horizon=select(motion_value(8),max(0.0,motion_value(16)-motion_value(13)),ledger);let d=state.zw*horizon;let radius=motion.minimum.z;let padding=radius+motion.material.w;var fraction=1.0;var prediction_count=motion.counts.w+1u;
  let lo=motion_cell(min(state.xy,state.xy+d)-padding);let hi=motion_cell(max(state.xy,state.xy+d)+padding);
  if(!ledger||rebuilding||motion_actors[i*2u+1u].y!=0u){
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let bucket=motion_bucket(vec2<i32>(x,y));let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);if(j==i||((!ledger||rebuilding||motion_actors[j*2u+1u].y!=0u)&&j<i)){continue;}let other=motion_state(j);let toi=motion_pair_toi(state.xy-other.xy,d-other.zw*horizon,2.0*radius+motion.material.w);prediction_count++;if(ledger){if(toi<1.0){motion_emit(motion_value(13)+toi*horizon,min(i,j),max(i,j),0u);}}else{fraction=min(fraction,toi);}}}}
  }
  // Always re-certify curved/rotating boundaries at the current pose. A retained
  // pair-event ledger must never substitute stale geometry certificates.
  let wall_d=state.zw*motion_value(8);
  var wall_fraction=1.0;
  let offset=state.xy-motion.wheel.xy;let limit=WHEEL_RADIUS-radius-WHEEL_BAR_HALF_WIDTH-motion.material.w;let a=dot(wall_d,wall_d);let b=dot(offset,wall_d);let c=(length(offset)-limit)*(length(offset)+limit);
  if(c>=0.0&&b>0.0){wall_fraction=0.0;}
  else if(a>1.0e-20&&b*b-a*c>=0.0){let exit=(-b+sqrt(max(0.0,b*b-a*c)))/a;if(exit>=0.0&&exit<1.0){wall_fraction=min(wall_fraction,exit);}}
  for(var segment=0u;segment< motion.counts.w;segment++){wall_fraction=min(wall_fraction,motion_bar_toi(state.xy,wall_d,segment,motion_value(1),motion_value(3)*motion_value(8)));}
  atomicAdd(&motion_control[21],prediction_count);
  fraction=min(fraction,wall_fraction);
  // Positive IEEE float bits are ordered; canonicalize negative zero too.
  atomicMin(&motion_control[4],bitcast<u32>(fraction)&0x7fffffffu);
}
@compute @workgroup_size(128) fn motion_find_toi(@builtin(global_invocation_id) gid:vec3<u32>){motion_find_body(gid.x);}
@compute @workgroup_size(128) fn motion_find_fallback(@builtin(global_invocation_id) gid:vec3<u32>){if(motion.counts.z==1u&&atomicLoad(&motion_control[29])!=0u){motion_find_body(gid.x);}}
fn motion_before(a:MotionEvent,b:MotionEvent)->bool{return a.time<b.time||(a.time==b.time&&(a.i<b.i||(a.i==b.i&&(a.j<b.j||(a.j==b.j&&a.kind<b.kind)))));}
fn motion_down(start:u32){var i=start;let item=motion_heap[i];let n=atomicLoad(&motion_control[14]);loop{let left=i*2u+1u;if(left>=n){break;}var child=left;if(left+1u<n&&motion_before(motion_heap[left+1u],motion_heap[left])){child=left+1u;}if(!motion_before(motion_heap[child],item)){break;}motion_heap[i]=motion_heap[child];i=child;}motion_heap[i]=item;}
fn motion_fallback(){atomicStore(&motion_control[29],1u);atomicStore(&motion_control[18],0u);atomicAdd(&motion_control[30],1u);}
@compute @workgroup_size(1) fn motion_update_ledger(){
  if(!motion_ledger()||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}
  let n=atomicLoad(&motion_control[15]);let capacity=arrayLength(&motion_heap);if(n>capacity){motion_fallback();return;}
  if(atomicLoad(&motion_control[18])==0u){atomicStore(&motion_control[14],n);for(var k=0u;k<n;k++){motion_heap[k]=motion_incoming[k];}var i=n/2u;loop{if(i==0u){break;}i--;motion_down(i);}atomicAdd(&motion_control[24],1u);atomicStore(&motion_control[18],1u);}
  else{
    var size=atomicLoad(&motion_control[14]);
    // Compact in-place before overflowing. Revisions, not probabilistic hashes,
    // determine validity. Rebuild heap ordering after stable compaction.
    if(size+n>capacity){var kept=0u;for(var k=0u;k<size;k++){let e=motion_heap[k];var valid=e.ri==motion_actors[e.i*2u+1u].x;if(e.j!=0xffffffffu){valid=valid&&e.rj==motion_actors[e.j*2u+1u].x;}if(valid){motion_heap[kept]=e;kept++;}}size=kept;atomicStore(&motion_control[14],size);var i=size/2u;loop{if(i==0u){break;}i--;motion_down(i);}}
    if(size+n>capacity){motion_fallback();return;}
    for(var k=0u;k<n;k++){let e=motion_incoming[k];var i=size;size++;loop{if(i==0u){break;}let parent=(i-1u)/2u;if(!motion_before(e,motion_heap[parent])){break;}motion_heap[i]=motion_heap[parent];i=parent;}motion_heap[i]=e;}atomicStore(&motion_control[14],size);
  }
  loop{
    let size=atomicLoad(&motion_control[14]);if(size==0u){return;}let e=motion_heap[0];var valid=e.ri==motion_actors[e.i*2u+1u].x;if(e.j!=0xffffffffu){valid=valid&&e.rj==motion_actors[e.j*2u+1u].x;}
    if(valid&&e.time>motion_value(13)+motion_value(8)){return;}
    atomicStore(&motion_control[14],size-1u);if(size>1u){motion_heap[0]=motion_heap[size-1u];motion_down(0u);}if(!valid){atomicAdd(&motion_control[23],1u);continue;}
    atomicStore(&motion_control[19],e.i);atomicStore(&motion_control[20],e.j);let fraction=clamp((e.time-motion_value(13))/max(motion_value(8),1.0e-20),0.0,1.0);atomicMin(&motion_control[4],bitcast<u32>(fraction)&0x7fffffffu);return;
  }
}
@compute @workgroup_size(128) fn motion_clear_dirty(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x){motion_actors[gid.x*2u+1u].y=0u;}}
@compute @workgroup_size(1) fn motion_consume_ledger(){if(motion_ledger()&&motion_value(0)>0.0){let i=atomicLoad(&motion_control[19]);let j=atomicLoad(&motion_control[20]);if(i!=0xffffffffu){motion_mark(i);}if(j!=0xffffffffu){motion_mark(j);}}}
fn motion_end_position(i:u32,duration:f32)->vec2<f32>{
  if(motion_ledger()&&atomicLoad(&motion_control[18])!=0u){let s=bitcast<vec4<f32>>(motion_actors[i*2u]);let t=bitcast<f32>(motion_actors[i*2u+1u].z);return s.xy+s.zw*(motion_value(13)+duration-t);}
  let s=motion_state(i);return s.xy+s.zw*duration;
}
fn motion_end_angle(duration:f32)->f32{if(motion_value(3)*duration==0.0){return motion_value(1);}let angle=motion_value(31)+motion_value(3)*(motion_value(13)+duration);let wrapped=angle-6.28318530718*floor((angle+3.14159265359)/6.28318530718);return select(angle,wrapped,abs(angle)>3.14159265359);}
@compute @workgroup_size(1) fn motion_prepare_guard(){atomicStore(&motion_control[10],atomicLoad(&motion_control[11]));atomicStore(&motion_control[11],0u);atomicStore(&motion_control[28],atomicLoad(&motion_control[4]));}
@compute @workgroup_size(128) fn motion_guard_end(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u||atomicLoad(&motion_control[10])==0u){return;}
  let fraction=motion_value(28);let duration=motion_value(8)*select(fraction*.999,1.0,fraction>=1.0);let p=motion_end_position(i,duration);let offset=p-motion.wheel.xy;let radius=motion.minimum.z;let clearance=radius+WHEEL_BAR_HALF_WIDTH+motion.material.w*.5;var safe=length(offset)<=WHEEL_RADIUS-clearance;var reason=select(1u,0u,safe);
  let local=motion_rotate(offset,-motion_end_angle(duration));for(var k=0u;k<motion.counts.w;k++){let bar=baffle(k);let clear=motion_distance(local,bar.xy,bar.zw)>=clearance;if(!clear){reason|=2u;}safe=safe&&clear;}
  let own=motion_state(i);let padding=radius+motion.material.w;let lo=motion_cell(min(own.xy,p)-padding);let hi=motion_cell(max(own.xy,p)+padding);
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let bucket=motion_bucket(vec2<i32>(x,y));let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);for(var k=0u;k<count;k++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+k]);if(j>i){let clear=length(p-motion_end_position(j,duration))>=2.0*radius+motion.material.w*.25;if(!clear){reason|=4u;}safe=safe&&clear;}}}}
  motion_actors[i*2u+1u].w=reason;
  // CCD certifies the swept path. This additional check protects the actual
  // float-buffer endpoint before any position/pose is committed. Never repair.
  if(!safe){atomicMin(&motion_control[4],bitcast<u32>(fraction*.5)&0x7fffffffu);atomicStore(&motion_control[11],1u);atomicAdd(&motion_control[26],1u);}
}
@compute @workgroup_size(1)
fn motion_advance_control(){
  if(atomicLoad(&motion_control[6])!=0u||motion_value(0)==0.0){return;}let fraction=motion_value(4);let duration=select(motion_value(8)*select(fraction*0.999,1.0,fraction>=1.0),0.0,atomicLoad(&motion_control[11])!=0u);motion_store(9,duration);atomicAdd(&motion_control[25],1u);atomicStore(&motion_control[22],0u);
  motion_store(0,max(0.0,motion_value(0)-duration));let delta=motion_value(3)*duration;
  // Never re-wrap an unchanged pose: repeated GPU trig introduced measurable
  // stationary drift and invalidated certified baffle geometry.
  if(delta!=0.0){motion_store(1,motion_end_angle(duration));motion_store(12,motion_value(12)+delta);}
  motion_store(13,motion_value(13)+duration);
  if(atomicLoad(&motion_control[7])!=0u){motion_store(2,motion_value(2)+duration);}
}
@compute @workgroup_size(128)
fn motion_advance_particles(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&atomicLoad(&motion_control[6])==0u){var state=motion_state(gid.x);if(!motion_ledger()||atomicLoad(&motion_control[18])==0u){state=vec4<f32>(state.xy+state.zw*motion_value(9),state.zw);}motion_particles[gid.x*motion.counts.y]=bitcast<vec4<u32>>(state);}}
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
        let bounce=select(0.0,motion.material.x,vn < -0.3);let separating_speed=motion.material.w/max(motion.schedule.x,1.0e-6);let jn=(separating_speed-(1.0+bounce)*vn)*0.5;let tangent=relative-vn*normal;let speed=length(tangent);let impulse=normal*jn-tangent/max(speed,1.0e-20)*min(speed*0.5,motion.material.y*jn);
        motion_write(i,vec4<f32>(own.xy,own.zw+impulse));motion_write(j,vec4<f32>(other.xy,other.zw-impulse));
      }
    }}
    let own=motion_state(i);let offset=own.xy-motion.wheel.xy;let radius=length(offset);let clearance=motion.minimum.z+WHEEL_BAR_HALF_WIDTH;
    if(radius>=WHEEL_RADIUS-clearance-motion.material.w*8.0){motion_wall_impulse(i,-offset/max(radius,1.0e-20),vec2<f32>(0.0),motion.material.z);}
    let local=motion_rotate(own.xy-motion.wheel.xy,-motion_value(1));
    for(var segment=0u;segment<motion.counts.w;segment++){let bar=baffle(segment);let edge=bar.zw-bar.xy;let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let separation=local-closest;let distance=length(separation);if(distance<=clearance+motion.material.w*8.0&&distance>1.0e-12){let normal=motion_rotate(separation/distance,motion_value(1));let point=motion_rotate(closest,motion_value(1));let surface=vec2<f32>(-point.y,point.x)*motion_value(3);motion_wall_impulse(i,normal,surface,motion.schedule.w);}}
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
  const capacity=Math.max(4096,count*16);
  if(capacity*32>maximumBytes)throw new Error('Collision ledger exceeds this GPU device limit');
  const uniform=make(96,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST),cells=make(side*side*4,GPUBufferUsage.STORAGE),items=make(side*side*64*4,GPUBufferUsage.STORAGE),control=make(128,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),readback=make(128,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),actors=make(count*32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),incoming=make(capacity*32,GPUBufferUsage.STORAGE),heap=make(capacity*32,GPUBufferUsage.STORAGE);
  const emptyLayout=device.createBindGroupLayout({entries:[]}),forceLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),forceGroup=device.createBindGroup({layout:forceLayout,entries:[{binding:0,resource:{buffer:uniform}},{binding:4,resource:{buffer:control}}]});
  const initialControl=new Float32Array(32);initialControl[1]=world.geometry.rotation;const reset=()=>device.queue.writeBuffer(control,0,initialControl);reset();
  const emptyGroup=device.createBindGroup({layout:emptyLayout,entries:[]});
  return {uniform,cells,items,control,readback,actors,incoming,heap,emptyLayout,emptyGroup,forceLayout,forceGroup,side,minimum,radius,cellWidth,margin,count,reset,destroy(){for(const b of [uniform,cells,items,control,readback,actors,incoming,heap])b.destroy();}};
}

export async function createPreventiveParticleContactGpu(device,world,physics,source,resources){
  const r=resources,layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,layout]}),module=device.createShaderModule({code:source});
  const entries=['motion_pause_velocity','motion_begin','motion_reset_speed','motion_measure_speed','motion_prepare','motion_ledger_initialize','motion_update_ledger','motion_find_fallback','motion_clear_dirty','motion_consume_ledger','motion_prepare_guard','motion_guard_end','motion_clear','motion_fill_swept','motion_fill_static','motion_sort','motion_find_toi','motion_advance_control','motion_advance_particles',...Array.from({length:9},(_,i)=>`motion_contacts_${i}`)];
  const pipelines=Object.fromEntries(await Promise.all(entries.map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})])));
  const group=device.createBindGroup({layout,entries:[r.uniform,physics.particleBuffer,r.cells,r.items,r.control,r.actors,r.incoming,r.heap].map((buffer,binding)=>({binding,resource:{buffer}}))}),empty=r.emptyGroup;
  const params=new ArrayBuffer(96),u=new Uint32Array(params),f=new Float32Array(params);u.set([r.count,world.kind==='granular'?2:3,0,world.geometry.segments.length,r.side,r.side,64,0]);f.set([...r.minimum,r.radius,r.cellWidth],8);f.set([...world.geometry.center,0,0],12);const sand=world.kind==='granular';f.set([world.properties.restitution??0,sand?(world.properties.friction??.9):0,sand?(world.properties.wall_friction??.78):0,r.margin],16);
  const dispatch=(pass,name,count)=>{pass.setPipeline(pipelines[name]);pass.dispatchWorkgroups(Math.ceil(count/128));};
  const policy=new ContactSchedulerPolicy(),timestamp=device.features?.has('timestamp-query')===true;
  const query=timestamp?device.createQuerySet({type:'timestamp',count:16}):null,queryBuffer=timestamp?device.createBuffer({size:128,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null,queryReadback=timestamp?device.createBuffer({size:128,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}):null;
  let queryIndex=0,started=0,selected='parallel',moving=false,eventBudget=8,fixedBudget=null;
  return {
    resources:r,
    setSchedulerMode:mode=>policy.force(mode),
    eventBudget:()=>fixedBudget??eventBudget,
    setEventBudget(value){if(value!==null&&(!Number.isInteger(value)||value<1||value>64))throw new Error('Contact event budget must be 1–64 or null');fixedBudget=value;},
    beginFrame(encoder,{requestedDelta,windowDuration,timeLimit,paused}){moving=Math.abs(requestedDelta)>1e-7;selected=policy.choose(moving);u[2]=selected==='ledger'?1:0;queryIndex=0;started=performance.now();f[14]=requestedDelta;f[15]=windowDuration;f.set([paused?windowDuration:world.time_step,timeLimit,paused?0:1,sand?(world.properties.wall_friction??.78):0],20);device.queue.writeBuffer(r.uniform,0,params);encoder.clearBuffer(r.control,48,4);encoder.clearBuffer(r.control,84,32);encoder.clearBuffer(r.control,116,8);},
    advance(encoder,{events=8}={}){if(!Number.isInteger(events)||events<1||events>8||queryIndex>14)throw new Error('Contact submission exceeds its bounded event/query budget');const timing=timestamp?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:queryIndex++,endOfPassWriteIndex:queryIndex++}}:{};const pass=encoder.beginComputePass({label:'VKF preventive contact events',...timing});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);pass.setBindGroup(2,group);dispatch(pass,'motion_pause_velocity',r.count);pass.setPipeline(pipelines.motion_begin);pass.dispatchWorkgroups(1);
      for(let event=0;event<events;event++){
        pass.setPipeline(pipelines.motion_reset_speed);pass.dispatchWorkgroups(1);dispatch(pass,'motion_measure_speed',r.count);pass.setPipeline(pipelines.motion_prepare);pass.dispatchWorkgroups(1);
        if(selected==='ledger')dispatch(pass,'motion_ledger_initialize',r.count);
        dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_swept',r.count);dispatch(pass,'motion_sort',r.side*r.side);dispatch(pass,'motion_find_toi',r.count);
        if(selected==='ledger'){pass.setPipeline(pipelines.motion_update_ledger);pass.dispatchWorkgroups(1);dispatch(pass,'motion_find_fallback',r.count);}
        for(let guard=0;guard<4;guard++){pass.setPipeline(pipelines.motion_prepare_guard);pass.dispatchWorkgroups(1);dispatch(pass,'motion_guard_end',r.count);}dispatch(pass,'motion_clear_dirty',r.count);pass.setPipeline(pipelines.motion_advance_control);pass.dispatchWorkgroups(1);dispatch(pass,'motion_advance_particles',r.count);
        if(selected==='ledger'){pass.setPipeline(pipelines.motion_consume_ledger);pass.dispatchWorkgroups(1);}
        dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_static',r.count);dispatch(pass,'motion_sort',r.side*r.side);for(let color=0;color<9;color++)dispatch(pass,`motion_contacts_${color}`,r.side*r.side);
      }pass.end();
    },
    finishFrame(encoder){encoder.copyBufferToBuffer(r.control,0,r.readback,0,128);if(timestamp&&queryIndex){encoder.resolveQuerySet(query,0,queryIndex,queryBuffer,0);encoder.copyBufferToBuffer(queryBuffer,0,queryReadback,0,queryIndex*8);}},
    async inspect(){await r.readback.mapAsync(GPUMapMode.READ);const bytes=r.readback.getMappedRange().slice(0);r.readback.unmap();const f=new Float32Array(bytes),u=new Uint32Array(bytes);if(u[6]||![f[0],f[1],f[2],f[12]].every(Number.isFinite))throw new Error(`Preventive contact state/broadphase failure ${u[6]}; no unchecked motion tail was advanced.`);let gpuMs=null;if(timestamp&&queryIndex){await queryReadback.mapAsync(GPUMapMode.READ);const q=new BigUint64Array(queryReadback.getMappedRange().slice(0));queryReadback.unmap();gpuMs=0;for(let i=0;i<queryIndex;i+=2)gpuMs+=Number(q[i+1]-q[i])/1e6;}const receipt={remainingTime:f[0],angle:f[1],time:f[2],angularDelta:f[12],scheduler:selected,gpuMs,completedMs:performance.now()-started,predictions:u[21],heapEvents:u[14],staleEvents:u[23],heapRebuilds:u[24],events:u[25],guardDeferrals:u[26],dirtyParticles:u[27]/Math.max(1,u[25]*r.count),fallbacks:u[30]};if(receipt.events>0){const cost=(gpuMs??receipt.completedMs)/receipt.events;if(cost>0)eventBudget=Math.max(1,Math.min(64,Math.floor(10/cost)));}policy.observe({...receipt,mode:selected,moving});return receipt;},
    reset(){r.reset();policy.reset();eventBudget=8;},destroy(){r.destroy();query?.destroy();queryBuffer?.destroy();queryReadback?.destroy();},
  };
}
