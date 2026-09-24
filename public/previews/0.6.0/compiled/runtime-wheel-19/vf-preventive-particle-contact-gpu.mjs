// Contact-core motion adapter. Position writes are preceded by continuous
// collision bounds. A bounded event budget defers time; it never consumes an
// unchecked motion tail or uses depenetration as a physical contact response.
import {DENSE_CONTACT_WGSL} from './vf-dense-contact-wgsl.mjs';
import {CONTACT_PRIMAL_BARRIER_WGSL} from './vf-contact-primal-barrier-wgsl.mjs';
import {CONTACT_DUAL_COMPONENT_CANDIDATES_WGSL} from './vf-contact-dual-component-candidates-wgsl.mjs';
import {CONTACT_AUGMENTED_VELOCITY_WGSL} from './vf-contact-augmented-velocity-wgsl.mjs';
import {CONTACT_TILED_PROJECTION_WGSL} from './vf-contact-tiled-projection-wgsl.mjs';
import {CONTACT_CACHED_STAR_WGSL} from './vf-contact-cached-star-wgsl.mjs';
import {ContactSchedulerPolicy} from './vf-contact-scheduler-policy.mjs';
import {createCheckedGpuPipeline,createOptionalGpuPipeline} from './vf-gpu-pipeline-errors.mjs';
import {createStaticWgslModuleCache} from './vf-wgsl-static-overrides.mjs';
import {PAUSED_WORLD_POSE_WGSL} from './vf-paused-world-pose-wgsl.mjs';
export const PREVENTIVE_PARTICLE_CONTACT_WGSL = /* wgsl */`
struct MotionParams { counts:vec4<u32>,grid:vec4<u32>,minimum:vec4<f32>,wheel:vec4<f32>,material:vec4<f32>,schedule:vec4<f32> };
override CONTACT_PERSISTENT_SKIN=false;
override MOTION_FULL_WINDOW=false;
override MOTION_ROTATING_TRAJECTORY=false;
@group(2) @binding(0) var<uniform> motion:MotionParams;
@group(2) @binding(1) var<storage,read_write> motion_particles:array<vec4<u32>>;
@group(2) @binding(2) var<storage,read_write> motion_cells:array<atomic<u32>>;
@group(2) @binding(3) var<storage,read_write> motion_items:array<atomic<u32>>;
@group(2) @binding(4) var<storage,read_write> motion_control:array<atomic<u32>>;
@group(2) @binding(5) var<storage,read_write> motion_actors:array<vec4<u32>>;
struct MotionEvent { time:f32, i:u32, j:u32, kind:u32, ri:u32, rj:u32, spare:vec2<u32> };
@group(2) @binding(6) var<storage,read_write> motion_incoming:array<MotionEvent>;
@group(2) @binding(7) var<storage,read_write> motion_heap:array<MotionEvent>;
@group(2) @binding(8) var<storage,read_write> motion_dispatch_args:array<atomic<u32>>;
@group(2) @binding(9) var<storage,read_write> motion_precision_output:array<vec2<f32>>;
// control: remainder, wrapped pose, law time, omega, TOI, speed, fault,
// operation-law flag, horizon, event duration, reserved, reserved, frame angle.
fn motion_value(i:u32)->f32{return bitcast<f32>(atomicLoad(&motion_control[i]));}
fn motion_store(i:u32,v:f32){atomicStore(&motion_control[i],bitcast<u32>(v));}
fn motion_force_ready()->bool{return motion.schedule.z>0.5&&motion_value(0)==0.0&&motion_value(2)+motion.schedule.x<=motion.schedule.y+1.0e-7&&atomicLoad(&motion_control[6])==0u;}
fn motion_ledger()->bool{return motion.counts.z==1u&&atomicLoad(&motion_control[29])==0u;}
// Material particle-particle behavior belongs to the shared grid Laws. This
// contact adapter can therefore specialize to rigid boundaries without
// rebuilding and solving a second particle graph.
fn motion_pair_contact()->bool{return motion.counts.z<2u;}
fn motion_two_sum(a:vec2<f32>,b:vec2<f32>)->vec4<f32>{let s=a+b;let v=s-a;return vec4<f32>(s,(a-(s-v))+(b-v));}
// WGSL fma may be implemented as a rounded multiply followed by add. On the
// physical Intel adapter fma(a,b,-(a*b)) returned zero for every probe. Split
// significands by bits instead: each 12-bit high product is exactly stored.
fn motion_product_error(a:vec2<f32>,b:vec2<f32>,product:vec2<f32>)->vec2<f32>{
 let ah=bitcast<vec2<f32>>(bitcast<vec2<u32>>(a)&vec2<u32>(0xfffff000u));let al=a-ah;
 let bh=bitcast<vec2<f32>>(bitcast<vec2<u32>>(b)&vec2<u32>(0xfffff000u));let bl=b-bh;
 return ((ah*bh-product)+ah*bl+al*bh)+al*bl;
}
fn motion_product_error1(a:f32,b:f32,product:f32)->f32{return motion_product_error(vec2<f32>(a),vec2<f32>(b),vec2<f32>(product)).x;}
fn motion_compensated(hi:vec2<f32>,lo:vec2<f32>,delta:vec2<f32>)->vec4<f32>{let first=motion_two_sum(hi,delta);return motion_two_sum(first.xy,lo+first.zw);}
fn motion_integrated(hi:vec2<f32>,lo:vec2<f32>,velocity:vec2<f32>,duration:f32)->vec4<f32>{let step=velocity*duration;let error=motion_product_error(velocity,vec2<f32>(duration),step);let integrated=motion_compensated(hi,lo,step);return motion_two_sum(integrated.xy,integrated.zw+error);}
// Authoritative positions are high+low; embeddings may consume high alone.
// Small motion must not disappear merely because world-coordinate ULP is large.
fn motion_position_parts(i:u32,duration:f32)->vec4<f32>{
 if(motion_ledger()&&atomicLoad(&motion_control[18])!=0u){let s=bitcast<vec4<f32>>(motion_actors[i*128u]);let t=bitcast<f32>(motion_actors[i*128u+1u].z);let low=bitcast<vec4<f32>>(motion_actors[i*128u+122u]).zw;return motion_integrated(s.xy,low,s.zw,motion_value(13)+duration-t);}
 let s=bitcast<vec4<f32>>(motion_particles[i*motion.counts.y]);let low=bitcast<vec4<f32>>(motion_actors[i*128u+122u]).xy;let velocity=select(s.zw,bitcast<vec4<f32>>(motion_actors[i*128u+125u]).zw,atomicLoad(&motion_control[17])!=0u);return motion_integrated(s.xy,low,velocity,duration);
}
fn motion_separation_parts(i:u32,j:u32)->vec4<f32>{let a=motion_position_parts(i,0.0);let b=motion_position_parts(j,0.0);let difference=motion_compensated(a.xy,a.zw,-b.xy);return motion_two_sum(difference.xy,difference.zw-b.zw);}
fn motion_separation(i:u32,j:u32)->vec2<f32>{let p=motion_separation_parts(i,j);return p.xy+p.zw;}
fn motion_state(i:u32)->vec4<f32>{
  let p=motion_position_parts(i,0.0);if(motion_ledger()&&atomicLoad(&motion_control[18])!=0u){return vec4<f32>(p.xy,bitcast<vec4<f32>>(motion_actors[i*128u]).zw);}let velocity=select(bitcast<vec4<f32>>(motion_particles[i*motion.counts.y]).zw,bitcast<vec4<f32>>(motion_actors[i*128u+125u]).zw,atomicLoad(&motion_control[17])!=0u);return vec4<f32>(p.xy,velocity);
}
fn motion_mark(i:u32){
  var m=motion_actors[i*128u+1u];if(m.y==0u){atomicAdd(&motion_control[22],1u);atomicAdd(&motion_control[27],1u);}m.x++;m.y=1u;m.z=bitcast<u32>(motion_value(13));motion_actors[i*128u+1u]=m;motion_actors[i*128u]=motion_particles[i*motion.counts.y];let low=bitcast<vec4<f32>>(motion_actors[i*128u+122u]).xy;motion_actors[i*128u+122u]=bitcast<vec4<u32>>(vec4<f32>(low,low));
}
fn motion_write(i:u32,s:vec4<f32>){if(atomicLoad(&motion_control[17])!=0u){motion_actors[i*128u+125u]=bitcast<vec4<u32>>(s);return;}let old=motion_particles[i*motion.counts.y];let parts=motion_position_parts(i,0.0);let previous=motion_actors[i*128u+122u];motion_actors[i*128u+122u]=vec4<u32>(bitcast<vec2<u32>>(parts.zw),previous.zw);motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(s);if(any(old.zw!=bitcast<vec2<u32>>(s.zw))){motion_mark(i);}}
@compute @workgroup_size(128) fn motion_materialize_state(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||atomicLoad(&motion_control[17])!=0u){return;}let s=motion_state(i);let parts=motion_position_parts(i,0.0);motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(s);let previous=motion_actors[i*128u+122u];motion_actors[i*128u+122u]=vec4<u32>(bitcast<vec2<u32>>(parts.zw),previous.zw);}
// Trial velocities are not physical state. A rejected/zero-duration motion
// keeps only the solver's private work, not its impulse in the next free state.
@compute @workgroup_size(1) fn motion_velocity_trial_begin(){atomicStore(&motion_control[18],0u);atomicStore(&motion_control[17],1u);}
@compute @workgroup_size(128) fn motion_velocity_trial_seed(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x){motion_actors[gid.x*128u+125u]=motion_particles[gid.x*motion.counts.y];}}
@compute @workgroup_size(1) fn motion_velocity_trial_end(){atomicStore(&motion_control[17],0u);if(motion_value(9)==0.0){atomicStore(&motion_control[18],0u);}}
@compute @workgroup_size(128) fn motion_precision_reset(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&atomicLoad(&motion_control[50])!=0u){motion_actors[gid.x*128u+122u]=vec4<u32>(0u);primal_scratch[primal_slot(gid.x,89u)]=vec4<u32>(0u);primal_put(gid.x,90u,vec4<f32>(0.0));}}
@compute @workgroup_size(1) fn motion_precision_reset_end(){atomicStore(&motion_control[50],0u);}
@compute @workgroup_size(128) fn motion_export_precision(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x){motion_precision_output[gid.x]=bitcast<vec4<f32>>(motion_actors[gid.x*128u+122u]).xy;}}
fn motion_cell(p:vec2<f32>)->vec2<i32>{return vec2<i32>(floor((p-motion.minimum.xy)/motion.minimum.w));}
fn motion_valid(c:vec2<i32>)->bool{return all(c>=vec2<i32>(0))&&all(c<vec2<i32>(motion.grid.xy));}
fn motion_bucket(c:vec2<i32>)->u32{return u32(c.x)+motion.grid.x*u32(c.y);}
// Swept AABBs occupy rectangular cell sets. Every intersecting pair shares
// exactly one lower-left intersection cell, even if the query is a prefix of
// one sweep. Use the stored INSERTION bounds of the other actor, not its centre.
fn motion_first_shared_cell(cell:vec2<i32>,query_lo:vec2<i32>,other:u32)->bool{
 let other_lo=bitcast<vec4<i32>>(motion_actors[other*128u+123u]).xy;
 return all(cell==max(query_lo,other_lo));
}
fn motion_sqrt(q:f32)->f32{let s=sqrt(max(0.0,q));if(s<=1.0e-20){return s;}return .5*(s+q/s);}
fn motion_length(v:vec2<f32>)->f32{return motion_sqrt(dot(v,v));}
// Metre-scale contact clearances cannot use implementation-dependent native
// trigonometric approximation. Bounded polynomial rotation avoids the measured
// geometry discrepancy on Intel gen-9 and keeps physics/guard coordinates equal.
fn motion_rotate(p:vec2<f32>,angle:f32)->vec2<f32>{
  let a=angle-6.28318530718*floor((angle+3.14159265359)/6.28318530718);let folded=abs(a)>1.57079632679;let x=select(a,sign(a)*3.14159265359-a,folded);let q=x*x;
  let s=x*(1.0+q*(-0.166666666666667+q*(0.00833333333333333+q*(-0.000198412698412698+q*(0.00000275573192239859+q*(-0.0000000250521083854417+q*0.000000000160590438368216))))));
  let c=(1.0+q*(-0.5+q*(0.0416666666666667+q*(-0.00138888888888889+q*(0.0000248015873015873+q*(-0.000000275573192239859+q*(0.00000000208767569878681-q*0.0000000000114707455977297)))))))*select(1.0,-1.0,folded);
  return vec2<f32>(c*p.x-s*p.y,s*p.x+c*p.y);
}
fn motion_distance(p:vec2<f32>,a:vec2<f32>,b:vec2<f32>)->f32{let edge=b-a;return motion_length(p-a-edge*clamp(dot(p-a,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0));}
fn motion_reserve()->f32{return motion_value(49);}
// Bounds apply to arithmetic on the exact stored f32 state, not native length
// approximation. Boundary arithmetic additionally includes rotation/constant
// conversion error. Never request expansion of an already admissible packing.
fn motion_pair_error(s:vec2<f32>)->f32{return 0.00000095367431640625*max(motion_length(s),2.0*motion.minimum.z);}
// Subtract in squared-distance space with compensated product/sum errors.
// sqrt(distance2)-diameter discards the remaining gap at dense contact.
fn motion_squared_gap(s:vec2<f32>,radius:f32)->f32{
 let x=s.x*s.x;let y=s.y*s.y;let rr=radius*radius;
 let sum=motion_two_sum(vec2<f32>(x),vec2<f32>(y));
 let difference=motion_two_sum(sum.xy,vec2<f32>(-rr));
 let error=motion_product_error1(s.x,s.x,x)+motion_product_error1(s.y,s.y,y)-motion_product_error1(radius,radius,rr);
 return difference.x+(difference.z+sum.z+error);
}
fn motion_pair_gap(s:vec2<f32>)->f32{let diameter=2.0*motion.minimum.z;return motion_squared_gap(s,diameter)/(motion_length(s)+diameter)-motion_pair_error(s);}
fn motion_radius_gap_parts(parts:vec4<f32>,radius:f32)->f32{
 let q=motion_squared_gap(parts.xy,radius)+2.0*dot(parts.xy,parts.zw)+dot(parts.zw,parts.zw);
 return q/(motion_length(parts.xy)+radius);
}
fn motion_pair_gap_parts(parts:vec4<f32>)->f32{return motion_radius_gap_parts(parts,2.0*motion.minimum.z)-motion_pair_error(parts.xy);}
fn motion_solve_reserve()->f32{return motion_value(52);}
// The adaptive pair reserve must not collapse the nonlinear boundary target.
// Leave enough wall clearance for a rounded finite velocity solve, without
// requesting expansion of a dense particle-particle packing.
fn motion_boundary_solve_reserve(gap:f32)->f32{if(CONTACT_PERSISTENT_SKIN){return motion_solve_reserve();}return select(motion_solve_reserve(),min(2.0*motion.material.w,max(0.0,gap*0.25)),motion.grid.w==6u);}
// Compensated support-plane slack. Forming it as an endpoint gap plus an
// order-one velocity dot product loses the static feasible point at contact.
fn motion_plane_gap_parts(p:vec4<f32>,n:vec2<f32>,radius:f32,buffer:f32)->f32{
 let norm_hi=motion_length(n);let norm_lo=motion_radius_gap_parts(vec4<f32>(n,0.0,0.0),1.0)-(norm_hi-1.0);
 let products=n*p.xy;let first=motion_two_sum(vec2<f32>(products.x),vec2<f32>(products.y));let radial=radius*norm_hi;let second=motion_two_sum(first.xy,vec2<f32>(radial));let padding=buffer*norm_hi;let third=motion_two_sum(second.xy,vec2<f32>(-padding));
 let low=first.z+second.z+third.z+motion_product_error1(n.x,p.x,products.x)+motion_product_error1(n.y,p.y,products.y)+dot(n,p.zw)+motion_product_error1(radius,norm_hi,radial)-motion_product_error1(buffer,norm_hi,padding)+(radius-buffer)*norm_lo;
 return third.x+low;
}
fn motion_offset_parts(parts:vec4<f32>)->vec4<f32>{return motion_compensated(parts.xy,parts.zw,-motion.wheel.xy);}
fn motion_vector_rotated_parts(parts:vec4<f32>,angle:f32)->vec4<f32>{
 let basis=motion_rotate(vec2<f32>(1.0,0.0),angle);let a=basis.x*parts.xy;let perpendicular=vec2<f32>(-parts.y,parts.x);let b=basis.y*perpendicular;
 let error=motion_product_error(vec2<f32>(basis.x),parts.xy,a)+motion_product_error(vec2<f32>(basis.y),perpendicular,b);let sum=motion_two_sum(a,b);
 // Restore unit norm in high+low precision. A single-f32 basis can expand
 // the entire packed configuration even though every local contact is safe.
 let correction=-0.5*motion_squared_gap(basis,1.0);
 return motion_two_sum(sum.xy,error+sum.zw+motion_rotate(parts.zw,angle)+sum.xy*correction);
}
fn motion_rim_error()->f32{return 0.0000002384185791015625*(WHEEL_RADIUS+abs(motion.wheel.x)+abs(motion.wheel.y));}
fn motion_rim_gap(offset:vec2<f32>)->f32{return WHEEL_RADIUS-motion.minimum.z-WHEEL_BAR_HALF_WIDTH-motion_length(offset)-motion_rim_error();}
fn motion_rim_gap_parts(offset:vec4<f32>)->f32{return -motion_radius_gap_parts(offset,WHEEL_RADIUS-motion.minimum.z-WHEEL_BAR_HALF_WIDTH)-motion_rim_error();}
fn motion_bar_error(bar:vec4<f32>)->f32{return 0.0000019073486328125*(WHEEL_RADIUS+motion_length(bar.zw-bar.xy));}
fn motion_bar_gap(offset:vec2<f32>,angle:f32,bar:vec4<f32>)->f32{let local=motion_rotate(offset,-angle);return motion_distance(local,bar.xy,bar.zw)-motion.minimum.z-WHEEL_BAR_HALF_WIDTH-motion_bar_error(bar);}
fn motion_bar_gap_parts(offset:vec4<f32>,angle:f32,bar:vec4<f32>)->f32{
 let local=motion_vector_rotated_parts(offset,-angle);let edge=bar.zw-bar.xy;let closest=bar.xy+edge*clamp(dot(local.xy-bar.xy,edge)/dot(edge,edge),0.0,1.0);
 let separation=motion_compensated(local.xy,local.zw,-closest);
 return motion_radius_gap_parts(separation,motion.minimum.z+WHEEL_BAR_HALF_WIDTH)-motion_bar_error(bar);
}
@compute @workgroup_size(1) fn motion_clearance_begin(){motion_store(48,motion.material.w*16.0);}
@compute @workgroup_size(128) fn motion_clearance(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)<=0.0){return;}let own=motion_state(i);let offset=motion_offset_parts(motion_position_parts(i,0.0));var gap=motion_rim_gap_parts(offset);
 for(var k=0u;k<motion.counts.w;k++){gap=min(gap,motion_bar_gap_parts(offset,motion_value(1),baffle(k)));}
 if(motion_pair_contact()){let cell=motion_cell(own.xy);for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let c=cell+vec2<i32>(x,y);if(!motion_valid(c)){continue;}let b=motion_bucket(c);let count=min(atomicLoad(&motion_cells[b]),motion.grid.z);for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[b*motion.grid.z+slot]);if(j>i){gap=min(gap,motion_pair_gap_parts(motion_separation_parts(i,j)));}}}}}
 if(gap<=0.0){atomicOr(&motion_control[6],512u);return;}atomicMin(&motion_control[48],bitcast<u32>(gap));
}
@compute @workgroup_size(1) fn motion_clearance_end(){
 if(motion_value(0)==0.0){return;}
 if(CONTACT_PERSISTENT_SKIN){let gap=motion_value(48);if(gap<=motion.material.w*0.5){atomicOr(&motion_control[6],512u);return;}motion_store(52,motion.material.w*0.5);motion_store(49,motion.material.w*0.25);return;}
 let gap=motion_value(48);motion_store(52,select(min(motion.material.w*2.0,gap*0.25),motion.material.w*2.0,motion.grid.w==1u||motion.grid.w==3u));motion_store(49,min(motion_solve_reserve()*0.25,gap*0.25));
}
// A feasible-start inertial proposal for constant-twist 2D boundaries. This is
// a solver STARTING GUESS, not an animation or a force added to the World.
@compute @workgroup_size(1) fn motion_twist_begin(){
 let omega=abs(motion_value(3));let speed=max(motion_value(5),omega*WHEEL_RADIUS);
 motion_store(53,2.0*omega*speed+omega*omega*(WHEEL_RADIUS+speed*motion_value(8)));
}
@compute @workgroup_size(128) fn motion_twist_horizon(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;let omega=abs(motion_value(3));if(i>=motion.counts.x||motion_value(0)==0.0||omega==0.0){return;}
 let current=motion_position_parts(i,0.0);let offset=motion_offset_parts(current);let r=motion_length(offset.xy);let gap=motion_rim_gap_parts(offset)-motion_solve_reserve();
 var h=min(motion_value(8),0.9*sqrt(max(0.0,gap*(2.0*r+gap)))/max(omega*r,1.0e-20));
 for(var k=0u;k<motion.counts.w;k++){let clear=motion_bar_gap_parts(offset,motion_value(1),baffle(k))-motion_solve_reserve();h=min(h,0.9*sqrt(max(0.0,2.0*clear)/max(motion_value(53),1.0e-20)));}
 // Validate the rounded starting endpoint through the same arithmetic as the
 // optimizer. The analytic tangent bound alone cannot certify f32 rounding.
 let velocity=motion_value(3)*vec2<f32>(-offset.y,offset.x);
 for(var retry=0u;retry<16u;retry++){let endpoint=motion_offset_parts(motion_compensated(current.xy,current.zw,velocity*h));if(motion_rim_gap_parts(endpoint)>motion_solve_reserve()){break;}h*=0.5;}
 atomicMin(&motion_control[8],bitcast<u32>(max(0.0,h)));
}
var<workgroup> motion_twist_valid:array<u32,128>;
var<workgroup> motion_twist_work:u32;
// An individually valid rounded pilot is not necessarily valid after another
// actor reduces the global interval: compensated f32 gaps are not monotonic.
// Recheck the ONE shared interval before building its constraints.
@compute @workgroup_size(128) fn motion_twist_validate(@builtin(local_invocation_index) lane:u32){
 for(var retry=0u;retry<32u;retry++){
  let h=motion_value(8);var bad=0u;
  for(var i=lane;i<motion.counts.x;i+=128u){let p=motion_position_parts(i,0.0);let offset=motion_offset_parts(p);let v=motion_value(3)*vec2<f32>(-offset.y,offset.x);let endpoint=motion_offset_parts(motion_compensated(p.xy,p.zw,v*h));if(motion_rim_gap_parts(endpoint)<=motion_solve_reserve()){bad=1u;}}
  motion_twist_valid[lane]=bad;workgroupBarrier();if(lane==0u){var any_bad=0u;for(var k=0u;k<128u;k++){any_bad|=motion_twist_valid[k];}motion_twist_work=any_bad;if(any_bad!=0u){motion_store(8,h*0.5);}}storageBarrier();
  if(workgroupUniformLoad(&motion_twist_work)==0u){break;}
 }
}
fn motion_pair_toi_parts(parts:vec4<f32>,d:vec2<f32>,radius:f32)->f32{
  let s=parts.xy+parts.zw;let b=dot(s,d);if(b>=0.0){return 1.0;}let length_s=motion_length(s);var reserve=motion_reserve();var gap=motion_pair_gap_parts(parts)-reserve;let a=dot(d,d);
  // Certify the minimum directly before forming a cancellation-prone quadratic
  // discriminant. The Lipschitz padding covers rounding of the minimizer.
  let t=clamp(-b/max(a,1.0e-30),0.0,1.0);let step=d*t;let closest=motion_compensated(parts.xy,parts.zw,step);let certified=motion_two_sum(closest.xy,closest.zw+motion_product_error(d,vec2<f32>(t),step));
  if(motion_pair_gap_parts(certified)>=reserve+0.000003814697265625*motion_length(d)){return 1.0;}
  if(gap<=0.0){
    // A rounded position may already be inside the outer NUMERICAL reserve,
    // while still separated physically. Find its remaining certified interval;
    // returning zero merely because a full horizon cannot fit freezes the world.
    // The inner boundary still retains a positive physical clearance.
    reserve*=.5;
    let closest=motion_compensated(parts.xy,parts.zw,d*clamp(-b/max(a,1.0e-20),0.0,1.0));
    gap=motion_pair_gap_parts(parts)-reserve;
    if(gap>=0.0&&motion_pair_gap_parts(closest)>=reserve){return 1.0;}
    if(gap<=0.0){return 0.0;}
  }
  let c=gap*(length_s+radius+motion_pair_error(s)+reserve);
  let discriminant=b*b-a*c;if(a<=1.0e-20||discriminant<=0.0){return 1.0;}
  return clamp(c/(-b+sqrt(discriminant)),0.0,1.0);
}
// Certified intervals for an actual rotating segment, not endpoint sampling or
// a straight chord substituted for its arc. Distance is 1-Lipschitz in point
// motion and the segment's Hausdorff motion. Uncertified intervals are deferred.
fn motion_bar_toi_parts(parts:vec4<f32>,d:vec2<f32>,segment:u32,angle:f32,delta:f32)->f32{
  let bar=baffle(segment);let radius=motion.minimum.z+WHEEL_BAR_HALF_WIDTH+motion_reserve()+motion_bar_error(bar);
  if(delta==0.0){
    let offset_parts=motion_offset_parts(parts);let local=motion_vector_rotated_parts(offset_parts,-angle);let edge=bar.zw-bar.xy;
    let closest=bar.xy+edge*clamp(dot(local.xy-bar.xy,edge)/dot(edge,edge),0.0,1.0);let separation=motion_compensated(local.xy,local.zw,-closest);let n=(separation.xy+separation.zw)/max(motion_length(separation.xy),1.0e-20);
    // A stationary capsule admits the complete point segment when the two
    // segments are disjoint and all four endpoint-to-segment distances clear
    // its radius. Tangent-plane extrapolation alone can report false stops.
    let travel=motion_rotate(d,-angle);let start=local.xy+local.zw;let end=start+travel;let qa=start-bar.xy;let qb=end-bar.xy;
    let sides=vec2<f32>(edge.x*qa.y-edge.y*qa.x,edge.x*qb.y-edge.y*qb.x);
    let pa=bar.xy-start;let pb=bar.zw-start;let pathSides=vec2<f32>(travel.x*pa.y-travel.y*pa.x,travel.x*pb.y-travel.y*pb.x);
    let sideError=0.000003814697265625*max(motion_length(edge)*max(motion_length(qa),motion_length(qb)),1.0e-20);let pathError=0.000003814697265625*max(motion_length(travel)*max(motion_length(pa),motion_length(pb)),1.0e-20);
    let disjoint=all(sides>vec2<f32>(sideError))||all(sides<vec2<f32>(-sideError))||all(pathSides>vec2<f32>(pathError))||all(pathSides<vec2<f32>(-pathError));
    if(disjoint){let distance=min(min(motion_distance(start,bar.xy,bar.zw),motion_distance(end,bar.xy,bar.zw)),min(motion_distance(bar.xy,start,end),motion_distance(bar.zw,start,end)));if(distance>=radius){return 1.0;}}
    let gap=motion_bar_gap_parts(offset_parts,angle,bar)-motion_reserve();let speed=dot(n,motion_rotate(d,-angle));
    if(speed>=0.0){return 1.0;}return clamp(gap/max(-speed,1.0e-30),0.0,1.0);
  }
  let offset_parts=motion_offset_parts(parts);let offset=offset_parts.xy+offset_parts.zw;
  let perpendicular=vec2<f32>(-offset.y,offset.x);
  let speed=motion_length(d-delta*perpendicular)+abs(delta)*motion_length(d);
  let acceleration=2.0*abs(delta)*motion_length(d)+delta*delta*(motion_length(offset)+motion_length(d));
  let edge=bar.zw-bar.xy;
  let normal=vec2<f32>(-edge.y,edge.x)/max(motion_length(edge),1.0e-20);
  if(speed==0.0){return 1.0;}
  var stack:array<vec2<f32>,32>;var top=1u;stack[0]=vec2<f32>(0.0,1.0);
  for(var visit=0u;visit<96u;visit++){
    if(top==0u){return 1.0;}top--;let interval=stack[top];let mid=(interval.x+interval.y)*0.5;
    let at_parts=motion_offset_parts(motion_compensated(parts.xy,parts.zw,d*mid));
    let local_parts=motion_vector_rotated_parts(at_parts,-angle-delta*mid);let local=local_parts.xy;
    let half=(interval.y-interval.x)*0.5;
    let at=offset+d*mid;
    let derivative=motion_rotate(d-delta*vec2<f32>(-at.y,at.x),-angle-delta*mid);
    let plane_parts=motion_compensated(local_parts.xy,local_parts.zw,-bar.xy);
    let plane_clearance=abs(dot(plane_parts.xy,normal)+dot(plane_parts.zw,normal))-abs(dot(derivative,normal))*half-0.5*acceleration*half*half;
    let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let separation=motion_compensated(local_parts.xy,local_parts.zw,-closest);let distance=motion_length(separation.xy);let support=(separation.xy+separation.zw)/max(distance,1.0e-20);
    // A convex capsule's nearest-point supporting plane also certifies its
    // rounded ends. The old face-only/Lipschitz test stalled tangential motion.
    let capsule_gap=motion_radius_gap_parts(separation,motion.minimum.z+WHEEL_BAR_HALF_WIDTH)-motion_bar_error(bar)-motion_reserve();
    let capsule_motion=abs(dot(derivative,support))*half+0.5*acceleration*half*half;
    if(plane_clearance>=radius||capsule_gap>=capsule_motion||capsule_gap>=speed*half){continue;}
    if(interval.y-interval.x<=1.0e-6||top+2u>32u){return interval.x;}
    stack[top]=vec2<f32>(mid,interval.y);stack[top+1u]=vec2<f32>(interval.x,mid);top+=2u;
  }
  if(top>0u){return stack[top-1u].x;}return 1.0;
}
@compute @workgroup_size(128)
fn motion_pause_velocity(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x>=motion.counts.x||motion.schedule.z>0.5||motion_value(0)>0.0||abs(motion.wheel.z)<1.0e-7){return;}let state=motion_state(gid.x);motion_write(gid.x,vec4<f32>(state.xy,0.0,0.0));}
@compute @workgroup_size(1)
fn motion_begin(){
  if(atomicLoad(&motion_control[6])!=0u){return;}
  if(motion_value(0)>0.0){
    if(MOTION_ROTATING_TRAJECTORY){if(abs(motion.wheel.z-motion_value(12))<=1.0e-7&&motion_value(3)!=0.0){motion_store(31,motion_value(1));motion_store(3,0.0);atomicStore(&motion_control[18],0u);}return;}
    let omega=(motion.wheel.z-motion_value(12))/max(motion.schedule.x,motion.wheel.w-(motion_value(2)-motion_value(61)));
    if((bitcast<u32>(omega)&0x7f800000u)==0x7f800000u){atomicOr(&motion_control[6],1u);return;}
    if(omega!=motion_value(3)){motion_store(31,motion_value(1)-omega*motion_value(13));motion_store(3,omega);atomicStore(&motion_control[18],0u);}
    return;
  }
  if(motion_force_ready()){atomicAdd(&motion_control[55],1u);motion_store(0,motion.schedule.x);atomicStore(&motion_control[7],1u);}
  else if(motion.schedule.z<0.5&&abs(motion.wheel.z-motion_value(12))>1.0e-7){motion_store(0,motion.schedule.x);atomicStore(&motion_control[7],0u);}
  else{return;}
  let credit=max(motion.schedule.x,motion.wheel.w-(motion_value(2)-motion_value(61)));
  let angular_velocity=select((motion.wheel.z-motion_value(12))/credit,select(motion_value(3),0.0,abs(motion.wheel.z-motion_value(12))<=1.0e-7),MOTION_ROTATING_TRAJECTORY);
  if((bitcast<u32>(angular_velocity)&0x7f800000u)==0x7f800000u){atomicOr(&motion_control[6],1u);return;}
  motion_store(3,angular_velocity);
  motion_store(13,0.0);motion_store(59,motion_value(2));motion_store(60,0.0);motion_store(62,motion_value(0));motion_store(31,motion_value(1));atomicStore(&motion_control[18],0u);
}
@compute @workgroup_size(1) fn motion_input_omega(){motion_store(61,motion_value(2));let omega=(motion.wheel.z-motion_value(12))/max(motion.schedule.x,motion.wheel.w);if((bitcast<u32>(omega)&0x7f800000u)==0x7f800000u){atomicOr(&motion_control[6],1u);return;}if(omega!=motion_value(3)){motion_store(31,motion_value(1)-omega*motion_value(13));motion_store(3,omega);atomicStore(&motion_control[18],0u);}}

@compute @workgroup_size(1) fn motion_frame_begin(){let omega=motion_value(3);motion_store(58,omega);atomicStore(&motion_control[57],select(0u,1u,omega!=0.0&&motion_value(0)>0.0));if(atomicLoad(&motion_control[57])!=0u){motion_store(3,0.0);atomicStore(&motion_control[18],0u);}}
@compute @workgroup_size(128) fn motion_enter_frame(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||atomicLoad(&motion_control[57])==0u){return;}let s=motion_state(i);let h=max(motion_value(8),1.0e-12);let offset=s.xy-motion.wheel.xy;let angle=-motion_value(58)*h;let basis=motion_rotate(vec2<f32>(1.0,0.0),angle);let half=motion_rotate(vec2<f32>(1.0,0.0),angle*.5).y;let change=(-2.0*half*half)*offset+basis.y*vec2<f32>(-offset.y,offset.x);let mapped=motion_rotate(s.zw,angle)+change/h;motion_actors[i*128u+120u]=bitcast<vec4<u32>>(s);motion_actors[i*128u+121u]=bitcast<vec4<u32>>(vec4<f32>(mapped,0.0,0.0));motion_write(i,vec4<f32>(s.xy,mapped));}
@compute @workgroup_size(128) fn motion_enter_rotating_frame(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||atomicLoad(&motion_control[57])==0u){return;}let s=motion_state(i);let offset=motion_offset_parts(motion_position_parts(i,0.0));let h=motion_value(8);let angle=-motion_end_rotation(h);
 // Exact inertial endpoint expressed in the shared moving reference. The
 // mass objective is the inertial displacement, not an approximate force
 // from the coordinate system. Contacts alone change the free trajectory.
 let rotated=motion_vector_rotated_parts(offset,angle);let difference=motion_compensated(rotated.xy,rotated.zw,-offset.xy);let change=motion_two_sum(difference.xy,difference.zw-offset.zw);
 let mapped=motion_rotate(s.zw,angle)+(change.xy+change.zw)/h;
 motion_actors[i*128u+120u]=bitcast<vec4<u32>>(s);motion_actors[i*128u+121u]=bitcast<vec4<u32>>(vec4<f32>(mapped,0.0,0.0));motion_write(i,vec4<f32>(s.xy,mapped));
}
@compute @workgroup_size(1) fn motion_frame_end(){if(atomicLoad(&motion_control[57])!=0u){atomicStore(&motion_control[18],0u);atomicStore(&motion_control[57],0u);}}
// The coframe is only a velocity-solve coordinate system. Convert the proposal
// to an inertial constant velocity BEFORE swept-path admission. Partial steps
// must not reuse a secant map prepared for a different duration.
@compute @workgroup_size(128) fn motion_exit_frame(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||atomicLoad(&motion_control[57])==0u){return;}
 let s=motion_state(i);let free=bitcast<vec4<f32>>(motion_actors[i*128u+120u]);let mapped=bitcast<vec4<f32>>(motion_actors[i*128u+121u]).xy;
 let velocity=free.zw+motion_rotate(s.zw-mapped,motion_value(58)*motion_value(8));motion_write(i,vec4<f32>(s.xy,velocity));
}
@compute @workgroup_size(1) fn motion_exit_frame_end(){if(atomicLoad(&motion_control[57])!=0u){motion_store(3,motion_value(58));atomicStore(&motion_control[57],0u);atomicStore(&motion_control[18],0u);}}
@compute @workgroup_size(1) fn motion_dispatch_prepare(){let pending=motion_value(0)>0.0&&atomicLoad(&motion_control[6])==0u;let groups=select(0u,(motion.counts.x+127u)/128u,pending);atomicStore(&motion_dispatch_args[0],groups);atomicStore(&motion_dispatch_args[1],1u);atomicStore(&motion_dispatch_args[2],1u);atomicStore(&motion_dispatch_args[3],groups);atomicStore(&motion_dispatch_args[4],1u);atomicStore(&motion_dispatch_args[5],1u);atomicStore(&motion_dispatch_args[6],select(0u,1u,pending));atomicStore(&motion_dispatch_args[7],1u);atomicStore(&motion_dispatch_args[8],1u);atomicStore(&motion_dispatch_args[16],select(0u,(motion.grid.x*motion.grid.y+127u)/128u,pending));atomicStore(&motion_dispatch_args[17],1u);atomicStore(&motion_dispatch_args[18],1u);}
// Pending contact time already has its force prediction. GPU-side dispatch
// admission must use the same law clock, not a host estimate or a second tick.
@compute @workgroup_size(1) fn motion_force_dispatch(){
 let ready=motion_force_ready();atomicStore(&motion_dispatch_args[9],select(0u,(motion.counts.x+127u)/128u,ready));
 atomicStore(&motion_dispatch_args[10],1u);atomicStore(&motion_dispatch_args[11],1u);
 atomicStore(&motion_dispatch_args[12],select(0u,atomicLoad(&motion_dispatch_args[15]),ready));
 atomicStore(&motion_dispatch_args[13],1u);atomicStore(&motion_dispatch_args[14],1u);
}
// Representation may read the accepted law clock, never advance it. Counter
// 63 is its prior publication clock; 56 is the elapsed publication interval.
@compute @workgroup_size(1) fn motion_representation_clock(){let time=motion_value(2);motion_store(56,max(0.0,time-motion_value(63)));motion_store(63,time);}

@compute @workgroup_size(1)
fn motion_reset_speed(){atomicStore(&motion_control[5],0u);}
@compute @workgroup_size(128)
fn motion_measure_speed(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x&&motion_value(0)>0.0){let state=motion_state(gid.x);if(any((bitcast<vec4<u32>>(state)&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))){atomicOr(&motion_control[6],1u);}else{atomicMax(&motion_control[5],bitcast<u32>(length(state.zw)));}}}
@compute @workgroup_size(1)
fn motion_prepare(){
  motion_store(4,1.0);motion_store(9,0.0);atomicStore(&motion_control[11],1u);atomicStore(&motion_control[19],0xffffffffu);atomicStore(&motion_control[20],0xffffffffu);
  let pair_cfl=motion_pair_contact()&&!MOTION_FULL_WINDOW;let transport_bound=select(motion_value(0),motion.minimum.z*1.2/max(motion_value(5)+abs(motion_value(3))*WHEEL_RADIUS,1.0e-12),pair_cfl);
  var horizon=min(motion_value(0),min(transport_bound,select(0.02,0.25,MOTION_FULL_WINDOW)/max(abs(motion_value(3)),1.0e-12)));
  if(MOTION_ROTATING_TRAJECTORY&&motion_value(3)!=0.0){let input_time=(motion.wheel.z-motion_value(12))/motion_value(3);if(input_time>0.0){horizon=min(horizon,input_time);}}
  // A bounded endpoint search must continue at its untested smaller interval
  // next time, not restart the same failed four trials indefinitely. This is
  // an integration step bound only; every new proposal is re-certified.
  if(motion_value(54)>0.0){horizon=min(horizon,motion_value(54));}
  if(!motion_ledger()){atomicStore(&motion_control[18],0u);}
  else if(atomicLoad(&motion_control[18])!=0u){let remaining=motion_value(16)-motion_value(13);if(remaining<=1.0e-8){atomicStore(&motion_control[18],0u);}else{horizon=min(horizon,remaining);}}
  if(motion_ledger()&&atomicLoad(&motion_control[18])==0u){motion_store(16,motion_value(13)+horizon);}
  motion_store(8,horizon);motion_store(51,horizon);atomicStore(&motion_control[15],0u);
}
// Corrected velocities can be faster than free velocities. Keep swept-cell
// storage bounded. A twist pilot's feasible-start interval is NOT a bound on
// physical progress: independently re-certify the actual inertial trajectory
// over the original motion interval. Its pilot supporting planes are proposals,
// not certificates of the expanded path. No position is committed here.
@compute @workgroup_size(1) fn motion_bound_proposal_horizon(){
 let requested=select(motion_value(8),motion_value(51),motion.grid.w==2u||motion.grid.w==4u);
 let h=select(requested,min(requested,motion.minimum.z*1.2/max(motion_value(5)+abs(motion_value(3))*WHEEL_RADIUS,1.0e-12)),motion_pair_contact()&&!MOTION_FULL_WINDOW);
 if(h!=motion_value(8)){motion_store(8,h);atomicStore(&motion_control[18],0u);motion_store(16,motion_value(13)+h);}
}
@compute @workgroup_size(128) fn motion_ledger_initialize(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<motion.counts.x&&motion_value(0)>0.0&&motion_ledger()&&atomicLoad(&motion_control[18])==0u){motion_actors[i*128u]=bitcast<vec4<u32>>(motion_state(i));motion_actors[i*128u+1u]=vec4<u32>(0u,1u,bitcast<u32>(motion_value(13)),0u);let low=bitcast<vec4<f32>>(motion_actors[i*128u+122u]).xy;motion_actors[i*128u+122u]=bitcast<vec4<u32>>(vec4<f32>(low,low));}}
fn motion_emit(time:f32,i:u32,j:u32,kind:u32){
  if(time>=motion_value(16)){return;}let k=atomicAdd(&motion_control[15],1u);if(k<arrayLength(&motion_incoming)){let ri=motion_actors[i*128u+1u].x;var rj=0u;if(j!=0xffffffffu){rj=motion_actors[j*128u+1u].x;}motion_incoming[k]=MotionEvent(time,i,j,kind,ri,rj,vec2<u32>(0u));}
}
@compute @workgroup_size(128)
fn motion_clear(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.grid.x*motion.grid.y&&motion_value(0)>0.0){atomicStore(&motion_cells[gid.x],0u);}}
fn motion_insert(i:u32,swept:bool){
  if(i>=motion.counts.x||motion_value(0)==0.0){return;}let state=motion_state(i);let horizon=select(motion_value(8),max(0.0,motion_value(16)-motion_value(13)),motion_ledger());let end=state.xy+state.zw*select(0.0,horizon,swept);
  let padding=select(0.0,motion.minimum.z+motion.material.w,swept);let lo=motion_cell(min(state.xy,end)-padding);let hi=motion_cell(max(state.xy,end)+padding);
  if(!motion_valid(lo)||!motion_valid(hi)){atomicOr(&motion_control[6],2u);return;}
  motion_actors[i*128u+123u]=bitcast<vec4<u32>>(vec4<i32>(lo,hi));
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
  if(motion_pair_contact()&&(!ledger||rebuilding||motion_actors[i*128u+1u].y!=0u)){
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let cell=vec2<i32>(x,y);let bucket=motion_bucket(cell);let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);if(j==i||((!ledger||rebuilding||motion_actors[j*128u+1u].y!=0u)&&j<i)||!motion_first_shared_cell(cell,lo,j)){continue;}let other=motion_state(j);let toi=motion_pair_toi_parts(motion_separation_parts(i,j),d-other.zw*horizon,2.0*radius);prediction_count++;if(ledger){if(toi<1.0){motion_emit(motion_value(13)+toi*horizon,min(i,j),max(i,j),0u);}}else{fraction=min(fraction,toi);}}}}
  }
  // Always re-certify curved/rotating boundaries at the current pose. A retained
  // pair-event ledger must never substitute stale geometry certificates.
  let wall_d=state.zw*motion_value(8);
  var wall_fraction=1.0;
  let offset_parts=motion_offset_parts(motion_position_parts(i,0.0));let offset=offset_parts.xy+offset_parts.zw;let limit=WHEEL_RADIUS-radius-WHEEL_BAR_HALF_WIDTH-motion_reserve()-motion_rim_error();let a=dot(wall_d,wall_d);let b=dot(offset,wall_d);let gap=motion_rim_gap_parts(offset_parts)-motion_reserve();let c=-gap*(motion_length(offset)+limit);
  // A circle is convex: two certified endpoints certify the complete straight
  // segment. Otherwise rationalize the positive root when b>=0; subtracting
  // almost equal sqrt(b*b-a*c) and b rounded safe near-contact exits to zero.
  let end_offset=motion_offset_parts(motion_position_parts(i,motion_value(8)));
  let rim_segment_safe=gap>=0.0&&motion_rim_gap_parts(end_offset)>=motion_reserve();
  if(!rim_segment_safe){
   if(c>=0.0&&b>0.0){wall_fraction=0.0;}
   else if(a>1.0e-20&&b*b-a*c>=0.0){let root=sqrt(max(0.0,b*b-a*c));var exit=(-b+root)/a;if(b>=0.0&&c<=0.0){exit=(-c)/max(b+root,1.0e-30);}if(exit>=0.0&&exit<1.0){wall_fraction=min(wall_fraction,exit);}}
  }
  for(var segment=0u;segment< motion.counts.w;segment++){wall_fraction=min(wall_fraction,motion_bar_toi_parts(motion_position_parts(i,0.0),wall_d,segment,motion_value(1),motion_value(3)*motion_value(8)));}
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
    if(size+n>capacity){var kept=0u;for(var k=0u;k<size;k++){let e=motion_heap[k];var valid=e.ri==motion_actors[e.i*128u+1u].x;if(e.j!=0xffffffffu){valid=valid&&e.rj==motion_actors[e.j*128u+1u].x;}if(valid){motion_heap[kept]=e;kept++;}}size=kept;atomicStore(&motion_control[14],size);var i=size/2u;loop{if(i==0u){break;}i--;motion_down(i);}}
    if(size+n>capacity){motion_fallback();return;}
    for(var k=0u;k<n;k++){let e=motion_incoming[k];var i=size;size++;loop{if(i==0u){break;}let parent=(i-1u)/2u;if(!motion_before(e,motion_heap[parent])){break;}motion_heap[i]=motion_heap[parent];i=parent;}motion_heap[i]=e;}atomicStore(&motion_control[14],size);
  }
  loop{
    let size=atomicLoad(&motion_control[14]);if(size==0u){return;}let e=motion_heap[0];var valid=e.ri==motion_actors[e.i*128u+1u].x;if(e.j!=0xffffffffu){valid=valid&&e.rj==motion_actors[e.j*128u+1u].x;}
    if(valid&&e.time>motion_value(13)+motion_value(8)){return;}
    atomicStore(&motion_control[14],size-1u);if(size>1u){motion_heap[0]=motion_heap[size-1u];motion_down(0u);}if(!valid){atomicAdd(&motion_control[23],1u);continue;}
    atomicStore(&motion_control[19],e.i);atomicStore(&motion_control[20],e.j);let fraction=clamp((e.time-motion_value(13))/max(motion_value(8),1.0e-20),0.0,1.0);atomicMin(&motion_control[4],bitcast<u32>(fraction)&0x7fffffffu);return;
  }
}
@compute @workgroup_size(128) fn motion_clear_dirty(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<motion.counts.x){motion_actors[gid.x*128u+1u].y=0u;}}
@compute @workgroup_size(1) fn motion_consume_ledger(){if(motion_ledger()&&motion_value(0)>0.0){let i=atomicLoad(&motion_control[19]);let j=atomicLoad(&motion_control[20]);if(i!=0xffffffffu){motion_mark(i);}if(j!=0xffffffffu){motion_mark(j);}}}
fn motion_local_end_position(i:u32,duration:f32)->vec2<f32>{return motion_position_parts(i,duration).xy;}
fn motion_rotated_parts(parts:vec4<f32>,angle:f32)->vec4<f32>{let rotated=motion_vector_rotated_parts(motion_offset_parts(parts),angle);return motion_compensated(rotated.xy,rotated.zw,motion.wheel.xy);}
fn motion_end_rotation(duration:f32)->f32{let delta=motion_end_angle(duration)-motion_value(1);return delta-6.28318530718*floor((delta+3.14159265359)/6.28318530718);}
fn motion_end_parts(i:u32,duration:f32)->vec4<f32>{let p=motion_position_parts(i,duration);if(atomicLoad(&motion_control[57])==0u){return p;}return motion_rotated_parts(p,motion_end_rotation(duration));}
fn motion_end_position(i:u32,duration:f32)->vec2<f32>{return motion_end_parts(i,duration).xy;}
fn motion_end_separation_parts(i:u32,j:u32,duration:f32)->vec4<f32>{let a=motion_end_parts(i,duration);let b=motion_end_parts(j,duration);let difference=motion_compensated(a.xy,a.zw,-b.xy);return motion_two_sum(difference.xy,difference.zw-b.zw);}
fn motion_end_angle(duration:f32)->f32{let omega=select(motion_value(3),motion_value(58),atomicLoad(&motion_control[57])!=0u);if(omega*duration==0.0){return motion_value(1);}let angle=select(motion_value(31)+omega*(motion_value(13)+duration),motion_value(1)+omega*duration,MOTION_ROTATING_TRAJECTORY);let wrapped=angle-6.28318530718*floor((angle+3.14159265359)/6.28318530718);return select(angle,wrapped,abs(angle)>3.14159265359);}
@compute @workgroup_size(1) fn motion_prepare_guard(){atomicStore(&motion_control[10],atomicLoad(&motion_control[11]));atomicStore(&motion_control[11],0u);atomicStore(&motion_control[28],atomicLoad(&motion_control[4]));}
@compute @workgroup_size(128) fn motion_guard_end(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u||atomicLoad(&motion_control[10])==0u){return;}
  let fraction=motion_value(28);let duration=motion_value(8)*select(fraction*.999,1.0,fraction>=1.0);let offset=motion_offset_parts(motion_end_parts(i,duration));let radius=motion.minimum.z;var safe=motion_rim_gap_parts(offset)>=motion_reserve();var reason=select(1u,0u,safe);
  for(var k=0u;k<motion.counts.w;k++){let clear=motion_bar_gap_parts(offset,motion_end_angle(duration),baffle(k))>=motion_reserve();if(!clear){reason|=2u;}safe=safe&&clear;}
  if(motion_pair_contact()){let own=motion_state(i);let padding=radius+motion.material.w;let q=motion_local_end_position(i,duration);let lo=motion_cell(min(own.xy,q)-padding);let hi=motion_cell(max(own.xy,q)+padding);
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){let cell=vec2<i32>(x,y);let bucket=motion_bucket(cell);let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);for(var k=0u;k<count;k++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+k]);if(j>i&&motion_first_shared_cell(cell,lo,j)){let clear=motion_pair_gap_parts(motion_end_separation_parts(i,j,duration))>=motion_reserve();if(!clear){reason|=4u;}safe=safe&&clear;}}}}}
  motion_actors[i*128u+1u].w=reason;
  if((reason&1u)!=0u){atomicAdd(&motion_control[43],1u);}if((reason&2u)!=0u){atomicAdd(&motion_control[44],1u);}if((reason&4u)!=0u){atomicAdd(&motion_control[45],1u);}
  // CCD certifies the swept path. This additional check protects the actual
  // float-buffer endpoint before any position/pose is committed. Never repair.
  if(!safe){atomicMin(&motion_control[4],bitcast<u32>(fraction*.5)&0x7fffffffu);atomicStore(&motion_control[11],1u);atomicAdd(&motion_control[26],1u);}
}
@compute @workgroup_size(1)
fn motion_advance_control(){
  if(atomicLoad(&motion_control[6])!=0u||motion_value(0)==0.0){return;}let fraction=motion_value(4);let shortened=MOTION_ROTATING_TRAJECTORY&&fraction<1.0;var duration=select(motion_value(8)*select(fraction*0.999,1.0,fraction>=1.0),0.0,atomicLoad(&motion_control[11])!=0u||shortened);
  // A near-TOI rotating retry must move materially inside the certified
  // interval. A 0.999 retry can converge on the same rounded contact forever
  // while pointer input keeps increasing the requested angular velocity.
  if((atomicLoad(&motion_control[11])!=0u||shortened)&&fraction>0.0){motion_store(54,motion_value(8)*fraction*0.9);}else if(duration>0.0){motion_store(54,0.0);}
  motion_store(9,duration);atomicAdd(&motion_control[25],1u);atomicStore(&motion_control[22],0u);
  // Generation for the optional cold-on-physical-change cache experiment.
  // Warm guesses are private; a generation counter never advances law time.
  if(duration>0.0){atomicAdd(&motion_control[55],1u);}
  let corrected=duration-motion_value(60);let local=motion_value(13)+corrected;motion_store(60,(local-motion_value(13))-corrected);motion_store(0,max(0.0,motion_value(62)-local));if(atomicLoad(&motion_control[57])!=0u){motion_store(3,motion_value(58));}let delta=motion_value(3)*duration;
  // Never re-wrap an unchanged pose: repeated GPU trig introduced measurable
  // stationary drift and invalidated certified baffle geometry.
  let acceptedRotation=motion_end_rotation(duration);motion_store(42,acceptedRotation);
  if(delta!=0.0){motion_store(1,motion_end_angle(duration));motion_store(12,motion_value(12)+acceptedRotation);}
  motion_store(13,local);
  if(atomicLoad(&motion_control[7])!=0u){motion_store(2,motion_value(59)+local);}
}
@compute @workgroup_size(128)
fn motion_advance_particles(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<motion.counts.x&&atomicLoad(&motion_control[6])==0u&&motion_value(9)>0.0){var state=motion_state(i);let duration=select(motion_value(9),0.0,motion_ledger()&&atomicLoad(&motion_control[18])!=0u);let start=motion_position_parts(i,0.0);var parts=motion_position_parts(i,duration);if(atomicLoad(&motion_control[57])!=0u){let delta=motion_value(42);parts=motion_rotated_parts(parts,delta);let difference=motion_compensated(parts.xy,parts.zw,-start.xy);let displacement=motion_two_sum(difference.xy,difference.zw-start.zw);state=vec4<f32>(parts.xy,(displacement.xy+displacement.zw)/motion_value(9));}else{state=vec4<f32>(parts.xy,state.zw);}motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(state);let previous=motion_actors[i*128u+122u];motion_actors[i*128u+122u]=vec4<u32>(bitcast<vec2<u32>>(parts.zw),previous.zw);}}
fn motion_wall_impulse(i:u32,normal:vec2<f32>,surface:vec2<f32>,friction:f32){
  let state=motion_state(i);var velocity=state.zw;let relative=velocity-surface;let vn=dot(relative,normal);
  // A tiny separating velocity keeps curved tangential contacts from an exact-
  // touching zero-TOI deadlock in finite precision. No position is repaired.
  let separating_speed=motion.material.w/max(motion.schedule.x,1.0e-6);
  if(vn>=separating_speed){return;}let jn=separating_speed-vn;velocity+=normal*jn;let tangent=relative-normal*vn;let speed=length(tangent);velocity-=tangent/max(speed,1.0e-20)*min(speed,friction*jn);motion_write(i,vec4<f32>(state.xy,velocity));
}
@compute @workgroup_size(128) fn motion_solve_boundaries(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}
  var own=motion_state(i);let horizon=max(motion_value(8),1.0e-12);let clearance=motion.minimum.z+WHEEL_BAR_HALF_WIDTH;
  // The rim is convex. Projecting the complete endpoint inside it certifies
  // the whole straight coframe path and lets every independent wall impact be
  // handled in parallel instead of serializing the global clock by earliest TOI.
  let offset=own.xy-motion.wheel.xy;let endpoint=offset+own.zw*horizon;let limit=WHEEL_RADIUS-clearance-motion_reserve()-motion_rim_error()-motion.material.w;
  let endpoint_radius=motion_length(endpoint);
  if(endpoint_radius>=limit){let capped=endpoint*(limit/max(endpoint_radius,1.0e-20));let velocity=(capped-offset)/horizon;motion_write(i,vec4<f32>(own.xy,velocity));own=motion_state(i);}
  // A baffle capsule is convex. Its first swept contact supplies a supporting
  // plane; removing inward velocity at that plane certifies the remaining path.
  for(var segment=0u;segment<motion.counts.w;segment++){
    let travel=own.zw*horizon;let toi=motion_bar_toi_parts(motion_position_parts(i,0.0),travel,segment,motion_value(1),motion_value(3)*horizon);
    if(toi<1.0){let angle=motion_value(1)+motion_value(3)*horizon*toi;let contact=own.xy+travel*toi;let local=motion_rotate(contact-motion.wheel.xy,-angle);let bar=baffle(segment);let edge=bar.zw-bar.xy;let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let separation=local-closest;let distance=motion_length(separation);if(distance>1.0e-12){let normal=motion_rotate(separation/distance,angle);let point=motion_rotate(closest,angle);let surface=vec2<f32>(-point.y,point.x)*motion_value(3);motion_wall_impulse(i,normal,surface,motion.schedule.w);own=motion_state(i);}}
  }
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
` + DENSE_CONTACT_WGSL + CONTACT_PRIMAL_BARRIER_WGSL + CONTACT_DUAL_COMPONENT_CANDIDATES_WGSL + CONTACT_AUGMENTED_VELOCITY_WGSL + CONTACT_TILED_PROJECTION_WGSL + CONTACT_CACHED_STAR_WGSL + PAUSED_WORLD_POSE_WGSL;

export const CONTACT_PRIMAL_SCRATCH_FIELDS=99;
export function createPreventiveContactResources(device,world,count){
  const radius=world.kind==='granular'?world.properties.radius:world.properties.spacing*.46,cellWidth=radius*2*1.1,margin=1e-6;
  if(!(radius>margin*64)||!Number.isSafeInteger(count)||count<1)throw new Error('Hard-contact radius/count is outside this metre-scale specialization');
  const side=Math.ceil((world.geometry.radius*2+radius*8)/cellWidth),minimum=world.geometry.center.map(v=>v-world.geometry.radius-radius*4),make=(size,usage)=>device.createBuffer({size,usage});
  const maximumBytes=Math.min(device.limits?.maxBufferSize??Number.MAX_SAFE_INTEGER,device.limits?.maxStorageBufferBindingSize??Number.MAX_SAFE_INTEGER);
  if(!Number.isSafeInteger(side)||side<1||!Number.isSafeInteger(side*side*64*4)||side*side*64*4>maximumBytes||Math.ceil(side*side/128)>(device.limits?.maxComputeWorkgroupsPerDimension??65535))throw new Error('Preventive contact broadphase exceeds this GPU device limit');
  const capacity=Math.max(4096,count*16);
  if(capacity*32>maximumBytes)throw new Error('Collision ledger exceeds this GPU device limit');
  if(count*CONTACT_PRIMAL_SCRATCH_FIELDS*16>maximumBytes||count*2048>maximumBytes)throw new Error('Padded contact specialization exceeds this GPU binding limit; a compact batched backend is required');
  const uniform=make(96,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),cells=make(side*side*4,GPUBufferUsage.STORAGE),items=make(side*side*64*4,GPUBufferUsage.STORAGE),control=make(256,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST),readback=make(256,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),actors=make(count*2048,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),incoming=make(capacity*32,GPUBufferUsage.STORAGE),heap=make(capacity*32,GPUBufferUsage.STORAGE),dispatchArgs=make(80,GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST);
  const emptyLayout=device.createBindGroupLayout({entries:[]}),forceLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),forceGroup=device.createBindGroup({layout:forceLayout,entries:[{binding:0,resource:{buffer:uniform}},{binding:4,resource:{buffer:control}}]});
  const precisionOutput=make(count*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),primalScratchFields=CONTACT_PRIMAL_SCRATCH_FIELDS,primalScratch=make(count*primalScratchFields*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const initialControl=new Float32Array(64);initialControl[1]=world.geometry.rotation;initialControl[50]=1;const reset=()=>device.queue.writeBuffer(control,0,initialControl);reset();
  const emptyGroup=device.createBindGroup({layout:emptyLayout,entries:[]});
  const pausedPose=make((1+count*4)*16,GPUBufferUsage.STORAGE);
  return {uniform,cells,items,control,readback,actors,incoming,heap,dispatchArgs,precisionOutput,primalScratch,pausedPose,primalScratchFields,emptyLayout,emptyGroup,forceLayout,forceGroup,side,minimum,radius,cellWidth,margin,count,reset,destroy(){for(const b of [uniform,cells,items,control,readback,actors,incoming,heap,dispatchArgs,precisionOutput,primalScratch,pausedPose])b.destroy();}};
}

export async function createPreventiveParticleContactGpu(device,world,physics,source,resources){
  if(resources.rotatingTrajectory&&(!['barrier','tiled'].includes(resources.experimentalSolve)||!source.includes('fn motion_enter_rotating_frame')))throw new Error('Rotating-reference experiment requires its compiled contact specialization');
  if(resources.sandBarrier&&(world.kind!=='granular'||!resources.rotatingTrajectory||!source.includes('override PRIMAL_COULOMB_DISSIPATION')))throw new Error('Granular barrier candidate requires its friction binding and matched rotating trajectory');
  if(!Number.isInteger(resources.starSweeps??1)||(resources.starSweeps??1)<1||(resources.starSweeps??1)>64)throw new Error('Star sweep experiment requires an integer from 1 to 64');
  if(resources.starNonlinearRim&&(resources.experimentalSolve!=='star'||!source.includes('override DENSE_NONLINEAR_RIM')||!source.includes('fn contact_star_rim_project')))throw new Error('Nonlinear rim experiment requires its compiled star specialization');
  if(resources.persistentSkin&&!source.includes('override CONTACT_PERSISTENT_SKIN'))throw new Error('Persistent contact clearance requires its compiled shader specialization');
  if(resources.cacheEpoch&&!source.includes('override DUAL_CACHE_EPOCH'))throw new Error('Contact cache-epoch experiment requires its compiled shader specialization');
  if(resources.predictedContacts&&!source.includes('override DUAL_PREDICTED_CONTACTS'))throw new Error('Predicted contact selection requires its compiled shader specialization');
  device.queue.writeBuffer(resources.dispatchArgs,60,new Uint32Array([Math.ceil((physics.gridCellCount??0)/128)]));
  const r=resources,baseEntries=[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))],layout=device.createBindGroupLayout({entries:[...baseEntries,{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,layout]}),staticModule=createStaticWgslModuleCache(device,source);
  const entries=['contact_residual','motion_pause_velocity','motion_begin','motion_input_omega','motion_frame_begin','motion_enter_frame','motion_frame_end','motion_dispatch_prepare','motion_reset_speed','motion_measure_speed','motion_prepare','motion_ledger_initialize','motion_update_ledger','motion_find_fallback','motion_clear_dirty','motion_consume_ledger','motion_prepare_guard','motion_guard_end','motion_clear','motion_fill_swept','motion_fill_static','motion_sort','motion_find_toi','motion_advance_control','motion_advance_particles','motion_clearance_begin','motion_clearance','motion_clearance_end','motion_solve_boundaries','motion_precision_reset','motion_precision_reset_end','motion_materialize_state','motion_export_precision','dense_begin','dense_build','dense_links',...Array.from({length:9},(_,i)=>`contact_project_${i}`),...Array.from({length:9},(_,i)=>`contact_repair_${i}`)];
  entries.push('motion_exit_frame','motion_exit_frame_end');
  if(world.kind==='liquid'||r.sandBarrier||r.experimentalSolve==='tiled'){
    entries.push('primal_begin','primal_cool','primal_initialize','primal_gradient','primal_operator','primal_alpha','primal_update','primal_beta','primal_conjugate','primal_line_begin','primal_line_bound','primal_trial_begin','primal_trial','primal_trial_validate','primal_trial_reduce','primal_line_apply');
    entries.push('primal_cg_begin','primal_weights','primal_component','primal_finish','primal_energy','primal_energy_validate');
  }
  if(source.includes('fn motion_bound_proposal_horizon'))entries.push('motion_bound_proposal_horizon');
  const requestedSolve=r.experimentalSolve??'barrier';
  if(r.rotatingTrajectory)entries.push('motion_enter_rotating_frame');
  const wantsDual=['dual','mprgp','accelerated','blockpivot','tiled','star'].includes(requestedSolve);
  if(['augmented','affine'].includes(requestedSolve)&&source.includes('fn al_component'))entries.push('al_component','al_affine_component');
  if(requestedSolve==='projected'&&source.includes('fn contact_project_component'))entries.push('contact_project_component');
  if(source.includes('fn motion_force_dispatch'))entries.push('motion_force_dispatch');
  if(source.includes('fn motion_representation_clock'))entries.push('motion_representation_clock');
  if(source.includes('fn motion_twist_horizon'))entries.push('motion_twist_begin','motion_twist_horizon','motion_twist_validate');
  if(world.kind==='liquid'&&r.count<=1908&&source.includes('fn primal_cached_component'))entries.push('primal_cached_component');
  if((world.kind==='liquid'||r.sandBarrier)&&source.includes('fn primal_adaptive_limit'))entries.push('primal_adaptive_limit');
  if(requestedSolve==='dual'&&source.includes('fn primal_dual_component'))entries.push('primal_dual_component','primal_dual_compact_component');
  if(wantsDual&&source.includes('fn primal_dual_seed'))entries.push('primal_dual_seed');
  if(wantsDual&&r.orderedContacts===true&&source.includes('fn primal_dual_sort_active'))entries.push('primal_dual_sort_active');
  if(requestedSolve==='mprgp'&&source.includes('fn primal_mprgp_component'))entries.push('primal_mprgp_component');
  if(requestedSolve==='accelerated'&&source.includes('fn primal_accelerated_component'))entries.push('primal_accelerated_component');
  if(requestedSolve==='blockpivot'&&source.includes('fn primal_blockpivot_component'))entries.push('primal_blockpivot_component');
  if(['tiled','star'].includes(requestedSolve)&&source.includes('fn contact_tiled_warm'))entries.push('contact_tiled_warm','contact_tiled_save');
  if(requestedSolve==='tiled'&&source.includes('fn contact_tiled_project'))entries.push(...Array.from({length:9},(_,i)=>`contact_tiled_project_${i}`));
  if(requestedSolve==='star'&&source.includes('fn contact_star_project'))entries.push(...Array.from({length:9},(_,i)=>`contact_star_project_${i}`));
  if(requestedSolve==='star'&&source.includes('fn contact_star_compact'))entries.push('contact_star_reset','contact_star_compact');
  if(requestedSolve==='star'&&r.starNonlinearRim)entries.push('contact_star_rim_seed','contact_star_rim_project');
  if(source.includes('fn motion_velocity_trial_begin'))entries.push('motion_velocity_trial_begin','motion_velocity_trial_seed','motion_velocity_trial_end');
  const parallelDualEntries=requestedSolve==='dual'&&r.experimentalDispatch==='parallel-dual'&&source.includes('fn primal_dual_global_begin')?['solution','direction','start','operator','bound','update','conjugate','refine','warm_end','begin','alpha','beta'].map(name=>`primal_dual_global_${name}`):[];
  entries.push(...parallelDualEntries);
  const parallelDualControllers=new Set(['primal_dual_global_begin','primal_dual_global_alpha','primal_dual_global_beta']);
  const hasStar=requestedSolve==='star'&&source.includes('fn contact_star_compact');
  const starCells=hasStar?device.createBuffer({label:'VKF compact contact cells',size:9*Math.ceil(r.side/3)**2*4,usage:GPUBufferUsage.STORAGE}):null;
  const starArgs=hasStar?device.createBuffer({label:'VKF compact contact dispatch',size:108,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT}):null;
  const starProjectLayout=hasStar?device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}):null;
  const starCompactLayout=hasStar?device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[2,3,4,5,11,12].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}):null;
  const starProjectPipelineLayout=hasStar?device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,starProjectLayout]}):null;
  const starCompactPipelineLayout=hasStar?device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,starCompactLayout]}):null;
  const starProjectGroup=hasStar?device.createBindGroup({layout:starProjectLayout,entries:[[0,r.uniform],[1,physics.particleBuffer],[2,r.cells],[3,r.items],[4,r.control],[5,r.actors],[11,starCells]].map(([binding,buffer])=>({binding,resource:{buffer}}))}):null;
  const starCompactGroup=hasStar?device.createBindGroup({layout:starCompactLayout,entries:[[0,r.uniform],[2,r.cells],[3,r.items],[4,r.control],[5,r.actors],[11,starCells],[12,starArgs]].map(([binding,buffer])=>({binding,resource:{buffer}}))}):null;
  const parallelDualLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[4,8,10].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}),parallelDualPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,parallelDualLayout]}),parallelDualGroup=device.createBindGroup({layout:parallelDualLayout,entries:[[0,r.uniform],[4,r.control],[8,r.dispatchArgs],[10,r.primalScratch]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
  const controllerEntries=new Set(['motion_force_dispatch','motion_dispatch_prepare','dense_start_control','dense_alpha','primal_cg_begin','primal_alpha']);
  const controllerLayout=device.createBindGroupLayout({entries:[...baseEntries,{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]}),controllerPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,controllerLayout]});
  const exportLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[5,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]}),exportPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,exportLayout]}),exportGroup=device.createBindGroup({layout:exportLayout,entries:[[0,r.uniform],[5,r.actors],[9,r.precisionOutput]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
  const candidateConstants={
    ...(source.includes('override PRIMAL_BLOCK_PRECONDITIONER')?{PRIMAL_BLOCK_PRECONDITIONER:r.blockPreconditioner===true}:{}),
    ...(source.includes('override MOTION_FULL_WINDOW')?{MOTION_FULL_WINDOW:r.fullWindow===true}:{}),
    ...(source.includes('override MOTION_ROTATING_TRAJECTORY')?{MOTION_ROTATING_TRAJECTORY:r.rotatingTrajectory===true}:{}),
    ...(source.includes('override PRIMAL_COULOMB_DISSIPATION')?{PRIMAL_COULOMB_DISSIPATION:r.sandBarrier===true}:{}),
    ...(source.includes('override DENSE_NONLINEAR_RIM')?{DENSE_NONLINEAR_RIM:r.starNonlinearRim===true}:{}),
    ...(source.includes('override DUAL_PREDICTED_CONTACTS')?{DUAL_PREDICTED_CONTACTS:r.predictedContacts===true}:{}),
    ...(source.includes('override CONTACT_PERSISTENT_SKIN')?{CONTACT_PERSISTENT_SKIN:r.persistentSkin===true}:{}),
    ...(source.includes('override DUAL_CACHE_EPOCH')?{DUAL_CACHE_EPOCH:r.cacheEpoch===true}:{}),
  };
  // Compile in order instead of queuing every large Metal specialization at
  // once. Liquid does not dispatch the granular colored-contact kernels.
  const boundaryOnly=r.particlePairContact===false;
  const componentBarrier=world.kind==='liquid'&&requestedSolve==='barrier'&&r.count<=4096&&r.experimentalDispatch!=='global';
  const globalPrimalEntries=new Set(['primal_begin','primal_cool','primal_initialize','primal_gradient','primal_operator','primal_alpha','primal_update','primal_beta','primal_conjugate','primal_line_begin','primal_line_bound','primal_trial_begin','primal_trial','primal_trial_validate','primal_trial_reduce','primal_line_apply','primal_cg_begin','primal_weights','primal_energy','primal_energy_validate','primal_adaptive_limit']);
  const boundaryEntries=new Set(['motion_pause_velocity','motion_begin','motion_input_omega',
    'motion_frame_begin','motion_enter_frame','motion_enter_rotating_frame','motion_frame_end',
    'motion_dispatch_prepare','motion_reset_speed','motion_measure_speed','motion_prepare',
    'motion_clear','motion_fill_static','motion_sort','motion_clear_dirty','motion_prepare_guard',
    'motion_guard_end','motion_find_toi','motion_advance_control','motion_advance_particles',
    'motion_clearance_begin','motion_clearance','motion_clearance_end','motion_solve_boundaries',
    'motion_precision_reset','motion_precision_reset_end','motion_materialize_state',
    'motion_export_precision','motion_exit_frame','motion_exit_frame_end','motion_force_dispatch',
    'motion_representation_clock','motion_bound_proposal_horizon','motion_velocity_trial_begin',
    'motion_velocity_trial_seed','motion_velocity_trial_end']);
  const compileEntries=entries.filter(name=>(!boundaryOnly||boundaryEntries.has(name))&&(world.kind!=='liquid'||!/^contact_(project|repair)_\d+$/.test(name))&&(!componentBarrier||!globalPrimalEntries.has(name))&&(!name.startsWith('motion_twist_')||requestedSolve==='twist'||requestedSolve==='adaptive'||name==='motion_twist_begin'&&['affine','dual','mprgp','accelerated','blockpivot'].includes(requestedSolve)));
  const pausedLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,9,13].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))]});
  const pausedPipelineLayout=device.createPipelineLayout({bindGroupLayouts:[r.emptyLayout,r.emptyLayout,pausedLayout]});
  const pausedGroup=device.createBindGroup({layout:pausedLayout,entries:[[0,r.uniform],[1,physics.particleBuffer],[2,r.cells],[3,r.items],[4,r.control],[5,r.actors],[9,r.precisionOutput],[13,r.pausedPose]].map(([binding,buffer])=>({binding,resource:{buffer}}))});
  if(source.includes('fn paused_capture'))compileEntries.push('paused_capture','paused_prepare','paused_propose','paused_validate','paused_commit','paused_finish');
  const pipelines={};
  for(const entryPoint of compileEntries){
    const star=entryPoint.startsWith('contact_star_project_'),tile=entryPoint.startsWith('contact_tiled_project_');
    const selectedLayout=entryPoint.startsWith('paused_')?pausedPipelineLayout:star?starProjectPipelineLayout:['contact_star_reset','contact_star_compact'].includes(entryPoint)?starCompactPipelineLayout:entryPoint==='motion_export_precision'?exportPipelineLayout:parallelDualControllers.has(entryPoint)?parallelDualPipelineLayout:controllerEntries.has(entryPoint)?controllerPipelineLayout:pipelineLayout;
    const constants={...candidateConstants,
      ...(star?{CONTACT_STAR_COLOUR:Number(entryPoint.split('_').at(-1)),CONTACT_STAR_CG:r.starCG===true,CONTACT_STAR_COMPACT:true}:{}),
      ...(tile?{CONTACT_TILE_COLOUR:Number(entryPoint.split('_').at(-1))}:{}),
      ...(entryPoint==='primal_blockpivot_component'?{DUAL_BLOCK_PIVOT:true}:entryPoint==='al_affine_component'?{AL_AFFINE_PRECONDITIONER:true}:entryPoint.startsWith('primal_dual_global_')?{DUAL_PARALLEL:true}:{}),
    };
    const descriptor={label:`VKF contact ${entryPoint}`,layout:selectedLayout,compute:{module:staticModule(constants),entryPoint:star?'contact_star_project':tile?'contact_tiled_project':entryPoint==='al_affine_component'?'al_component':entryPoint}};
    const optional=entryPoint==='primal_cached_component';
    pipelines[entryPoint]=await (optional?createOptionalGpuPipeline:createCheckedGpuPipeline)(device,'compute',descriptor);
  }
  const group=device.createBindGroup({layout,entries:[...[r.uniform,physics.particleBuffer,r.cells,r.items,r.control,r.actors,r.incoming,r.heap].map((buffer,binding)=>({binding,resource:{buffer}})),{binding:10,resource:{buffer:r.primalScratch}}]}),controllerGroup=device.createBindGroup({layout:controllerLayout,entries:[r.uniform,physics.particleBuffer,r.cells,r.items,r.control,r.actors,r.incoming,r.heap,r.dispatchArgs].map((buffer,binding)=>({binding,resource:{buffer}}))}),empty=r.emptyGroup;
  const params=new ArrayBuffer(96),u=new Uint32Array(params),f=new Float32Array(params);u.set([r.count,world.kind==='granular'?2:3,0,world.geometry.segments.length,r.side,r.side,64,0]);f.set([...r.minimum,r.radius,r.cellWidth],8);f.set([...world.geometry.center,0,0],12);const sand=world.kind==='granular',projectedSand=sand&&!r.sandBarrier;f.set([world.properties.restitution??0,sand?(world.properties.friction??.9):0,sand?(world.properties.wall_friction??.78):0,r.margin],16);
  const cellAdmission=source.includes('&motion_dispatch_args[16]');
  const admittedParticles=new Set(['motion_fill_static','motion_clearance','contact_residual','primal_finish','motion_fill_swept','motion_find_toi','motion_guard_end','motion_clear_dirty','motion_advance_particles']);
  const pipelineFor=name=>{const pipeline=pipelines[name];if(!pipeline)throw new Error(`Compiled contact pipeline ${name} is missing`);return pipeline;};
  const dispatch=(pass,name,count,solver=false)=>{pass.setBindGroup(2,['contact_star_reset','contact_star_compact'].includes(name)?starCompactGroup:name==='motion_export_precision'?exportGroup:parallelDualControllers.has(name)?parallelDualGroup:controllerEntries.has(name)?controllerGroup:group);pass.setPipeline(pipelineFor(name));if(cellAdmission&&(name==='motion_clear'||name==='motion_sort'||/^contact_(project|repair)_\d$/.test(name)))pass.dispatchWorkgroupsIndirect(r.dispatchArgs,64);else if(admittedParticles.has(name)||name.startsWith('dense_')||name.startsWith('primal_dual_global_')||['primal_operator','primal_update','primal_conjugate'].includes(name))pass.dispatchWorkgroupsIndirect(r.dispatchArgs,solver?12:0);else pass.dispatchWorkgroups(Math.ceil(count/128));};
  const single=(pass,name)=>{pass.setBindGroup(2,['contact_star_reset','contact_star_compact'].includes(name)?starCompactGroup:parallelDualControllers.has(name)?parallelDualGroup:controllerEntries.has(name)?controllerGroup:group);pass.setPipeline(pipelineFor(name));pass.dispatchWorkgroups(1);};
  const policy=new ContactSchedulerPolicy(),timestamp=device.features?.has('timestamp-query')===true;
  const query=timestamp?device.createQuerySet({type:'timestamp',count:128}):null,queryBuffer=timestamp?device.createBuffer({size:1024,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null,queryReadback=timestamp?device.createBuffer({size:1024,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}):null;
  let queryIndex=0,started=0,selected='parallel',moving=false,eventBudget=4,fixedBudget=r.rotatingTrajectory?8:null,liquidDispatch=r.rotatingTrajectory&&!sand&&r.count<=1908&&pipelines.primal_cached_component?'cached':'auto',liquidSolve='barrier';
  let stageQuery=null,stageBuffer=null,stageReadback=null;const stageLabels=[];
  const profileOpen=(encoder,label)=>{const index=stageLabels.length*2;if(index>=640)throw new Error('Stage profiling exceeds its 64-event capacity');stageLabels.push(label);return encoder.beginComputePass({label:`VKF contact ${label}`,timestampWrites:{querySet:stageQuery,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});};
  return {
    resources:r,
    rotatePaused(encoder,{delta,capture}){
      if(!pipelines.paused_capture)throw new Error('Compiled frozen-World pose kernels are missing');
      u[2]=0;f[14]=delta;f[15]=0;f.set([0,0,0,0],20);device.queue.writeBuffer(r.uniform,0,params);
      const pass=encoder.beginComputePass({label:'VKF frozen World pose, validated before commit'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);
      const run=(name,count)=>{pass.setBindGroup(2,pausedGroup);pass.setPipeline(pipelines[name]);pass.dispatchWorkgroups(Math.ceil(count/128));};
      if(capture)run('paused_capture',r.count);
      run('paused_prepare',1);single(pass,'motion_dispatch_prepare');dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_static',r.count);dispatch(pass,'motion_sort',r.side*r.side);
      run('paused_propose',r.count);run('paused_validate',r.count);run('paused_commit',r.count);run('paused_finish',1);pass.end();
      started=performance.now();queryIndex=0;
    },
    setStageProfiling(enabled){if(started!==0||typeof enabled!=='boolean')throw new Error('Stage profiling is startup-only');if(!enabled)return;if(!timestamp)throw new Error('Stage profiling requires physical GPU timestamps');if(!stageQuery){stageQuery=device.createQuerySet({type:'timestamp',count:640});stageBuffer=device.createBuffer({size:5120,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});stageReadback=device.createBuffer({size:5120,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}},
    prepareForces(encoder){r.forceIndirect=!!pipelines.motion_force_dispatch;if(!r.forceIndirect)return;const pass=encoder.beginComputePass({label:'VKF force dispatch admission'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);single(pass,'motion_force_dispatch');pass.end();},
    prepareRepresentation(encoder){r.hasRepresentationClock=!!pipelines.motion_representation_clock;if(!r.hasRepresentationClock)return;const pass=encoder.beginComputePass({label:'VKF accepted representation interval'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);single(pass,'motion_representation_clock');pass.end();},
    synchronizeExternalStep(time){
      if(!Number.isFinite(time)||time<0)throw new RangeError('External contact clock must be finite and nonnegative');
      const value=new Float32Array([time]),one=new Uint32Array([1]),zero=new Uint32Array([0]);
      device.queue.writeBuffer(r.control,2*4,value);
      device.queue.writeBuffer(r.control,61*4,value);
      device.queue.writeBuffer(r.control,63*4,value);
      device.queue.writeBuffer(r.control,50*4,one);
      device.queue.writeBuffer(r.control,18*4,zero);
    },
    setSchedulerMode:mode=>policy.force(mode),
    setLiquidDispatchMode(mode){if(!['auto','global','component','cached','parallel-dual'].includes(mode)||(mode==='global'&&!pipelines.primal_begin)||(mode==='component'&&(r.count>4096||!pipelines.primal_component))||(mode==='cached'&&(r.count>1908||!pipelines.primal_cached_component))||(mode==='parallel-dual'&&!pipelines.primal_dual_global_begin))throw new Error('Liquid dispatch mode is outside the bounded component contract');liquidDispatch=mode;},
    setLiquidSolveMode(mode){if(started!==0||!['barrier','augmented','affine','projected','twist','adaptive','dual','mprgp','accelerated','blockpivot','tiled','star'].includes(mode)||mode!=='barrier'&&(sand&&mode!=='tiled'||r.count>(['augmented','affine','dual','mprgp','accelerated','blockpivot'].includes(mode)?1908:['tiled','star'].includes(mode)?Number.MAX_SAFE_INTEGER:4096)||!(['tiled','star'].includes(mode)?pipelines.primal_dual_seed&&pipelines.contact_tiled_warm&&pipelines.contact_tiled_save&&pipelines[mode==='star'?'contact_star_project_0':'contact_tiled_project_0']&&pipelines.motion_velocity_trial_begin:mode==='blockpivot'?pipelines.primal_dual_seed&&pipelines.primal_blockpivot_component:mode==='accelerated'?pipelines.primal_dual_seed&&pipelines.primal_accelerated_component:mode==='mprgp'?pipelines.primal_dual_seed&&pipelines.primal_mprgp_component:mode==='dual'?pipelines.primal_dual_component:mode==='affine'?pipelines.al_affine_component:mode==='augmented'?pipelines.al_component:['twist','adaptive'].includes(mode)?pipelines.motion_twist_horizon&& (mode!=='adaptive'||pipelines.primal_adaptive_limit):pipelines.contact_project_component)))throw new Error('Contact solve is a startup-only bounded experiment requiring its compiled binding');liquidSolve=mode;},
    eventBudget:()=>fixedBudget??eventBudget,
    setEventBudget(value){if(value!==null&&(!Number.isInteger(value)||value<1||value>64))throw new Error('Contact event budget must be 1–64 or null');fixedBudget=value;},
    beginFrame(encoder,{requestedDelta,windowDuration,timeLimit,paused}){moving=Math.abs(requestedDelta)>1e-7;selected=r.rotatingTrajectory?"parallel":policy.choose(moving);u[2]=boundaryOnly?2:selected==='ledger'?1:0;u[7]=liquidSolve==='barrier'?0:liquidSolve==='twist'?2:liquidSolve==='affine'?3:liquidSolve==='adaptive'?4:['dual','mprgp','accelerated','blockpivot','tiled','star'].includes(liquidSolve)?6:1;queryIndex=0;started=performance.now();f[14]=requestedDelta;f[15]=windowDuration;f.set([paused?windowDuration:world.time_step,timeLimit,paused?0:1,sand?(world.properties.wall_friction??.78):0],20);device.queue.writeBuffer(r.uniform,0,params);encoder.clearBuffer(r.control,48,4);encoder.clearBuffer(r.control,84,32);encoder.clearBuffer(r.control,116,8);const pass=encoder.beginComputePass({label:'VKF wheel input velocity'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);dispatch(pass,'motion_precision_reset',r.count);single(pass,'motion_precision_reset_end');single(pass,'motion_input_omega');pass.end();},
    advance(encoder,{events=8}={}){if(!Number.isInteger(events)||events<1||events>8||queryIndex>126)throw new Error('Contact submission exceeds its bounded event/query budget');if(queryIndex===0)stageLabels.length=0;if(timestamp){const stamp=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:queryIndex++}});stamp.end();}const open=(label='prepare')=>{const next=stageQuery?profileOpen(encoder,label):encoder.beginComputePass({label:'VKF preventive contact events'});next.setBindGroup(0,empty);next.setBindGroup(1,empty);next.setBindGroup(2,group);return next;};let pass=open();const changeStage=label=>{if(stageQuery){pass.end();pass=open(label);}};dispatch(pass,'motion_pause_velocity',r.count);single(pass,'motion_begin');
      for(let event=0;event<events;event++){
        if(pipelines.motion_velocity_trial_begin){dispatch(pass,'motion_materialize_state',r.count);single(pass,'motion_velocity_trial_begin');dispatch(pass,'motion_velocity_trial_seed',r.count);}
        single(pass,'motion_reset_speed');dispatch(pass,'motion_measure_speed',r.count);single(pass,'motion_prepare');
        if(liquidSolve==='barrier'||liquidSolve==='tiled'&&r.rotatingTrajectory){single(pass,'motion_frame_begin');dispatch(pass,r.rotatingTrajectory?'motion_enter_rotating_frame':'motion_enter_frame',r.count);}single(pass,'motion_dispatch_prepare');
        single(pass,'motion_reset_speed');dispatch(pass,'motion_measure_speed',r.count);
        if(!boundaryOnly){dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_static',r.count);dispatch(pass,'motion_sort',r.side*r.side);}
        single(pass,'motion_clearance_begin');dispatch(pass,'motion_clearance',r.count);single(pass,'motion_clearance_end');
        if(liquidSolve==='twist'||liquidSolve==='adaptive'){single(pass,'motion_twist_begin');dispatch(pass,'motion_twist_horizon',r.count);single(pass,'motion_twist_validate');}
        if(['affine','dual','mprgp','accelerated','blockpivot'].includes(liquidSolve))single(pass,'motion_twist_begin');
        dispatch(pass,'motion_materialize_state',r.count);
        if(boundaryOnly){
          changeStage('boundary');dispatch(pass,'motion_solve_boundaries',r.count);dispatch(pass,'motion_solve_boundaries',r.count);
        }else{
        changeStage('graph');
        single(pass,'dense_begin');dispatch(pass,'dense_build',r.count);dispatch(pass,'dense_links',r.count);
        changeStage('proposal');
        if(liquidSolve==='tiled'||liquidSolve==='star'){
          dispatch(pass,'primal_dual_seed',r.count);dispatch(pass,'contact_tiled_warm',r.count);
          if(liquidSolve==='star'){single(pass,'contact_star_reset');dispatch(pass,'contact_star_compact',r.side*r.side);}
          if(liquidSolve==='star'&&r.starNonlinearRim){dispatch(pass,'contact_star_rim_seed',r.count);dispatch(pass,'contact_star_rim_project',r.count);}
          const columns=Math.ceil(Math.ceil(r.side/8)/3);
          for(let sweep=0;sweep<(liquidSolve==='star'?(r.starSweeps??1):1);sweep++){
            for(let colour=0;colour<9;colour++){pass.setBindGroup(2,liquidSolve==='star'?starProjectGroup:group);pass.setPipeline(pipelines[`contact_${liquidSolve==='star'?'star':'tiled'}_project_${colour}`]);if(liquidSolve==='star')pass.dispatchWorkgroupsIndirect(starArgs,colour*12);else pass.dispatchWorkgroups(columns*columns);}
            if(liquidSolve==='star'&&r.starNonlinearRim)dispatch(pass,'contact_star_rim_project',r.count);
          }
          dispatch(pass,'contact_tiled_save',r.count);
        }
        else if(liquidSolve==='projected'){pass.setBindGroup(2,group);pass.setPipeline(pipelines.contact_project_component);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,24);}
        else if(liquidSolve==='blockpivot'||liquidSolve==='accelerated'||liquidSolve==='mprgp'){dispatch(pass,'primal_dual_seed',r.count);if(pipelines.primal_dual_sort_active)single(pass,'primal_dual_sort_active');pass.setBindGroup(2,group);pass.setPipeline(pipelines[liquidSolve==='blockpivot'?'primal_blockpivot_component':liquidSolve==='accelerated'?'primal_accelerated_component':'primal_mprgp_component']);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,24);}
        else if(liquidSolve==='dual'){
          if(pipelines.primal_dual_seed)dispatch(pass,'primal_dual_seed',r.count);
          if(pipelines.primal_dual_sort_active)single(pass,'primal_dual_sort_active');
          if(liquidDispatch==='parallel-dual'){
            for(let outer=0;outer<3;outer++){
              single(pass,'primal_dual_global_begin');dispatch(pass,'primal_dual_global_solution',r.count);dispatch(pass,'primal_dual_global_start',r.count);
              for(let cg=0;cg<64;cg++){dispatch(pass,'primal_dual_global_direction',r.count,true);dispatch(pass,'primal_dual_global_operator',r.count,true);single(pass,'primal_dual_global_alpha');dispatch(pass,'primal_dual_global_bound',r.count,true);dispatch(pass,'primal_dual_global_update',r.count,true);single(pass,'primal_dual_global_beta');dispatch(pass,'primal_dual_global_conjugate',r.count,true);}
              dispatch(pass,'primal_dual_global_solution',r.count);dispatch(pass,'primal_dual_global_refine',r.count);
            }
            dispatch(pass,'primal_dual_global_warm_end',r.count);
          }else{pass.setBindGroup(2,group);pass.setPipeline(pipelines.primal_dual_compact_component??pipelines.primal_dual_component);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,24);}
        }
        else if(liquidSolve==='augmented'||liquidSolve==='affine'){pass.setBindGroup(2,group);pass.setPipeline(pipelines[liquidSolve==='affine'?'al_affine_component':'al_component']);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,24);}
        else if((!sand&&r.count<=4096||r.sandBarrier&&r.count<=16384)&&liquidDispatch!=='global'&&(liquidDispatch==='cached'?pipelines.primal_cached_component:pipelines.primal_component)){pass.setBindGroup(2,group);pass.setPipeline(liquidDispatch==='cached'?pipelines.primal_cached_component:pipelines.primal_component);pass.dispatchWorkgroupsIndirect(r.dispatchArgs,24);}
        else if(!projectedSand){
          single(pass,'primal_begin');dispatch(pass,'primal_initialize',r.count);
          if(liquidSolve==='adaptive')dispatch(pass,'primal_adaptive_limit',r.count);
          for(let outer=0;outer<3;outer++){
            for(let newton=0;newton<(r.count<=4096?2:4);newton++){
              dispatch(pass,'primal_weights',r.count);dispatch(pass,'primal_gradient',r.count);
              single(pass,'primal_cg_begin');
              for(let cg=0;cg<(r.count<=4096?8:12);cg++){dispatch(pass,'primal_operator',r.count);single(pass,'primal_alpha');dispatch(pass,'primal_update',r.count);single(pass,'primal_beta');dispatch(pass,'primal_conjugate',r.count);}
              single(pass,'primal_line_begin');dispatch(pass,'primal_line_bound',r.count);
              for(let retry=0;retry<4;retry++){single(pass,'primal_trial_begin');dispatch(pass,'primal_trial',r.count);dispatch(pass,'primal_trial_validate',r.count);dispatch(pass,'primal_energy',r.count);single(pass,'primal_energy_validate');if(retry<3)single(pass,'primal_trial_reduce');}
              dispatch(pass,'primal_line_apply',r.count);
            }
            single(pass,'primal_cool');
          }
        }else for(let iteration=0;iteration<8;iteration++)for(let color=0;color<9;color++)dispatch(pass,`contact_project_${color}`,r.side*r.side);
        // Repair on the actual rounded post-friction velocity, without releasing
        // support or applying friction afterward. Finite sweeps are not a
        // certificate: the actual audit and continuous guard remain authoritative.
        if(projectedSand)for(let iteration=0;iteration<16;iteration++)for(let color=0;color<9;color++)dispatch(pass,`contact_repair_${color}`,r.side*r.side);
        if(!projectedSand&&!['projected','tiled','star'].includes(liquidSolve))dispatch(pass,'primal_finish',r.count);
        dispatch(pass,'contact_residual',r.count);
        }
        if(!r.rotatingTrajectory){dispatch(pass,'motion_exit_frame',r.count);single(pass,'motion_exit_frame_end');}
        single(pass,'motion_reset_speed');dispatch(pass,'motion_measure_speed',r.count);
        if(pipelines.motion_bound_proposal_horizon)single(pass,'motion_bound_proposal_horizon');
        changeStage('swept');
        if(selected==='ledger')dispatch(pass,'motion_ledger_initialize',r.count);
        if(!boundaryOnly){dispatch(pass,'motion_clear',r.side*r.side);dispatch(pass,'motion_fill_swept',r.count);dispatch(pass,'motion_sort',r.side*r.side);}dispatch(pass,'motion_find_toi',r.count);
        if(selected==='ledger'){single(pass,'motion_update_ledger');dispatch(pass,'motion_find_fallback',r.count);}
        changeStage('admission');
        for(let guard=0;guard<4;guard++){single(pass,'motion_prepare_guard');dispatch(pass,'motion_guard_end',r.count);}dispatch(pass,'motion_clear_dirty',r.count);single(pass,'motion_advance_control');dispatch(pass,'motion_advance_particles',r.count);
        if(selected==='ledger'){single(pass,'motion_consume_ledger');}
        single(pass,'motion_frame_end');
        if(pipelines.motion_velocity_trial_end)single(pass,'motion_velocity_trial_end');
        if(event+1<events)changeStage('prepare');
      }pass.end();if(timestamp){const stamp=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:queryIndex++}});stamp.end();}
    },
    finishFrame(encoder){const pass=encoder.beginComputePass({label:'VKF authoritative position low components'});pass.setBindGroup(0,empty);pass.setBindGroup(1,empty);dispatch(pass,'motion_export_precision',r.count);pass.end();encoder.copyBufferToBuffer(r.control,0,r.readback,0,256);if(timestamp&&queryIndex){encoder.resolveQuerySet(query,0,queryIndex,queryBuffer,0);encoder.copyBufferToBuffer(queryBuffer,0,queryReadback,0,queryIndex*8);}if(stageQuery&&stageLabels.length){encoder.resolveQuerySet(stageQuery,0,stageLabels.length*2,stageBuffer,0);encoder.copyBufferToBuffer(stageBuffer,0,stageReadback,0,stageLabels.length*16);}},
    async inspectStages(){if(!stageQuery||!stageLabels.length)return null;await stageReadback.mapAsync(GPUMapMode.READ);const q=new BigUint64Array(stageReadback.getMappedRange().slice(0));stageReadback.unmap();const stages={};for(let i=0;i<stageLabels.length;i++)stages[stageLabels[i]]=(stages[stageLabels[i]]??0)+Number(q[i*2+1]-q[i*2])/1e6;return stages;},
    async inspect(){await r.readback.mapAsync(GPUMapMode.READ);const bytes=r.readback.getMappedRange().slice(0);r.readback.unmap();const f=new Float32Array(bytes),u=new Uint32Array(bytes);if(u[6]||![f[0],f[1],f[2],f[12]].every(Number.isFinite))throw new Error(`Preventive contact state/broadphase failure ${u[6]}; no unchecked motion tail was advanced.`);let gpuMs=null;if(timestamp&&queryIndex){await queryReadback.mapAsync(GPUMapMode.READ);const q=new BigUint64Array(queryReadback.getMappedRange().slice(0));queryReadback.unmap();gpuMs=0;for(let i=0;i<queryIndex;i+=2)gpuMs+=Number(q[i+1]-q[i])/1e6;}const receipt={remainingTime:f[0],angle:f[1],time:f[2],angularDelta:f[12],scheduler:selected,gpuMs,completedMs:performance.now()-started,predictions:u[21],heapEvents:u[14],staleEvents:u[23],heapRebuilds:u[24],events:u[25],guardDeferrals:u[26],dirtyParticles:u[27]/Math.max(1,u[25]*r.count),fallbacks:u[30]};if(receipt.events>0){const cost=(gpuMs??receipt.completedMs)/receipt.events;if(cost>0)eventBudget=Math.max(4,Math.min(8,Math.floor(10/cost)));}policy.observe({...receipt,mode:selected,moving});return receipt;},
    reset(){r.reset();policy.reset();eventBudget=4;},destroy(){r.destroy();starCells?.destroy();starArgs?.destroy();query?.destroy();queryBuffer?.destroy();queryReadback?.destroy();stageQuery?.destroy();stageBuffer?.destroy();stageReadback?.destroy();},
  };
}
