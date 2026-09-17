// Experimental dense velocity projection; CCD remains the commit authority.
export const DENSE_CONTACT_WGSL = /* wgsl */`
override DUAL_PREDICTED_CONTACTS=false;
override DENSE_NONLINEAR_RIM=false;
fn dense_row(i:u32,k:u32)->u32{return i*128u+24u+k*3u;}
fn dense_add(i:u32,j:u32,n:vec2<f32>,gamma:f32,mu:f32,surface:vec2<f32>,feature:u32){
  let k=motion_actors[i*128u+4u].x;if(k>=32u){atomicOr(&motion_control[6],16u);return;}
  let own=motion_state(i);var other=surface;if(j!=0xffffffffu){other=motion_state(j).zw;}
  let a=dense_row(i,k);let t=vec2<f32>(-n.y,n.x);let v=own.zw-other;
  motion_actors[a]=vec4<u32>(j,bitcast<u32>(n.x),bitcast<u32>(n.y),bitcast<u32>(mu));
  motion_actors[a+1u]=bitcast<vec4<u32>>(vec4<f32>(0.0,0.0,0.0,gamma-dot(v,n)));
  let beta=gamma+select(0.0,dot(surface,n),j==0xffffffffu);
  motion_actors[a+2u]=vec4<u32>(bitcast<u32>(beta),feature,0u,bitcast<u32>(-dot(v,t)));
  motion_actors[i*128u+4u].x=k+1u;
}
@compute @workgroup_size(1) fn dense_begin(){atomicStore(&motion_control[33],0u);atomicStore(&motion_control[34],1u);}
@compute @workgroup_size(128) fn dense_build(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}
  let s=motion_state(i);motion_actors[i*128u+2u]=bitcast<vec4<u32>>(s);motion_actors[i*128u+4u]=vec4<u32>(0u);
  let h=max(motion_value(8),1.0e-8);let diameter=2.0*motion.minimum.z;let reach=2.0*motion_value(5)*h+8.0*motion.material.w;let cell=motion_cell(s.xy);
  for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let c=cell+vec2<i32>(x,y);if(!motion_valid(c)){continue;}let b=motion_bucket(c);let count=min(atomicLoad(&motion_cells[b]),motion.grid.z);
    for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[b*motion.grid.z+slot]);if(j<=i){continue;}let other=motion_state(j);let sep=motion_separation(i,j);let d=motion_length(sep);if(d>diameter+reach||d<1.0e-12){continue;}
      // Include nearby nonclosing rows too: an impulse from another contact
      // can turn a formerly separating pair into the next blocking impact.
      // This proposal graph still does not replace complete swept admission.
      let gap=motion_pair_gap_parts(motion_separation_parts(i,j));let gamma=(motion_solve_reserve()-gap)/h;
      // Proposal active-set policy only. Every near-touching pair is retained,
      // even when separating. Distant free-predictor-safe rows may be added on
      // subsequent events; the COMPLETE swept graph still owns admission.
      if(DUAL_PREDICTED_CONTACTS&&motion.grid.w==6u&&gap>8.0*motion.material.w&&dot(sep/d,s.zw-other.zw)>gamma+motion.material.w/h){continue;}
      let mu=select(0.0,motion.material.y,d<=diameter+motion.material.w*8.0);dense_add(i,j,sep/d,gamma,mu,vec2<f32>(0.0),0u);
    }
  }}
  let offset=s.xy-motion.wheel.xy;let r=motion_length(offset);let clearance=motion.minimum.z+WHEEL_BAR_HALF_WIDTH+motion_reserve();
  // Liquid uses the exact circular barrier, not an additional tangent-plane
  // barrier. Counting both produces artificial normal work at a wet rim.
  if(!DENSE_NONLINEAR_RIM&&(motion.material.y>0.0||motion.grid.w==6u)&&r>=WHEEL_RADIUS-clearance-reach){var n=-offset/max(r,1.0e-20);
    // Tangent velocity changes during the solve. Do not bake the FREE tangent
    // into a linear row that can exclude the safe stationary trajectory.
    var gamma=(motion_solve_reserve()-motion_rim_gap_parts(motion_offset_parts(motion_position_parts(i,0.0))))/h;
    if(motion.grid.w==6u){let endpoint=motion_offset_parts(motion_position_parts(i,h));n=-endpoint.xy/max(motion_length(endpoint.xy),1.0e-20);let current=motion_offset_parts(motion_position_parts(i,0.0));let reserve=motion_boundary_solve_reserve(motion_rim_gap_parts(current));gamma=-motion_plane_gap_parts(current,n,WHEEL_RADIUS-motion.minimum.z-WHEEL_BAR_HALF_WIDTH,motion_rim_error()+reserve)/h;}
    dense_add(i,0xffffffffu,n,gamma,motion.material.z,vec2<f32>(0.0),1u);}
  let angle=motion_value(1);let local=motion_rotate(offset,-angle);let omega=motion_value(3);
  for(var k=0u;k<motion.counts.w;k++){let bar=baffle(k);let edge=bar.zw-bar.xy;let closest=bar.xy+edge*clamp(dot(local-bar.xy,edge)/max(dot(edge,edge),1.0e-20),0.0,1.0);let sep=local-closest;let d=motion_length(sep);let reach_bar=(motion_value(5)+abs(omega)*max(motion_length(bar.xy),motion_length(bar.zw)))*h+8.0*motion.material.w;
    if(d>clearance+reach_bar||d<1.0e-12){continue;}let n=motion_rotate(sep/d,angle);let a=motion_rotate(bar.xy,angle);let b=motion_rotate(bar.zw,angle);let point=motion_rotate(closest,angle);let surface=omega*vec2<f32>(-point.y,point.x);
    if(motion.grid.w==6u){
      // Signed normal curvature, not a global |q''| penalty. In the baffle
      // frame q=R(-omega*t)(x+v*t), q''(0)=-2 omega Jv-omega^2 x.
      // |q'''| <= B. Bound its cubic remainder by B*h*t^2/6, then
      // require the two nonconstant Bernstein coefficients of that quadratic
      // lower bound to be nonnegative. Both rows are affine in the unknown v.
      let raw_gap=motion_bar_gap_parts(motion_offset_parts(motion_position_parts(i,0.0)),angle,bar);let gap=raw_gap-motion_boundary_solve_reserve(raw_gap);
      let twist=omega*vec2<f32>(-offset.y,offset.x);let initial=dot(n,twist);
      let speed=max(motion_value(5),abs(omega)*WHEEL_RADIUS);let third=3.0*omega*omega*speed+abs(omega*omega*omega)*(WHEEL_RADIUS+speed*h);
      let curved=n+omega*h*vec2<f32>(-n.y,n.x);
      dense_add(i,0xffffffffu,n,initial-2.0*gap/h,0.0,vec2<f32>(0.0),2u+k);
      dense_add(i,0xffffffffu,curved,initial+0.5*omega*omega*h*dot(n,offset)-gap/h+third*h*h/6.0,0.0,vec2<f32>(0.0),9u+k);
      continue;
    }
    let va=dot(omega*vec2<f32>(-a.y,a.x),n)+.5*omega*omega*motion_length(a)*h-(dot(offset-a,n)-clearance-motion_bar_error(bar))/h;
    let vb=dot(omega*vec2<f32>(-b.y,b.x),n)+.5*omega*omega*motion_length(b)*h-(dot(offset-b,n)-clearance-motion_bar_error(bar))/h;
    var gamma=select(max(va,vb)-dot(surface,n),(motion_solve_reserve()-motion_bar_gap_parts(motion_offset_parts(motion_position_parts(i,0.0)),angle,bar))/h,omega==0.0);
    if(motion.grid.w==2u||motion.grid.w==3u||motion.grid.w==4u){let twist=omega*vec2<f32>(-offset.y,offset.x);gamma=dot(twist-surface,n)+0.5*motion_value(53)*h+(motion_solve_reserve()-motion_bar_gap_parts(motion_offset_parts(motion_position_parts(i,0.0)),angle,bar))/h;}
    dense_add(i,0xffffffffu,n,gamma,motion.schedule.w,surface,2u+k);
  }
}
fn dense_incident(i:u32,number:u32,row:u32){if(number>=64u){atomicOr(&motion_control[6],32u);return;}motion_actors[i*128u+8u+number/4u][number%4u]=row;}
@compute @workgroup_size(128) fn dense_links(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}var total=0u;let own_count=motion_actors[i*128u+4u].x;
  for(var k=0u;k<own_count;k++){dense_incident(i,total,i*32u+k);total++;}
  let cell=motion_cell(motion_state(i).xy);
  for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let c=cell+vec2<i32>(x,y);if(!motion_valid(c)){continue;}let b=motion_bucket(c);let count=min(atomicLoad(&motion_cells[b]),motion.grid.z);
    for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[b*motion.grid.z+slot]);if(j>=i){continue;}let rows=motion_actors[j*128u+4u].x;for(var k=0u;k<rows;k++){if(motion_actors[dense_row(j,k)].x==i){dense_incident(i,total,j*32u+k);total++;}}}
  }}
  motion_actors[i*128u+4u].y=total;atomicMax(&motion_control[34],total);
}

@compute @workgroup_size(1) fn dense_direction(){atomicStore(&motion_control[33],0u);}
@compute @workgroup_size(1) fn dense_solution(){atomicStore(&motion_control[33],1u);}
@compute @workgroup_size(1) fn dense_full_solution(){atomicStore(&motion_control[33],2u);}
@compute @workgroup_size(128) fn dense_gather(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}var dv=vec2<f32>(0.0);let count=motion_actors[i*128u+4u].y;let mode=atomicLoad(&motion_control[33]);
 for(var k=0u;k<count;k++){let id=motion_actors[i*128u+8u+k/4u][k%4u];let owner=id/32u;let a=dense_row(owner,id%32u);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let t=vec2<f32>(-n.y,n.x);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);let friction=bitcast<vec4<f32>>(motion_actors[a+2u]);var impulse=n*select(p.y,p.x,mode!=0u&&mode!=3u);if(mode==2u||mode==3u){impulse+=t*friction.z;}dv+=select(-1.0,1.0,owner==i)*impulse;}
 motion_actors[i*128u+3u]=bitcast<vec4<u32>>(vec4<f32>(dv,0.0,0.0));
}
fn dense_relative(i:u32,row:vec4<u32>)->vec2<f32>{var dv=bitcast<vec4<f32>>(motion_actors[i*128u+3u]).xy;if(row.x!=0xffffffffu){dv-=bitcast<vec4<f32>>(motion_actors[row.x*128u+3u]).xy;}return dv;}
@compute @workgroup_size(128) fn dense_start(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);let other=bitcast<vec4<f32>>(motion_actors[a+2u]);let residual=p.w-dot(n,dense_relative(i,row));let diag=select(2.0,1.0,row.x==0xffffffffu);let z=select(0.0,residual/diag,p.x>0.0||residual>0.0);motion_actors[a+1u]=bitcast<vec4<u32>>(vec4<f32>(p.x,z,residual,p.w));motion_actors[a+2u]=bitcast<vec4<u32>>(vec4<f32>(z,0.0,other.z,other.w));}
}
var<workgroup> dense_sums:array<vec2<f32>,128>;
@compute @workgroup_size(128) fn dense_dot(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 let i=gid.x;var sum=vec2<f32>(0.0);if(i<motion.counts.x&&motion_value(0)>0.0){let count=motion_actors[i*128u+4u].x;for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);var q=bitcast<vec4<f32>>(motion_actors[a+2u]);let ap=dot(n,dense_relative(i,row))+p.y*1.0e-8;q.y=ap;motion_actors[a+2u]=bitcast<vec4<u32>>(q);sum+=vec2<f32>(p.z*q.x,p.y*ap);}}
 dense_sums[lane]=sum;workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){dense_sums[lane]+=dense_sums[lane+width];}workgroupBarrier();}if(lane==0u){motion_actors[group.x*128u+6u]=bitcast<vec4<u32>>(vec4<f32>(dense_sums[0],0.0,0.0));}
}
fn dense_totals()->vec2<f32>{var total=vec2<f32>(0.0);for(var k=0u;k<(motion.counts.x+127u)/128u;k++){total+=bitcast<vec4<f32>>(motion_actors[k*128u+6u]).xy;}return total;}
@compute @workgroup_size(1) fn dense_start_control(){atomicStore(&motion_dispatch_args[3],atomicLoad(&motion_dispatch_args[0]));}
@compute @workgroup_size(1) fn dense_alpha(){let total=dense_totals();let active_work=total.x>1.0e-12&&total.y>1.0e-24;motion_store(36,select(0.0,max(0.0,total.x)/max(total.y,1.0e-30),active_work));motion_store(38,total.x);motion_store(40,1.0);atomicStore(&motion_control[35],0u);if(!active_work){atomicStore(&motion_dispatch_args[3],0u);}}
@compute @workgroup_size(128) fn dense_bounds(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;let alpha=motion_value(36);for(var k=0u;k<count;k++){let a=dense_row(i,k);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);if(p.y<0.0&&p.x+alpha*p.y<0.0){let fraction=clamp(p.x/max(-alpha*p.y,1.0e-30),0.0,1.0);atomicMin(&motion_control[40],bitcast<u32>(fraction));}}}
@compute @workgroup_size(128) fn dense_update(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;let alpha=motion_value(36)*motion_value(40);
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=motion_actors[a];let p=bitcast<vec4<f32>>(motion_actors[a+1u]);var q=bitcast<vec4<f32>>(motion_actors[a+2u]);var lambda=max(0.0,p.x+alpha*p.y);if(lambda<1.0e-10){lambda=0.0;}let residual=p.z-alpha*q.y;let was_active=p.x>0.0||p.z>0.0;let active_now=lambda>0.0||residual>0.0;if(was_active!=active_now){atomicStore(&motion_control[35],1u);}let diag=select(2.0,1.0,row.x==0xffffffffu);q.x=select(0.0,residual/diag,active_now);motion_actors[a+1u]=bitcast<vec4<u32>>(vec4<f32>(lambda,p.y,residual,p.w));motion_actors[a+2u]=bitcast<vec4<u32>>(q);}
}
@compute @workgroup_size(1) fn dense_beta(){let total=dense_totals();let beta=select(max(0.0,total.x)/max(motion_value(38),1.0e-30),0.0,atomicLoad(&motion_control[35])!=0u||motion_value(40)<.99999);motion_store(37,min(beta,2.0));}
@compute @workgroup_size(128) fn dense_conjugate(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;let beta=motion_value(37);for(var k=0u;k<count;k++){let a=dense_row(i,k);var p=bitcast<vec4<f32>>(motion_actors[a+1u]);let q=bitcast<vec4<f32>>(motion_actors[a+2u]);p.y=select(0.0,q.x+beta*p.y,p.x>0.0||p.z>0.0);motion_actors[a+1u]=bitcast<vec4<u32>>(p);}}
@compute @workgroup_size(128) fn dense_friction(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;let step=.48/f32(max(1u,atomicLoad(&motion_control[34])));
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let t=vec2<f32>(-n.y,n.x);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);var q=bitcast<vec4<f32>>(motion_actors[a+2u]);let limit=bitcast<f32>(row.w)*p.x;q.z=clamp(q.z+step*(q.w-dot(t,dense_relative(i,row))),-limit,limit);motion_actors[a+2u]=bitcast<vec4<u32>>(q);}
}
@compute @workgroup_size(128) fn dense_finish(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]);let dv=bitcast<vec4<f32>>(motion_actors[i*128u+3u]).xy;motion_write(i,vec4<f32>(free.xy,free.zw+dv));}
// Project the accumulated multipliers, not just each newly closing velocity.
// Releasing an earlier impulse is essential when neighboring constraints change.
fn contact_project_cell(bucket:u32,color:u32){
 if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}
 let cell=vec2<u32>(bucket%motion.grid.x,bucket/motion.grid.x);if(cell.x%3u+3u*(cell.y%3u)!=color){return;}
 let members=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);
 for(var slot=0u;slot<members;slot++){let i=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);let rows=motion_actors[i*128u+4u].x;let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]);
  for(var k=0u;k<rows;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let t=vec2<f32>(-n.y,n.x);var p=bitcast<vec4<f32>>(motion_actors[a+1u]);var q=bitcast<vec4<f32>>(motion_actors[a+2u]);let own=motion_state(i);var delta_v=own.zw-free.zw;var other=vec4<f32>(0.0);if(row.x!=0xffffffffu){other=motion_state(row.x);delta_v-=other.zw-bitcast<vec4<f32>>(motion_actors[row.x*128u+2u]).zw;}
   let diagonal=select(2.0,1.0,row.x==0xffffffffu);let normal_lambda=max(0.0,p.x+(p.w-dot(n,delta_v))/diagonal);let dn=normal_lambda-p.x;let bound=bitcast<f32>(row.w)*normal_lambda;let tangent_lambda=clamp(q.z+(q.w-dot(t,delta_v))/diagonal,-bound,bound);let impulse=n*dn+t*(tangent_lambda-q.z);p.x=normal_lambda;q.z=tangent_lambda;motion_actors[a+1u]=bitcast<vec4<u32>>(p);motion_actors[a+2u]=bitcast<vec4<u32>>(q);
   motion_write(i,vec4<f32>(own.xy,own.zw+impulse));if(row.x!=0xffffffffu){motion_write(row.x,vec4<f32>(other.xy,other.zw-impulse));}
  }
 }
}
@compute @workgroup_size(128) fn contact_project_0(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,0u);}
@compute @workgroup_size(128) fn contact_project_1(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,1u);}
@compute @workgroup_size(128) fn contact_project_2(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,2u);}
@compute @workgroup_size(128) fn contact_project_3(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,3u);}
@compute @workgroup_size(128) fn contact_project_4(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,4u);}
@compute @workgroup_size(128) fn contact_project_5(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,5u);}
@compute @workgroup_size(128) fn contact_project_6(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,6u);}
@compute @workgroup_size(128) fn contact_project_7(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,7u);}
@compute @workgroup_size(128) fn contact_project_8(@builtin(global_invocation_id) gid:vec3<u32>){contact_project_cell(gid.x,8u);}
@compute @workgroup_size(128) fn contact_residual(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let own=motion_state(i);let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]);let rows=motion_actors[i*128u+4u].x;
 for(var k=0u;k<rows;k++){let a=dense_row(i,k);let row=motion_actors[a];let p=bitcast<vec4<f32>>(motion_actors[a+1u]);let n=bitcast<vec2<f32>>(row.yz);var dv=own.zw-free.zw;if(row.x!=0xffffffffu){dv-=motion_state(row.x).zw-bitcast<vec4<f32>>(motion_actors[row.x*128u+2u]).zw;}let residual=max(0.0,p.w-dot(n,dv));atomicMax(&motion_control[46],bitcast<u32>(residual));}
}
// Feasibility repair acts on the final rounded velocities, never on positions.
// It only adds repulsive normal impulse; do not apply friction after this pass.
fn contact_repair_cell(bucket:u32,color:u32){
 if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)==0.0){return;}
 let cell=vec2<u32>(bucket%motion.grid.x,bucket/motion.grid.x);if(cell.x%3u+(cell.y%3u)*3u!=color){return;}
 let members=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);
 for(var slot=0u;slot<members;slot++){let i=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]);
  for(var k=0u;k<motion_actors[i*128u+4u].x;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);var p=bitcast<vec4<f32>>(motion_actors[a+1u]);let own=motion_state(i);var delta=own.zw-free.zw;var other=vec4<f32>(0.0);if(row.x!=0xffffffffu){other=motion_state(row.x);delta-=other.zw-bitcast<vec4<f32>>(motion_actors[row.x*128u+2u]).zw;}
   let impulse=max(0.0,p.w-dot(n,delta))/(select(2.0,1.0,row.x==0xffffffffu)*dot(n,n));motion_write(i,vec4<f32>(own.xy,own.zw+n*impulse));if(row.x!=0xffffffffu){motion_write(row.x,vec4<f32>(other.xy,other.zw-n*impulse));}p.x+=impulse;p.y=p.x;motion_actors[a+1u]=bitcast<vec4<u32>>(p);
  }
  // The circular constraint is nonlinear: neighboring contact/friction changes
  // its tangent speed. Recompute the endpoint envelope on the ACTUAL velocity,
  // not the free velocity frozen when the row was built. Only velocity changes.
  let own=motion_state(i);let offset_parts=motion_offset_parts(motion_position_parts(i,0.0));let offset=offset_parts.xy+offset_parts.zw;let r=motion_length(offset);let buffer=max(motion_reserve()*1.125,motion.material.w*2.0);let gap=motion_rim_gap_parts(offset_parts)-buffer;let limit=r+gap;let h=max(motion_value(8),1.0e-12);
  if(motion_rim_gap_parts(motion_offset_parts(motion_position_parts(i,h)))<buffer){let n=-offset/max(r,1.0e-20);let vn=dot(own.zw,n);let vt=own.zw-n*vn;let t2=dot(vt,vt)*h*h;
   if(t2<limit*limit){let gamma=(-gap*(2.0*r+gap)+t2)/(h*max(r+motion_sqrt(max(0.0,limit*limit-t2)),1.0e-20));motion_write(i,vec4<f32>(own.xy,vt+n*max(vn,gamma)));}
   else{motion_write(i,vec4<f32>(own.xy,vec2<f32>(0.0)));}
  }
 }
}
@compute @workgroup_size(128) fn contact_repair_0(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,0u);}
@compute @workgroup_size(128) fn contact_repair_1(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,1u);}
@compute @workgroup_size(128) fn contact_repair_2(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,2u);}
@compute @workgroup_size(128) fn contact_repair_3(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,3u);}
@compute @workgroup_size(128) fn contact_repair_4(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,4u);}
@compute @workgroup_size(128) fn contact_repair_5(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,5u);}
@compute @workgroup_size(128) fn contact_repair_6(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,6u);}
@compute @workgroup_size(128) fn contact_repair_7(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,7u);}
@compute @workgroup_size(128) fn contact_repair_8(@builtin(global_invocation_id) gid:vec3<u32>){contact_repair_cell(gid.x,8u);}
fn contact_color_bucket(index:u32,color:u32)->u32{
 let cx=color%3u;let cy=color/3u;let columns=(motion.grid.x+2u-cx)/3u;
 let x=cx+3u*(index%columns);let y=cy+3u*(index/columns);
 return x+motion.grid.x*y;
}
// Single bounded component: each color has disjoint one-cell write
// neighborhoods. Synchronize between colors; never race two impulses on a
// particle. The independent swept and endpoint guards still decide admission.
@compute @workgroup_size(256) fn contact_project_component(@builtin(local_invocation_index) lane:u32){
 for(var iteration=0u;iteration<8u;iteration++){
  for(var color=0u;color<9u;color++){
   let cells=((motion.grid.x+2u-color%3u)/3u)*((motion.grid.y+2u-color/3u)/3u);
   for(var k=lane;k<cells;k+=256u){contact_project_cell(contact_color_bucket(k,color),color);}storageBarrier();
  }
 }
 for(var iteration=0u;iteration<16u;iteration++){
  for(var color=0u;color<9u;color++){
   let cells=((motion.grid.x+2u-color%3u)/3u)*((motion.grid.y+2u-color/3u)/3u);
   for(var k=lane;k<cells;k+=256u){contact_repair_cell(contact_color_bucket(k,color),color);}storageBarrier();
  }
 }
}
@compute @workgroup_size(1) fn dense_acceleration_begin(){motion_store(36,1.0);motion_store(37,0.0);atomicStore(&motion_control[33],3u);}
@compute @workgroup_size(1) fn dense_acceleration_step(){let t=motion_value(36);let next=(1.0+sqrt(1.0+4.0*t*t))*.5;motion_store(37,(t-1.0)/next);motion_store(36,next);}
@compute @workgroup_size(128) fn dense_projected_gradient(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=motion_actors[i*128u+4u].x;let degree=f32(max(1u,atomicLoad(&motion_control[34])));
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=motion_actors[a];let n=bitcast<vec2<f32>>(row.yz);let p=bitcast<vec4<f32>>(motion_actors[a+1u]);let residual=p.w-dot(n,dense_relative(i,row));let diagonal=select(2.0,1.0,row.x==0xffffffffu);let lambda=max(0.0,p.y+residual/(diagonal*degree));motion_actors[a+1u]=bitcast<vec4<u32>>(vec4<f32>(lambda,lambda+motion_value(37)*(lambda-p.x),residual,p.w));}
}
`;
