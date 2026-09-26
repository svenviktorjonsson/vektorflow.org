// Feasible primal velocity iterates. Continuous guards still own admission.
export const CONTACT_PRIMAL_BARRIER_WGSL=/* wgsl */`
// The block experiment regressed motion and p95 on the physical fixture.
override PRIMAL_BLOCK_PRECONDITIONER=false;
// Bounded granular candidate. Off unless explicitly selected by the harness.
override PRIMAL_COULOMB_DISSIPATION=false;
@group(2) @binding(10) var<storage,read_write> primal_scratch:array<vec4<u32>>;
// Field-major working arrays coalesce adjacent lanes. Iterative proposals do
// not mutate published particle velocities or the padded actor records.
fn primal_slot(i:u32,k:u32)->u32{return k*motion.counts.x+i;}
fn primal_data(i:u32,k:u32)->vec4<f32>{return bitcast<vec4<f32>>(primal_scratch[primal_slot(i,k)]);}
fn primal_put(i:u32,k:u32,v:vec4<f32>){primal_scratch[primal_slot(i,k)]=bitcast<vec4<u32>>(v);}
fn primal_row(i:u32,k:u32)->vec4<u32>{return primal_scratch[primal_slot(i,9u+2u*k)];}
fn primal_weight(i:u32,k:u32)->f32{return primal_data(i,10u+2u*k).x;}
fn primal_friction(i:u32,k:u32)->vec3<f32>{return primal_data(i,10u+2u*k).yzw;}
fn primal_tangent_speed(owner:u32,row:vec4<u32>,a:u32,trial:bool)->f32{
 let slot=select(5u,4u,trial);var relative=primal_data(owner,slot).xy;
 if(row.x!=0xffffffffu){relative-=primal_data(row.x,slot).xy;}
 let n=bitcast<vec2<f32>>(row.yz);let t=vec2<f32>(-n.y,n.x);
 var surface=0.0;if(row.x==0xffffffffu){surface=dot(t,primal_free(owner))+bitcast<f32>(motion_actors[a+2u].w);}
 return dot(t,relative)-surface;
}
fn primal_huber(speed:f32)->vec3<f32>{
 let epsilon=0.00001;let absolute=abs(speed);
 if(absolute<epsilon){return vec3<f32>(0.5*speed*speed/epsilon,speed/epsilon,1.0/epsilon);}
 return vec3<f32>(absolute-0.5*epsilon,sign(speed),0.0);
}
fn primal_counts(i:u32)->vec2<u32>{return primal_scratch[primal_slot(i,89u)].xy;}
fn primal_incident(i:u32,k:u32)->u32{return primal_scratch[primal_slot(i,73u+k/4u)][k%4u];}
fn primal_block_inverse_positive(diagonal:vec2<f32>,off:f32,determinant:f32)->vec4<f32>{
 let scale=max(1.0,max(diagonal.x,diagonal.y));let delta=vec2<f32>((diagonal.x-diagonal.y)/scale,2.0*off/scale);let length_delta=length(delta);
 let direction=select(vec2<f32>(1.0,0.0),delta/max(length_delta,1.0e-30),length_delta>0.0);let mean=0.5*(diagonal.x/scale+diagonal.y/scale);let radius=0.5*length_delta;
 // The identity mass term guarantees eigenvalues >=1. Preserve it when
 // subtracting almost equal large eigenvalues loses that low-order term.
 let large=max(1.0,(mean+radius)*scale);let small=max(1.0,determinant/large);
 var basis=vec2<f32>(1.0,0.0);
 if(direction.x>=0.0){basis.x=sqrt(0.5*(1.0+direction.x));basis.y=direction.y/max(2.0*basis.x,1.0e-30);}
 else{basis.y=select(-1.0,1.0,direction.y>=0.0)*sqrt(0.5*(1.0-direction.x));basis.x=direction.y/(2.0*basis.y);}
 return vec4<f32>(normalize(basis),1.0/large,1.0/small);
}
fn primal_block_inverse(diagonal:vec2<f32>,off:f32)->vec4<f32>{return primal_block_inverse_positive(diagonal,off,max(1.0,fma(diagonal.x,diagonal.y,-off*off)));}
fn primal_block_apply(inverse:vec4<f32>,r:vec2<f32>)->vec2<f32>{let axis=inverse.xy;let tangent=vec2<f32>(-axis.y,axis.x);return axis*(dot(axis,r)*inverse.z)+tangent*(dot(tangent,r)*inverse.w);}
fn primal_rank(i:u32,k:u32)->vec3<f32>{let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);return vec3<f32>(bitcast<vec2<f32>>(row.yz),primal_weight(owner,id%32u));}
fn primal_cross(a:vec2<f32>,b:vec2<f32>)->f32{let p=a.x*b.y;let q=a.y*b.x;let high=p-q;return high+((p-high)-q)+(motion_product_error1(a.x,b.y,p)-motion_product_error1(a.y,b.x,q));}
fn primal_positive_determinant(i:u32,isotropic:f32,rim_outer:f32,endpoint:vec2<f32>)->f32{
 // det(a I + sum w n n^T) as a sum of nonnegative terms. Subtracting
 // Cxx*Cyy-Cxy*Cxy loses the mass/tangential mode at stiff wall contacts.
 var determinant=isotropic*(isotropic+rim_outer*dot(endpoint,endpoint));let count=primal_counts(i).y;
 for(var k=0u;k<count;k++){let rank=primal_rank(i,k);let c=primal_cross(rank.xy,endpoint);determinant+=rank.z*(isotropic*dot(rank.xy,rank.xy)+rim_outer*c*c);
  for(var j=0u;j<k;j++){let other=primal_rank(i,j);let cross=primal_cross(rank.xy,other.xy);determinant+=rank.z*other.z*cross*cross;}
 }
 return determinant;
}
fn primal_free(i:u32)->vec2<f32>{return primal_data(i,6u).xy;}
fn primal_velocity(i:u32)->vec2<f32>{return primal_data(i,5u).xy;}
fn primal_required(owner:u32,row:vec4<u32>,a:u32)->f32{
 // Keep the small gap speed independently of the order-one free velocity.
 // Reconstructing it as (gamma-dot(v,n))+dot(v,n) cancels dense clearances.
 return bitcast<f32>(row.w);
}
fn primal_project_slack(n:vec2<f32>,own:vec2<f32>,other:vec2<f32>,required:f32)->f32{
 let relative=motion_two_sum(own,-other);let products=n*relative.xy;
 let sum=motion_two_sum(vec2<f32>(products.x),vec2<f32>(products.y));let difference=motion_two_sum(sum.xy,vec2<f32>(-required));
 return difference.x+(difference.z+sum.z+dot(n,relative.zw)+motion_product_error1(n.x,relative.x,products.x)+motion_product_error1(n.y,relative.y,products.y));
}
fn primal_slack(owner:u32,row:vec4<u32>,a:u32)->f32{
 var other=vec2<f32>(0.0);if(row.x!=0xffffffffu){other=primal_velocity(row.x);}
 return primal_project_slack(bitcast<vec2<f32>>(row.yz),primal_velocity(owner),other,primal_required(owner,row,a));
}
fn primal_endpoint(i:u32)->vec4<f32>{let p=primal_data(i,7u);return motion_offset_parts(motion_integrated(p.xy,p.zw,primal_velocity(i),motion_value(8)));}
fn primal_rim_slack(i:u32)->f32{
 let endpoint=primal_endpoint(i);let r=motion_length(endpoint.xy);let gap=motion_rim_gap_parts(endpoint)-motion_solve_reserve();return gap*(2.0*r+gap);
}
@compute @workgroup_size(1) fn primal_begin(){motion_store(47,0.0001);}
@compute @workgroup_size(1) fn primal_cool(){motion_store(47,max(select(select(0.000001,0.0001,PRIMAL_COULOMB_DISSIPATION||CONTACT_PERSISTENT_SKIN),1.0e-30,motion.grid.w==4u),motion_value(47)*0.1));}
fn primal_initialize_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}var initial=vec2<f32>(0.0);if(motion.grid.w==2u||motion.grid.w==4u){let offset=motion_offset_parts(motion_position_parts(i,0.0));initial=motion_value(3)*vec2<f32>(-offset.y,offset.x);}primal_put(i,5u,vec4<f32>(initial,0.0,0.0));primal_put(i,6u,vec4<f32>(bitcast<vec4<f32>>(motion_actors[i*128u+2u]).zw,0.0,0.0));primal_put(i,7u,motion_position_parts(i,0.0));
 let counts=motion_actors[i*128u+4u].xy;primal_scratch[primal_slot(i,89u)]=vec4<u32>(counts,0u,0u);
 for(var k=0u;k<counts.x;k++){let a=dense_row(i,k);let row=motion_actors[a];primal_scratch[primal_slot(i,9u+2u*k)]=vec4<u32>(row.xyz,motion_actors[a+2u].x);}
 for(var k=0u;k<(counts.y+3u)/4u;k++){primal_scratch[primal_slot(i,73u+k)]=motion_actors[i*128u+8u+k];}
}
fn primal_adaptive_limit_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}var mu=0.0001;
 // Start on the feasible central path without introducing a large repulsive
 // impulse merely because the numerical clearance is small. The mass term is
 // identity here; bound the INITIAL barrier diagonal by one percent of mass.
 for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let slack=primal_slack(i,row,dense_row(i,k));if(slack<=0.0){atomicOr(&motion_control[6],1024u);return;}mu=min(mu,0.01*slack*slack/64.0);}
 let endpoint=primal_endpoint(i);let s=primal_rim_slack(i);let h=motion_value(8);if(s<=0.0){atomicOr(&motion_control[6],2048u);return;}mu=min(mu,0.01*s*s/max(h*h*(2.0*s+4.0*dot(endpoint.xy,endpoint.xy)),1.0e-30));
 atomicMin(&motion_control[47],bitcast<u32>(mu));
}
@compute @workgroup_size(128) fn primal_adaptive_limit(@builtin(global_invocation_id) gid:vec3<u32>){primal_adaptive_limit_one(gid.x);}
fn primal_weights_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}let count=primal_counts(i).x;let mu=motion_value(47);
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=primal_row(i,k);let slack=primal_slack(i,row,a);if(slack<=0.0){atomicOr(&motion_control[6],1024u);return;}
  var friction=vec3<f32>(0.0);
  if(PRIMAL_COULOMB_DISSIPATION){let capacity=max(0.0,bitcast<f32>(motion_actors[a].w))*mu/slack;let huber=primal_huber(primal_tangent_speed(i,row,a,false));friction=vec3<f32>(capacity,capacity*huber.z,capacity*huber.y);}
  primal_put(i,10u+2u*k,vec4<f32>(mu/(slack*slack),friction));
 }
}
fn primal_gradient_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}let mu=motion_value(47);let h=motion_value(8);var gradient=primal_velocity(i)-primal_free(i);var diagonal=vec2<f32>(1.0);var off=0.0;
 let count=primal_counts(i).y;
 for(var k=0u;k<count;k++){
  let id=primal_incident(i,k);let owner=id/32u;let a=dense_row(owner,id%32u);let row=primal_row(owner,id%32u);let n=bitcast<vec2<f32>>(row.yz);let orientation=select(-1.0,1.0,owner==i);let slack=primal_slack(owner,row,a);
  if(slack<=0.0){atomicOr(&motion_control[6],1024u);return;}
  let weight=primal_weight(owner,id%32u);gradient-=n*(orientation*mu/slack);diagonal+=n*n*weight;off+=n.x*n.y*weight;
  if(PRIMAL_COULOMB_DISSIPATION){let t=vec2<f32>(-n.y,n.x);let friction=primal_friction(owner,id%32u);gradient+=t*(orientation*friction.z);diagonal+=t*t*friction.y;off+=t.x*t.y*friction.y;}
 }
 let endpoint=primal_endpoint(i);let s=primal_rim_slack(i);if(s<=0.0){primal_put(i,90u,vec4<f32>(s,h,motion_rim_gap_parts(endpoint),-2048.0));atomicOr(&motion_control[6],2048u);return;}
 let rim_identity=2.0*mu*h*h/s;let rim_outer=4.0*mu*h*h/(s*s);
 let rim=vec3<f32>(rim_identity+rim_outer*endpoint.x*endpoint.x,rim_outer*endpoint.x*endpoint.y,rim_identity+rim_outer*endpoint.y*endpoint.y);
 primal_put(i,8u,vec4<f32>(rim,0.0));
 gradient+=2.0*mu*h*endpoint.xy/s;diagonal+=rim.xz;off+=rim.y;
 var determinant=1.0;if(PRIMAL_BLOCK_PRECONDITIONER){determinant=primal_positive_determinant(i,1.0+rim_identity,rim_outer,endpoint.xy);}let inverse=primal_block_inverse_positive(diagonal,off,determinant);primal_put(i,90u,inverse);
 let r=-gradient;var z=r/diagonal;if(PRIMAL_BLOCK_PRECONDITIONER){z=primal_block_apply(inverse,r);}
 primal_put(i,0u,vec4<f32>(gradient,diagonal));primal_put(i,1u,vec4<f32>(z,r));primal_put(i,2u,vec4<f32>(z,0.0,0.0));
}
var<workgroup> primal_sums:array<vec2<f32>,256>;
fn primal_operator_one(i:u32)->vec2<f32>{
 var sums=vec2<f32>(0.0);
 if(i<motion.counts.x&&motion_value(0)>0.0&&atomicLoad(&motion_control[6])==0u){
  let mu=motion_value(47);let h=motion_value(8);let p=primal_data(i,1u).xy;var ap=p;let count=primal_counts(i).y;
  for(var k=0u;k<count;k++){
   let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let n=bitcast<vec2<f32>>(row.yz);var other=vec2<f32>(0.0);let j=select(owner,row.x,owner==i);if(j!=0xffffffffu){other=primal_data(j,1u).xy;}
   ap+=n*primal_weight(owner,id%32u)*dot(n,p-other);
   if(PRIMAL_COULOMB_DISSIPATION){let t=vec2<f32>(-n.y,n.x);ap+=t*primal_friction(owner,id%32u).y*dot(t,p-other);}
  }
  let rim=primal_data(i,8u);ap+=vec2<f32>(rim.x*p.x+rim.y*p.y,rim.y*p.x+rim.z*p.y);
  primal_put(i,3u,vec4<f32>(ap,0.0,0.0));sums=vec2<f32>(dot(primal_data(i,1u).zw,primal_data(i,2u).xy),dot(p,ap));
 }
 return sums;
}
@compute @workgroup_size(128) fn primal_operator(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 primal_sums[lane]=primal_operator_one(gid.x);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_sums[lane]+=primal_sums[lane+width];}workgroupBarrier();}
 if(lane==0u){motion_actors[group.x*128u+6u]=bitcast<vec4<u32>>(vec4<f32>(primal_sums[0],0.0,0.0));}
}
@compute @workgroup_size(1) fn primal_cg_begin(){atomicStore(&motion_control[55],0u);atomicStore(&motion_dispatch_args[3],atomicLoad(&motion_dispatch_args[0]));}
@compute @workgroup_size(1) fn primal_alpha(){
 // Slot 54 is the CCD retry horizon, never a linear-solver residual.
 let total=dense_totals();if(atomicLoad(&motion_control[55])==0u){atomicStore(&motion_dispatch_args[19],bitcast<u32>(total.x));atomicStore(&motion_control[55],1u);}
 let working=total.x>max(1.0e-18,bitcast<f32>(atomicLoad(&motion_dispatch_args[19]))*0.00001)&&total.y>1.0e-24;
 motion_store(36,select(0.0,total.x/max(total.y,1.0e-30),working));motion_store(38,total.x);if(!working){atomicStore(&motion_dispatch_args[3],0u);}
}
fn primal_update_one(i:u32)->f32{
 var sum=0.0;if(i<motion.counts.x&&motion_value(0)>0.0&&atomicLoad(&motion_control[6])==0u){
  let a=motion_value(36);let old=primal_data(i,1u);let residual=old.zw-a*primal_data(i,3u).xy;var z=residual/primal_data(i,0u).zw;if(PRIMAL_BLOCK_PRECONDITIONER){z=primal_block_apply(primal_data(i,90u),residual);}let d=primal_data(i,2u).zw+a*old.xy;
  primal_put(i,1u,vec4<f32>(old.xy,residual));primal_put(i,2u,vec4<f32>(z,d));sum=dot(residual,z);
 }
 return sum;
}
@compute @workgroup_size(128) fn primal_update(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 primal_sums[lane]=vec2<f32>(primal_update_one(gid.x),0.0);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_sums[lane]+=primal_sums[lane+width];}workgroupBarrier();}if(lane==0u){motion_actors[group.x*128u+6u]=bitcast<vec4<u32>>(vec4<f32>(primal_sums[0],0.0,0.0));}
}
@compute @workgroup_size(1) fn primal_beta(){motion_store(37,dense_totals().x/max(motion_value(38),1.0e-30));}
fn primal_conjugate_one(i:u32){if(i>=motion.counts.x||motion_value(0)==0.0){return;}let old=primal_data(i,1u);primal_put(i,1u,vec4<f32>(primal_data(i,2u).xy+motion_value(37)*old.xy,old.zw));}
@compute @workgroup_size(1) fn primal_line_begin(){motion_store(40,1.0);}
fn primal_line_bound_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}var fraction=1.0;let d=primal_data(i,2u).zw;let count=primal_counts(i).x;
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=primal_row(i,k);var other=vec2<f32>(0.0);if(row.x!=0xffffffffu){other=primal_data(row.x,2u).zw;}let closing=-primal_project_slack(bitcast<vec2<f32>>(row.yz),d,other,0.0);if(closing>0.0){fraction=min(fraction,0.95*primal_slack(i,row,a)/closing);}}
 let endpoint=primal_endpoint(i);let delta=d*motion_value(8);let proposed=motion_compensated(endpoint.xy,endpoint.zw,delta);let gap=motion_rim_gap_parts(proposed)-motion_solve_reserve();
 if(gap<0.0){let b=dot(endpoint.xy,delta);let aa=dot(delta,delta);let c=primal_rim_slack(i);fraction=min(fraction,0.95*c/max(b+sqrt(max(0.0,b*b+aa*c)),1.0e-30));}
 atomicMin(&motion_control[40],bitcast<u32>(clamp(fraction,0.0,1.0)));
}
@compute @workgroup_size(1) fn primal_trial_begin(){atomicStore(&motion_control[39],0u);}
fn primal_trial_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}primal_put(i,4u,vec4<f32>(primal_velocity(i)+motion_value(40)*primal_data(i,2u).zw,0.0,0.0));
}
fn primal_trial_end_parts(i:u32)->vec4<f32>{let current=primal_data(i,7u);let endpoint=motion_integrated(current.xy,current.zw,primal_data(i,4u).xy,motion_value(8));if(atomicLoad(&motion_control[57])==0u){return endpoint;}return motion_rotated_parts(endpoint,motion_end_rotation(motion_value(8)));}
fn primal_trial_validate_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}let v=primal_data(i,4u).xy;let current=primal_data(i,7u);let endpoint=motion_offset_parts(motion_integrated(current.xy,current.zw,v,motion_value(8)));var safe=motion_rim_gap_parts(endpoint)>motion_solve_reserve();let count=primal_counts(i).x;
 for(var k=0u;k<count;k++){let a=dense_row(i,k);let row=primal_row(i,k);var other=vec2<f32>(0.0);if(row.x!=0xffffffffu){other=primal_data(row.x,4u).xy;}safe=safe&&primal_project_slack(bitcast<vec2<f32>>(row.yz),v,other,primal_required(i,row,a))>0.0;}
 if(CONTACT_PERSISTENT_SKIN){
  // The actual rounded coupled configuration belongs in the line search,
  // not merely in an after-the-solve guard which can reject every motion.
  // Keep this safety threshold outside the log barrier's feasible start.
  let floor=motion_solve_reserve();let own=primal_trial_end_parts(i);let offset=motion_offset_parts(own);
  safe=safe&&motion_rim_gap_parts(offset)>floor;
  for(var k=0u;k<motion.counts.w;k++){safe=safe&&motion_bar_gap_parts(offset,motion_end_angle(motion_value(8)),baffle(k))>floor;}
  for(var k=0u;k<count;k++){let row=primal_row(i,k);if(row.x==0xffffffffu){continue;}let other=primal_trial_end_parts(row.x);let difference=motion_compensated(own.xy,own.zw,-other.xy);let separation=motion_two_sum(difference.xy,difference.zw-other.zw);safe=safe&&motion_pair_gap_parts(separation)>floor;}
 }
 if(!safe){atomicStore(&motion_control[39],1u);}
}
fn primal_energy_one(i:u32)->vec2<f32>{
 if(i>=motion.counts.x||motion_value(0)==0.0){return vec2<f32>(0.0);}
 let old=primal_velocity(i);let trial=primal_data(i,4u).xy;let free=primal_free(i);let mu=motion_value(47);
 var energy=0.5*vec2<f32>(dot(old-free,old-free),dot(trial-free,trial-free));
 let count=primal_counts(i).x;
 for(var k=0u;k<count;k++){
  let a=dense_row(i,k);let row=primal_row(i,k);var other=vec2<f32>(0.0);if(row.x!=0xffffffffu){other=primal_data(row.x,4u).xy;}
  let slack=primal_project_slack(bitcast<vec2<f32>>(row.yz),trial,other,primal_required(i,row,a));
  energy-=mu*log(max(vec2<f32>(primal_slack(i,row,a),slack),vec2<f32>(1.0e-30)));
  if(PRIMAL_COULOMB_DISSIPATION){let capacity=primal_friction(i,k).x;energy+=capacity*vec2<f32>(primal_huber(primal_tangent_speed(i,row,a,false)).x,primal_huber(primal_tangent_speed(i,row,a,true)).x);}
 }
 let current=primal_data(i,7u);let endpoint=motion_offset_parts(motion_integrated(current.xy,current.zw,trial,motion_value(8)));
 let gap=motion_rim_gap_parts(endpoint)-motion_solve_reserve();let trialRim=gap*(2.0*motion_length(endpoint.xy)+gap);
 energy-=mu*log(max(vec2<f32>(primal_rim_slack(i),trialRim),vec2<f32>(1.0e-30)));
 // Compare against Armijo's directional decrease, never feasibility alone.
 energy.y-=0.0001*motion_value(40)*dot(primal_data(i,0u).xy,primal_data(i,2u).zw);
 return energy;
}
@compute @workgroup_size(128) fn primal_energy(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 primal_sums[lane]=primal_energy_one(gid.x);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_sums[lane]+=primal_sums[lane+width];}workgroupBarrier();}if(lane==0u){motion_actors[group.x*128u+6u]=bitcast<vec4<u32>>(vec4<f32>(primal_sums[0],0.0,0.0));}
}
@compute @workgroup_size(1) fn primal_energy_validate(){let energy=dense_totals();if(energy.y>energy.x){atomicStore(&motion_control[39],1u);}}
@compute @workgroup_size(1) fn primal_trial_reduce(){if(atomicLoad(&motion_control[39])!=0u){motion_store(40,motion_value(40)*0.5);}}
fn primal_line_apply_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u||atomicLoad(&motion_control[39])!=0u){return;}primal_put(i,5u,primal_data(i,4u));
}
@compute @workgroup_size(128) fn primal_finish(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0||atomicLoad(&motion_control[6])!=0u){return;}let s=motion_state(i);motion_write(i,vec4<f32>(s.xy,primal_velocity(i)));}
@compute @workgroup_size(128) fn primal_initialize(@builtin(global_invocation_id) gid:vec3<u32>){primal_initialize_one(gid.x);}
@compute @workgroup_size(128) fn primal_weights(@builtin(global_invocation_id) gid:vec3<u32>){primal_weights_one(gid.x);}
@compute @workgroup_size(128) fn primal_gradient(@builtin(global_invocation_id) gid:vec3<u32>){primal_gradient_one(gid.x);}
@compute @workgroup_size(128) fn primal_conjugate(@builtin(global_invocation_id) gid:vec3<u32>){primal_conjugate_one(gid.x);}
@compute @workgroup_size(128) fn primal_line_bound(@builtin(global_invocation_id) gid:vec3<u32>){primal_line_bound_one(gid.x);}
@compute @workgroup_size(128) fn primal_trial(@builtin(global_invocation_id) gid:vec3<u32>){primal_trial_one(gid.x);}
@compute @workgroup_size(128) fn primal_trial_validate(@builtin(global_invocation_id) gid:vec3<u32>){primal_trial_validate_one(gid.x);}
@compute @workgroup_size(128) fn primal_line_apply(@builtin(global_invocation_id) gid:vec3<u32>){primal_line_apply_one(gid.x);}

fn primal_reduce(lane:u32){workgroupBarrier();for(var width=128u;width>0u;width/=2u){if(lane<width){primal_sums[lane]+=primal_sums[lane+width];}workgroupBarrier();}}
var<workgroup> primal_work:u32;
// Portable 16-KiB bounded cache: 1908*8 + 128*8 + 4*8 + 16.
// This path retains the scalar preconditioner and all original iteration gates.
struct PrimalCacheControl {work:u32,alpha:f32,beta:f32,rr:f32,initial_rr:f32,free_rr:f32,chopped_rr:f32,step_kind:u32}
var<workgroup> primal_cache:array<vec2<f32>,1908>;
var<workgroup> primal_cached_sums:array<vec2<f32>,128>;
var<workgroup> primal_cached_partial:array<vec2<f32>,4>;
var<workgroup> primal_cached_control:PrimalCacheControl;
fn primal_cached_reduce(lane:u32){
 workgroupBarrier();if(lane<4u){var sum=vec2<f32>(0.0);for(var k=lane;k<128u;k+=4u){sum+=primal_cached_sums[k];}primal_cached_partial[lane]=sum;}workgroupBarrier();
 if(lane==0u){primal_cached_sums[0]=(primal_cached_partial[0]+primal_cached_partial[1])+(primal_cached_partial[2]+primal_cached_partial[3]);}
}
fn primal_cached_operator_one(i:u32)->vec2<f32>{
 if(atomicLoad(&motion_control[6])!=0u){return vec2<f32>(0.0);}let p=primal_cache[i];var ap=p;
 for(var k=0u;k<primal_counts(i).y;k++){let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let n=bitcast<vec2<f32>>(row.yz);let j=select(owner,row.x,owner==i);var other=vec2<f32>(0.0);if(j!=0xffffffffu){other=primal_cache[j];}ap+=n*primal_weight(owner,id%32u)*dot(n,p-other);if(PRIMAL_COULOMB_DISSIPATION){let t=vec2<f32>(-n.y,n.x);ap+=t*primal_friction(owner,id%32u).y*dot(t,p-other);}}
 let rim=primal_data(i,8u);ap+=vec2<f32>(rim.x*p.x+rim.y*p.y,rim.y*p.x+rim.z*p.y);primal_put(i,3u,vec4<f32>(ap,0.0,0.0));return vec2<f32>(dot(primal_data(i,1u).zw,primal_data(i,2u).xy),dot(p,ap));
}
fn primal_cached_update_one(i:u32)->f32{
 let residual=primal_data(i,1u).zw-primal_cached_control.alpha*primal_data(i,3u).xy;var z=residual/primal_data(i,0u).zw;if(PRIMAL_BLOCK_PRECONDITIONER){z=primal_block_apply(primal_data(i,90u),residual);}let d=primal_data(i,2u).zw+primal_cached_control.alpha*primal_cache[i];primal_put(i,1u,vec4<f32>(primal_cache[i],residual));primal_put(i,2u,vec4<f32>(z,d));return dot(residual,z);
}
@compute @workgroup_size(128) fn primal_cached_component(@builtin(local_invocation_index) lane:u32){
 for(var i=lane;i<motion.counts.x;i+=128u){primal_initialize_one(i);}storageBarrier();if(lane==0u){motion_store(47,0.0001);}storageBarrier();
 if(motion.grid.w==4u){for(var i=lane;i<motion.counts.x;i+=128u){primal_adaptive_limit_one(i);}storageBarrier();}
 for(var outer=0u;outer<3u;outer++){
  for(var newton=0u;newton<select(2u,8u,PRIMAL_BLOCK_PRECONDITIONER);newton++){
   for(var i=lane;i<motion.counts.x;i+=128u){primal_weights_one(i);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=128u){primal_gradient_one(i);}storageBarrier();
   for(var cg=0u;cg<select(8u,32u,PRIMAL_BLOCK_PRECONDITIONER);cg++){
    for(var i=lane;i<motion.counts.x;i+=128u){primal_cache[i]=primal_data(i,1u).xy;}workgroupBarrier();
    var sum=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=128u){sum+=primal_cached_operator_one(i);}primal_cached_sums[lane]=sum;primal_cached_reduce(lane);
    if(lane==0u){let t=primal_cached_sums[0];if(cg==0u){primal_cached_control.initial_rr=t.x;}let threshold=select(1.0e-20,max(1.0e-18,primal_cached_control.initial_rr*0.00001),motion.grid.w==4u);let work=t.x>threshold&&t.y>1.0e-24;primal_cached_control.work=select(0u,1u,work);primal_cached_control.alpha=select(0.0,t.x/max(t.y,1.0e-30),work);primal_cached_control.rr=t.x;}workgroupBarrier();
    if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
    var rr=0.0;for(var i=lane;i<motion.counts.x;i+=128u){rr+=primal_cached_update_one(i);}primal_cached_sums[lane]=vec2<f32>(rr,0.0);primal_cached_reduce(lane);
    if(lane==0u){primal_cached_control.beta=primal_cached_sums[0].x/max(primal_cached_control.rr,1.0e-30);}workgroupBarrier();
    for(var i=lane;i<motion.counts.x;i+=128u){let old=primal_data(i,1u);primal_put(i,1u,vec4<f32>(primal_data(i,2u).xy+primal_cached_control.beta*primal_cache[i],old.zw));}
   }
   if(lane==0u){motion_store(40,1.0);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=128u){primal_line_bound_one(i);}storageBarrier();
   for(var retry=0u;retry<4u;retry++){
    if(lane==0u){atomicStore(&motion_control[39],0u);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=128u){primal_trial_one(i);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=128u){primal_trial_validate_one(i);}storageBarrier();
    var energy=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=128u){energy+=primal_energy_one(i);}primal_cached_sums[lane]=energy;primal_cached_reduce(lane);
    if(lane==0u){if(primal_cached_sums[0].y>primal_cached_sums[0].x){atomicStore(&motion_control[39],1u);}primal_cached_control.work=atomicLoad(&motion_control[39]);}storageBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
    if(lane==0u&&retry<3u){motion_store(40,motion_value(40)*0.5);}storageBarrier();
   }
   for(var i=lane;i<motion.counts.x;i+=128u){primal_line_apply_one(i);}storageBarrier();
  }
  if(lane==0u){motion_store(47,max(select(select(0.000001,0.0001,CONTACT_PERSISTENT_SKIN),1.0e-30,motion.grid.w==4u),motion_value(47)*0.1));}storageBarrier();
 }
}
// The same per-body laws are used by the multi-workgroup entrypoints above.
@compute @workgroup_size(256) fn primal_component(@builtin(local_invocation_index) lane:u32){
 for(var i=lane;i<motion.counts.x;i+=256u){primal_initialize_one(i);}storageBarrier();
 if(lane==0u){motion_store(47,0.0001);}storageBarrier();
 if(motion.grid.w==4u){for(var i=lane;i<motion.counts.x;i+=256u){primal_adaptive_limit_one(i);}storageBarrier();}
 for(var outer=0u;outer<3u;outer++){
  for(var newton=0u;newton<select(2u,4u,PRIMAL_COULOMB_DISSIPATION);newton++){
   for(var i=lane;i<motion.counts.x;i+=256u){primal_weights_one(i);}storageBarrier();
   for(var i=lane;i<motion.counts.x;i+=256u){primal_gradient_one(i);}storageBarrier();
   for(var cg=0u;cg<select(8u,32u,PRIMAL_COULOMB_DISSIPATION);cg++){
    var sums=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=256u){sums+=primal_operator_one(i);}primal_sums[lane]=sums;storageBarrier();primal_reduce(lane);
    if(lane==0u){let t=primal_sums[0];let work=t.x>1.0e-20&&t.y>1.0e-24;primal_work=select(0u,1u,work);motion_store(36,select(0.0,t.x/max(t.y,1.0e-30),work));motion_store(38,t.x);}storageBarrier();
    // The previous implementation ran every remaining iteration with alpha=0.
    // No state update can follow a zero step; use a uniform termination gate.
    if(workgroupUniformLoad(&primal_work)==0u){break;}
    var rr=0.0;for(var i=lane;i<motion.counts.x;i+=256u){rr+=primal_update_one(i);}primal_sums[lane]=vec2<f32>(rr,0.0);storageBarrier();primal_reduce(lane);
    if(lane==0u){motion_store(37,primal_sums[0].x/max(motion_value(38),1.0e-30));}storageBarrier();
    for(var i=lane;i<motion.counts.x;i+=256u){primal_conjugate_one(i);}storageBarrier();
   }
   if(lane==0u){motion_store(40,1.0);}storageBarrier();
   for(var i=lane;i<motion.counts.x;i+=256u){primal_line_bound_one(i);}storageBarrier();
   for(var retry=0u;retry<4u;retry++){
    if(lane==0u){atomicStore(&motion_control[39],0u);}storageBarrier();
    for(var i=lane;i<motion.counts.x;i+=256u){primal_trial_one(i);}storageBarrier();
    for(var i=lane;i<motion.counts.x;i+=256u){primal_trial_validate_one(i);}storageBarrier();
    var energy=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=256u){energy+=primal_energy_one(i);}primal_sums[lane]=energy;storageBarrier();primal_reduce(lane);
    if(lane==0u){if(primal_sums[0].y>primal_sums[0].x){atomicStore(&motion_control[39],1u);}primal_work=select(0u,1u,atomicLoad(&motion_control[39])!=0u);}storageBarrier();
    if(workgroupUniformLoad(&primal_work)==0u){break;}
    if(lane==0u&&retry<3u&&atomicLoad(&motion_control[39])!=0u){motion_store(40,motion_value(40)*0.5);}storageBarrier();
   }
   for(var i=lane;i<motion.counts.x;i+=256u){primal_line_apply_one(i);}storageBarrier();
  }
  if(lane==0u){motion_store(47,max(select(select(0.000001,0.0001,CONTACT_PERSISTENT_SKIN),1.0e-30,motion.grid.w==4u),motion_value(47)*0.1));}storageBarrier();
 }
}
`;
