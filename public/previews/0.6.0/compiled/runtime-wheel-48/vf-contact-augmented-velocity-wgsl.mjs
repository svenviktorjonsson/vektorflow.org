// Experimental inertial velocity solve. Private proposals may violate the
// linearized model; ONLY continuous admission can publish physical motion.
export const CONTACT_AUGMENTED_VELOCITY_WGSL=/* wgsl */`
const AL_CACHE_ENABLED=false;
override AL_AFFINE_PRECONDITIONER=false;
fn al_rim_constraint(i:u32,v:vec2<f32>)->f32{
 let current=primal_data(i,7u);let h=max(motion_value(8),1.0e-12);
 let endpoint=motion_offset_parts(motion_compensated(current.xy,current.zw,v*h));
 return (motion_rim_gap_parts(endpoint)-motion_solve_reserve())/h;
}
fn al_initialize_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}
 let next_counts=motion_actors[i*128u+4u].xy;let old_count=primal_counts(i).x;
 // Match before overwriting headers. Temporary x never overwrites old lambda y.
 for(var k=0u;k<next_counts.x;k++){
  let a=dense_row(i,k);let row=motion_actors[a];let key=motion_actors[a+2u].y;var lambda=0.0;
  for(var j=0u;j<min(old_count,32u);j++){
   let old=primal_row(i,j);let state=primal_data(i,10u+2u*j);
   if(AL_CACHE_ENABLED&&old.x==row.x&&bitcast<u32>(state.w)==key&&dot(bitcast<vec2<f32>>(old.yz),bitcast<vec2<f32>>(row.yz))>0.9){lambda=max(0.0,state.y);break;}
  }
  var state=primal_data(i,10u+2u*k);state.x=lambda;primal_put(i,10u+2u*k,state);
 }
 let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]).zw;
 primal_put(i,5u,vec4<f32>(free,0.0,0.0));primal_put(i,6u,vec4<f32>(free,0.0,0.0));primal_put(i,7u,motion_position_parts(i,0.0));
 for(var k=0u;k<next_counts.x;k++){
  let a=dense_row(i,k);let row=motion_actors[a];let lambda=primal_data(i,10u+2u*k).x;
  primal_scratch[primal_slot(i,9u+2u*k)]=vec4<u32>(row.xyz,motion_actors[a+2u].x);
  primal_put(i,10u+2u*k,vec4<f32>(0.0,lambda,0.0,bitcast<f32>(motion_actors[a+2u].y)));
 }
 for(var k=0u;k<(next_counts.y+3u)/4u;k++){primal_scratch[primal_slot(i,73u+k)]=motion_actors[i*128u+8u+k];}
 primal_scratch[primal_slot(i,89u)]=vec4<u32>(next_counts,0u,0u);
 if(!AL_CACHE_ENABLED){primal_put(i,90u,vec4<f32>(0.0));}
}
fn al_seed_velocity_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}var v=primal_free(i);
 for(var k=0u;k<primal_counts(i).y;k++){let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);v+=select(-1.0,1.0,owner==i)*bitcast<vec2<f32>>(row.yz)*primal_data(owner,10u+2u*(id%32u)).y;}
 let p=motion_offset_parts(primal_data(i,7u));v-=max(0.0,primal_data(i,90u).x)*p.xy/max(motion_length(p.xy),1.0e-12);primal_put(i,5u,vec4<f32>(v,0.0,0.0));
}
fn al_weights_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}let mu=motion_value(47);
 for(var k=0u;k<primal_counts(i).x;k++){
  let row=primal_row(i,k);let c=primal_slack(i,row,dense_row(i,k));var state=primal_data(i,10u+2u*k);
  state.x=select(0.0,mu,c-state.y/mu<=0.0);state.z=max(0.0,state.y-mu*c);primal_put(i,10u+2u*k,state);
 }
}
fn al_gradient_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}
 let mu=motion_value(47);let h=max(motion_value(8),1.0e-12);var gradient=primal_velocity(i)-primal_free(i);var diagonal=vec2<f32>(1.0);var off=0.0;
 for(var k=0u;k<primal_counts(i).y;k++){
  let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let n=bitcast<vec2<f32>>(row.yz);let state=primal_data(owner,10u+2u*(id%32u));
  gradient-=select(-1.0,1.0,owner==i)*n*state.z;diagonal+=state.x*n*n;off+=state.x*n.x*n.y;
 }
 let p=primal_endpoint(i);let radius=max(motion_length(p.xy),1.0e-12);let n=p.xy/radius;let c=al_rim_constraint(i,primal_velocity(i));let lambda=max(0.0,primal_data(i,90u).x);let force=max(0.0,lambda-mu*c);
 let engaged=c-lambda/mu<=0.0;let weight=select(0.0,mu,engaged);let tangent=force*h/radius;
 let rim=vec3<f32>(weight*n.x*n.x+tangent*n.y*n.y,(weight-tangent)*n.x*n.y,weight*n.y*n.y+tangent*n.x*n.x);
 primal_put(i,8u,vec4<f32>(rim,0.0));gradient+=force*n;diagonal+=rim.xz;off+=rim.y;
 if(any((bitcast<vec2<u32>>(gradient)&vec2<u32>(0x7f800000u))==vec2<u32>(0x7f800000u))||any((bitcast<vec2<u32>>(diagonal)&vec2<u32>(0x7f800000u))==vec2<u32>(0x7f800000u))){atomicOr(&motion_control[6],32u);return;}
 let r=-gradient;let z=primal_block_apply(primal_block_inverse(diagonal,off),r);primal_put(i,0u,vec4<f32>(gradient,diagonal));primal_put(i,1u,vec4<f32>(z,r));primal_put(i,2u,vec4<f32>(z,0.0,0.0));primal_put(i,3u,vec4<f32>(0.0,0.0,off,0.0));
}
fn al_energy_one(i:u32)->vec2<f32>{
 if(i>=motion.counts.x||motion_value(0)==0.0){return vec2<f32>(0.0);}
 let old=primal_velocity(i);let trial=primal_data(i,4u).xy;let free=primal_free(i);let mu=motion_value(47);var energy=0.5*vec2<f32>(dot(old-free,old-free),dot(trial-free,trial-free));
 for(var k=0u;k<primal_counts(i).x;k++){
  let row=primal_row(i,k);var relative=trial;if(row.x!=0xffffffffu){relative-=primal_data(row.x,4u).xy;}
  let lambda=primal_data(i,10u+2u*k).y;let c=vec2<f32>(primal_slack(i,row,dense_row(i,k)),dot(bitcast<vec2<f32>>(row.yz),relative)-bitcast<f32>(row.w));let residual=min(c-vec2<f32>(lambda/mu),vec2<f32>(0.0));energy+=0.5*mu*residual*residual;
 }
 let lambda=max(0.0,primal_data(i,90u).x);let residual=min(vec2<f32>(al_rim_constraint(i,old),al_rim_constraint(i,trial))-vec2<f32>(lambda/mu),vec2<f32>(0.0));energy+=0.5*mu*residual*residual;
 energy.y-=0.0001*motion_value(40)*dot(primal_data(i,0u).xy,primal_data(i,2u).zw);return energy;
}
fn al_multipliers_one(i:u32){
 if(i>=motion.counts.x||motion_value(0)==0.0){return;}let mu=motion_value(47);
 for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);var state=primal_data(i,10u+2u*k);state.y=max(0.0,state.y-mu*primal_slack(i,row,dense_row(i,k)));primal_put(i,10u+2u*k,state);}
 let lambda=max(0.0,primal_data(i,90u).x);primal_put(i,90u,vec4<f32>(max(0.0,lambda-mu*al_rim_constraint(i,primal_velocity(i))),0.0,0.0,0.0));
}
struct AlController {work:u32,alpha:f32,beta:f32,initial_rr:f32}
// Portable cache plus 6-DOF additive affine preconditioner fits 16 KiB.
// The host rejects larger components; no silent truncation or batching claim.
var<workgroup> al_vectors:array<vec2<f32>,1908>;
var<workgroup> al_sums:array<vec2<f32>,64>;
var<workgroup> al_partial:array<vec2<f32>,4>;
var<workgroup> al_controller:AlController;
var<workgroup> al_coarse:array<f32,88>;
fn al_basis(i:u32)->vec3<f32>{let p=motion_offset_parts(primal_data(i,7u));return vec3<f32>(1.0,(p.xy+p.zw)/WHEEL_RADIUS);}
fn al_upper(a:u32,b:u32)->u32{return a*(13u-a)/2u+b-a;}
fn al_coarse_build(lane:u32){
 // Reuse the direction-vector cache while no CG direction is live. This avoids
 // float atomics and leaves the standard 16-KiB workgroup limit unchanged.
 var matrix:array<f32,21>;for(var k=0u;k<21u;k++){matrix[k]=0.0;}
 for(var i=lane;i<motion.counts.x;i+=64u){let basis=al_basis(i);let rim=primal_data(i,8u);
  for(var a=0u;a<6u;a++){for(var b=a;b<6u;b++){let ax=a%2u;let bx=b%2u;var value=rim.y;if(ax==bx){value=1.0+select(rim.x,rim.z,ax==1u);}matrix[al_upper(a,b)]+=basis[a/2u]*basis[b/2u]*value;}}
  for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var difference=basis;if(row.x!=0xffffffffu){difference-=al_basis(row.x);}let weight=primal_weight(i,k);
   for(var a=0u;a<6u;a++){for(var b=a;b<6u;b++){matrix[al_upper(a,b)]+=weight*difference[a/2u]*n[a%2u]*difference[b/2u]*n[b%2u];}}
  }
 }
 for(var k=0u;k<11u;k++){var y=0.0;if(2u*k+1u<21u){y=matrix[2u*k+1u];}al_vectors[k*64u+lane]=vec2<f32>(matrix[2u*k],y);}workgroupBarrier();
 if(lane==0u){for(var a=0u;a<6u;a++){for(var b=a;b<6u;b++){let k=al_upper(a,b);var value=0.0;for(var l=0u;l<64u;l++){value+=al_vectors[(k/2u)*64u+l][k%2u];}al_coarse[a*6u+b]=value;al_coarse[b*6u+a]=value;}}
  var scale=1.0;for(var a=0u;a<6u;a++){scale=max(scale,al_coarse[a*6u+a]);}var jitter=scale*0.00001;var success=false;
  for(var retry=0u;retry<10u;retry++){success=true;for(var a=0u;a<6u;a++){for(var b=0u;b<=a;b++){var value=al_coarse[a*6u+b];if(a==b){value+=jitter;}for(var k=0u;k<b;k++){value-=al_coarse[36u+a*6u+k]*al_coarse[36u+b*6u+k];}if(a==b){success=success&&value>0.0;al_coarse[36u+a*6u+b]=sqrt(max(value,1.0e-20));}else{al_coarse[36u+a*6u+b]=value/al_coarse[36u+b*6u+b];}}}if(success){break;}jitter*=2.0;}
  // Failure falls back to the block preconditioner, never to unchecked motion.
  al_coarse[84]=select(0.0,1.0,success);
 }workgroupBarrier();
}
fn al_coarse_apply(lane:u32){
 workgroupBarrier();var rhs:array<f32,6>;for(var k=0u;k<6u;k++){rhs[k]=0.0;}
 for(var i=lane;i<motion.counts.x;i+=64u){let basis=al_basis(i);let r=primal_data(i,1u).zw;for(var k=0u;k<6u;k++){rhs[k]+=basis[k/2u]*r[k%2u];}}
 for(var k=0u;k<3u;k++){al_vectors[k*64u+lane]=vec2<f32>(rhs[2u*k],rhs[2u*k+1u]);}workgroupBarrier();
 if(lane==0u){for(var k=0u;k<6u;k++){var value=0.0;for(var l=0u;l<64u;l++){value+=al_vectors[(k/2u)*64u+l][k%2u];}al_coarse[72u+k]=value;}
  for(var a=0u;a<6u;a++){var value=al_coarse[72u+a];for(var k=0u;k<a;k++){value-=al_coarse[36u+a*6u+k]*al_coarse[78u+k];}al_coarse[78u+a]=value/al_coarse[36u+a*6u+a];}
  for(var step=0u;step<6u;step++){let a=5u-step;var value=al_coarse[78u+a];for(var k=a+1u;k<6u;k++){value-=al_coarse[36u+k*6u+a]*al_coarse[78u+k];}al_coarse[78u+a]=value/al_coarse[36u+a*6u+a];}
 }workgroupBarrier();
 for(var i=lane;i<motion.counts.x;i+=64u){let basis=al_basis(i);var correction=vec2<f32>(0.0);for(var k=0u;k<3u;k++){correction+=basis[k]*vec2<f32>(al_coarse[78u+2u*k],al_coarse[79u+2u*k]);}let state=primal_data(i,2u);primal_put(i,2u,vec4<f32>(state.xy+correction*al_coarse[84],state.zw));}
}
fn al_reduce(lane:u32){
 workgroupBarrier();if(lane<4u){var total=vec2<f32>(0.0);for(var k=lane;k<64u;k+=4u){total+=al_sums[k];}al_partial[lane]=total;}workgroupBarrier();
 // The caller's lane-zero controller is followed by an execution barrier.
 // Other lanes never consume this scalar sum before that controller finishes.
 if(lane==0u){al_sums[0]=(al_partial[0]+al_partial[1])+(al_partial[2]+al_partial[3]);}
}
fn al_cached_operator_one(i:u32)->vec2<f32>{
 let p=al_vectors[i];var ap=p;
 for(var k=0u;k<primal_counts(i).y;k++){
  let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let n=bitcast<vec2<f32>>(row.yz);let j=select(owner,row.x,owner==i);var other=vec2<f32>(0.0);if(j!=0xffffffffu){other=al_vectors[j];}
  ap+=n*primal_weight(owner,id%32u)*dot(n,p-other);
 }
 let rim=primal_data(i,8u);ap+=vec2<f32>(rim.x*p.x+rim.y*p.y,rim.y*p.x+rim.z*p.y);primal_put(i,3u,vec4<f32>(ap,primal_data(i,3u).z,0.0));return vec2<f32>(dot(primal_data(i,1u).zw,primal_data(i,2u).xy),dot(p,ap));
}
fn al_cached_update_one(i:u32)->f32{
 let residual=primal_data(i,1u).zw-al_controller.alpha*primal_data(i,3u).xy;let z=primal_block_apply(primal_block_inverse(primal_data(i,0u).zw,primal_data(i,3u).z),residual);let d=primal_data(i,2u).zw+al_controller.alpha*al_vectors[i];primal_put(i,1u,vec4<f32>(al_vectors[i],residual));primal_put(i,2u,vec4<f32>(z,d));return dot(residual,z);
}
@compute @workgroup_size(64) fn al_component(@builtin(local_invocation_index) lane:u32){
 for(var i=lane;i<motion.counts.x;i+=64u){al_initialize_one(i);}storageBarrier();if(lane==0u){motion_store(47,100000.0);}storageBarrier();
 for(var i=lane;i<motion.counts.x;i+=64u){al_seed_velocity_one(i);}storageBarrier();
 for(var outer=0u;outer<4u;outer++){
  for(var newton=0u;newton<2u;newton++){
   for(var i=lane;i<motion.counts.x;i+=64u){al_weights_one(i);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=64u){al_gradient_one(i);}storageBarrier();
   if(AL_AFFINE_PRECONDITIONER){al_coarse_build(lane);al_coarse_apply(lane);for(var i=lane;i<motion.counts.x;i+=64u){let state=primal_data(i,1u);primal_put(i,1u,vec4<f32>(primal_data(i,2u).xy,state.zw));}}
   for(var cg=0u;cg<64u;cg++){
    for(var i=lane;i<motion.counts.x;i+=64u){al_vectors[i]=primal_data(i,1u).xy;}workgroupBarrier();
    var sums=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=64u){sums+=al_cached_operator_one(i);}al_sums[lane]=sums;al_reduce(lane);
    if(lane==0u){let t=al_sums[0];if(cg==0u){al_controller.initial_rr=t.x;}let work=t.x>max(1.0e-20,al_controller.initial_rr*0.0000000001)&&t.y>1.0e-24;al_controller.work=select(0u,1u,work);al_controller.alpha=select(0.0,t.x/max(t.y,1.0e-30),work);}workgroupBarrier();
    if(workgroupUniformLoad(&al_controller.work)==0u){break;}
    for(var i=lane;i<motion.counts.x;i+=64u){al_cached_update_one(i);}if(AL_AFFINE_PRECONDITIONER){al_coarse_apply(lane);}
    var rr=0.0;for(var i=lane;i<motion.counts.x;i+=64u){rr+=dot(primal_data(i,1u).zw,primal_data(i,2u).xy);}al_sums[lane]=vec2<f32>(rr,sums.x);al_reduce(lane);
    if(lane==0u){al_controller.beta=al_sums[0].x/max(al_sums[0].y,1.0e-30);}workgroupBarrier();for(var i=lane;i<motion.counts.x;i+=64u){let old=primal_data(i,1u);primal_put(i,1u,vec4<f32>(primal_data(i,2u).xy+al_controller.beta*old.xy,old.zw));}
   }
   if(lane==0u){motion_store(40,1.0);}storageBarrier();
   for(var retry=0u;retry<16u;retry++){
    if(lane==0u){atomicStore(&motion_control[39],0u);}storageBarrier();for(var i=lane;i<motion.counts.x;i+=64u){primal_trial_one(i);}storageBarrier();
    var energy=vec2<f32>(0.0);for(var i=lane;i<motion.counts.x;i+=64u){energy+=al_energy_one(i);}al_sums[lane]=energy;al_reduce(lane);
    if(lane==0u){let failed=al_sums[0].y>al_sums[0].x;atomicStore(&motion_control[39],select(0u,1u,failed));al_controller.work=select(0u,1u,failed);if(retry<15u&&failed){motion_store(40,motion_value(40)*0.5);}}storageBarrier();
    if(workgroupUniformLoad(&al_controller.work)==0u){break;}
   }
   for(var i=lane;i<motion.counts.x;i+=64u){primal_line_apply_one(i);}storageBarrier();
  }
  for(var i=lane;i<motion.counts.x;i+=64u){al_multipliers_one(i);}storageBarrier();
 }
}
`;
