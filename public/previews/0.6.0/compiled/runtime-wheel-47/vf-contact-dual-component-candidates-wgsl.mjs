// Engineering-only dual/component contact candidates, not release accepted.
export const CONTACT_DUAL_COMPONENT_CANDIDATES_WGSL=/* wgsl */`
// Bounded contact-component specialization. Not a million-body component.
// Direct nonnegative impulse QP candidate. Scratch proposals are not physical
// positions; the same swept and rounded endpoint admission remains mandatory.
override DUAL_PARALLEL=false;
override DUAL_BLOCK_PIVOT=false;
override DUAL_CACHE_EPOCH=false;
fn primal_dual_active(i:u32,k:u32,state:vec4<f32>)->bool{if(DUAL_BLOCK_PIVOT){return (primal_scratch[primal_slot(i,90u)].x&(1u<<k))!=0u;}return state.x>0.0||state.z>0.0;}
// Field-major packed low impulses: one owner writes its own four-lane words.
// Keep compensated arithmetic without fetching 2 KiB-strided actor records.
fn primal_dual_low(i:u32,k:u32)->f32{return bitcast<vec4<f32>>(primal_scratch[primal_slot(i,91u+k/4u)])[k%4u];}
fn primal_dual_put_low(i:u32,k:u32,value:f32){let at=primal_slot(i,91u+k/4u);var word=primal_scratch[at];word[k%4u]=bitcast<u32>(value);primal_scratch[at]=word;}
fn primal_dual_delta(i:u32)->vec2<f32>{if(DUAL_PARALLEL){return primal_data(i,2u).xy;}return primal_cache[i];}
fn primal_dual_alpha()->f32{if(DUAL_PARALLEL){return motion_value(36);}return primal_cached_control.alpha;}
fn primal_dual_gather(i:u32,direction:bool)->vec2<f32>{
 var value=vec4<f32>(0.0);for(var k=0u;k<primal_counts(i).y;k++){let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let state=primal_data(owner,10u+2u*(id%32u));let n=select(-1.0,1.0,owner==i)*bitcast<vec2<f32>>(row.yz);let lambda=select(state.x,state.y,direction);var low=0.0;if(!direction){low=primal_dual_low(owner,id%32u);}let product=n*lambda;let error=fma(n,vec2<f32>(lambda),-product)+n*low;let sum=motion_two_sum(value.xy,product);value=motion_two_sum(sum.xy,sum.zw+value.zw+error);}return value.xy+value.zw;
}
fn primal_dual_start(i:u32){
 for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var free=primal_free(i);var delta=primal_dual_delta(i);if(row.x!=0xffffffffu){free-=primal_free(row.x);delta-=primal_dual_delta(row.x);}let state=primal_data(i,10u+2u*k);let residual=bitcast<f32>(row.w)-dot(n,free+delta);let diag=select(2.0,1.0,row.x==0xffffffffu)*dot(n,n);let z=select(0.0,residual/max(diag,1.0e-20),primal_dual_active(i,k,vec4<f32>(state.xy,residual,state.w)));primal_put(i,10u+2u*k,vec4<f32>(state.x,z,residual,z));}
}
fn primal_dual_operator(i:u32)->vec2<f32>{
 var result=vec2<f32>(0.0);for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var delta=primal_dual_delta(i);if(row.x!=0xffffffffu){delta-=primal_dual_delta(row.x);}let state=primal_data(i,10u+2u*k);let ap=dot(n,delta);result+=vec2<f32>(state.z*state.w,state.y*ap);primal_put(i,10u+2u*k,vec4<f32>(state.xyz,ap));}return result;
}
fn primal_dual_bound(i:u32){for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let lambda=state.x+primal_dual_low(i,k);let change=primal_dual_alpha()*state.y;if(change<0.0&&lambda+change<0.0){atomicMin(&motion_control[40],bitcast<u32>(clamp(lambda/max(-change,1.0e-30),0.0,1.0)));}}}
fn primal_dual_update(i:u32)->f32{
 var result=0.0;let alpha=primal_dual_alpha()*motion_value(40);for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);let old=primal_data(i,10u+2u*k);let change=alpha*old.y;let sum=motion_two_sum(vec2<f32>(old.x),vec2<f32>(change));var parts=motion_two_sum(sum.xy,sum.zw+vec2<f32>(primal_dual_low(i,k)+fma(alpha,old.y,-change)));let bound_rounding=0.000000476837158203125*(abs(old.x)+abs(change));if(!DUAL_BLOCK_PIVOT&&(parts.x+parts.z<=0.0||(change<0.0&&motion_value(40)<0.99999&&parts.x+parts.z<=bound_rounding))){parts=vec4<f32>(0.0);}let lambda=parts.x;primal_dual_put_low(i,k,parts.z);let residual=fma(-alpha,old.w,old.z);let old_active=primal_dual_active(i,k,old);let new_active=primal_dual_active(i,k,vec4<f32>(lambda,old.y,residual,old.w));if(old_active!=new_active){atomicStore(&motion_control[35],1u);}let diag=select(2.0,1.0,row.x==0xffffffffu)*dot(n,n);let z=select(0.0,residual/max(diag,1.0e-20),new_active);primal_put(i,10u+2u*k,vec4<f32>(lambda,old.y,residual,z));result+=residual*z;}return result;
}
fn primal_dual_refine_rim(i:u32){
 let endpoint=primal_endpoint(i);let n=-endpoint.xy/max(motion_length(endpoint.xy),1.0e-20);let current=motion_offset_parts(primal_data(i,7u));let reserve=motion_boundary_solve_reserve(motion_rim_gap_parts(current));let beta=-motion_plane_gap_parts(current,n,WHEEL_RADIUS-motion.minimum.z-WHEEL_BAR_HALF_WIDTH,motion_rim_error()+reserve)/max(motion_value(8),1.0e-12);
 for(var k=0u;k<primal_counts(i).x;k++){if(motion_actors[dense_row(i,k)+2u].y==1u){let row=primal_row(i,k);primal_scratch[primal_slot(i,9u+2u*k)]=vec4<u32>(row.x,bitcast<vec2<u32>>(n),bitcast<u32>(beta));}}
}
@compute @workgroup_size(128) fn primal_dual_seed(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}let generation=atomicLoad(&motion_control[55]);let old_count=select(0u,primal_counts(i).x,!DUAL_CACHE_EPOCH||motion_actors[i*128u+124u].x==generation);let old_keys=primal_scratch[primal_slot(i,0u)];let next_count=motion_actors[i*128u+4u].x;
 // Temporary zw preserves ALL old high/low impulses in xy while matching.
 for(var k=0u;k<next_count;k++){let at=dense_row(i,k);let row=motion_actors[at];let key=motion_actors[at+2u].y;var guess=vec2<f32>(0.0);for(var j=0u;j<min(old_count,32u);j++){let old=primal_row(i,j);let old_key=(old_keys[j/8u]>>(4u*(j%8u)))&15u;if(key<15u&&old_key==key&&old.x==row.x&&dot(bitcast<vec2<f32>>(old.yz),bitcast<vec2<f32>>(row.yz))>0.99){guess=primal_data(i,10u+2u*j).xy;break;}}let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.xy,guess));}
 primal_initialize_one(i);primal_put(i,5u,vec4<f32>(primal_free(i),0.0,0.0));var keys=vec4<u32>(0u);
 for(var k=0u;k<next_count;k++){let at=dense_row(i,k);let guess=primal_data(i,10u+2u*k).zw;primal_put(i,10u+2u*k,vec4<f32>(max(0.0,guess.x),0.0,0.0,0.0));primal_dual_put_low(i,k,guess.y);let key=min(15u,motion_actors[at+2u].y);keys[k/8u]|=key<<(4u*(k%8u));}
 primal_scratch[primal_slot(i,0u)]=keys;
 motion_actors[i*128u+124u].x=generation;
 if(primal_counts(i).y>0u){let slot=atomicAdd(&motion_control[33],1u);motion_actors[slot*128u+6u].x=i;}
}
fn primal_dual_index(k:u32,compacted:bool)->u32{if(compacted){return motion_actors[k*128u+6u].x;}return k;}
// Ordered compaction avoids atomics scrambling neighboring owners across SIMD
// lanes. This changes iteration storage/order only, never the constraint set.
@compute @workgroup_size(128) fn primal_dual_sort_active(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){primal_cached_control.work=0u;}workgroupBarrier();
 for(var base=0u;base<motion.counts.x;base+=128u){let i=base+lane;var has_contacts=false;if(i<motion.counts.x){has_contacts=primal_counts(i).y>0u;}primal_cached_sums[lane]=vec2<f32>(select(0.0,1.0,has_contacts),0.0);workgroupBarrier();
  if(lane==0u){var offset=primal_cached_control.work;for(var k=0u;k<128u;k++){primal_cached_sums[k].y=f32(offset);offset+=u32(primal_cached_sums[k].x);}primal_cached_control.work=offset;}workgroupBarrier();
  if(has_contacts){motion_actors[u32(primal_cached_sums[lane].y)*128u+6u].x=i;}storageBarrier();workgroupBarrier();
 }
 if(lane==0u){atomicStore(&motion_control[33],primal_cached_control.work);}
}
fn primal_dual_solve(lane:u32,compacted:bool){
 let actors=select(motion.counts.x,atomicLoad(&motion_control[33]),compacted);
 if(!compacted){for(var i=lane;i<motion.counts.x;i+=128u){primal_initialize_one(i);for(var k=0u;k<primal_counts(i).x;k++){primal_put(i,10u+2u*k,vec4<f32>(0.0));primal_dual_put_low(i,k,0.0);}}storageBarrier();}
 for(var outer=0u;outer<3u;outer++){
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){primal_dual_start(primal_dual_index(at,compacted));}storageBarrier();
  for(var cg=0u;cg<64u;cg++){
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);primal_cache[i]=primal_dual_gather(i,true);}workgroupBarrier();var sums=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){sums+=primal_dual_operator(primal_dual_index(at,compacted));}primal_cached_sums[lane]=sums;primal_cached_reduce(lane);
   if(lane==0u){let t=primal_cached_sums[0];if(cg==0u){primal_cached_control.initial_rr=t.x;}let epsilon=0.25*motion_solve_reserve()/max(motion_value(8),1.0e-12);let tolerance=max(1.0e-20,min(primal_cached_control.initial_rr*0.00001,epsilon*epsilon*0.5));let work=t.x>tolerance&&t.y>1.0e-24;primal_cached_control.work=select(0u,1u,work);primal_cached_control.alpha=select(0.0,t.x/max(t.y,1.0e-30),work);primal_cached_control.rr=t.x;motion_store(40,1.0);atomicStore(&motion_control[35],0u);}storageBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
   for(var at=lane;at<actors;at+=128u){primal_dual_bound(primal_dual_index(at,compacted));}storageBarrier();var rr=0.0;for(var at=lane;at<actors;at+=128u){rr+=primal_dual_update(primal_dual_index(at,compacted));}primal_cached_sums[lane]=vec2<f32>(rr,0.0);primal_cached_reduce(lane);storageBarrier();
   if(lane==0u){primal_cached_control.beta=select(max(0.0,primal_cached_sums[0].x)/max(primal_cached_control.rr,1.0e-30),0.0,atomicLoad(&motion_control[35])!=0u||motion_value(40)<0.99999);}workgroupBarrier();
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let p=select(0.0,state.w+primal_cached_control.beta*state.y,primal_dual_active(i,k,state));primal_put(i,10u+2u*k,vec4<f32>(state.x,p,state.zw));}}storageBarrier();
  }
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);primal_put(i,5u,vec4<f32>(primal_free(i)+primal_cache[i],0.0,0.0));primal_dual_refine_rim(i);}storageBarrier();
 }
 for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,compacted);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let low=primal_dual_low(i,k);primal_put(i,10u+2u*k,vec4<f32>(state.x,low,state.zw));}}
}
@compute @workgroup_size(128) fn primal_dual_component(@builtin(local_invocation_index) lane:u32){primal_dual_solve(lane,false);}
@compute @workgroup_size(128) fn primal_dual_compact_component(@builtin(local_invocation_index) lane:u32){primal_dual_solve(lane,true);}
// Accelerated projected dual gradient candidate. Unlike bound-truncated CG,
// Block principal-pivot candidate: solve one complete free face before
// pivoting its violating multipliers together, rather than clipping each CG
// step at the first bound. Negative guesses never become physical impulses.
@compute @workgroup_size(128) fn primal_blockpivot_component(@builtin(local_invocation_index) lane:u32){
 let actors=atomicLoad(&motion_control[33]);
 for(var outer=0u;outer<3u;outer++){
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);var mask=0u;for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let state=primal_data(i,10u+2u*k);var velocity=primal_free(i)+primal_cache[i];if(row.x!=0xffffffffu){velocity-=primal_free(row.x)+primal_cache[row.x];}let residual=bitcast<f32>(row.w)-dot(bitcast<vec2<f32>>(row.yz),velocity);if(state.x>0.0||residual>=-max(0.00000001,motion.material.w/max(motion_value(8),1.0e-12))){mask|=1u<<k;}}primal_scratch[primal_slot(i,90u)]=vec4<u32>(mask,0u,0u,0u);}storageBarrier();
  for(var face=0u;face<8u;face++){
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){primal_dual_start(primal_dual_index(at,true));}storageBarrier();
   for(var cg=0u;cg<64u;cg++){
    for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,true);}workgroupBarrier();var sums=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){sums+=primal_dual_operator(primal_dual_index(at,true));}primal_cached_sums[lane]=sums;primal_cached_reduce(lane);
    if(lane==0u){let t=primal_cached_sums[0];if(cg==0u){primal_cached_control.initial_rr=t.x;}let epsilon=0.25*motion_solve_reserve()/max(motion_value(8),1.0e-12);let tolerance=max(1.0e-20,min(primal_cached_control.initial_rr*0.00001,epsilon*epsilon*0.5));let work=t.x>tolerance&&t.y>1.0e-24;primal_cached_control.work=select(0u,1u,work);primal_cached_control.alpha=select(0.0,t.x/max(t.y,1.0e-30),work);primal_cached_control.rr=t.x;motion_store(40,1.0);}storageBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
    var rr=0.0;for(var at=lane;at<actors;at+=128u){rr+=primal_dual_update(primal_dual_index(at,true));}primal_cached_sums[lane]=vec2<f32>(rr,0.0);primal_cached_reduce(lane);storageBarrier();if(lane==0u){primal_cached_control.beta=max(0.0,primal_cached_sums[0].x)/max(primal_cached_control.rr,1.0e-30);}workgroupBarrier();
    for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let p=select(0.0,state.w+primal_cached_control.beta*state.y,primal_dual_active(i,k,state));primal_put(i,10u+2u*k,vec4<f32>(state.x,p,state.zw));}}storageBarrier();
   }
   if(lane==0u){primal_cached_control.step_kind=0u;}workgroupBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();var changed=0.0;
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);let old_mask=primal_scratch[primal_slot(i,90u)].x;var mask=old_mask;for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let lambda=state.x+primal_dual_low(i,k);let bit=1u<<k;if((old_mask&bit)!=0u&&lambda<0.0){mask&=~bit;primal_dual_put_low(i,k,0.0);primal_put(i,10u+2u*k,vec4<f32>(0.0));}else if((old_mask&bit)==0u){let row=primal_row(i,k);var velocity=primal_free(i)+primal_cache[i];if(row.x!=0xffffffffu){velocity-=primal_free(row.x)+primal_cache[row.x];}let residual=bitcast<f32>(row.w)-dot(bitcast<vec2<f32>>(row.yz),velocity);if(residual>0.00000001){mask|=bit;}}}changed+=select(0.0,1.0,mask!=old_mask);primal_scratch[primal_slot(i,90u)].x=mask;}
   primal_cached_sums[lane]=vec2<f32>(changed,0.0);storageBarrier();primal_cached_reduce(lane);if(lane==0u){primal_cached_control.step_kind=select(0u,1u,primal_cached_sums[0].x>0.0);}workgroupBarrier();if(workgroupUniformLoad(&primal_cached_control.step_kind)==0u){break;}
  }
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);if(state.x+primal_dual_low(i,k)<0.0){primal_dual_put_low(i,k,0.0);primal_put(i,10u+2u*k,vec4<f32>(0.0));}}}storageBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_put(i,5u,vec4<f32>(primal_free(i)+primal_cache[i],0.0,0.0));primal_dual_refine_rim(i);}storageBarrier();
 }
 for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,primal_dual_low(i,k),state.zw));}}
}
// Accelerated projected dual gradient candidate. Unlike bound-truncated CG,
// this crosses active faces without global restarts. All output remains private.
fn primal_accelerated_gather(i:u32)->vec2<f32>{
 var value=vec4<f32>(0.0);for(var k=0u;k<primal_counts(i).y;k++){let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let state=primal_data(owner,10u+2u*(id%32u));let n=select(-1.0,1.0,owner==i)*bitcast<vec2<f32>>(row.yz);let product=n*state.y;let error=fma(n,vec2<f32>(state.y),-product)+n*state.w;let sum=motion_two_sum(value.xy,product);value=motion_two_sum(sum.xy,sum.zw+value.zw+error);}return value.xy+value.zw;
}
@compute @workgroup_size(128) fn primal_accelerated_component(@builtin(local_invocation_index) lane:u32){
 let actors=atomicLoad(&motion_control[33]);for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);atomicMax(&motion_control[34],primal_counts(i).y);}storageBarrier();
 // ||J J^T|| <= 2 max_degree max_row_norm_squared. Curved rows can
 // have a norm above one, so reduce that maximum rather than assuming it.
 var maximum=0.0;for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let n=bitcast<vec2<f32>>(primal_row(i,k).yz);maximum=max(maximum,dot(n,n));}}primal_cached_sums[lane]=vec2<f32>(maximum,0.0);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_cached_sums[lane].x=max(primal_cached_sums[lane].x,primal_cached_sums[lane+width].x);}workgroupBarrier();}
 if(lane==0u){primal_cached_control.alpha=0.49/max(1.0,f32(atomicLoad(&motion_control[34]))*primal_cached_sums[0].x);}workgroupBarrier();
 for(var outer=0u;outer<3u;outer++){
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,state.x,0.0,primal_dual_low(i,k)));}}storageBarrier();if(lane==0u){primal_cached_control.rr=1.0;}workgroupBarrier();
  for(var iteration=0u;iteration<512u;iteration++){
   if(lane==0u){let t=primal_cached_control.rr;let next=0.5*(1.0+sqrt(1.0+4.0*t*t));primal_cached_control.beta=(t-1.0)/next;primal_cached_control.rr=next;}workgroupBarrier();
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_accelerated_gather(i);}workgroupBarrier();var sums=vec2<f32>(0.0);
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var velocity=primal_free(i)+primal_cache[i];if(row.x!=0xffffffffu){velocity-=primal_free(row.x)+primal_cache[row.x];}let state=primal_data(i,10u+2u*k);let low=primal_dual_low(i,k);let residual=bitcast<f32>(row.w)-dot(n,velocity);let change=primal_cached_control.alpha*residual;let sum=motion_two_sum(vec2<f32>(state.y),vec2<f32>(change));var next=motion_two_sum(sum.xy,sum.zw+vec2<f32>(state.w+fma(primal_cached_control.alpha,residual,-change)));if(next.x+next.z<=0.0){next=vec4<f32>(0.0);}let difference=motion_two_sum(vec2<f32>(next.x),vec2<f32>(-state.x));let delta=motion_two_sum(difference.xy,difference.zw+vec2<f32>(next.z-low));let extrapolation=primal_cached_control.beta*delta.x;let momentum=motion_two_sum(vec2<f32>(next.x),vec2<f32>(extrapolation));let result=motion_two_sum(momentum.xy,momentum.zw+vec2<f32>(next.z+primal_cached_control.beta*delta.z+fma(primal_cached_control.beta,delta.x,-extrapolation)));primal_dual_put_low(i,k,next.z);primal_put(i,10u+2u*k,vec4<f32>(next.x,result.x,residual,result.z));let violation=select(max(0.0,residual),abs(residual),next.x>0.0);sums.x=max(sums.x,violation);sums.y+=delta.x*(state.y-next.x);}}
   primal_cached_sums[lane]=sums;storageBarrier();workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_cached_sums[lane].x=max(primal_cached_sums[lane].x,primal_cached_sums[lane+width].x);primal_cached_sums[lane].y+=primal_cached_sums[lane+width].y;}workgroupBarrier();}
   if(lane==0u){let tolerance=max(0.00000001,0.25*motion_solve_reserve()/max(motion_value(8),1.0e-12));primal_cached_control.work=select(0u,1u,primal_cached_sums[0].x>tolerance);primal_cached_control.step_kind=select(0u,1u,primal_cached_sums[0].y>0.0);if(primal_cached_control.step_kind!=0u){primal_cached_control.rr=1.0;}}workgroupBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
   if(workgroupUniformLoad(&primal_cached_control.step_kind)!=0u){for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,state.x,state.z,primal_dual_low(i,k)));}}storageBarrier();}
  }
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_put(i,5u,vec4<f32>(primal_free(i)+primal_cache[i],0.0,0.0));primal_dual_refine_rim(i);}storageBarrier();
 }
 for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,primal_dual_low(i,k),state.zw));}}
}
// MPRGP candidate: CG explores only the current free multiplier face. A
// proportioning step releases violated zero multipliers; a boundary hit is
// followed by a projected gradient expansion. Merely restarting ordinary CG
// at every bound hit is not this algorithm. See arXiv:2002.06077, Algorithms 1-4.
fn primal_mprgp_gradient(i:u32)->vec2<f32>{
 var result=vec2<f32>(0.0);for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var relative=primal_free(i)+primal_cache[i];if(row.x!=0xffffffffu){relative-=primal_free(row.x)+primal_cache[row.x];}let state=primal_data(i,10u+2u*k);let residual=bitcast<f32>(row.w)-dot(n,relative);let free=select(0.0,residual,state.x>0.0);let chopped=select(max(0.0,residual),0.0,state.x>0.0);primal_put(i,10u+2u*k,vec4<f32>(state.xy,residual,free));result+=vec2<f32>(free*free,chopped*chopped);}return result;
}
fn primal_mprgp_direction(i:u32,restart:bool){for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);var p=state.y;if(primal_cached_control.step_kind==1u){p=select(max(0.0,state.z),0.0,state.x>0.0);}else if(restart){p=state.w;}else{p=select(0.0,state.w+primal_cached_control.beta*p,state.x>0.0);}primal_put(i,10u+2u*k,vec4<f32>(state.x,p,state.zw));}}
fn primal_mprgp_operator(i:u32)->vec2<f32>{var result=vec2<f32>(0.0);for(var k=0u;k<primal_counts(i).x;k++){let row=primal_row(i,k);let n=bitcast<vec2<f32>>(row.yz);var delta=primal_cache[i];if(row.x!=0xffffffffu){delta-=primal_cache[row.x];}let state=primal_data(i,10u+2u*k);let ap=dot(n,delta);result+=vec2<f32>(state.z*state.y,state.y*ap);primal_put(i,10u+2u*k,vec4<f32>(state.xyz,ap));}return result;}
fn primal_mprgp_apply(i:u32,alpha:f32,expansion:bool){
 for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let direction=select(state.y,state.w,expansion);let change=alpha*direction;let sum=motion_two_sum(vec2<f32>(state.x),vec2<f32>(change));var parts=motion_two_sum(sum.xy,sum.zw+vec2<f32>(primal_dual_low(i,k)+fma(alpha,direction,-change)));
  // Roundoff at a maximal feasible bound must not leave a spurious tiny
  // positive multiplier forever on the wrong free face. Only private guesses
  // change; published positions still require continuous/endpoint admission.
  let rounding=0.000000476837158203125*(abs(state.x)+abs(change));if(parts.x+parts.z<=0.0||(!expansion&&change<0.0&&motion_value(40)<0.99999&&parts.x+parts.z<=rounding)){parts=vec4<f32>(0.0);}let next=parts.x;if((state.x>0.0)!=(next>0.0)){atomicStore(&motion_control[35],1u);}primal_dual_put_low(i,k,parts.z);primal_put(i,10u+2u*k,vec4<f32>(next,state.yzw));
 }
}
@compute @workgroup_size(128) fn primal_mprgp_component(@builtin(local_invocation_index) lane:u32){
 let actors=atomicLoad(&motion_control[33]);for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);atomicMax(&motion_control[34],primal_counts(i).y);}storageBarrier();
 for(var outer=0u;outer<3u;outer++){
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();var initial=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){initial+=primal_mprgp_gradient(primal_dual_index(at,true));}primal_cached_sums[lane]=initial;primal_cached_reduce(lane);if(lane==0u){primal_cached_control.free_rr=primal_cached_sums[0].x;primal_cached_control.chopped_rr=primal_cached_sums[0].y;primal_cached_control.initial_rr=primal_cached_sums[0].x+primal_cached_sums[0].y;primal_cached_control.beta=0.0;}storageBarrier();
  for(var iteration=0u;iteration<256u;iteration++){
   if(lane==0u){let epsilon=0.25*motion_solve_reserve()/max(motion_value(8),1.0e-12);let tolerance=max(1.0e-20,min(primal_cached_control.initial_rr*0.00001,epsilon*epsilon*0.5));primal_cached_control.work=select(0u,1u,primal_cached_control.free_rr+primal_cached_control.chopped_rr>tolerance);primal_cached_control.step_kind=select(0u,1u,primal_cached_control.chopped_rr>primal_cached_control.free_rr);motion_store(40,1.0);atomicStore(&motion_control[35],0u);}storageBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}
   for(var at=lane;at<actors;at+=128u){primal_mprgp_direction(primal_dual_index(at,true),iteration==0u||primal_cached_control.beta==0.0);}storageBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,true);}workgroupBarrier();var dots=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){dots+=primal_mprgp_operator(primal_dual_index(at,true));}primal_cached_sums[lane]=dots;primal_cached_reduce(lane);
   if(lane==0u){let t=primal_cached_sums[0];primal_cached_control.alpha=max(0.0,t.x)/max(t.y,1.0e-30);primal_cached_control.work=select(0u,1u,t.x>0.0&&t.y>1.0e-24);}workgroupBarrier();if(workgroupUniformLoad(&primal_cached_control.work)==0u){break;}for(var at=lane;at<actors;at+=128u){primal_dual_bound(primal_dual_index(at,true));}storageBarrier();
   for(var at=lane;at<actors;at+=128u){primal_mprgp_apply(primal_dual_index(at,true),primal_cached_control.alpha*motion_value(40),false);}storageBarrier();
   for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();var norms=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){norms+=primal_mprgp_gradient(primal_dual_index(at,true));}primal_cached_sums[lane]=norms;primal_cached_reduce(lane);storageBarrier();
   if(lane==0u){primal_cached_control.work=select(0u,1u,primal_cached_control.step_kind==0u&&motion_value(40)<0.99999);}workgroupBarrier();
   if(workgroupUniformLoad(&primal_cached_control.work)!=0u){
    let alpha=0.49/f32(max(1u,atomicLoad(&motion_control[34])));for(var at=lane;at<actors;at+=128u){primal_mprgp_apply(primal_dual_index(at,true),alpha,true);}storageBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();norms=vec2<f32>(0.0);for(var at=lane;at<actors;at+=128u){norms+=primal_mprgp_gradient(primal_dual_index(at,true));}primal_cached_sums[lane]=norms;primal_cached_reduce(lane);storageBarrier();
   }
   if(lane==0u){let next=primal_cached_sums[0];let restart=primal_cached_control.step_kind!=0u||motion_value(40)<0.99999||atomicLoad(&motion_control[35])!=0u;primal_cached_control.beta=select(next.x/max(primal_cached_control.free_rr,1.0e-30),0.0,restart);primal_cached_control.free_rr=next.x;primal_cached_control.chopped_rr=next.y;}workgroupBarrier();
  }
  for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_cache[i]=primal_dual_gather(i,false);}workgroupBarrier();for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);primal_put(i,5u,vec4<f32>(primal_free(i)+primal_cache[i],0.0,0.0));primal_dual_refine_rim(i);}storageBarrier();
 }
 for(var at=lane;at<actors;at+=128u){let i=primal_dual_index(at,true);for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,primal_dual_low(i,k),state.zw));}}
}
// Parallel implementation of the same QP. Dispatch boundaries synchronize
// storage globally; workgroup barriers are never used as a cross-group fence.
fn primal_dual_global_active(i:u32)->bool{return i<motion.counts.x&&motion_value(0)>0.0;}
@compute @workgroup_size(128) fn primal_dual_global_solution(@builtin(global_invocation_id) gid:vec3<u32>){if(primal_dual_global_active(gid.x)){primal_put(gid.x,2u,vec4<f32>(primal_dual_gather(gid.x,false),0.0,0.0));}}
@compute @workgroup_size(128) fn primal_dual_global_direction(@builtin(global_invocation_id) gid:vec3<u32>){if(primal_dual_global_active(gid.x)){primal_put(gid.x,2u,vec4<f32>(primal_dual_gather(gid.x,true),0.0,0.0));}}
@compute @workgroup_size(128) fn primal_dual_global_start(@builtin(global_invocation_id) gid:vec3<u32>){if(primal_dual_global_active(gid.x)){primal_dual_start(gid.x);}}
fn primal_dual_global_reduce(lane:u32,group:u32,value:vec2<f32>){primal_cached_sums[lane]=value;workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){primal_cached_sums[lane]+=primal_cached_sums[lane+width];}workgroupBarrier();}if(lane==0u){primal_put(group,1u,vec4<f32>(primal_cached_sums[0],0.0,0.0));}}
@compute @workgroup_size(128) fn primal_dual_global_operator(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){var value=vec2<f32>(0.0);if(primal_dual_global_active(gid.x)){value=primal_dual_operator(gid.x);}primal_dual_global_reduce(lane,group.x,value);}
@compute @workgroup_size(128) fn primal_dual_global_bound(@builtin(global_invocation_id) gid:vec3<u32>){if(primal_dual_global_active(gid.x)){primal_dual_bound(gid.x);}}
@compute @workgroup_size(128) fn primal_dual_global_update(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){var rr=0.0;if(primal_dual_global_active(gid.x)){rr=primal_dual_update(gid.x);}primal_dual_global_reduce(lane,group.x,vec2<f32>(rr,0.0));}
@compute @workgroup_size(128) fn primal_dual_global_conjugate(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(!primal_dual_global_active(i)){return;}for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);let p=select(0.0,state.w+motion_value(37)*state.y,primal_dual_active(i,k,state));primal_put(i,10u+2u*k,vec4<f32>(state.x,p,state.zw));}}
@compute @workgroup_size(128) fn primal_dual_global_refine(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(primal_dual_global_active(i)){primal_put(i,5u,vec4<f32>(primal_free(i)+primal_data(i,2u).xy,0.0,0.0));primal_dual_refine_rim(i);}}
@compute @workgroup_size(128) fn primal_dual_global_warm_end(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(!primal_dual_global_active(i)){return;}for(var k=0u;k<primal_counts(i).x;k++){let state=primal_data(i,10u+2u*k);primal_put(i,10u+2u*k,vec4<f32>(state.x,primal_dual_low(i,k),state.zw));}}
fn primal_dual_global_totals()->vec2<f32>{var total=vec2<f32>(0.0);for(var g=0u;g<(motion.counts.x+127u)/128u;g++){total+=primal_data(g,1u).xy;}return total;}
@compute @workgroup_size(1) fn primal_dual_global_begin(){atomicStore(&motion_control[39],0u);atomicStore(&motion_dispatch_args[3],atomicLoad(&motion_dispatch_args[0]));}
@compute @workgroup_size(1) fn primal_dual_global_alpha(){if(atomicLoad(&motion_dispatch_args[3])==0u){return;}let total=primal_dual_global_totals();if(atomicLoad(&motion_control[39])==0u){motion_store(41,total.x);}let epsilon=0.25*motion_solve_reserve()/max(motion_value(8),1.0e-12);let tolerance=max(1.0e-20,min(motion_value(41)*0.00001,epsilon*epsilon*0.5));let has_work=total.x>tolerance&&total.y>1.0e-24;motion_store(36,select(0.0,total.x/max(total.y,1.0e-30),has_work));motion_store(38,total.x);motion_store(40,1.0);atomicStore(&motion_control[35],0u);if(!has_work){atomicStore(&motion_dispatch_args[3],0u);}}
@compute @workgroup_size(1) fn primal_dual_global_beta(){if(atomicLoad(&motion_dispatch_args[3])==0u){return;}let rr=primal_dual_global_totals().x;let restart=atomicLoad(&motion_control[35])!=0u||motion_value(40)<0.99999;motion_store(37,select(max(0.0,rr)/max(motion_value(38),1.0e-30),0.0,restart));atomicAdd(&motion_control[39],1u);}
`;
