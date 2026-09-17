// Cached local normal-contact QP candidate. Liquid only; not release accepted.
export const CONTACT_CACHED_STAR_WGSL=/* wgsl */`
override CONTACT_STAR_COLOUR=0u;
override CONTACT_STAR_ITERATIONS=64u;
override CONTACT_STAR_CG=false;
override CONTACT_STAR_COMPACT=false;
@group(2) @binding(11) var<storage,read_write> contact_star_cells:array<u32>;
@group(2) @binding(12) var<storage,read_write> contact_star_args:array<atomic<u32>>;
@compute @workgroup_size(1) fn contact_star_reset(){
 for(var colour=0u;colour<9u;colour++){atomicStore(&contact_star_args[3u*colour],0u);atomicStore(&contact_star_args[3u*colour+1u],1u);atomicStore(&contact_star_args[3u*colour+2u],1u);}
}
@compute @workgroup_size(128) fn contact_star_compact(@builtin(global_invocation_id) gid:vec3<u32>){
 let bucket=gid.x;if(bucket>=motion.grid.x*motion.grid.y||motion_value(0)<=0.0||atomicLoad(&motion_control[6])!=0u){return;}
 let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);var has_rows=false;for(var k=0u;k<count;k++){let id=atomicLoad(&motion_items[bucket*motion.grid.z+k]);has_rows=has_rows||motion_actors[id*128u+4u].x>0u;}if(!has_rows){return;}
 let cell=vec2<u32>(bucket%motion.grid.x,bucket/motion.grid.x);let colour=cell.x%3u+3u*(cell.y%3u);let columns=(motion.grid.x+2u)/3u;let stride=columns*((motion.grid.y+2u)/3u);let at=atomicAdd(&contact_star_args[3u*colour],1u);if(at>=stride){atomicOr(&motion_control[6],1024u);return;}contact_star_cells[colour*stride+at]=bucket;
}
@compute @workgroup_size(128) fn contact_star_rim_seed(@builtin(global_invocation_id) gid:vec3<u32>){
 if(gid.x<motion.counts.x){motion_actors[gid.x*128u+126u]=vec4<u32>(0u);}
}
// One convex dual-coordinate/Dykstra disk update. It changes private velocity,
// not position. Circle convexity certifies the straight path only AFTER the
// complete coupled solve; independent swept/rounded admission still follows.
@compute @workgroup_size(128) fn contact_star_rim_project(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)<=0.0||atomicLoad(&motion_control[6])!=0u){return;}
 let h=max(motion_value(8),1.0e-12);let state=motion_state(i);let old=bitcast<vec4<f32>>(motion_actors[i*128u+126u]).xy;let free=state.zw-old;
 let current=motion_position_parts(i,0.0);let endpoint=motion_offset_parts(motion_compensated(current.xy,current.zw,free*h));let radius=WHEEL_RADIUS-motion.minimum.z-WHEEL_BAR_HALF_WIDTH-motion_rim_error()-2.0*motion.material.w;
 let gap=motion_radius_gap_parts(endpoint,radius);var impulse=vec2<f32>(0.0);if(gap>0.0){let norm=max(motion_length(endpoint.xy),1.0e-20);let normal=(endpoint.xy+endpoint.zw)/norm;impulse=-normal*(gap/h);}
 motion_actors[i*128u+126u]=bitcast<vec4<u32>>(vec4<f32>(impulse,0.0,0.0));motion_write(i,vec4<f32>(state.xy,free+impulse));
}
var<workgroup> contact_star_id:array<u32,36>;
var<workgroup> contact_star_count:array<u32,9>;
var<workgroup> contact_star_incident:array<vec4<u32>,36>;
var<workgroup> contact_star_degree:array<u32,36>;
var<workgroup> contact_star_header:array<vec4<f32>,128>;
var<workgroup> contact_star_x:array<f32,128>;
var<workgroup> contact_star_y:array<f32,128>;
var<workgroup> contact_star_base:array<vec2<f32>,36>;
var<workgroup> contact_star_velocity:array<vec2<f32>,36>;
var<workgroup> contact_star_sums:array<vec2<f32>,128>;
var<workgroup> contact_star_residual:array<vec2<f32>,128>;
var<workgroup> contact_star_changed:atomic<u32>;
var<workgroup> contact_star_bucket:u32;
struct ContactStarControl {work:u32,restart:u32,alpha:f32,beta:f32,t:f32,gradient_step:f32}
var<workgroup> contact_star_control:ContactStarControl;
fn contact_star_local(cell:vec2<i32>,origin:vec2<i32>,id:u32)->u32{
 let local=cell-origin;if(any(local<vec2<i32>(0))||any(local>=vec2<i32>(3))){return 255u;}let bucket=u32(local.x+3*local.y);for(var k=0u;k<contact_star_count[bucket];k++){let at=bucket*4u+k;if(contact_star_id[at]==id){return at;}}return 255u;
}
fn contact_star_gather(at:u32,momentum:bool)->vec2<f32>{
 var sum=vec4<f32>(0.0);let mask=contact_star_incident[at];for(var word=0u;word<4u;word++){var bits=mask[word];while(bits!=0u){let row=word*32u+firstTrailingBit(bits);bits&=bits-1u;let h=contact_star_header[row];let owner=u32(h.w)&255u;let n=select(-1.0,1.0,owner==at)*h.xy;let lambda=select(contact_star_x[row],contact_star_y[row],momentum);let product=n*lambda;let value=motion_two_sum(sum.xy,product);sum=motion_two_sum(value.xy,value.zw+sum.zw+fma(n,vec2<f32>(lambda),-product));}}return sum.xy+sum.zw;
}
@compute @workgroup_size(128) fn contact_star_project(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 if(lane==0u){contact_star_control.work=select(0u,1u,motion_value(0)>0.0&&atomicLoad(&motion_control[6])==0u);}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){return;}
 let columns=(motion.grid.x+2u)/3u;var cell=vec2<u32>((group.x%columns)*3u+CONTACT_STAR_COLOUR%3u,(group.x/columns)*3u+CONTACT_STAR_COLOUR/3u);if(CONTACT_STAR_COMPACT){let stride=columns*((motion.grid.y+2u)/3u);if(lane==0u){contact_star_bucket=contact_star_cells[CONTACT_STAR_COLOUR*stride+group.x];}workgroupBarrier();let bucket=workgroupUniformLoad(&contact_star_bucket);cell=vec2<u32>(bucket%motion.grid.x,bucket/motion.grid.x);}if(any(cell>=motion.grid.xy)){return;}let origin=vec2<i32>(cell)-vec2<i32>(1);
 if(lane<9u){let c=origin+vec2<i32>(i32(lane%3u),i32(lane/3u));var count=0u;if(motion_valid(c)){count=atomicLoad(&motion_cells[motion_bucket(c)]);}if(count>4u){atomicOr(&motion_control[6],64u);}contact_star_count[lane]=min(count,4u);for(var k=0u;k<min(count,4u);k++){let at=lane*4u+k;let id=atomicLoad(&motion_items[motion_bucket(c)*motion.grid.z+k]);contact_star_id[at]=id;contact_star_velocity[at]=motion_state(id).zw;}}workgroupBarrier();
 if(lane==0u){contact_star_control.work=contact_star_count[4];}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){return;}
 let slot=lane/32u;let k=lane%32u;var h=vec4<f32>(0.0,0.0,0.0,-1.0);var lambda=0.0;if(slot<contact_star_count[4]){let id=contact_star_id[16u+slot];let count=motion_actors[id*128u+4u].x;if(k<count){let at=dense_row(id,k);let row=motion_actors[at];let mu=bitcast<f32>(row.w);if(mu!=0.0){atomicOr(&motion_control[6],256u);}var other=255u;if(row.x!=0xffffffffu){other=contact_star_local(motion_cell(motion_state(row.x).xy),origin,row.x);if(other==255u){atomicOr(&motion_control[6],128u);}}h=vec4<f32>(bitcast<vec2<f32>>(row.yz),bitcast<f32>(motion_actors[at+2u].x),f32((16u+slot)|(other<<8u)));lambda=bitcast<f32>(motion_actors[at+1u].x);}}
 contact_star_header[lane]=h;contact_star_x[lane]=lambda;contact_star_y[lane]=lambda;workgroupBarrier();
 // A block already satisfying its dual KKT conditions has no correction.
 // Check before building its incident matrix, without dropping any row.
 var initial_violation=0.0;if(h.w>=0.0){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}let r=h.z-dot(h.xy,relative);initial_violation=select(max(0.0,r),abs(r),lambda>0.0);}contact_star_sums[lane]=vec2<f32>(initial_violation,0.0);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane].x=max(contact_star_sums[lane].x,contact_star_sums[lane+width].x);}workgroupBarrier();}
 if(lane==0u){let tolerance=max(0.00000001,0.125*motion.material.w/max(motion_value(8),1.0e-12));contact_star_control.work=select(0u,1u,contact_star_sums[0].x>tolerance);}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){return;}
 var degree=0u;if(lane<36u){var mask=vec4<u32>(0u);let bucket=lane/4u;if(lane%4u<contact_star_count[bucket]){for(var row=0u;row<128u;row++){let word=u32(contact_star_header[row].w);if(contact_star_header[row].w<0.0){continue;}if((word&255u)==lane||((word>>8u)&255u)==lane){mask[row/32u]|=1u<<(row%32u);degree++;}}}if(degree>64u){atomicOr(&motion_control[6],32u);}contact_star_incident[lane]=mask;contact_star_degree[lane]=degree;}workgroupBarrier();
 var norm=0.0;if(h.w>=0.0){norm=dot(h.xy,h.xy);}contact_star_sums[lane]=vec2<f32>(f32(degree),norm);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane]=max(contact_star_sums[lane],contact_star_sums[lane+width]);}workgroupBarrier();}
 if(lane==0u){contact_star_control.alpha=.49/max(1.0,contact_star_sums[0].x*contact_star_sums[0].y);contact_star_control.gradient_step=contact_star_control.alpha;contact_star_control.t=1.0;}workgroupBarrier();
 if(lane<36u&&lane%4u<contact_star_count[lane/4u]){contact_star_base[lane]=contact_star_velocity[lane]-contact_star_gather(lane,false);}workgroupBarrier();
 if(CONTACT_STAR_CG){
  // Local face_active-face CG. All matrix products use cached shared-memory rows;
  // clipping or a face change restarts conjugacy. This is a private proposal.
  var face_active=false;var z=0.0;
  if(h.w>=0.0){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}let r=h.z-dot(h.xy,relative);let diagonal=select(2.0,1.0,other==255u)*dot(h.xy,h.xy);contact_star_residual[lane]=vec2<f32>(r,diagonal);face_active=contact_star_x[lane]>0.0||r>0.0;z=select(0.0,r/max(diagonal,1.0e-20),face_active);contact_star_y[lane]=z;}
  workgroupBarrier();
  for(var iteration=0u;iteration<128u;iteration++){
   if(lane<36u&&lane%4u<contact_star_count[lane/4u]){contact_star_velocity[lane]=contact_star_gather(lane,true);}workgroupBarrier();
   var product=0.0;if(h.w>=0.0&&face_active){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}product=dot(h.xy,relative);}
   contact_star_sums[lane]=vec2<f32>(contact_star_residual[lane].x*z,contact_star_y[lane]*product);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane]+=contact_star_sums[lane+width];}workgroupBarrier();}
   if(lane==0u){let s=contact_star_sums[0];contact_star_control.work=select(0u,1u,s.x>1.0e-20);contact_star_control.alpha=s.x/max(s.y,1.0e-30);contact_star_control.beta=select(0.0,1.0,s.y<=1.0e-24);contact_star_control.t=s.x;atomicStore(&contact_star_changed,0u);}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){break;}
   var fraction=1.0;if(h.w>=0.0&&contact_star_y[lane]<0.0&&contact_star_control.beta==0.0){fraction=min(1.0,contact_star_x[lane]/max(-contact_star_control.alpha*contact_star_y[lane],1.0e-30));}contact_star_sums[lane]=vec2<f32>(fraction,0.0);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane].x=min(contact_star_sums[lane].x,contact_star_sums[lane+width].x);}workgroupBarrier();}
   if(lane==0u){let f=contact_star_sums[0].x;let projected=contact_star_control.beta==1.0||f<.00001;contact_star_control.alpha=select(contact_star_control.alpha*f,contact_star_control.gradient_step,projected);contact_star_control.beta=select(0.0,1.0,projected);contact_star_control.restart=select(0u,1u,f<.99999||projected);}workgroupBarrier();
   // A zero-length free-face step must leave that face, not repeat forever.
   // Projected gradient releases blocking zero multipliers together.
   if(h.w>=0.0){let old=contact_star_x[lane];let direction=select(contact_star_y[lane],contact_star_residual[lane].x,contact_star_control.beta==1.0);let change=contact_star_control.alpha*direction;var next=max(0.0,fma(contact_star_control.alpha,direction,old));if(contact_star_control.restart!=0u&&change<0.0&&next<=0.000000476837158203125*(abs(old)+abs(change))){next=0.0;}contact_star_x[lane]=next;}workgroupBarrier();
   if(lane<36u&&lane%4u<contact_star_count[lane/4u]){contact_star_velocity[lane]=contact_star_base[lane]+contact_star_gather(lane,false);}workgroupBarrier();
   var kkt=0.0;if(h.w>=0.0){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}let r=h.z-dot(h.xy,relative);let next_active=contact_star_x[lane]>0.0||r>0.0;if(next_active!=face_active){atomicOr(&contact_star_changed,1u);}face_active=next_active;contact_star_residual[lane].x=r;z=select(0.0,r/max(contact_star_residual[lane].y,1.0e-20),face_active);kkt=select(max(0.0,r),abs(r),contact_star_x[lane]>0.0);}
   contact_star_sums[lane]=vec2<f32>(contact_star_residual[lane].x*z,kkt);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane].x+=contact_star_sums[lane+width].x;contact_star_sums[lane].y=max(contact_star_sums[lane].y,contact_star_sums[lane+width].y);}workgroupBarrier();}
   if(lane==0u){let s=contact_star_sums[0];let tolerance=max(0.00000001,0.125*motion.material.w/max(motion_value(8),1.0e-12));contact_star_control.beta=select(s.x/max(contact_star_control.t,1.0e-30),0.0,contact_star_control.restart!=0u||atomicLoad(&contact_star_changed)!=0u);contact_star_control.work=select(0u,1u,s.y>tolerance);}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){break;}
   contact_star_y[lane]=select(0.0,fma(contact_star_control.beta,contact_star_y[lane],z),face_active);workgroupBarrier();
  }
 }
 // CG termination is not a KKT certificate. Recompute the actual cached
 // velocity and use projected-gradient cleanup when the face solve is short.
 if(CONTACT_STAR_CG){
  if(lane<36u&&lane%4u<contact_star_count[lane/4u]){contact_star_velocity[lane]=contact_star_base[lane]+contact_star_gather(lane,false);}workgroupBarrier();
  var violation=0.0;if(h.w>=0.0){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}let r=h.z-dot(h.xy,relative);violation=select(max(0.0,r),abs(r),contact_star_x[lane]>0.0);}contact_star_sums[lane]=vec2<f32>(violation,0.0);workgroupBarrier();for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane].x=max(contact_star_sums[lane].x,contact_star_sums[lane+width].x);}workgroupBarrier();}
  if(lane==0u){let tolerance=max(0.00000001,0.125*motion.material.w/max(motion_value(8),1.0e-12));contact_star_control.work=select(0u,1u,contact_star_sums[0].x>tolerance);contact_star_control.alpha=contact_star_control.gradient_step;contact_star_control.t=1.0;}contact_star_y[lane]=contact_star_x[lane];workgroupBarrier();
 }
 if(!CONTACT_STAR_CG||workgroupUniformLoad(&contact_star_control.work)!=0u){for(var iteration=0u;iteration<CONTACT_STAR_ITERATIONS;iteration++){
  if(lane==0u){let t=contact_star_control.t;let next=.5*(1.0+sqrt(1.0+4.0*t*t));contact_star_control.beta=(t-1.0)/next;contact_star_control.t=next;}workgroupBarrier();
  if(lane<36u&&lane%4u<contact_star_count[lane/4u]){contact_star_velocity[lane]=contact_star_base[lane]+contact_star_gather(lane,true);}workgroupBarrier();
  var violation=0.0;var restart=0.0;if(h.w>=0.0){let word=u32(h.w);let owner=word&255u;let other=(word>>8u)&255u;var relative=contact_star_velocity[owner];if(other!=255u){relative-=contact_star_velocity[other];}let residual=h.z-dot(h.xy,relative);let old=contact_star_x[lane];let old_y=contact_star_y[lane];let next=max(0.0,fma(contact_star_control.alpha,residual,old_y));contact_star_x[lane]=next;contact_star_y[lane]=fma(contact_star_control.beta,next-old,next);violation=select(max(0.0,residual),abs(residual),next>0.0);restart=(next-old)*(old_y-next);}contact_star_sums[lane]=vec2<f32>(violation,restart);workgroupBarrier();
  for(var width=64u;width>0u;width/=2u){if(lane<width){contact_star_sums[lane].x=max(contact_star_sums[lane].x,contact_star_sums[lane+width].x);contact_star_sums[lane].y+=contact_star_sums[lane+width].y;}workgroupBarrier();}
  if(lane==0u){let tolerance=max(0.00000001,0.125*motion.material.w/max(motion_value(8),1.0e-12));contact_star_control.work=select(0u,1u,contact_star_sums[0].x>tolerance);contact_star_control.restart=select(0u,1u,contact_star_sums[0].y>0.0);if(contact_star_control.restart!=0u){contact_star_control.t=1.0;}}workgroupBarrier();if(workgroupUniformLoad(&contact_star_control.work)==0u){break;}if(workgroupUniformLoad(&contact_star_control.restart)!=0u){contact_star_y[lane]=contact_star_x[lane];}workgroupBarrier();
 }}
 if(lane<36u&&lane%4u<contact_star_count[lane/4u]){let id=contact_star_id[lane];let state=motion_state(id);let velocity=contact_star_base[lane]+contact_star_gather(lane,false);motion_write(id,vec4<f32>(state.xy,velocity));}
 if(h.w>=0.0){let id=contact_star_id[16u+slot];let at=dense_row(id,k)+1u;var p=motion_actors[at];p.x=bitcast<u32>(contact_star_x[lane]);motion_actors[at]=p;}
}
`;
