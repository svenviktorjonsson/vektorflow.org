// Domain-decomposed velocity projection candidate. Not release acceptance.
export const CONTACT_TILED_PROJECTION_WGSL=/* wgsl */`
override CONTACT_TILE_COLOUR=0u;
override CONTACT_TILE_ITERATIONS=32u;
var<workgroup> contact_tile_velocity:array<vec2<f32>,400>;
var<workgroup> contact_tile_id:array<u32,400>;
var<workgroup> contact_tile_count:array<u32,100>;
var<workgroup> contact_tile_work:u32;
@compute @workgroup_size(128) fn contact_tiled_warm(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}var impulse=vec4<f32>(0.0);
 for(var k=0u;k<primal_counts(i).y;k++){let id=primal_incident(i,k);let owner=id/32u;let row=primal_row(owner,id%32u);let n=select(-1.0,1.0,owner==i)*bitcast<vec2<f32>>(row.yz);let lambda=primal_data(owner,10u+2u*(id%32u)).x;let product=n*lambda;let sum=motion_two_sum(impulse.xy,product);impulse=motion_two_sum(sum.xy,sum.zw+impulse.zw+fma(n,vec2<f32>(lambda),-product));}
 for(var k=0u;k<primal_counts(i).x;k++){let at=dense_row(i,k)+1u;var p=motion_actors[at];p.x=bitcast<u32>(primal_data(i,10u+2u*k).x);motion_actors[at]=p;}
 let state=motion_state(i);motion_write(i,vec4<f32>(state.xy,primal_free(i)+impulse.xy+impulse.zw));
}
@compute @workgroup_size(128) fn contact_tiled_save(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||motion_value(0)==0.0){return;}for(var k=0u;k<primal_counts(i).x;k++){let at=dense_row(i,k);let normal=bitcast<vec4<f32>>(motion_actors[at+1u]).x;let tangent=bitcast<vec4<f32>>(motion_actors[at+2u]).z;primal_put(i,10u+2u*k,vec4<f32>(normal,0.0,tangent,0.0));}
}
fn contact_tile_local(cell:vec2<i32>,origin:vec2<i32>,id:u32)->u32{
 let local=cell-origin;if(any(local<vec2<i32>(0))||any(local>=vec2<i32>(10))){return 0xffffffffu;}let bucket=u32(local.x+10*local.y);
 for(var k=0u;k<contact_tile_count[bucket];k++){let at=bucket*4u+k;if(contact_tile_id[at]==id){return at;}}return 0xffffffffu;
}
@compute @workgroup_size(128) fn contact_tiled_project(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 if(lane==0u){contact_tile_work=select(0u,1u,motion_value(0)>0.0&&atomicLoad(&motion_control[6])==0u);}workgroupBarrier();if(workgroupUniformLoad(&contact_tile_work)==0u){return;}
 let tiles=(motion.grid.x+7u)/8u;let columns=(tiles+2u)/3u;let tile=vec2<u32>((group.x%columns)*3u+CONTACT_TILE_COLOUR%3u,(group.x/columns)*3u+CONTACT_TILE_COLOUR/3u);
 if(any(tile>=vec2<u32>(tiles))){return;}let origin=vec2<i32>(tile*8u)-vec2<i32>(1);
 if(lane<100u){let cell=origin+vec2<i32>(i32(lane%10u),i32(lane/10u));var count=0u;if(motion_valid(cell)){count=atomicLoad(&motion_cells[motion_bucket(cell)]);}
  // A static cell has width 1.1 diameters. Dividing it into four squares
  // gives diagonal .55*sqrt(2)<1 diameter; exclusion implies <=4 centres.
  // Never truncate a violation of that proven packing bound.
  if(count>4u){atomicOr(&motion_control[6],64u);}contact_tile_count[lane]=min(count,4u);
  for(var k=0u;k<min(count,4u);k++){let id=atomicLoad(&motion_items[motion_bucket(cell)*motion.grid.z+k]);contact_tile_id[lane*4u+k]=id;contact_tile_velocity[lane*4u+k]=motion_state(id).zw;}
 }workgroupBarrier();
 if(lane==0u){var count=0u;for(var k=0u;k<100u;k++){count+=contact_tile_count[k];}contact_tile_work=count;}workgroupBarrier();if(workgroupUniformLoad(&contact_tile_work)==0u){return;}
 for(var iteration=0u;iteration<CONTACT_TILE_ITERATIONS;iteration++){
  for(var colour=0u;colour<9u;colour++){
   if(lane<100u){let local=vec2<u32>(lane%10u,lane/10u);
    // Owners are interior cells; their complete one-cell neighborhoods are
    // loaded in the halo. Nine mini-colours have disjoint write neighborhoods.
    if(all(local>=vec2<u32>(1))&&all(local<=vec2<u32>(8))&&local.x%3u+3u*(local.y%3u)==colour){
     for(var slot=0u;slot<contact_tile_count[lane];slot++){let own_at=lane*4u+slot;let i=contact_tile_id[own_at];let rows=motion_actors[i*128u+4u].x;
      for(var k=0u;k<rows;k++){let at=dense_row(i,k);let row=motion_actors[at];let n=bitcast<vec2<f32>>(row.yz);let t=vec2<f32>(-n.y,n.x);var p=bitcast<vec4<f32>>(motion_actors[at+1u]);var q=bitcast<vec4<f32>>(motion_actors[at+2u]);var other_at=0xffffffffu;var other=vec2<f32>(0.0);
       if(row.x!=0xffffffffu){other_at=contact_tile_local(motion_cell(motion_state(row.x).xy),origin,row.x);if(other_at==0xffffffffu){atomicOr(&motion_control[6],128u);continue;}other=contact_tile_velocity[other_at];}
       let relative=contact_tile_velocity[own_at]-other;let diagonal=select(2.0,1.0,row.x==0xffffffffu)*dot(n,n);
       let normal=max(0.0,p.x+(bitcast<f32>(motion_actors[at+2u].x)-dot(n,relative))/max(diagonal,1.0e-20));let bound=bitcast<f32>(row.w)*normal;
       var tangent_target=0.0;if(row.x==0xffffffffu){let free=bitcast<vec4<f32>>(motion_actors[i*128u+2u]).zw;tangent_target=q.w+dot(t,free);}
       let tangent=clamp(q.z+(tangent_target-dot(t,relative))/max(diagonal,1.0e-20),-bound,bound);let impulse=n*(normal-p.x)+t*(tangent-q.z);p.x=normal;q.z=tangent;
       contact_tile_velocity[own_at]+=impulse;if(other_at!=0xffffffffu){contact_tile_velocity[other_at]-=impulse;}motion_actors[at+1u]=bitcast<vec4<u32>>(p);motion_actors[at+2u]=bitcast<vec4<u32>>(q);
      }
     }
    }
   // Only this owner invocation reads/writes its multiplier records. Neighbor
   // coupling uses shared velocities, not those records: a shared-memory
   // fence suffices here. Dispatch boundaries fence the next macro-colour.
   }workgroupBarrier();
  }
 }
 // Same macro-colour tile windows are 24 cells apart and only 10 wide.
 // Halo commits cannot race another workgroup. These are PRIVATE velocities;
 // physical positions are still committed only by full continuous admission.
 for(var at=lane;at<400u;at+=128u){let cell=at/4u;if(at%4u<contact_tile_count[cell]){let id=contact_tile_id[at];let state=motion_state(id);motion_write(id,vec4<f32>(state.xy,contact_tile_velocity[at]));}}
}
`;
