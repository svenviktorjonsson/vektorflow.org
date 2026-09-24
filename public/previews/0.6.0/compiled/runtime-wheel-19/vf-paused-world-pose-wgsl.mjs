// Frozen World configuration editing, not a time-integration/contact shortcut.
// An immutable captured configuration is transformed, then independently
// checked against actual rounded endpoints before any physical state is written.
export const PAUSED_WORLD_POSE_WGSL = /* wgsl */`
@group(2) @binding(13) var<storage,read_write> paused_pose:array<vec4<f32>>;
fn paused_slot(i:u32,field:u32)->u32{return 1u+i*4u+field;}
fn paused_parts(i:u32)->vec4<f32>{return vec4<f32>(paused_pose[paused_slot(i,2u)].xy,paused_pose[paused_slot(i,3u)].xy);}
@compute @workgroup_size(128) fn paused_capture(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x){return;}let parts=motion_position_parts(i,0.0);let state=motion_state(i);
 paused_pose[paused_slot(i,0u)]=vec4<f32>(parts.xy,state.zw);paused_pose[paused_slot(i,1u)]=vec4<f32>(parts.zw,0.0,0.0);
 if(i==0u){paused_pose[0]=vec4<f32>(motion_value(1),0.0,0.0,0.0);}
}
@compute @workgroup_size(1) fn paused_prepare(){motion_store(0,1.0);atomicStore(&motion_control[11],0u);}
@compute @workgroup_size(128) fn paused_propose(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x){return;}let reference=paused_pose[paused_slot(i,0u)];let low=paused_pose[paused_slot(i,1u)].xy;
 let parts=motion_rotated_parts(vec4<f32>(reference.xy,low),motion.wheel.z);
 paused_pose[paused_slot(i,2u)]=vec4<f32>(parts.xy,motion_rotate(reference.zw,motion.wheel.z));paused_pose[paused_slot(i,3u)]=vec4<f32>(parts.zw,0.0,0.0);
}
@compute @workgroup_size(128) fn paused_validate(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x){return;}let parts=paused_parts(i);let offset=motion_offset_parts(parts);let pose=paused_pose[0].x+motion.wheel.z;
 var safe=all(abs(parts)<vec4<f32>(1.0e10))&&motion_rim_gap_parts(offset)>=0.0;
 for(var k=0u;k<motion.counts.w;k++){safe=safe&&motion_bar_gap_parts(offset,pose,baffle(k))>=0.0;}
 // Rigidly transformed close pairs were close in the captured/current state.
 // The unchanged static grid is conservative for this isometry; validate the
 // actual proposed high+low positions, not the ideal rotation's distances.
 let cell=motion_cell(motion_state(i).xy);
 for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let c=cell+vec2<i32>(x,y);if(!motion_valid(c)){continue;}let bucket=motion_bucket(c);let count=min(atomicLoad(&motion_cells[bucket]),motion.grid.z);
  for(var slot=0u;slot<count;slot++){let j=atomicLoad(&motion_items[bucket*motion.grid.z+slot]);if(j<=i){continue;}let other=paused_parts(j);let difference=motion_compensated(parts.xy,parts.zw,-other.xy);let separation=motion_two_sum(difference.xy,difference.zw-other.zw);safe=safe&&motion_pair_gap_parts(separation)>=0.0;}
 }}
 if(!safe){atomicStore(&motion_control[11],1u);}
}
@compute @workgroup_size(128) fn paused_commit(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=motion.counts.x||atomicLoad(&motion_control[11])!=0u){return;}let state=paused_pose[paused_slot(i,2u)];let low=paused_pose[paused_slot(i,3u)].xy;
 motion_particles[i*motion.counts.y]=bitcast<vec4<u32>>(state);motion_actors[i*128u+122u]=bitcast<vec4<u32>>(vec4<f32>(low,low));motion_precision_output[i]=low;
}
@compute @workgroup_size(1) fn paused_finish(){
 if(atomicLoad(&motion_control[11])==0u){let pose=paused_pose[0].x+motion.wheel.z;motion_store(1,pose-6.28318530718*floor((pose+3.14159265359)/6.28318530718));motion_store(12,motion.wheel.z-paused_pose[0].y);paused_pose[0].y=motion.wheel.z;}else{motion_store(12,0.0);}
 motion_store(0,0.0);motion_store(13,0.0);motion_store(54,0.0);motion_store(60,0.0);motion_store(62,0.0);atomicStore(&motion_control[7],0u);atomicStore(&motion_control[17],0u);atomicStore(&motion_control[18],0u);atomicStore(&motion_control[57],0u);
}
`;
