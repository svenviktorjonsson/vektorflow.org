// Geometry-only acceleration. Positions and the boundary pose are immutable
// throughout one force prediction. Never reuse this packet after admission.
export const LIQUID_PRESSURE_GEOMETRY_CACHE_WGSL=/* wgsl */`
@group(2) @binding(11) var<storage,read_write> pressure_geometry:array<vec4<u32>>;
fn pressure_geometry_slot(i:u32,k:u32)->u32{return k*params.counts.x+i;}
@compute @workgroup_size(128) fn preventive_cache_pressure_geometry(@builtin(global_invocation_id) gid:vec3<u32>){
 let index=gid.x;if(index>=params.counts.x||!motion_force_ready()){return;}
 let position=particles[index].position;let inverse_density=1.0/params.material.x;let inverse_mass=1.0/params.fluid.w;
 var density=params.fluid.w*sample_kernel(vec2<f32>(0.0)).weight;var center_gradient=vec2<f32>(0.0);var norm_sum=0.0;var neighbor_count=0u;let center_cell=cell_coordinate(position);
 for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let cell=center_cell+vec2<i32>(x,y);if(!valid_cell(cell)){continue;}let bucket=cell_index(cell);let count=min(atomicLoad(&cell_counts[bucket]),params.counts.w);
  for(var slot=0u;slot<count;slot++){let other=atomicLoad(&cell_items[bucket*params.counts.w+slot]);if(other>=params.counts.x||other==index||!particle_occupies_cell(other,cell)){continue;}let kernel=sample_kernel(position-particles[other].position);if(kernel.supported==0u){continue;}let gradient=params.fluid.w*inverse_density*kernel.gradient;
   // At most nine buckets, each with the declared validated occupancy. The
   // adapter allocates all 9*occupancy slots; no neighbor can be truncated.
   pressure_geometry[pressure_geometry_slot(index,2u+neighbor_count)]=vec4<u32>(other,bitcast<vec2<u32>>(gradient),0u);neighbor_count++;
   density+=params.fluid.w*kernel.weight;center_gradient+=gradient;norm_sum+=inverse_mass*dot(gradient,gradient);
  }
 }}
 var boundary_gradient=vec2<f32>(0.0);
 if(valid_solid_cell(center_cell)){let bucket=solid_cell_index(center_cell);let count=boundary_cell_count(bucket);for(var slot=0u;slot<count;slot++){let other=boundary_cell_item(bucket,slot);if(other>=params.limits.z){continue;}let solid=boundary_particle(other);let kernel=sample_kernel(position-solid.xy);if(kernel.supported==0u){continue;}let gradient=solid.z*kernel.gradient;density+=params.material.x*solid.z*kernel.weight;center_gradient+=gradient;boundary_gradient+=gradient;}}
 let wheel=wheel_boundary_support(position);density+=params.material.x*wheel.density_factor;center_gradient+=wheel.gradient;boundary_gradient+=wheel.gradient;
 let offset=position-params.terrain.zw;let wall_velocity=params.terrain.y*vec2<f32>(-offset.y,offset.x);let wall_rate=dot(wheel.gradient,wall_velocity);let denominator=max(inverse_mass*dot(center_gradient,center_gradient)+norm_sum,1.0e-8);
 pressure_geometry[pressure_geometry_slot(index,0u)]=bitcast<vec4<u32>>(vec4<f32>(density,denominator,center_gradient));
 pressure_geometry[pressure_geometry_slot(index,1u)]=vec4<u32>(bitcast<vec2<u32>>(boundary_gradient),bitcast<u32>(wall_rate),neighbor_count);
}
fn cached_pressure_sample(index:u32)->ConstraintSample{
 let fixed=bitcast<vec4<f32>>(pressure_geometry[pressure_geometry_slot(index,0u)]);let boundary=pressure_geometry[pressure_geometry_slot(index,1u)];var neighbor_velocity=0.0;
 for(var k=0u;k<boundary.w;k++){let row=pressure_geometry[pressure_geometry_slot(index,2u+k)];neighbor_velocity+=dot(bitcast<vec2<f32>>(row.yz),particles[row.x].velocity);}
 let rate=dot(fixed.zw,particles[index].velocity)-neighbor_velocity-bitcast<f32>(boundary.z);
 return ConstraintSample(fixed.x,rate,fixed.y,boundary.w,fixed.zw);
}
@compute @workgroup_size(128) fn preventive_cached_divergence_lambda(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=params.counts.x||!motion_force_ready()){return;}let sample=cached_pressure_sample(i);let residual=max(0.0,sample.density_rate-params.material.y);particles[i].lambda=-residual/sample.denominator;particles[i].density=sample.density;particles[i].neighbor_count=sample.neighbor_count;}
@compute @workgroup_size(128) fn preventive_cached_density_lambda(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=params.counts.x||!motion_force_ready()){return;}let sample=cached_pressure_sample(i);let error=sample.density/params.material.x-1.0;let residual=max(0.0,error+params.fluid.x*sample.density_rate-params.material.z)/params.fluid.x;particles[i].lambda=-residual/sample.denominator;particles[i].density=sample.density;particles[i].neighbor_count=sample.neighbor_count;}
@compute @workgroup_size(128) fn preventive_cached_apply_pressure(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.counts.x||!motion_force_ready()){return;}let own=particles[i].lambda;let inverse_mass=1.0/params.fluid.w;let boundary=pressure_geometry[pressure_geometry_slot(i,1u)];var correction=vec2<f32>(0.0);
 for(var k=0u;k<boundary.w;k++){let row=pressure_geometry[pressure_geometry_slot(i,2u+k)];correction+=inverse_mass*(own+particles[row.x].lambda)*bitcast<vec2<f32>>(row.yz);}
 // Static and analytic boundary gradients share the same scalar own lambda.
 // Grouping them changes only f32 summation order, checked independently.
 correction+=inverse_mass*own*bitcast<vec2<f32>>(boundary.xy);
 particles[i].velocity+=correction*params.force.z;
}
`;
