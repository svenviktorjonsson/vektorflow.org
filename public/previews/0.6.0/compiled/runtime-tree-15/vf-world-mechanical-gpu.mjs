// Dense World laws. The compiler publishes this kernel in the application's WASM.
// JS below allocates/transfers buffers; it does not integrate mechanical state.
import {ELASTIC_BLADE_POSE_WGSL} from './vf-elastic-blade-pose-wgsl.mjs';
export const MECHANICAL_WORLD_WGSL = /* wgsl */`
struct Params { gravity_dt:vec4<f32>, counts:vec4<u32>, wind:vec4<f32>, domain_min:vec4<f32>, domain_span:vec4<f32>, grid:vec4<u32>, held:vec4<f32>, material:vec4<f32>, grass:vec4<f32> };
struct Body { p:vec4<f32>, v:vec4<f32>, q:vec4<f32>, w:vec4<f32>, info:vec4<f32> };
struct Parcel { p:vec4<f32>, v:vec4<f32> };
struct Node { d:vec4<f32>, v:vec4<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> bodies:array<Body>;
@group(0) @binding(2) var<storage,read_write> parcels:array<Parcel>;
@group(0) @binding(3) var<storage,read_write> nodes:array<Node>;
@group(0) @binding(4) var<storage,read_write> impulses:array<atomic<i32>>;
@group(0) @binding(5) var<storage,read> geometry:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read> prior_nodes:array<Node>;
@group(0) @binding(7) var<storage,read_write> rotated_geometry:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read> prior_bodies:array<Body>;
fn rotate(q:vec4<f32>,p:vec3<f32>)->vec3<f32>{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
fn inverse_inertia_world(i:u32,torque:vec3<f32>)->vec3<f32>{
  let q=bodies[i].q;let local=rotate(vec4<f32>(-q.xyz,q.w),torque);
  let base=params.counts.x*params.counts.w+i*3u;
  return rotate(q,vec3<f32>(dot(geometry[base].xyz,local),dot(geometry[base+1u].xyz,local),dot(geometry[base+2u].xyz,local)));
}
fn contact_matrix_component(i:u32,r:vec3<f32>,left:vec3<f32>,right:vec3<f32>,enabled:bool)->f32{
  if(!enabled){return 0.0;}
  return inverse_mass(i)*dot(left,right)+
    dot(cross(r,left),inverse_inertia_world(i,cross(r,right)));
}
fn contact_linear_delta(i:u32,r:vec3<f32>,linear:vec3<f32>,angular:vec3<f32>,enabled:bool)->vec3<f32>{
  if(!enabled){return vec3<f32>(0.0);}
  return inverse_mass(i)*linear+cross(inverse_inertia_world(i,cross(r,linear)+angular),r);
}
fn contact_angular_delta(i:u32,r:vec3<f32>,linear:vec3<f32>,angular:vec3<f32>,enabled:bool)->vec3<f32>{
  if(!enabled){return vec3<f32>(0.0);}
  return inverse_inertia_world(i,cross(r,linear)+angular);
}
struct ContactImpulse { linear:vec3<f32>, angular:vec3<f32>, normal:f32 };
fn rigid_contact_impulse(i:u32,j:u32,ri:vec3<f32>,rj:vec3<f32>,n:vec3<f32>,
  relative:vec3<f32>,relative_omega:vec3<f32>,first_enabled:bool,second_enabled:bool,
  restitution:f32,contact_radius:f32,support_load:f32)->ContactImpulse{
  let vn=dot(relative,n);
  if(vn>=0.0&&support_load<=0.0){return ContactImpulse(vec3<f32>(0.0),vec3<f32>(0.0),0.0);}
  let knn=contact_matrix_component(i,ri,n,n,first_enabled)+
    contact_matrix_component(j,rj,n,n,second_enabled);
  if(knn<=1.0e-8){return ContactImpulse(vec3<f32>(0.0),vec3<f32>(0.0),0.0);}
  let jn=max(0.0,-(1.0+restitution)*vn/knn);
  let friction_budget=max(jn,support_load);
  let rolling_velocity=relative_omega-n*dot(relative_omega,n);
  let rolling_speed=length(rolling_velocity);
  let rolling_axis=rolling_velocity/max(rolling_speed,1.0e-8);
  let rolling_mass=dot(rolling_axis,contact_angular_delta(i,ri,vec3<f32>(0.0),rolling_axis,first_enabled)+
    contact_angular_delta(j,rj,vec3<f32>(0.0),rolling_axis,second_enabled));
  let rolling_magnitude=-min(rolling_speed/max(rolling_mass,1.0e-8),params.material.z*contact_radius*friction_budget);
  let rolling_impulse=rolling_axis*rolling_magnitude;
  let normal_impulse=n*jn;
  let velocity_after=relative+contact_linear_delta(i,ri,normal_impulse,rolling_impulse,first_enabled)+
    contact_linear_delta(j,rj,normal_impulse,rolling_impulse,second_enabled);
  let angular_after=relative_omega+contact_angular_delta(i,ri,normal_impulse,rolling_impulse,first_enabled)+
    contact_angular_delta(j,rj,normal_impulse,rolling_impulse,second_enabled);
  let tangent_velocity=velocity_after-n*dot(velocity_after,n);
  let tangent_speed=length(tangent_velocity);
  if(tangent_speed<=1.0e-8&&abs(dot(angular_after,n))<=1.0e-8)
    {return ContactImpulse(normal_impulse,rolling_impulse,jn);}
  let fallback=select(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),abs(n.x)>0.8);
  let t=select(normalize(cross(n,fallback)),tangent_velocity/max(tangent_speed,1.0e-8),tangent_speed>1.0e-8);
  let a=contact_matrix_component(i,ri,t,t,first_enabled)+contact_matrix_component(j,rj,t,t,second_enabled);
  let b=dot(t,contact_linear_delta(i,ri,vec3<f32>(0.0),n,first_enabled)+
    contact_linear_delta(j,rj,vec3<f32>(0.0),n,second_enabled));
  let c=dot(n,contact_angular_delta(i,ri,t,vec3<f32>(0.0),first_enabled)+
    contact_angular_delta(j,rj,t,vec3<f32>(0.0),second_enabled));
  let d=dot(n,contact_angular_delta(i,ri,vec3<f32>(0.0),n,first_enabled)+
    contact_angular_delta(j,rj,vec3<f32>(0.0),n,second_enabled));
  let determinant=a*d-b*c;
  if(determinant<=1.0e-8){return ContactImpulse(normal_impulse,rolling_impulse,jn);}
  let tangent_speed_after=dot(velocity_after,t);
  let spin_speed_after=dot(angular_after,n);
  var jt=-(d*tangent_speed_after-b*spin_speed_after)/determinant;
  var js=-(a*spin_speed_after-c*tangent_speed_after)/determinant;
  let static_spin=min(params.material.w,params.material.y)*contact_radius*friction_budget;
  if(abs(jt)>params.material.y*friction_budget||abs(js)>static_spin){
    jt=clamp(jt,-params.grass.z*friction_budget,params.grass.z*friction_budget);
    let dynamic_spin=static_spin*params.grass.z/max(params.material.y,1.0e-8);
    js=clamp(js,-dynamic_spin,dynamic_spin);
  }
  return ContactImpulse(normal_impulse+t*jt,rolling_impulse+n*js,jn);
}
fn support(i:u32,n:vec3<f32>)->f32{
  var best=-1.0e10;
  if(params.counts.y==0u){for(var k=0u;k<params.counts.w;k++){best=max(best,dot(rotated_geometry[i*params.counts.w+k].xyz,n));}}
  else{let group_base=params.counts.x*(params.counts.w+3u)+i*params.counts.y;
    for(var group=0u;group<params.counts.y;group++){
      let bound=rotated_geometry[group_base+group];
      if(dot(bound.xyz,n)+bound.w+1.0e-6<=best){continue;}
      for(var lane=0u;lane<8u;lane++){best=max(best,dot(rotated_geometry[i*params.counts.w+group*8u+lane].xyz,n));}
    }
  }
  return best;
}
fn support_point(i:u32,n:vec3<f32>)->vec3<f32>{
  var best=-1.0e10;var point=vec3<f32>(0.0);
  if(params.counts.y==0u){for(var k=0u;k<params.counts.w;k++){let p=rotated_geometry[i*params.counts.w+k].xyz;let projection=dot(p,n);if(projection>best){best=projection;point=p;}}}
  else{let group_base=params.counts.x*(params.counts.w+3u)+i*params.counts.y;
    for(var group=0u;group<params.counts.y;group++){
      let bound=rotated_geometry[group_base+group];
      if(dot(bound.xyz,n)+bound.w+1.0e-6<=best){continue;}
      for(var lane=0u;lane<8u;lane++){let p=rotated_geometry[i*params.counts.w+group*8u+lane].xyz;let projection=dot(p,n);if(projection>best){best=projection;point=p;}}
    }
  }
  return point;
}
@compute @workgroup_size(1)
fn place_held(){if(params.held.x>=0.0&&u32(params.held.x)<params.counts.x){let i=u32(params.held.x);bodies[i].info.w=0.0;bodies[i].p.z=params.held.y;bodies[i].v=vec4<f32>(0.0);bodies[i].w=vec4<f32>(0.0);}}
fn inverse_mass(i:u32)->f32 {if(f32(i)==params.held.x){return 0.0;} return 1.0/bodies[i].info.x;}
fn contact_velocity(b:Body,r:vec3<f32>)->vec3<f32>{return b.v.xyz+cross(b.w.xyz,r);}
fn impulse_body_general(i:u32,r:vec3<f32>,linear:vec3<f32>,angular:vec3<f32>){
  if(inverse_mass(i)==0.0){return;}
  if(dot(linear,linear)+dot(angular,angular)>1.0e-8){bodies[i].info.w=0.0;}
  bodies[i].v=vec4<f32>(bodies[i].v.xyz+linear*inverse_mass(i),bodies[i].v.w);
  bodies[i].w=vec4<f32>(bodies[i].w.xyz+inverse_inertia_world(i,cross(r,linear)+angular),bodies[i].w.w);
}
struct SweepWindow { enter:f32, exit:f32, normal:vec3<f32>, valid:u32 };
fn sweep_axis(i:u32,j:u32,raw:vec3<f32>,start:vec3<f32>,travel:vec3<f32>,window:SweepWindow)->SweepWindow{
  var result=window;let axis_length=length(raw);
  if(axis_length<1.0e-5||result.valid==0u){return result;}
  let axis=raw/axis_length;
  let low=-support(i,-axis)-support(j,axis);
  let high=support(i,axis)+support(j,-axis);
  let initial=dot(start,axis);let motion=dot(travel,axis);
  if(abs(motion)<1.0e-7){if(initial<low||initial>high){result.valid=0u;}return result;}
  let a=(low-initial)/motion;let b=(high-initial)/motion;
  let enter=min(a,b);let leave=max(a,b);
  if(enter>result.enter){result.enter=enter;result.normal=select(-axis,axis,motion<0.0);}
  result.exit=min(result.exit,leave);
  if(result.enter>result.exit){result.valid=0u;}
  return result;
}
fn swept_pair(i:u32,j:u32)->SweepWindow{
  var window=SweepWindow(0.0,1.0,vec3<f32>(0.0,0.0,1.0),1u);
  let start=prior_bodies[j].p.xyz-prior_bodies[i].p.xyz;
  let end=bodies[j].p.xyz-bodies[i].p.xyz;
  let travel=end-start;let distance_squared=dot(travel,travel);
  // The overlap solver already resolves motion smaller than a conservative
  // fraction of the smaller collision envelope. Skip costly swept hull axes
  // unless a body can advance appreciably through another in one step.
  let small_motion=0.15*min(bodies[i].info.z,bodies[j].info.z);
  if(distance_squared<small_motion*small_motion){window.valid=0u;return window;}
  let nearest=clamp(-dot(start,travel)/distance_squared,0.0,1.0);
  let limit=bodies[i].info.z+bodies[j].info.z;
  if(dot(start+travel*nearest,start+travel*nearest)>limit*limit){window.valid=0u;return window;}
  for(var axis=0u;axis<9u;axis++){
    var basis=vec3<f32>(0.0);basis[axis%3u]=1.0;var direction=basis;
    if(axis>=3u&&axis<6u){direction=rotate(bodies[i].q,basis);}
    if(axis>=6u){direction=rotate(bodies[j].q,basis);}
    window=sweep_axis(i,j,direction,start,travel,window);
    if(window.valid==0u){return window;}
  }
  for(var ai=0u;ai<3u;ai++){
    var axis_i=vec3<f32>(0.0);axis_i[ai]=1.0;
    for(var aj=0u;aj<3u;aj++){
      var axis_j=vec3<f32>(0.0);axis_j[aj]=1.0;
      window=sweep_axis(i,j,cross(rotate(bodies[i].q,axis_i),rotate(bodies[j].q,axis_j)),start,travel,window);
      if(window.valid==0u){return window;}
    }
  }
  return window;
}
fn resolve_swept_pair(i:u32,j:u32){
  // Continuous contact for fast motion. Keep the ordinary overlap solver for
  // resting contacts; this path only handles a first impact within the step.
  if(f32(i)==params.held.x||f32(j)==params.held.x){return;}
  let window=swept_pair(i,j);
  if(window.valid==0u||window.enter<=1.0e-5||window.enter>=1.0){return;}
  let n=window.normal;
  let start_i=prior_bodies[i].p.xyz;let start_j=prior_bodies[j].p.xyz;
  let end_i=bodies[i].p.xyz;let end_j=bodies[j].p.xyz;
  bodies[i].p=vec4<f32>(mix(start_i,end_i,window.enter),0.0);
  bodies[j].p=vec4<f32>(mix(start_j,end_j,window.enter),0.0);
  let surface_i=support_point(i,n);let surface_j=support_point(j,-n);
  let contact=0.5*(bodies[i].p.xyz+surface_i+bodies[j].p.xyz+surface_j);
  let ri=contact-bodies[i].p.xyz;let rj=contact-bodies[j].p.xyz;
  let im=inverse_mass(i);let jm=inverse_mass(j);
  let solve_im=im;let solve_jm=jm;
  let relative=contact_velocity(bodies[j],rj)-contact_velocity(bodies[i],ri);
  let vn=dot(relative,n);
  if(vn<0.0&&solve_im+solve_jm>0.0){
    let total=solve_im+solve_jm;
    if(total>0.0){
      let impulse=rigid_contact_impulse(i,j,ri,rj,n,relative,bodies[j].w.xyz-bodies[i].w.xyz,
        solve_im>0.0,solve_jm>0.0,select(0.0,params.material.x,vn < -0.3),min(bodies[i].info.z,bodies[j].info.z),0.0);
      if(solve_im>0.0){impulse_body_general(i,ri,-impulse.linear,-impulse.angular);}
      if(solve_jm>0.0){impulse_body_general(j,rj,impulse.linear,impulse.angular);}
    }
  }
  let remaining=params.gravity_dt.w*(1.0-window.enter);
  if(solve_im>0.0){bodies[i].p=vec4<f32>(bodies[i].p.xyz+bodies[i].v.xyz*remaining,0.0);}
  if(solve_jm>0.0){bodies[j].p=vec4<f32>(bodies[j].p.xyz+bodies[j].v.xyz*remaining,0.0);}
  bodies[i].w.w=1.0;bodies[j].w.w=1.0;
}
@compute @workgroup_size(1)
fn advance_bodies(){
  let dt=params.gravity_dt.w;
  for(var i=0u;i<params.counts.x;i++){
    if(f32(i)==params.held.x){bodies[i].p.z=params.held.y;bodies[i].v=vec4<f32>(0.0);bodies[i].w=vec4<f32>(0.0);continue;}
    if(bodies[i].info.w>0.5){continue;}
    bodies[i].v=vec4<f32>(bodies[i].v.xyz+params.gravity_dt.xyz*dt,bodies[i].v.w);
    bodies[i].p=vec4<f32>(bodies[i].p.xyz+bodies[i].v.xyz*dt,0.0);
    let q=bodies[i].q;let w=bodies[i].w.xyz;
    bodies[i].q=normalize(q+vec4<f32>(q.w*w+cross(w,q.xyz),-dot(w,q.xyz))*(0.5*dt));
    bodies[i].w=vec4<f32>(w*0.999,0.0);
  }
}
@compute @workgroup_size(64)
fn rotate_support(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;let hull_total=params.counts.x*params.counts.w;
  if(i<hull_total){rotated_geometry[i]=vec4<f32>(rotate(bodies[i/params.counts.w].q,geometry[i].xyz),0.0);}
  else{let group_index=i-hull_total;
    if(params.counts.y>0u&&group_index<params.counts.x*params.counts.y){
      let source=hull_total+params.counts.x*3u+group_index;
      rotated_geometry[source]=vec4<f32>(rotate(bodies[group_index/params.counts.y].q,geometry[source].xyz),geometry[source].w);
    }
  }
}
@compute @workgroup_size(1)
fn rigid_step(){
  for(var iteration=0u;iteration<3u;iteration++){
    for(var i=0u;i<params.counts.x;i++){
      let im=inverse_mass(i);let up=vec3<f32>(0.0,0.0,1.0);let ground_point=support_point(i,-up);
      let depth=dot(ground_point,-up)-bodies[i].p.z;
      if(depth>-0.0005){bodies[i].w.w=1.0;}
      if(depth>-0.0005&&im>0.0&&bodies[i].info.w<0.5){
        bodies[i].p.z+=max(depth,0.0);
        let r=ground_point;let cv=contact_velocity(bodies[i],r);
        // Each of the three contact iterations carries one third of the
        // floor's gravity reaction, including at zero normal velocity.
        let support_load=bodies[i].info.x*max(0.0,-dot(params.gravity_dt.xyz,up))*params.gravity_dt.w/3.0;
        let impulse=rigid_contact_impulse(i,i,r,vec3<f32>(0.0),up,cv,bodies[i].w.xyz,true,false,
          select(0.0,params.material.x,cv.z < -0.3),bodies[i].info.z,support_load);
        impulse_body_general(i,r,impulse.linear,impulse.angular);
        bodies[i].w.w=1.0;
      }
      for(var j=i+1u;j<params.counts.x;j++){
        if(bodies[i].info.w>0.5&&bodies[j].info.w>0.5){continue;}
        if(iteration==0u){resolve_swept_pair(i,j);}
        let delta=bodies[j].p.xyz-bodies[i].p.xyz;let distance=length(delta);
        if(distance>bodies[i].info.z+bodies[j].info.z){continue;}
        var n=select(delta/max(distance,1.0e-8),up,distance<1.0e-7);var penetration=support(i,n)+support(j,-n)-dot(delta,n);
        if(penetration<=0.0){continue;}
        // A center-axis test alone spuriously collides adjacent irregular shapes.
        for(var axis=0u;axis<9u;axis++){var basis=vec3<f32>(0.0);basis[axis%3u]=1.0;var direction=basis;
          if(axis>=3u&&axis<6u){direction=rotate(bodies[i].q,basis);}if(axis>=6u){direction=rotate(bodies[j].q,basis);}
          direction=select(direction,-direction,dot(delta,direction)<0.0);
          let depth=support(i,direction)+support(j,-direction)-dot(delta,direction);if(depth<penetration){penetration=depth;n=direction;}
          if(penetration<=0.0){break;}
        }
        if(penetration<=0.0){continue;}
        // The nine face axes miss rotated edge-edge separation. Cross axes
        // close that SAT gap without changing the shared rigid impulse law.
        for(var ai=0u;ai<3u;ai++){var axis_i=vec3<f32>(0.0);axis_i[ai]=1.0;
          for(var aj=0u;aj<3u;aj++){var axis_j=vec3<f32>(0.0);axis_j[aj]=1.0;
            let cross_axis=cross(rotate(bodies[i].q,axis_i),rotate(bodies[j].q,axis_j));let axis_length=length(cross_axis);
            if(axis_length<1.0e-5){continue;}var direction=cross_axis/axis_length;
            direction=select(direction,-direction,dot(delta,direction)<0.0);
            let depth=support(i,direction)+support(j,-direction)-dot(delta,direction);if(depth<penetration){penetration=depth;n=direction;}
            if(penetration<=0.0){break;}
          }
          if(penetration<=0.0){break;}
        }
        if(penetration>-0.0005){bodies[i].w.w=1.0;bodies[j].w.w=1.0;}
        if(penetration<=0.0){continue;}
        let surface_i=support_point(i,n);let surface_j=support_point(j,-n);
        let overlap=penetration;let solve_im=im;let solve_jm=inverse_mass(j);
        let total=solve_im+solve_jm;
        if(overlap<=0.0||total==0.0){continue;}
        if(solve_im>0.0){bodies[i].info.w=0.0;}if(solve_jm>0.0){bodies[j].info.w=0.0;}
        bodies[i].p=vec4<f32>(bodies[i].p.xyz-n*overlap*solve_im/total,0.0);
        bodies[j].p=vec4<f32>(bodies[j].p.xyz+n*overlap*solve_jm/total,0.0);
        let contact=0.5*(bodies[i].p.xyz+surface_i+bodies[j].p.xyz+surface_j);
        let ri=contact-bodies[i].p.xyz;let rj=contact-bodies[j].p.xyz;
        let relative=contact_velocity(bodies[j],rj)-contact_velocity(bodies[i],ri);let vn=dot(relative,n);
        if(vn<0.0){
          let impulse=rigid_contact_impulse(i,j,ri,rj,n,relative,bodies[j].w.xyz-bodies[i].w.xyz,
            solve_im>0.0,solve_jm>0.0,select(0.0,params.material.x,vn < -0.3),min(bodies[i].info.z,bodies[j].info.z),0.0);
          if(solve_im>0.0){impulse_body_general(i,ri,-impulse.linear,-impulse.angular);}
          if(solve_jm>0.0){impulse_body_general(j,rj,impulse.linear,impulse.angular);}
        }
        bodies[i].w.w=1.0;bodies[j].w.w=1.0;
      }
    }
  }
  // Contact-qualified sleep removes settled bodies from later pair work.
  // The timer lives in v.w; w.w is a per-step contact bit reset by advance.
  for(var i=0u;i<params.counts.x;i++){
    if(f32(i)==params.held.x||bodies[i].info.w>0.5){continue;}
    // Linear threshold exceeds one gravity impulse at 60 Hz; otherwise a
    // perfectly resting contact can never qualify before its support impulse.
    let slow=length(bodies[i].v.xyz)<0.22&&length(bodies[i].w.xyz)<0.60;
    let contact_delta=select(-params.gravity_dt.w*0.25,params.gravity_dt.w,bodies[i].w.w>0.5);
    bodies[i].v.w=select(0.0,max(0.0,bodies[i].v.w+contact_delta),slow);
    if(bodies[i].v.w>=0.5){bodies[i].v=vec4<f32>(0.0);bodies[i].w=vec4<f32>(0.0);bodies[i].info.w=1.0;}
  }
}
fn hash(x:u32)->u32 {var v=x;v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn unit(x:u32)->f32{return f32(hash(x)&0x00ffffffu)/16777216.0;}
const IMPULSE_SCALE:f32=16777216.0;
fn cell_coordinates(p:vec3<f32>)->vec3<u32>{
  let t=clamp((p-params.domain_min.xyz)/params.domain_span.xyz,vec3<f32>(0.0),vec3<f32>(0.99999));
  return vec3<u32>(t*vec3<f32>(params.grid.xyz));
}
fn cell_index(c:vec3<u32>)->u32{return c.x+params.grid.x*(c.y+params.grid.y*c.z);}
fn cell(p:vec3<f32>)->u32{return cell_index(cell_coordinates(p));}
fn occupied(at:vec3<i32>)->f32{
  if(any(at<vec3<i32>(0))||any(at>=vec3<i32>(params.grid.xyz))){return 0.0;}
  let kind=geometry[cell_index(vec3<u32>(at))].w;
  return select(0.0,1.0,kind==1.0||kind==3.0);
}
fn air_velocity(position:vec3<f32>,time:f32)->vec3<f32>{
  let speed=max(params.wind.x,0.0);let scale=max(params.material.y,0.25);
  let q=(position-params.domain_min.xyz)/scale;let phase=time*speed/scale;
  // Each Fourier amplitude is perpendicular to its wave vector, so these
  // coherent modes are divergence-free instead of independent random kicks.
  let mode1=vec3<f32>(0.78,0.40,-0.5754386)*sin(dot(vec3<f32>(0.0,0.82,0.57),q)+phase*0.71);
  let mode2=vec3<f32>(-0.30,0.88,0.4676471)*sin(dot(vec3<f32>(0.53,0.0,0.34),q)-phase*0.47+1.7);
  let mode3=vec3<f32>(0.58,-0.3708197,0.72)*sin(dot(vec3<f32>(0.39,0.61,0.0),q)+phase*0.93+3.1);
  let mode4=vec3<f32>(0.44,0.72,0.0)*sin(dot(vec3<f32>(0.72,-0.44,0.37),q)-phase*1.31+2.29)*0.35;
  let mode5=vec3<f32>(0.63,0.0,0.31)*sin(dot(vec3<f32>(0.31,0.27,-0.63),q)+phase*1.73+4.37)*0.25;
  let height=max(0.0,position.z-params.domain_min.z);
  let boundary_layer=0.62+0.38*(1.0-exp(-height/1.5));
  return vec3<f32>(speed*boundary_layer,0.0,0.0)+(mode1+mode2+mode3+mode4+mode5)*(speed*params.material.x*0.34);
}
fn obstacle_normal(coordinate:vec3<u32>,kind:f32,owner:u32,fallback:vec3<f32>)->vec3<f32>{
  if(kind==3.0){let leaf=geometry[params.counts.z*4u+owner].xyz;if(dot(leaf,leaf)>1.0e-8){return normalize(leaf);}}
  if(kind==2.0){return vec3<f32>(0.0,0.0,1.0);}
  let at=vec3<i32>(coordinate);
  let gradient=vec3<f32>(occupied(at-vec3<i32>(1,0,0))-occupied(at+vec3<i32>(1,0,0)),occupied(at-vec3<i32>(0,1,0))-occupied(at+vec3<i32>(0,1,0)),occupied(at-vec3<i32>(0,0,1))-occupied(at+vec3<i32>(0,0,1)));
  let fallback_length=length(fallback);let fallback_normal=select(vec3<f32>(-1.0,0.0,0.0),-fallback/max(fallback_length,1.0e-8),fallback_length>1.0e-8);
  return select(fallback_normal,normalize(gradient),dot(gradient,gradient)>1.0e-8);
}
@compute @workgroup_size(64)
fn clear_impulses(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<params.counts.z*3u){atomicStore(&impulses[gid.x],0);}}
@compute @workgroup_size(64)
fn wind_step(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.y){return;}let dt=params.gravity_dt.w;
  var p=parcels[i];p.p.w=max(0.0,p.p.w-dt);p.v.w=max(0.0,p.v.w-dt);
  let old=p.p.xyz;p.p=vec4<f32>(old+p.v.xyz*dt,p.p.w);
  let coordinate=cell_coordinates(p.p.xyz);let c=cell_index(coordinate);let object=geometry[c];
  let inside=all(p.p.xyz>=params.domain_min.xyz)&&all(p.p.xyz<params.domain_min.xyz+params.domain_span.xyz);
  let collides=object.w==1.0||object.w==3.0||(object.w==2.0&&p.p.z<0.22);
  if(inside&&collides&&p.v.w==0.0){
    let incoming=p.v.xyz;
    let owner=u32(geometry[params.counts.z*3u+c].w);
    let attached=geometry[params.counts.z*3u+owner];
    let cellArea=params.domain_span.y/f32(params.grid.y)*params.domain_span.z/f32(params.grid.z);
    let coverage=select(1.0,clamp(attached.z/cellArea,0.0,1.0),object.w==3.0);
    let impacted=select(owner,c+params.grid.x*params.grid.y,object.w==2.0);
    let structure_velocity=nodes[impacted].v.xyz;
    let relative=incoming-structure_velocity;let relative_speed=length(relative);
    let normal=obstacle_normal(coordinate,object.w,owner,relative);
    let normal_speed=dot(relative,normal);
    // The rigid support cache is idle for wind; its vec4 slots hold four
    // immutable Particle masses without a ninth portable storage binding.
    let parcel_mass=rotated_geometry[i/4u][i%4u];
    let parcel_length=pow(max(parcel_mass/max(params.wind.z,1.0e-6),1.0e-9),1.0/3.0);
    let drag_fraction=clamp(0.5*params.material.z*coverage*abs(normal_speed)*dt/parcel_length,0.0,0.82);
    let tangent_velocity=relative-normal*normal_speed;
    var outgoing_relative=relative-normal*normal_speed*drag_fraction-tangent_velocity*(drag_fraction*0.12);
    var swirl=cross(normal,vec3<f32>(0.0,0.0,1.0));if(dot(swirl,swirl)<0.02){swirl=cross(normal,vec3<f32>(0.0,1.0,0.0));}
    let wake_phase=dot((p.p.xyz-params.domain_min.xyz)/max(params.material.y,0.25),vec3<f32>(0.71,1.13,0.43))+params.wind.w*params.wind.x/max(params.material.y,0.25);
    outgoing_relative+=normalize(swirl)*sin(wake_phase)*abs(normal_speed)*drag_fraction*0.16;
    // Porous drag may redirect resolved flow, but never adds kinetic energy.
    outgoing_relative*=min(1.0,relative_speed/max(length(outgoing_relative),1.0e-8));
    let outgoing=structure_velocity+outgoing_relative;
    p.v=vec4<f32>(outgoing,max(dt,parcel_length/max(relative_speed,0.1)));
    p.p.w=0.25;
    let impulse=(incoming-outgoing)*parcel_mass;
    for(var a=0u;a<3u;a++){atomicAdd(&impulses[impacted*3u+a],i32(clamp(impulse[a]*IMPULSE_SCALE,-1.0e7,1.0e7)));}
  }
  let out=params.domain_min.xyz+params.domain_span.xyz;
  if(p.p.x>=out.x||p.p.x<params.domain_min.x||p.p.y<params.domain_min.y||p.p.y>=out.y||p.p.z<0.0||p.p.z>=out.z){
    let generation=u32(params.wind.w*60.0);
    p.p=vec4<f32>(params.domain_min.x,params.domain_min.y+unit(i*31u+generation)*params.domain_span.y,unit(i*47u+generation)*params.domain_span.z,0.0);
    p.v=vec4<f32>(air_velocity(p.p.xyz,params.wind.w),0.0);
  }
  // Pressure-scale relaxation restores the coherent inlet field. Collision
  // wakes persist for an integral-scale turnover time instead of disappearing
  // into a frame-driven visual animation.
  // When the user lowers the target wind, retain physical inertia but use the
  // parcel's current speed for the turnover scale so calm does not take tens
  // of seconds to become visibly calm.
  let characteristic_speed=max(params.wind.x,length(p.v.xyz));
  let turnover=max(params.material.y/max(characteristic_speed,0.1),dt);
  let relaxation=1.0-exp(-params.material.w*dt/turnover);
  p.v=vec4<f32>(mix(p.v.xyz,air_velocity(p.p.xyz,params.wind.w),relaxation),p.v.w);
  parcels[i]=p;
}
fn coupled_neighbor(index:u32,mode:f32)->vec4<f32>{
  let neighbor_mode=geometry[params.counts.z+index].w;
  let connected=neighbor_mode>0.5&&abs(neighbor_mode-mode)<0.25;
  return vec4<f32>(select(vec3<f32>(0.0),prior_nodes[index].d.xyz,connected),select(0.0,1.0,connected));
}
@compute @workgroup_size(64)
fn elastic_step(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.z){return;}let dt=params.gravity_dt.w;
  let z=i/(params.grid.x*params.grid.y);
  if(z==0u){nodes[i]=Node(vec4<f32>(0.0),vec4<f32>(0.0));return;}
  var force=vec3<f32>(f32(atomicLoad(&impulses[i*3u])),f32(atomicLoad(&impulses[i*3u+1u])),f32(atomicLoad(&impulses[i*3u+2u])))/(IMPULSE_SCALE*dt);
  let material=geometry[params.counts.z+i];
  let mode=material.w;if(mode<0.5){nodes[i]=Node(vec4<f32>(0.0),vec4<f32>(0.0));return;}
  if(mode>1.5){
    // Sub-grid grass is loaded by the same coherent air field as the parcels.
    // Blade reconfiguration reduces projected area as wind flattens the grass.
    let relative=air_velocity(geometry[i].xyz,params.wind.w)-nodes[i].v.xyz;
    let speed=length(relative);
    let projected=params.grass.x/(1.0+pow(speed/max(params.grass.y,0.1),2.0));
    force+=0.5*params.wind.z*params.material.z*projected*speed*relative;
  }
  let compliance=select(1.0,0.3+f32(z)/f32(params.grid.z),mode<1.5);
  let k=material.y/compliance;
  let x=i%params.grid.x;let y=(i/params.grid.x)%params.grid.y;var coupling=vec3<f32>(0.0);var neighbors=0.0;
  let stride=params.grid.x*params.grid.y;
  if(x>0u){let neighbor=coupled_neighbor(i-1u,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}if(x+1u<params.grid.x){let neighbor=coupled_neighbor(i+1u,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}
  if(y>0u){let neighbor=coupled_neighbor(i-params.grid.x,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}if(y+1u<params.grid.y){let neighbor=coupled_neighbor(i+params.grid.x,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}
  if(z>0u){let neighbor=coupled_neighbor(i-stride,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}if(z+1u<params.grid.z){let neighbor=coupled_neighbor(i+stride,mode);coupling+=neighbor.xyz;neighbors+=neighbor.w;}
  // Implicit local spring/damper with a prior-state neighbor boundary. This
  // remains stable with wood stiffness instead of relying on displacement caps.
  let coupling_weight=select(0.12,0.35,mode<1.5);let total_k=k*(1.0+neighbors*coupling_weight);
  let old_v=nodes[i].v.xyz;
  var v=(material.x*old_v+(force+coupling*k*coupling_weight-total_k*nodes[i].d.xyz)*dt)/(material.x+dt*material.z+dt*dt*total_k);
  // Solve branch translation and attached leaf rotation together. The
  // off-diagonal mass term carries leaf momentum and spring reaction into
  // the branch; no arbitrary fraction of the impact is discarded.
  if(arrayLength(&nodes)>params.counts.z){
    let leaf=geometry[params.counts.z*2u+i];let j=params.counts.z+i;
    if(leaf.x>0.0&&leaf.y>0.0){
      let attached=geometry[params.counts.z*3u+i];let mass=attached.x;let moment=attached.y;
      let lever=moment/max(mass,1.0e-8);let old=nodes[j].d.xyz;let angular_v=nodes[j].v.xyz;
      let diagonal=material.x+mass+dt*material.z+dt*dt*total_k;
      let rhs=(material.x+mass)*old_v+moment*angular_v+(force+coupling*k*coupling_weight-total_k*nodes[i].d.xyz)*dt;
      let inertia=leaf.x+dt*leaf.z-moment*moment/diagonal;
      let leaf_rhs=leaf.x*angular_v*dt+force*lever*dt*dt-moment*dt*(rhs/diagonal-old_v);
      var low=vec3<f32>(-1.570795);var high=vec3<f32>(1.570795);
      for(var iteration=0u;iteration<24u;iteration++){
        let angle=(low+high)*0.5;let residual=inertia*(angle-old)+dt*dt*leaf.y*tan(angle)-leaf_rhs;
        high=select(high,angle,residual>vec3<f32>(0.0));low=select(angle,low,residual>vec3<f32>(0.0));
      }
      let angle=(low+high)*0.5;let angular=(angle-old)/dt;
      v=(rhs-moment*angular)/diagonal;
      nodes[j].v=vec4<f32>(angular,0.0);nodes[j].d=vec4<f32>(angle,0.0);
    }
  }
  nodes[i].v=vec4<f32>(v,0.0);nodes[i].d=vec4<f32>(nodes[i].d.xyz+v*dt,0.0);
}
` + ELASTIC_BLADE_POSE_WGSL;

export async function createMechanicalWorldGpu(device,world,initial,shaderSource) {
  const make=(label,data)=>{const buffer=device.createBuffer({label,size:Math.max(80,data.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(buffer,0,data);return buffer;};
  const parcelMasses=initial.parcelMasses??new Float32Array(Math.max(4,Math.ceil(initial.parcelCount/4)*4)).fill(world.properties.parcel_mass??0.002);
  const buffers=[make('VKF rigid state',initial.bodies),make('VKF inlet parcels',initial.parcels),make('VKF elastic state',initial.nodes),make('VKF contact impulse ledger',new Int32Array(Math.max(3,initial.nodeCount*3))),make('VKF added collision geometry',initial.geometry),make('VKF Jacobi prior elastic state',initial.nodes),make(world.kind==='rigid'?'VKF rotated support cache':'VKF air parcel mass',world.kind==='rigid'?initial.geometry:parcelMasses),make('VKF prior rigid state',world.kind==='rigid'?initial.bodies:new Float32Array(20))];
  const uniform=device.createBuffer({label:'VKF World properties',size:144,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...buffers.map((_,i)=>({binding:i+1,visibility:GPUShaderStage.COMPUTE,buffer:{type:i===4||i===5||i===7?'read-only-storage':'storage'}}))]});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniform}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});
  const module=device.createShaderModule({label:'Compiled VKF World mechanical laws',code:shaderSource});
  const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
  if(errors.length)throw new Error(errors.map(error=>`mechanical WGSL ${error.lineNum}:${error.linePos}: ${error.message}`).join('\n'));
  const entries=world.kind==='rigid'?['advance_bodies','rotate_support','rigid_step']:['clear_impulses','wind_step','elastic_step'];
  const pipelines=await Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})));
  const heldPipeline=world.kind==='rigid'?await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'place_held'}}):null;
  const data=new ArrayBuffer(144),f=new Float32Array(data),u=new Uint32Array(data);
  const p=world.properties;
  f.set([...world.gravity,world.time_step],0);u.set([initial.bodyCount,world.kind==='rigid'?(initial.groupCount??0):initial.parcelCount,initial.nodeCount,initial.hullCount??0],4);
  f.set([p.speed??3.5,p.parcel_mass??0.002,p.density??1.225,0],8);
  f.set([...(p.domain_min??[-6,-6,0]),0],12);f.set([...(p.domain_span??[12,12,9]),0],16);
  u.set([...(p.grid??[12,12,18]),0],20);f.set([-1,0,0,0],24);
  f.set(world.kind==='wind'?[p.turbulence_intensity??0.12,p.integral_scale??2.4,p.drag_coefficient??1.05,p.eddy_decay??0.75]:[p.restitution??0.10,p.static_friction??p.friction??0.72,p.rolling_friction??0.08,p.spin_friction??0.035],28);
  f.set([p.grass_wind_area??0.6,p.grass_reconfiguration_speed??6,p.dynamic_friction??p.friction??0.72,0],32);
  let time=0;
  return {device,buffers,initial,
    get time(){return time;},setHeld(id,z){f[24]=id;f[25]=z;},setSpeed(speed){f[8]=Math.max(0,Math.min(20,Number.isFinite(speed)?speed:0));},
    releaseHeld(){f[24]=-1;},
    placeHeld(encoder){if(!heldPipeline||f[24]<0)return;device.queue.writeBuffer(uniform,0,data);const pass=encoder.beginComputePass();pass.setBindGroup(0,group);pass.setPipeline(heldPipeline);pass.dispatchWorkgroups(1);pass.end();},
    step(encoder){time+=world.time_step;f[11]=time;device.queue.writeBuffer(uniform,0,data);
      if(world.kind==='wind')encoder.copyBufferToBuffer(buffers[2],0,buffers[5],0,initial.nodes.byteLength);
      else encoder.copyBufferToBuffer(buffers[0],0,buffers[7],0,initial.bodies.byteLength);
      const pass=encoder.beginComputePass();pass.setBindGroup(0,group);
      for(let i=0;i<pipelines.length;i++){pass.setPipeline(pipelines[i]);pass.dispatchWorkgroups(world.kind==='rigid'?(i===1?Math.ceil(initial.bodyCount*(initial.hullCount+(initial.groupCount??0))/64):1):Math.ceil((i===0?initial.nodeCount*3:i===1?initial.parcelCount:initial.nodeCount)/64));}pass.end();},
    reset(){time=0;f[24]=-1;device.queue.writeBuffer(buffers[0],0,initial.bodies);device.queue.writeBuffer(buffers[1],0,initial.parcels);device.queue.writeBuffer(buffers[2],0,initial.nodes);},
    async readBodies(){const out=device.createBuffer({size:Math.max(80,initial.bodies.byteLength),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffers[0],0,out,0,initial.bodies.byteLength);device.queue.submit([encoder.finish()]);await out.mapAsync(GPUMapMode.READ);const copy=new Float32Array(out.getMappedRange()).slice();out.unmap();out.destroy();return copy;},
    async inspectLeafModes(){
      if(world.kind!=='wind')throw Error('Leaf modes require wind World');
      const size=initial.nodeCount*32,out=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffers[2],size,out,0,size);device.queue.submit([encoder.finish()]);await out.mapAsync(GPUMapMode.READ);
      const state=new Float32Array(out.getMappedRange());let maxDisplacement=0,maxVelocity=0;const finite=state.every(Number.isFinite);
      for(let i=0;i<initial.nodeCount;i++){maxDisplacement=Math.max(maxDisplacement,Math.hypot(...state.subarray(i*8,i*8+3)));maxVelocity=Math.max(maxVelocity,Math.hypot(...state.subarray(i*8+4,i*8+7)));}
      out.unmap();out.destroy();return {finite,maxAngle:maxDisplacement,maxAngularVelocity:maxVelocity};
    },
    async inspectGrassWave(){
      if(world.kind!=='wind')throw Error('Grass modes require wind World');
      const size=initial.nodeCount*32,out=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffers[2],0,out,0,size);device.queue.submit([encoder.finish()]);await out.mapAsync(GPUMapMode.READ);
      const state=new Float32Array(out.getMappedRange()),grid=p.grid??[12,12,18],stride=grid[0]*grid[1];
      let total=0,count=0,correlation=0,pairs=0,minimum=Infinity,maximum=-Infinity;
      for(let i=stride;i<Math.min(2*stride,initial.nodeCount);i++){
        if(initial.geometry[(initial.nodeCount+i)*4+3]!==2)continue;
        const x=state[i*8],y=state[i*8+1],z=state[i*8+2],magnitude=Math.hypot(x,y,z);
        total+=magnitude;count++;minimum=Math.min(minimum,x);maximum=Math.max(maximum,x);
        const j=i+1;if(i%grid[0]===grid[0]-1||initial.geometry[(initial.nodeCount+j)*4+3]!==2)continue;
        const xx=state[j*8],yy=state[j*8+1],zz=state[j*8+2],other=Math.hypot(xx,yy,zz);
        if(magnitude>1e-5&&other>1e-5){correlation+=(x*xx+y*yy+z*zz)/(magnitude*other);pairs++;}
      }
      out.unmap();out.destroy();return {meanGrassDisplacement:total/Math.max(count,1),grassNeighborCorrelation:correlation/Math.max(pairs,1),grassSpatialRange:maximum-minimum};
    },
    async inspect(){const sizes=[initial.bodies.byteLength,initial.parcels.byteLength,initial.nodes.byteLength],total=sizes.reduce((sum,n)=>sum+n,0),out=device.createBuffer({size:total,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const encoder=device.createCommandEncoder();let offset=0;for(let i=0;i<3;i++){encoder.copyBufferToBuffer(buffers[i],0,out,offset,sizes[i]);offset+=sizes[i];}device.queue.submit([encoder.finish()]);await out.mapAsync(GPUMapMode.READ);const data=new Float32Array(out.getMappedRange()).slice();out.unmap();out.destroy();const finite=data.every(Number.isFinite),bodyData=data.subarray(0,sizes[0]/4),parcelData=data.subarray(sizes[0]/4,(sizes[0]+sizes[1])/4),nodeData=data.subarray((sizes[0]+sizes[1])/4);let visible=0,maxDisplacement=0,maxCanopyDisplacement=0,transverseSpeed=0,streamwiseSpeed=0,maxParcelSpeed=0;for(let i=0;i<initial.parcelCount;i++){if(parcelData[i*8+3]>0)visible++;const vx=parcelData[i*8+4],vy=parcelData[i*8+5],vz=parcelData[i*8+6];streamwiseSpeed+=vx;transverseSpeed+=Math.hypot(vy,vz);maxParcelSpeed=Math.max(maxParcelSpeed,Math.hypot(vx,vy,vz));}for(let i=0;i<initial.nodeCount;i++){const d=Math.hypot(...nodeData.subarray(i*8,i*8+3));maxDisplacement=Math.max(maxDisplacement,d);const kind=initial.geometry[i*4+3];if(initial.geometry[i*4+2]>2&&(kind===1||kind===3))maxCanopyDisplacement=Math.max(maxCanopyDisplacement,d);}return {finite,time,bodyCenters:Array.from({length:initial.bodyCount},(_,i)=>Array.from(bodyData.subarray(i*20,i*20+3))),visibleParcels:visible,maxDisplacement,maxCanopyDisplacement,meanStreamwiseSpeed:streamwiseSpeed/Math.max(1,initial.parcelCount),meanTransverseSpeed:transverseSpeed/Math.max(1,initial.parcelCount),maxParcelSpeed,parcelMass:p.parcel_mass,beam:p.beam};},
    destroy(){uniform.destroy();for(const b of buffers)b.destroy();}};
}
