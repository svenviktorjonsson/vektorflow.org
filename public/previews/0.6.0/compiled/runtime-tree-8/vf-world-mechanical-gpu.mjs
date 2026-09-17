// Dense World laws. The compiler publishes this kernel in the application's WASM.
// JS below allocates/transfers buffers; it does not integrate mechanical state.
export const MECHANICAL_WORLD_WGSL = /* wgsl */`
struct Params { gravity_dt:vec4<f32>, counts:vec4<u32>, wind:vec4<f32>, domain_min:vec4<f32>, domain_span:vec4<f32>, grid:vec4<u32>, held:vec4<f32>, material:vec4<f32> };
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
fn rotate(q:vec4<f32>,p:vec3<f32>)->vec3<f32>{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
fn support(i:u32,n:vec3<f32>)->f32{
  let b=bodies[i]; var best=-1.0e10;
  for(var k=0u;k<params.counts.w;k++){best=max(best,dot(rotated_geometry[i*params.counts.w+k].xyz,n));}
  return best;
}
fn support_point(i:u32,n:vec3<f32>)->vec3<f32>{
  let b=bodies[i];var best=-1.0e10;var point=vec3<f32>(0.0);
  for(var k=0u;k<params.counts.w;k++){let p=rotated_geometry[i*params.counts.w+k].xyz;let projection=dot(p,n);if(projection>best){best=projection;point=p;}}
  return point;
}
@compute @workgroup_size(1)
fn place_held(){if(params.held.x>=0.0&&u32(params.held.x)<params.counts.x){let i=u32(params.held.x);bodies[i].info.w=0.0;bodies[i].p.z=params.held.y;bodies[i].v=vec4<f32>(0.0);bodies[i].w=vec4<f32>(0.0);}}
fn inverse_mass(i:u32)->f32 {if(f32(i)==params.held.x){return 0.0;} return 1.0/bodies[i].info.x;}
fn contact_velocity(b:Body,r:vec3<f32>)->vec3<f32>{return b.v.xyz+cross(b.w.xyz,r);}
fn impulse_body(i:u32,r:vec3<f32>,j:vec3<f32>){
  if(inverse_mass(i)==0.0){return;}
  if(dot(j,j)>1.0e-8){bodies[i].info.w=0.0;}
  bodies[i].v=vec4<f32>(bodies[i].v.xyz+j*inverse_mass(i),0.0);
  bodies[i].w=vec4<f32>(bodies[i].w.xyz+cross(r,j)/bodies[i].info.y,0.0);
}
@compute @workgroup_size(1)
fn advance_bodies(){
  let dt=params.gravity_dt.w;
  for(var i=0u;i<params.counts.x;i++){
    if(f32(i)==params.held.x){bodies[i].p.z=params.held.y;bodies[i].v=vec4<f32>(0.0);bodies[i].w=vec4<f32>(0.0);continue;}
    if(bodies[i].info.w>0.5){continue;}
    bodies[i].v=vec4<f32>(bodies[i].v.xyz+params.gravity_dt.xyz*dt,0.0);
    bodies[i].p=vec4<f32>(bodies[i].p.xyz+bodies[i].v.xyz*dt,0.0);
    let q=bodies[i].q;let w=bodies[i].w.xyz;
    bodies[i].q=normalize(q+vec4<f32>(q.w*w+cross(w,q.xyz),-dot(w,q.xyz))*(0.5*dt));
    bodies[i].w=vec4<f32>(w*0.999,0.0);
  }
}
@compute @workgroup_size(64)
fn rotate_support(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<params.counts.x*params.counts.w){rotated_geometry[i]=vec4<f32>(rotate(bodies[i/params.counts.w].q,geometry[i].xyz),0.0);}}
@compute @workgroup_size(1)
fn rigid_step(){
  for(var iteration=0u;iteration<6u;iteration++){
    for(var i=0u;i<params.counts.x;i++){
      let im=inverse_mass(i);let up=vec3<f32>(0.0,0.0,1.0);
      let depth=support(i,-up)-bodies[i].p.z;
      if(depth>0.0&&im>0.0&&bodies[i].info.w<0.5){
        bodies[i].p.z+=depth;
        let r=support_point(i,-up);let cv=contact_velocity(bodies[i],r);
        if(cv.z<0.0){
          let lever=cross(r,up);let jn=-(1.0+select(0.0,params.material.x,cv.z < -0.3))*cv.z/(im+dot(lever,lever)/bodies[i].info.y);
          let tangent=vec3<f32>(cv.x,cv.y,0.0);let speed=length(tangent);
          let jt=min(params.material.y*jn,speed/(im+dot(r,r)/bodies[i].info.y));
          impulse_body(i,r,up*jn-tangent/max(speed,1.0e-8)*jt);
          // Rolling resistance removes spin at a resting rough contact; it
          // does not attract or pin the body to a pre-recorded pile position.
          let spin=length(bodies[i].w.xyz);let reduction=params.material.y*0.25*jn*bodies[i].info.z/bodies[i].info.y;
          bodies[i].w=vec4<f32>(bodies[i].w.xyz*max(0.0,1.0-reduction/max(spin,1.0e-8)),0.0);
        }
      }
      for(var j=i+1u;j<params.counts.x;j++){
        if(bodies[i].info.w>0.5&&bodies[j].info.w>0.5){continue;}
        let delta=bodies[j].p.xyz-bodies[i].p.xyz;let distance=length(delta);
        if(distance>bodies[i].info.z+bodies[j].info.z){continue;}
        var n=select(delta/max(distance,1.0e-8),up,distance<1.0e-7);var penetration=support(i,n)+support(j,-n)-dot(delta,n);
        // A center-axis test alone spuriously collides adjacent irregular shapes.
        for(var axis=0u;axis<9u;axis++){var basis=vec3<f32>(0.0);basis[axis%3u]=1.0;var direction=basis;
          if(axis>=3u&&axis<6u){direction=rotate(bodies[i].q,basis);}if(axis>=6u){direction=rotate(bodies[j].q,basis);}
          direction=select(direction,-direction,dot(delta,direction)<0.0);
          let depth=support(i,direction)+support(j,-direction)-dot(delta,direction);if(depth<penetration){penetration=depth;n=direction;}
        }
        if(penetration<=0.0){continue;}
        let ri=support_point(i,n);let rj=support_point(j,-n);
        let overlap=penetration;let jm=inverse_mass(j);let total=im+jm;
        if(overlap<=0.0||total==0.0){continue;}
        if(im>0.0){bodies[i].info.w=0.0;}if(jm>0.0){bodies[j].info.w=0.0;}
        bodies[i].p=vec4<f32>(bodies[i].p.xyz-n*overlap*im/total,0.0);
        bodies[j].p=vec4<f32>(bodies[j].p.xyz+n*overlap*jm/total,0.0);
        let relative=contact_velocity(bodies[j],rj)-contact_velocity(bodies[i],ri);
        let vn=dot(relative,n);
        if(vn<0.0){
          let li=cross(ri,n);let lj=cross(rj,n);let denominator=total+select(dot(li,li)/bodies[i].info.y,0.0,im==0.0)+select(dot(lj,lj)/bodies[j].info.y,0.0,jm==0.0);
          let jn=-(1.0+select(0.0,params.material.x,vn < -0.3))*vn/denominator;
          let tangent=relative-vn*n;let speed=length(tangent);
          let jt=min(params.material.y*jn,speed/(total+dot(ri,ri)/bodies[i].info.y+dot(rj,rj)/bodies[j].info.y));
          let impulse=n*jn-tangent/max(speed,1.0e-8)*jt;
          impulse_body(i,ri,-impulse);impulse_body(j,rj,impulse);
        }
      }
    }
  }
}
fn hash(x:u32)->u32 {var v=x;v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn unit(x:u32)->f32{return f32(hash(x)&0x00ffffffu)/16777216.0;}
fn cell(p:vec3<f32>)->u32{
  let t=clamp((p-params.domain_min.xyz)/params.domain_span.xyz,vec3<f32>(0.0),vec3<f32>(0.99999));
  let c=vec3<u32>(t*vec3<f32>(params.grid.xyz));return c.x+params.grid.x*(c.y+params.grid.y*c.z);
}
@compute @workgroup_size(64)
fn clear_impulses(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<params.counts.z*3u){atomicStore(&impulses[gid.x],0);}}
@compute @workgroup_size(64)
fn wind_step(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.y){return;}let dt=params.gravity_dt.w;
  var p=parcels[i];p.p.w=max(0.0,p.p.w-dt);p.v.w=max(0.0,p.v.w-dt);
  let old=p.p.xyz;p.p=vec4<f32>(old+p.v.xyz*dt,p.p.w);
  let c=cell(p.p.xyz);let object=geometry[c];
  let inside=all(p.p.xyz>=params.domain_min.xyz)&&all(p.p.xyz<params.domain_min.xyz+params.domain_span.xyz);
  if(inside&&object.w>0.0&&(object.w<1.5||p.p.z<0.22)&&p.v.w==0.0){
    let incoming=p.v.xyz;
    p.v=vec4<f32>(incoming.x*0.55,incoming.y+(unit(i*17u)-0.5)*params.wind.x*0.18,incoming.z+params.wind.x*0.035,0.10);
    p.p.w=0.25;
    let impulse=(incoming-p.v.xyz)*params.wind.y;
    let impacted=select(c,c+params.grid.x*params.grid.y,object.w>1.5);
    for(var a=0u;a<3u;a++){atomicAdd(&impulses[impacted*3u+a],i32(clamp(impulse[a]*65536.0,-1.0e7,1.0e7)));}
  }
  let out=params.domain_min.xyz+params.domain_span.xyz;
  if(p.p.x>=out.x||p.p.x<params.domain_min.x||p.p.y<params.domain_min.y||p.p.y>=out.y||p.p.z<0.0||p.p.z>=out.z){
    let generation=u32(params.wind.w*60.0);
    p.p=vec4<f32>(params.domain_min.x,params.domain_min.y+unit(i*31u+generation)*params.domain_span.y,unit(i*47u+generation)*params.domain_span.z,0.0);
    p.v=vec4<f32>(params.wind.x,0.0,0.0,0.0);
  }
  // Prescribed inlet velocity; no parcel-parcel cohesion or artificial gust animation.
  p.v.x+=clamp(params.wind.x-p.v.x,-dt*params.wind.x,dt*params.wind.x);
  parcels[i]=p;
}
@compute @workgroup_size(64)
fn elastic_step(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.counts.z){return;}let dt=params.gravity_dt.w;
  let z=i/(params.grid.x*params.grid.y);
  if(z==0u){nodes[i]=Node(vec4<f32>(0.0),vec4<f32>(0.0));return;}
  let force=vec3<f32>(f32(atomicLoad(&impulses[i*3u])),f32(atomicLoad(&impulses[i*3u+1u])),f32(atomicLoad(&impulses[i*3u+2u])))/(65536.0*dt);
  let compliance=0.3+f32(z)/f32(params.grid.z);
  let material=geometry[params.counts.z+i];
  let k=material.y/compliance;
  let x=i%params.grid.x;let y=(i/params.grid.x)%params.grid.y;var coupling=vec3<f32>(0.0);var neighbors=0.0;
  let stride=params.grid.x*params.grid.y;
  if(x>0u){coupling+=prior_nodes[i-1u].d.xyz;neighbors+=1.0;}if(x+1u<params.grid.x){coupling+=prior_nodes[i+1u].d.xyz;neighbors+=1.0;}
  if(y>0u){coupling+=prior_nodes[i-params.grid.x].d.xyz;neighbors+=1.0;}if(y+1u<params.grid.y){coupling+=prior_nodes[i+params.grid.x].d.xyz;neighbors+=1.0;}
  if(z>0u){coupling+=prior_nodes[i-stride].d.xyz;neighbors+=1.0;}if(z+1u<params.grid.z){coupling+=prior_nodes[i+stride].d.xyz;neighbors+=1.0;}
  // Implicit local spring/damper with a prior-state neighbor boundary. This
  // remains stable with wood stiffness instead of relying on displacement caps.
  let total_k=k*(1.0+neighbors*0.3);
  var wood_force=force;
  if(arrayLength(&nodes)>params.counts.z){if(geometry[params.counts.z*2u+i].x>0.0){wood_force=force*0.35;}}
  let rhs=wood_force+coupling*k*0.3-total_k*nodes[i].d.xyz;
  let v=(material.x*nodes[i].v.xyz+rhs*dt)/(material.x+dt*material.z+dt*dt*total_k);
  nodes[i].v=vec4<f32>(v,0.0);nodes[i].d=vec4<f32>(nodes[i].d.xyz+v*dt,0.0);
  // Independent compliant leaf mode, forced by the same local parcel impacts.
  // Geometry coefficients aggregate the actual accepted leaves in this cell.
  if(arrayLength(&nodes)>params.counts.z){
    let leaf=geometry[params.counts.z*2u+i];let j=params.counts.z+i;
    if(leaf.x>0.0&&leaf.y>0.0){
      let old=nodes[j].d.xyz;
      let cellArea=params.domain_span.y/f32(params.grid.y)*params.domain_span.z/f32(params.grid.z);
      let density=params.wind.y*f32(params.counts.y)/(params.domain_span.x*params.domain_span.y*params.domain_span.z);
      let available=0.5*density*params.wind.x*params.wind.x*cellArea;
      // A coarse parcel impact cannot deliver more sustained drag than the
      // intercepted flow's dynamic pressure. Resolve that momentum into the
      // area's angular mode without a whole-parcel impulse on a single leaf.
      let magnitude=length(force);
      let resolved=force/max(magnitude,1.0e-8)*available*tanh(magnitude/max(available,1.0e-8));
      let torque=resolved*leaf.w;
      // Backward-Euler angular spring with a convex tan-angle potential.
      // Solve inside its physical domain, rather than integrate huge metre
      // displacements and hide them behind an embedding angle clamp.
      let inertia=leaf.x+dt*leaf.z;
      let leaf_rhs=leaf.x*nodes[j].v.xyz*dt+torque*dt*dt;
      var low=vec3<f32>(-1.570795);var high=vec3<f32>(1.570795);
      for(var iteration=0u;iteration<24u;iteration++){
        let angle=(low+high)*0.5;let residual=inertia*(angle-old)+dt*dt*leaf.y*tan(angle)-leaf_rhs;
        high=select(high,angle,residual>vec3<f32>(0.0));low=select(angle,low,residual>vec3<f32>(0.0));
      }
      let angle=(low+high)*0.5;nodes[j].v=vec4<f32>((angle-old)/dt,0.0);nodes[j].d=vec4<f32>(angle,0.0);
    }
  }
}
`;

export async function createMechanicalWorldGpu(device,world,initial,shaderSource) {
  const make=(label,data)=>{const buffer=device.createBuffer({label,size:Math.max(80,data.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(buffer,0,data);return buffer;};
  const buffers=[make('VKF rigid state',initial.bodies),make('VKF inlet parcels',initial.parcels),make('VKF elastic state',initial.nodes),make('VKF contact impulse ledger',new Int32Array(Math.max(3,initial.nodeCount*3))),make('VKF added collision geometry',initial.geometry),make('VKF Jacobi prior elastic state',initial.nodes),make('VKF rotated support cache',world.kind==='rigid'?initial.geometry:new Float32Array(4))];
  const uniform=device.createBuffer({label:'VKF World properties',size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...buffers.map((_,i)=>({binding:i+1,visibility:GPUShaderStage.COMPUTE,buffer:{type:i===4||i===5?'read-only-storage':'storage'}}))]});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniform}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});
  const module=device.createShaderModule({label:'Compiled VKF World mechanical laws',code:shaderSource});
  const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
  if(errors.length)throw new Error(errors.map(error=>`mechanical WGSL ${error.lineNum}:${error.linePos}: ${error.message}`).join('\n'));
  const entries=world.kind==='rigid'?['advance_bodies','rotate_support','rigid_step']:['clear_impulses','wind_step','elastic_step'];
  const pipelines=await Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})));
  const heldPipeline=world.kind==='rigid'?await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'place_held'}}):null;
  const data=new ArrayBuffer(128),f=new Float32Array(data),u=new Uint32Array(data);
  const p=world.properties;
  f.set([...world.gravity,world.time_step],0);u.set([initial.bodyCount,initial.parcelCount,initial.nodeCount,initial.hullCount??0],4);
  f.set([p.speed??3.5,p.parcel_mass??0.002,p.node_mass??1,0],8);
  f.set([...(p.domain_min??[-6,-6,0]),0],12);f.set([...(p.domain_span??[12,12,9]),0],16);
  u.set([...(p.grid??[12,12,18]),0],20);f.set([-1,0,0,0],24);
  f.set([p.restitution??0.08,p.friction??0.65,p.spring_constant??60,p.damping??12],28);
  let time=0;
  return {device,buffers,initial,
    get time(){return time;},setHeld(id,z){f[24]=id;f[25]=z;},setSpeed(speed){f[8]=speed;},
    placeHeld(encoder){if(!heldPipeline||f[24]<0)return;device.queue.writeBuffer(uniform,0,data);const pass=encoder.beginComputePass();pass.setBindGroup(0,group);pass.setPipeline(heldPipeline);pass.dispatchWorkgroups(1);pass.end();},
    step(encoder){time+=world.time_step;f[11]=time;device.queue.writeBuffer(uniform,0,data);
      if(world.kind==='wind')encoder.copyBufferToBuffer(buffers[2],0,buffers[5],0,initial.nodes.byteLength);
      const pass=encoder.beginComputePass();pass.setBindGroup(0,group);
      for(let i=0;i<pipelines.length;i++){pass.setPipeline(pipelines[i]);pass.dispatchWorkgroups(world.kind==='rigid'?(i===1?Math.ceil(initial.bodyCount*initial.hullCount/64):1):Math.ceil((i===0?initial.nodeCount*3:i===1?initial.parcelCount:initial.nodeCount)/64));}pass.end();},
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
    async inspect(){const sizes=[initial.bodies.byteLength,initial.parcels.byteLength,initial.nodes.byteLength],total=sizes.reduce((sum,n)=>sum+n,0),out=device.createBuffer({size:total,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const encoder=device.createCommandEncoder();let offset=0;for(let i=0;i<3;i++){encoder.copyBufferToBuffer(buffers[i],0,out,offset,sizes[i]);offset+=sizes[i];}device.queue.submit([encoder.finish()]);await out.mapAsync(GPUMapMode.READ);const data=new Float32Array(out.getMappedRange()).slice();out.unmap();out.destroy();const finite=data.every(Number.isFinite),bodyData=data.subarray(0,sizes[0]/4),parcelData=data.subarray(sizes[0]/4,(sizes[0]+sizes[1])/4),nodeData=data.subarray((sizes[0]+sizes[1])/4);let visible=0,maxDisplacement=0,maxCanopyDisplacement=0;for(let i=0;i<initial.parcelCount;i++)if(parcelData[i*8+3]>0)visible++;for(let i=0;i<initial.nodeCount;i++){const d=Math.hypot(...nodeData.subarray(i*8,i*8+3));maxDisplacement=Math.max(maxDisplacement,d);if(initial.geometry[i*4+2]>2&&initial.geometry[i*4+3]===1)maxCanopyDisplacement=Math.max(maxCanopyDisplacement,d);}return {finite,time,bodyCenters:Array.from({length:initial.bodyCount},(_,i)=>Array.from(bodyData.subarray(i*20,i*20+3))),visibleParcels:visible,maxDisplacement,maxCanopyDisplacement,parcelMass:p.parcel_mass,beam:p.beam};},
    destroy(){uniform.destroy();for(const b of buffers)b.destroy();}};
}
