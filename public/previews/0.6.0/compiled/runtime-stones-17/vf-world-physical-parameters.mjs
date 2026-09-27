// Geometry/material characterization for a reduced-order cantilever mode.
// This initializes World coefficients; it never integrates state or animates.
export function cantileverParameters({density,young_modulus,branch_radius:r,branch_length:L,damping_ratio:zeta=0.2,elasticity=1,mass:overrideMass,spring_constant:overrideStiffness,damping:overrideDamping}){
  if(![density,young_modulus,r,L,elasticity].every(x=>Number.isFinite(x)&&x>0)||!(zeta>=0))throw new Error('Invalid cantilever properties');
  const inertia=Math.PI*r**4/4,volume=Math.PI*r*r*L;
  const mass=overrideMass??0.236*density*volume,stiffness=overrideStiffness??3*young_modulus*inertia/(L**3*elasticity),damping=overrideDamping??2*zeta*Math.sqrt(mass*stiffness);
  if(!(mass>0&&stiffness>0&&damping>=0&&[mass,stiffness,damping].every(Number.isFinite)))throw new Error('Invalid cantilever overrides');
  return {mass,stiffness,damping,inertia,volume};
}
// Independent double-precision reference for the GPU air Law. The three mode
// amplitudes are perpendicular to their wave vectors, making their divergence
// analytically zero. It verifies physical structure; it does not animate Views.
export function coherentAirVelocityReference(position,time,{speed=3.5,turbulence_intensity:intensity=.12,integral_scale:scale=2.4,domain_min:min=[-6,-6,0]}={}){
  if(!(position?.length===3&&position.every(Number.isFinite)&&Number.isFinite(time)&&speed>=0&&intensity>=0&&scale>0))throw new Error('Invalid coherent air parameters');
  const q=position.map((value,axis)=>(value-min[axis])/scale),phase=time*speed/scale;
  const mode=(amplitude,wave,offset,rate)=>amplitude.map(value=>value*Math.sin(wave.reduce((sum,k,axis)=>sum+k*q[axis],0)+phase*rate+offset));
  const modes=[mode([.78,.40,-.5754386],[0,.82,.57],0,.71),mode([-.30,.88,.4676471],[.53,0,.34],1.7,-.47),mode([.58,-.3708197,.72],[.39,.61,0],3.1,.93),mode([.44*.35,.72*.35,0],[.72,-.44,.37],2.29,-1.31),mode([.63*.25,0,.31*.25],[.31,.27,-.63],4.37,1.73)];
  const boundary=.62+.38*(1-Math.exp(-Math.max(0,position[2]-min[2])/1.5));
  return [speed*boundary,0,0].map((base,axis)=>base+modes.reduce((sum,value)=>sum+value[axis],0)*speed*intensity*.34);
}
export function airObstacleExchangeReference({incoming,structure_velocity:structure=[0,0,0],normal,parcel_mass:mass,density=1.225,drag_coefficient:drag=1.05,coverage=1,dt=1/120}){
  if(![mass,density,drag,coverage,dt].every(Number.isFinite)||!(mass>0&&density>0&&drag>=0&&coverage>=0&&dt>0))throw new Error('Invalid air exchange parameters');
  const magnitude=value=>Math.hypot(...value),unit=value=>value.map(component=>component/Math.max(magnitude(value),1e-12)),n=unit(normal),relative=incoming.map((value,axis)=>value-structure[axis]);
  const normalSpeed=relative.reduce((sum,value,axis)=>sum+value*n[axis],0),length=Math.cbrt(mass/density),fraction=Math.min(.82,.5*drag*coverage*Math.abs(normalSpeed)*dt/length);
  const tangent=relative.map((value,axis)=>value-n[axis]*normalSpeed),candidate=relative.map((value,axis)=>value-n[axis]*normalSpeed*fraction-tangent[axis]*fraction*.12),bounded=candidate.map(value=>value*Math.min(1,magnitude(relative)/Math.max(magnitude(candidate),1e-12)));
  const outgoing=bounded.map((value,axis)=>value+structure[axis]),solidImpulse=incoming.map((value,axis)=>(value-outgoing[axis])*mass);
  return {outgoing,solidImpulse,airMomentumChange:solidImpulse.map(value=>-value),dragFraction:fraction};
}
export function zoomCamera(camera,ratio){
  if(!(Number.isFinite(ratio)&&ratio>0))return;
  const v=camera.pos.map((p,i)=>p-camera.target[i]),old=Math.hypot(...v),distance=Math.max(camera.minDistance??2,Math.min(camera.maxDistance??60,old*ratio));
  camera.pos=camera.target.map((p,i)=>p+v[i]*distance/old);
}
export function emissiveSphere(center,properties){
  const r=properties.radius,emissivity=properties.emissivity??0,color=properties.color??[1,1,1,1],emission=properties.emission??color;
  if(!(r>0&&emissivity>=0&&emissivity<=1))throw new Error('Invalid emissive sphere properties');
  const vertices=[],indices=[],rows=24,columns=48;
  for(let j=0;j<=rows;j++)for(let i=0;i<=columns;i++){
    const theta=j*Math.PI/rows,phi=i*2*Math.PI/columns,n=[Math.sin(theta)*Math.cos(phi),Math.sin(theta)*Math.sin(phi),Math.cos(theta)];
    vertices.push(...center.map((p,a)=>p+n[a]*r),...n,...color,-1,0,...emission.slice(0,3).map(x=>x*emissivity*(properties.radiance??180)),0,0,0,0,0,0,0,0,0);
    if(j<rows&&i<columns){const k=j*(columns+1)+i;indices.push(k,k+1,k+columns+1,k+1,k+columns+2,k+columns+1);}
  }
  return {shadow:false,vertices:new Float32Array(vertices),indices:new Uint32Array(indices),light:{position:center,radius:r,radiance:(properties.radiance??180)*emissivity,color:emission.slice(0,3),orbitSpeed:properties.orbit_speed??0,orbitTarget:properties.orbit_target??[0,0,0]}};
}
