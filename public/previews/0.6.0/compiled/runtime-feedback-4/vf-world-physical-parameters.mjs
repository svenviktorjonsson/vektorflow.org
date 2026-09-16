// Geometry/material characterization for a reduced-order cantilever mode.
// This initializes World coefficients; it never integrates state or animates.
export function cantileverParameters({density,young_modulus,branch_radius:r,branch_length:L,damping_ratio:zeta=0.2,elasticity=1,mass:overrideMass,spring_constant:overrideStiffness,damping:overrideDamping}){
  if(![density,young_modulus,r,L,elasticity].every(x=>Number.isFinite(x)&&x>0)||!(zeta>=0))throw new Error('Invalid cantilever properties');
  const inertia=Math.PI*r**4/4,volume=Math.PI*r*r*L;
  const mass=overrideMass??0.236*density*volume,stiffness=overrideStiffness??3*young_modulus*inertia/(L**3*elasticity),damping=overrideDamping??2*zeta*Math.sqrt(mass*stiffness);
  if(!(mass>0&&stiffness>0&&damping>=0&&[mass,stiffness,damping].every(Number.isFinite)))throw new Error('Invalid cantilever overrides');
  return {mass,stiffness,damping,inertia,volume};
}
export function zoomCamera(camera,ratio){
  if(!(Number.isFinite(ratio)&&ratio>0))return;
  const v=camera.pos.map((p,i)=>p-camera.target[i]),old=Math.hypot(...v),distance=Math.max(2,Math.min(60,old*ratio));
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
  return {shadow:false,vertices:new Float32Array(vertices),indices:new Uint32Array(indices),light:{position:center,radius:r,radiance:(properties.radiance??180)*emissivity,color:emission.slice(0,3)}};
}
