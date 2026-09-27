// Immutable initial shape producer, not a simulation or runtime integrator.
import { mkdir,writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { stoneShape } from '../../vektor-flow/build/branches/pre-gen/web/vf-ui/vf-stone-shape.mjs';
const surfaceContacts=!process.argv.includes('--legacy');
const targetLargestMassKg=30;
const baseCenters=[[-1.60,-.10,.69],[0,.18,.91],[1.78,-.08,.65],[-.85,-.18,2.15],[.45,.20,2.04]];
const baseSizes=[.68,.82,.62,.50,.42];
const colors=[[.50,.49,.47],[.18,.20,.22],[.58,.34,.28],[.70,.68,.61],[.34,.40,.37]];
const densities=[2650,3000,2630,2650,2750],species=['gray-granite','basalt','red-granite','quartzite','gneiss'];
const hash=x=>{x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;};
const unit=x=>hash(x)/4294967296;
const meshMassProperties=(positions,triangles,density)=>{
  let signedVolume=0;const first=[0,0,0],second=Array.from({length:3},()=>[0,0,0]);
  for(let face=0;face<triangles.length;face+=3){
    const a=positions[triangles[face]],b=positions[triangles[face+1]],c=positions[triangles[face+2]];
    const volume=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6;
    signedVolume+=volume;
    const sum=a.map((v,k)=>v+b[k]+c[k]);
    for(let row=0;row<3;row++){
      first[row]+=volume*sum[row]/4;
      for(let column=0;column<3;column++)
        second[row][column]+=volume*(a[row]*a[column]+b[row]*b[column]+c[row]*c[column]+sum[row]*sum[column])/20;
    }
  }
  if(!(Math.abs(signedVolume)>1e-12))throw Error('Generated rigid mesh has zero signed volume');
  const center=first.map(v=>v/signedVolume),mass=Math.abs(signedVolume)*density,sign=Math.sign(signedVolume);
  const moment=second.map((row,r)=>row.map((value,c)=>sign*density*value-mass*center[r]*center[c]));
  const trace=moment[0][0]+moment[1][1]+moment[2][2];
  const inertiaTensor=moment.map((row,r)=>row.map((value,c)=>(r===c?trace:0)-value));
  if(inertiaTensor.some(row=>row.some(value=>!Number.isFinite(value))))throw Error('Generated rigid inertia is not finite');
  return {mass,center,inertiaTensor};
};
// Gilbert distance gives an upper bound from every visible vertex to the
// convex hull of the reduced contact supports. The largest bound is a
// geometry-derived collision skin, not a material or scene-specific fudge.
const supportError=(positions,hull)=>{
  const supports=Array.from({length:hull.length/4},(_,i)=>hull.slice(i*4,i*4+3));
  let largest=0;
  for(const point of positions){
    let closest=supports[0],distance=Infinity;
    for(const candidate of supports){const d=candidate.reduce((sum,v,a)=>sum+(point[a]-v)**2,0);
      if(d<distance){distance=d;closest=candidate;}}
    if(distance<1e-15)continue;
    let q=closest.slice();
    for(let step=0;step<256;step++){
      const direction=point.map((v,a)=>v-q[a]);let best=-Infinity,extreme=supports[0];
      for(const candidate of supports){const projection=direction.reduce((sum,v,a)=>sum+v*candidate[a],0);
        if(projection>best){best=projection;extreme=candidate;}}
      const edge=extreme.map((v,a)=>v-q[a]);const gap=direction.reduce((sum,v,a)=>sum+v*edge[a],0);
      if(gap<=1e-11)break;
      const edgeSquare=edge.reduce((sum,v)=>sum+v*v,0);
      const fraction=Math.min(1,gap/edgeSquare);
      q=q.map((v,a)=>v+fraction*edge[a]);
    }
    largest=Math.max(largest,Math.hypot(...point.map((v,a)=>v-q[a])));
  }
  return largest+2e-5;
};
const signedVolume=positions=>{let volume=0;for(let y=0;y<32;y++)for(let x=0;x<48;x++){const a=y*49+x,b=a+49,triangles=y===0?[[a+1,b,b+1]]:y===31?[[a,b,a+1]]:[[a,b,a+1],[a+1,b,b+1]];for(const triangle of triangles){const [p,q,r]=triangle.map(index=>positions[index]);volume+=(p[0]*(q[1]*r[2]-q[2]*r[1])+p[1]*(q[2]*r[0]-q[0]*r[2])+p[2]*(q[0]*r[1]-q[1]*r[0]))/6;}}return Math.abs(volume);};
const baselineMasses=baseSizes.map((size,i)=>{const point=stoneShape(hash(8187+i*1193),size,i),positions=[];for(let y=0;y<=32;y++)for(let x=0;x<=48;x++)positions.push(point(y*Math.PI/32,x*Math.PI*2/48));return signedVolume(positions)*densities[i];});
const physicalScale=Math.cbrt(targetLargestMassKg/Math.max(...baselineMasses));
const centers=baseCenters.map(center=>center.map(value=>value*physicalScale)),sizes=baseSizes.map(size=>size*physicalScale);
const chunks=[],supports=[],runtimeMasses=[],proxyErrors=[];let vertices=0,indices=0;
for(let i=0;i<5;i++){
  const center=centers[i],lat=32,lon=48,seed=hash(8187+i*1193),positions=[],triangles=[];
  const point=stoneShape(seed,sizes[i],i);
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++)positions.push(point(y*Math.PI/lat,x*Math.PI*2/lon));
  for(let y=0;y<lat;y++)for(let x=0;x<lon;x++){const a=y*(lon+1)+x,b=a+lon+1;if(y>0)triangles.push(a,b,a+1);if(y<lat-1)triangles.push(a+1,b,b+1);}
  const massProperties=meshMassProperties(positions,triangles,2700);
  const data=new Float32Array(positions.length*10);
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++){
    const j=y*(lon+1)+x,theta=y*Math.PI/lat,phi=x*Math.PI*2/lon,p=positions[j],e=.0001;
    const a=point(Math.max(e,Math.min(Math.PI-e,theta+e)),phi),b=point(Math.max(e,Math.min(Math.PI-e,theta-e)),phi),c=point(Math.max(e,Math.min(Math.PI-e,theta)),phi+e),d=point(Math.max(e,Math.min(Math.PI-e,theta)),phi-e);
    const t=a.map((v,k)=>v-b[k]),u=c.map((v,k)=>v-d[k]);let n=[t[1]*u[2]-t[2]*u[1],t[2]*u[0]-t[0]*u[2],t[0]*u[1]-t[1]*u[0]],length=Math.hypot(...n);
    if(length<1e-10){n=[0,0,y===0?1:-1];length=1;}if(n.reduce((s,v,k)=>s+v*p[k],0)<0)n=n.map(v=>-v);
    data.set([...p,...n.map(v=>v/length),...colors[i],1],j*10);
  }
  const source={indices:new Uint32Array(triangles)};
  // Initial packing uses the same convex support data as the rigid solver.
  // Find the first non-penetrating height; no per-stone hover offset.
  if(surfaceContacts)center[2]=-Math.min(...positions.map(p=>p[2]));
  const hull=[];let radius=0;for(let j=0;j<data.length;j+=10)radius=Math.max(radius,Math.hypot(data[j],data[j+1],data[j+2]));
  // Contact support must cover the visible mineral surface in many directions;
  // an overly sparse hull lets rendered protrusions enter a neighbour.
  const supportCount=256,normals=[];
  for(let k=0;k<supportCount;k++){const z=1-2*k/(supportCount-1),theta=k*2.399963229728653,r=Math.sqrt(1-z*z),n=[r*Math.cos(theta),r*Math.sin(theta),z];let best=-Infinity,at=0;
    normals.push(n);
    for(let j=0;j<data.length;j+=10){const d=data[j]*n[0]+data[j+1]*n[1]+data[j+2]*n[2];if(d>best){best=d;at=j;}}
    hull.push(data[at],data[at+1],data[at+2],0);
  }
  const group=(entries)=>{
    if(entries.length<=8)return [entries];
    const spread=[0,1,2].map(axis=>Math.max(...entries.map(e=>e.point[axis]))-Math.min(...entries.map(e=>e.point[axis])));
    const axis=spread.indexOf(Math.max(...spread));entries.sort((a,b)=>a.point[axis]-b.point[axis]);
    const middle=entries.length/2;return [...group(entries.slice(0,middle)),...group(entries.slice(middle))];
  };
  const groups=group(Array.from({length:supportCount},(_,k)=>({point:hull.slice(k*4,k*4+3),normal:normals[k]})));
  hull.splice(0,hull.length,...groups.flatMap(cluster=>cluster.flatMap(entry=>[...entry.point,0])));
  normals.splice(0,normals.length,...groups.flatMap(cluster=>cluster.map(entry=>entry.normal)));
  const hullGroups=groups.flatMap(cluster=>{
    const center=[0,1,2].map(axis=>cluster.reduce((sum,entry)=>sum+entry.point[axis],0)/cluster.length);
    const radius=Math.max(...cluster.map(entry=>Math.hypot(...entry.point.map((v,axis)=>v-center[axis]))))+2e-6;
    return [...center,radius];
  });
  if(surfaceContacts&&supports.length){
    const penetrating=height=>positions.some(point=>supports.some(base=>{
      const relative=[point[0]+center[0]-base.center[0],point[1]+center[1]-base.center[1],point[2]+height-base.center[2]];
      if(Math.hypot(...relative)>base.radius)return false;
      return base.normals.every((normal,k)=>normal.reduce((sum,value,axis)=>sum+value*(base.hull[k*4+axis]-relative[axis]),0)>1e-6);
    }));
    const floorHeight=center[2];
    if(penetrating(floorHeight)){
      let low=floorHeight,high=floorHeight+0.01;
      while(penetrating(high))high+=Math.max(0.01,high-floorHeight);
      for(let iteration=0;iteration<24;iteration++){const middle=(low+high)/2;if(penetrating(middle))low=middle;else high=middle;}
      center[2]=high;
    }
  }
  supports.push({center:center.slice(),hull,normals,radius:radius*2});
  const proxyError=supportError(positions,hull);proxyErrors.push(proxyError);
  const referenceDensity=2700,mass=massProperties.mass;
  runtimeMasses.push(mass*densities[i]/referenceDensity);
  const inertia=massProperties.inertiaTensor.reduce((sum,row,index)=>sum+row[index],0)/3;
  const meta=Buffer.from(JSON.stringify({id:`stone-${i}`,species:species[i],collision:{center,mass,inertia,inertia_tensor:massProperties.inertiaTensor,center_of_mass:massProperties.center,proxy_error:proxyError,radius,hull,hull_groups:hullGroups,hull_normals:normals.flat(),density:densities[i],reference_density:referenceDensity},shape:'direction-space radial Fourier',seed}));
  const header=Buffer.alloc(20);[meta.length,data.length,source.indices.length,0,0].forEach((n,a)=>header.writeUInt32LE(n,a*4));
  chunks.push(header,meta,Buffer.alloc((4-meta.length%4)%4),Buffer.from(data.buffer),Buffer.from(source.indices.buffer,source.indices.byteOffset,source.indices.byteLength));vertices+=data.length/10;indices+=source.indices.length;
}
const header=Buffer.alloc(20);header.write('VFTREE02');header.writeUInt32LE(5,8);header.writeUInt32LE(vertices,12);header.writeUInt32LE(indices,16);
await mkdir(new URL('../public/previews/0.6.0/live/rocks/assets/',import.meta.url),{recursive:true});
const output=gzipSync(Buffer.concat([header,...chunks]),{level:9});await writeFile(new URL('../public/previews/0.6.0/live/rocks/assets/'+(surfaceContacts?'rigid-stones-30kg-12.bin.gz':'rigid-stones.bin.gz'),import.meta.url),output);
if(Math.abs(Math.max(...runtimeMasses)-targetLargestMassKg)>1e-6)throw Error('Largest stone mass drifted from target: '+JSON.stringify(runtimeMasses));
console.log(JSON.stringify({centers,sizes,runtimeMasses,proxyErrors,physicalScale,vertices,indices,compressedBytes:output.length}));
