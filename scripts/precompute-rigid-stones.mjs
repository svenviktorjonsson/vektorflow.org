// Immutable initial shape producer, not a simulation or runtime integrator.
import { mkdir,writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { stoneShape } from '../../vektor-flow/build/branches/pre-gen/web/vf-ui/vf-stone-shape.mjs';
const surfaceContacts=!process.argv.includes('--legacy');
const centers=[[-1.12,-.10,.69],[0,.18,.91],[1.15,-.08,.65],[-.55,-.18,2.15],[.58,.20,2.04]];
const sizes=[.68,.82,.62,.50,.42];
const colors=[[.50,.49,.47],[.18,.20,.22],[.58,.34,.28],[.70,.68,.61],[.34,.40,.37]];
const densities=[2650,3000,2630,2650,2750],species=['gray-granite','basalt','red-granite','quartzite','gneiss'];
const hash=x=>{x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;};
const unit=x=>hash(x)/4294967296;
const chunks=[],supports=[];let vertices=0,indices=0;
for(let i=0;i<5;i++){
  const center=centers[i],lat=40,lon=64,seed=hash(8187+i*1193),positions=[],triangles=[];
  const point=stoneShape(seed,sizes[i],i);
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++)positions.push(point(y*Math.PI/lat,x*Math.PI*2/lon));
  for(let y=0;y<lat;y++)for(let x=0;x<lon;x++){const a=y*(lon+1)+x,b=a+lon+1;if(y>0)triangles.push(a,b,a+1);if(y<lat-1)triangles.push(a+1,b,b+1);}
  const data=new Float32Array(positions.length*10);
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++){
    const j=y*(lon+1)+x,theta=y*Math.PI/lat,phi=x*Math.PI*2/lon,p=positions[j],e=.0001;
    const a=point(Math.max(e,Math.min(Math.PI-e,theta+e)),phi),b=point(Math.max(e,Math.min(Math.PI-e,theta-e)),phi),c=point(Math.max(e,Math.min(Math.PI-e,theta)),phi+e),d=point(Math.max(e,Math.min(Math.PI-e,theta)),phi-e);
    const t=a.map((v,k)=>v-b[k]),u=c.map((v,k)=>v-d[k]);let n=[t[1]*u[2]-t[2]*u[1],t[2]*u[0]-t[0]*u[2],t[0]*u[1]-t[1]*u[0]],length=Math.hypot(...n);
    if(length<1e-10){n=[0,0,y===0?1:-1];length=1;}if(n.reduce((s,v,k)=>s+v*p[k],0)<0)n=n.map(v=>-v);
    data.set([...p,...n.map(v=>v/length),...colors[i],1],j*10);
  }
  const source={indices:new Uint32Array(triangles)};
  // Place the initial pile on its actual surface, not a spherical height proxy.
  // This is an offline initial-condition calculation, never an animation.
  if(surfaceContacts)center[2]=-Math.min(...positions.map(p=>p[2]));
  if(surfaceContacts&&i>=3)for(const p of positions){const x=p[0]+center[0],y=p[1]+center[1];for(const base of supports){for(let t=0;t<base.indices.length;t+=3){
    const a=base.positions[base.indices[t]],b=base.positions[base.indices[t+1]],c=base.positions[base.indices[t+2]],px=x-base.center[0],py=y-base.center[1];
    if(px<Math.min(a[0],b[0],c[0])||px>Math.max(a[0],b[0],c[0])||py<Math.min(a[1],b[1],c[1])||py>Math.max(a[1],b[1],c[1]))continue;
    const denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(denominator)<1e-12)continue;
    const u=((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/denominator,v=((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/denominator;
    if(u>=0&&v>=0&&u+v<=1)center[2]=Math.max(center[2],base.center[2]+u*a[2]+v*b[2]+(1-u-v)*c[2]-p[2]);
  }}}
  supports.push({positions,indices:source.indices,center});
  const hull=[];let radius=0;for(let j=0;j<data.length;j+=10)radius=Math.max(radius,Math.hypot(data[j],data[j+1],data[j+2]));
  for(let k=0;k<96;k++){const z=1-2*k/95,theta=k*2.399963229728653,r=Math.sqrt(1-z*z),n=[r*Math.cos(theta),r*Math.sin(theta),z];let best=-Infinity,at=0;
    for(let j=0;j<data.length;j+=10){const d=data[j]*n[0]+data[j+1]*n[1]+data[j+2]*n[2];if(d>best){best=d;at=j;}}
    hull.push(data[at],data[at+1],data[at+2],0);
  }
  let volume=0;for(let j=0;j<source.indices.length;j+=3){const a=source.indices[j]*10,b=source.indices[j+1]*10,c=source.indices[j+2]*10;
    volume+=(data[a]*(data[b+1]*data[c+2]-data[b+2]*data[c+1])+data[a+1]*(data[b+2]*data[c]-data[b]*data[c+2])+data[a+2]*(data[b]*data[c+1]-data[b+1]*data[c]))/6;
  }
  const referenceDensity=2700,mass=Math.max(.1,Math.abs(volume)*referenceDensity);
  const extents=[0,1,2].map(axis=>Math.max(...positions.map(p=>Math.abs(p[axis]))));
  const inertia=2*mass*(extents[0]**2+extents[1]**2+extents[2]**2)/15;
  const meta=Buffer.from(JSON.stringify({id:`stone-${i}`,species:species[i],collision:{center,mass,inertia,radius,hull,density:densities[i],reference_density:referenceDensity},shape:'direction-space radial Fourier',seed}));
  const header=Buffer.alloc(20);[meta.length,data.length,source.indices.length,0,0].forEach((n,a)=>header.writeUInt32LE(n,a*4));
  chunks.push(header,meta,Buffer.alloc((4-meta.length%4)%4),Buffer.from(data.buffer),Buffer.from(source.indices.buffer,source.indices.byteOffset,source.indices.byteLength));vertices+=data.length/10;indices+=source.indices.length;
}
const header=Buffer.alloc(20);header.write('VFTREE02');header.writeUInt32LE(5,8);header.writeUInt32LE(vertices,12);header.writeUInt32LE(indices,16);
await mkdir(new URL('../public/previews/0.6.0/live/rocks/assets/',import.meta.url),{recursive:true});
const output=gzipSync(Buffer.concat([header,...chunks]),{level:9});await writeFile(new URL('../public/previews/0.6.0/live/rocks/assets/'+(surfaceContacts?'rigid-stones-mixed-8.bin.gz':'rigid-stones.bin.gz'),import.meta.url),output);
console.log(JSON.stringify({centers,vertices,indices,compressedBytes:output.length}));
