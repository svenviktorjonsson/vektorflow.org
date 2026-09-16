// Immutable initial shape producer, not a simulation or runtime integrator.
import { mkdir,writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
const centers=[[-1.12,0,.69],[0,.08,.91],[1.15,0,.65],[-.63,0,2.15],[.64,0,2.04]];
const sizes=[.72,.91,.64,.54,.44],colors=[[.39,.40,.38],[.24,.25,.23],[.49,.39,.31],[.58,.56,.49],[.36,.38,.35]];
const hash=x=>{x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;};
const unit=x=>hash(x)/4294967296;
const chunks=[];let vertices=0,indices=0;
for(let i=0;i<5;i++){
  const center=centers[i],lat=40,lon=64,seed=hash(8187+i*1193),positions=[],triangles=[];
  const terms=Array.from({length:5},(_,k)=>({n:1+k%3,m:1+k%4,a:(.045+unit(seed+k)*.055)*(k%2?1:-1),phase:unit(seed+k*17)*Math.PI*2}));
  const point=(theta,phi)=>{
    // Pi-periodic polar amplitudes vanish at the poles. Integer azimuthal
    // frequencies guarantee identical values and tangents across the seam.
    const r=sizes[i]*(1+terms.reduce((sum,t)=>sum+t.a*Math.sin(theta)**2*Math.cos(2*t.n*theta+t.m*phi+t.phase),0));
    return [r*Math.sin(theta)*Math.cos(phi)*(1+unit(seed+2)*.09),r*Math.sin(theta)*Math.sin(phi)*(1-unit(seed+3)*.08),r*Math.cos(theta)*(i>2?.89:1)];
  };
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++)positions.push(point(y*Math.PI/lat,x*Math.PI*2/lon));
  for(let y=0;y<lat;y++)for(let x=0;x<lon;x++){const a=y*(lon+1)+x,b=a+lon+1;if(y>0)triangles.push(a,b,a+1);if(y<lat-1)triangles.push(a+1,b,b+1);}
  const data=new Float32Array(positions.length*10);
  for(let y=0;y<=lat;y++)for(let x=0;x<=lon;x++){
    const j=y*(lon+1)+x,theta=y*Math.PI/lat,phi=x*Math.PI*2/lon,p=positions[j],e=.0001;
    const a=point(Math.max(e,Math.min(Math.PI-e,theta+e)),phi),b=point(Math.max(e,Math.min(Math.PI-e,theta-e)),phi),c=point(Math.max(e,Math.min(Math.PI-e,theta)),phi+e),d=point(Math.max(e,Math.min(Math.PI-e,theta)),phi-e);
    const t=a.map((v,k)=>v-b[k]),u=c.map((v,k)=>v-d[k]);let n=[t[1]*u[2]-t[2]*u[1],t[2]*u[0]-t[0]*u[2],t[0]*u[1]-t[1]*u[0]],length=Math.hypot(...n);
    if(length<1e-10){n=[0,0,y===0?1:-1];length=1;}if(n.reduce((s,v,k)=>s+v*p[k],0)<0)n=n.map(v=>-v);
    const noise=(unit(seed+Math.round((x%lon)*193+y*773))-.5)*.055;
    data.set([...p,...n.map(v=>v/length),...colors[i].map(v=>v+noise),1],j*10);
  }
  const source={indices:new Uint32Array(triangles)};
  const hull=[];let radius=0;for(let j=0;j<data.length;j+=10)radius=Math.max(radius,Math.hypot(data[j],data[j+1],data[j+2]));
  for(let k=0;k<96;k++){const z=1-2*k/95,theta=k*2.399963229728653,r=Math.sqrt(1-z*z),n=[r*Math.cos(theta),r*Math.sin(theta),z];let best=-Infinity,at=0;
    for(let j=0;j<data.length;j+=10){const d=data[j]*n[0]+data[j+1]*n[1]+data[j+2]*n[2];if(d>best){best=d;at=j;}}
    hull.push(data[at],data[at+1],data[at+2],0);
  }
  let volume=0;for(let j=0;j<source.indices.length;j+=3){const a=source.indices[j]*10,b=source.indices[j+1]*10,c=source.indices[j+2]*10;
    volume+=(data[a]*(data[b+1]*data[c+2]-data[b+2]*data[c+1])+data[a+1]*(data[b+2]*data[c]-data[b]*data[c+2])+data[a+2]*(data[b]*data[c+1]-data[b+1]*data[c]))/6;
  }
  const mass=Math.max(.1,Math.abs(volume)*2700),inertia=.4*mass*radius*radius;
  const meta=Buffer.from(JSON.stringify({id:`stone-${i}`,collision:{center,mass,inertia,radius,hull},shape:'seam-continuous radial Fourier',seed}));
  const header=Buffer.alloc(20);[meta.length,data.length,source.indices.length,0,0].forEach((n,a)=>header.writeUInt32LE(n,a*4));
  chunks.push(header,meta,Buffer.alloc((4-meta.length%4)%4),Buffer.from(data.buffer),Buffer.from(source.indices.buffer,source.indices.byteOffset,source.indices.byteLength));vertices+=data.length/10;indices+=source.indices.length;
}
const header=Buffer.alloc(20);header.write('VFTREE02');header.writeUInt32LE(5,8);header.writeUInt32LE(vertices,12);header.writeUInt32LE(indices,16);
await mkdir(new URL('../public/previews/0.6.0/live/rocks/assets/',import.meta.url),{recursive:true});
const output=gzipSync(Buffer.concat([header,...chunks]),{level:9});await writeFile(new URL('../public/previews/0.6.0/live/rocks/assets/rigid-stones.bin.gz',import.meta.url),output);
console.log(JSON.stringify({centers,vertices,indices,compressedBytes:output.length}));
