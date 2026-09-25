// Exercise the real published HTML entry point, including startup and rendering.
// Isolated physical-GPU Chrome; no CDP, software adapter or personal profile.
import {createServer} from 'node:http';
import {spawn} from 'node:child_process';
import {readFile,mkdtemp,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const root=path.join(site,'public');
await mkdir(path.join(site,'.work'),{recursive:true});
const work=await mkdtemp(path.join(site,'.work','wheel-page-gpu-'));
let complete;
const result=new Promise(resolve=>complete=resolve);
const probe=`<script>
let auditDevice;const requestDevice=GPUAdapter.prototype.requestDevice;
GPUAdapter.prototype.requestDevice=async function(...args){auditDevice=await requestDevice.apply(this,args);return auditDevice;};
async function checkExclusion(current,{assert=true}={}){
 const world=current.world,n=current.contact.resources.count,stride=world.kind==='liquid'?12:8;
 const read=auditDevice.createBuffer({size:n*(stride+2)*4+256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,n*stride*4);e.copyBufferToBuffer(current.contact.resources.precisionOutput,0,read,n*stride*4,n*8);e.copyBufferToBuffer(current.contact.resources.control,0,read,n*(stride+2)*4,256);auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 // The CPU receipt may advance while mapAsync yields. Use the wheel pose copied
 // in the same GPU submission as the particles, or fast-turn geometry audits
 // report a false penetration against a different frame's baffles.
 const sampledAngle=raw[n*(stride+2)+1];
 const r=world.kind==='liquid'?world.properties.spacing*.46:world.properties.radius,d=2*r,grid=new Map(),c=Math.cos(sampledAngle),s=Math.sin(sampledAngle);let pairGap=Infinity,boundaryGap=Infinity,worstBoundary=null,overlapPairs=0,overlapAreaEstimate=0,worstPair=null,worstCoords=null,kinetic=0,maximumSpeed=0,minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,sumY=0;const surfaceTopByBin=Array(16).fill(-Infinity);
 for(let i=0;i<n;i++){const x=raw[i*stride]+raw[n*stride+2*i]-world.geometry.center[0],y=raw[i*stride+1]+raw[n*stride+2*i+1]-world.geometry.center[1],cx=Math.floor(x/d),cy=Math.floor(y/d);
  minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);sumY+=y;const surfaceBin=Math.max(0,Math.min(15,Math.floor((x+.5)*16)));surfaceTopByBin[surfaceBin]=Math.max(surfaceTopByBin[surfaceBin],y);
  for(let yy=cy-1;yy<=cy+1;yy++)for(let xx=cx-1;xx<=cx+1;xx++)for(const p of grid.get(xx+','+yy)||[]){const distance=Math.hypot(x-p[0],y-p[1]),gap=distance-d;if(gap<0){overlapPairs++;if(world.kind==='granular')overlapAreaEstimate+=2*r*r*Math.acos(Math.min(1,distance/d))-.5*distance*Math.sqrt(Math.max(0,d*d-distance*distance));}if(gap<pairGap){pairGap=gap;worstPair=[p[2],i];worstCoords=[[p[0],p[1]],[x,y]];}}
  const key=cx+','+cy;if(!grid.has(key))grid.set(key,[]);grid.get(key).push([x,y,i]);const rimGap=world.geometry.radius-world.geometry.half_width-r-Math.hypot(x,y);if(rimGap<boundaryGap){boundaryGap=rimGap;worstBoundary={kind:'rim',particle:i,position:[x,y]};}
  const speed2=raw[i*stride+2]**2+raw[i*stride+3]**2;kinetic+=speed2;maximumSpeed=Math.max(maximumSpeed,Math.sqrt(speed2));
  for(let segment=0;segment<world.geometry.segments.length;segment++){const [ax,ay,bx,by]=world.geometry.segments[segment],a=c*ax-s*ay,b=s*ax+c*ay,ex=c*(bx-ax)-s*(by-ay),ey=s*(bx-ax)+c*(by-ay),t=Math.max(0,Math.min(1,((x-a)*ex+(y-b)*ey)/(ex*ex+ey*ey))),gap=Math.hypot(x-a-t*ex,y-b-t*ey)-r-world.geometry.half_width;if(gap<boundaryGap){boundaryGap=gap;worstBoundary={kind:'baffle',segment,particle:i,position:[x,y]};}}
 }
 const worstPairParts=worstPair?.map(i=>({high:[raw[i*stride],raw[i*stride+1]],low:[raw[n*stride+2*i],raw[n*stride+2*i+1]]}));
 if(assert&&!(boundaryGap>=0&&(world.kind==='liquid'||pairGap>=0)))throw Error('Paused configuration overlap: '+JSON.stringify({pairGap,boundaryGap}));return {sampledAngle,pairGap,boundaryGap,worstBoundary,overlapPairs,overlapAreaEstimate,diskAreaFraction:1-overlapAreaEstimate/(n*Math.PI*r*r),worstPair,worstCoords,worstPairParts,meanSpeedSquared:kinetic/n,maximumSpeed,meanY:sumY/n,bounds:[minX,minY,maxX,maxY],surfaceTopByBin,vertices:n};
}
async function checkDiffuseWheel(current){
 const n=current.physics.diffuseCapacity,bytes=n*16;
 const read=auditDevice.createBuffer({size:bytes+256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();
 e.copyBufferToBuffer(current.physics.diffuseRenderBuffer,0,read,0,bytes);
 e.copyBufferToBuffer(current.contact.resources.control,0,read,bytes,256);
 auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 const world=current.world,angle=raw[n*4+1],c=Math.cos(angle),s=Math.sin(angle),center=world.geometry.center;
 let active=0,worstGap=Infinity,worst=null;
 for(let i=0;i<n;i++){
  const x=raw[i*4],y=raw[i*4+1],radius=raw[i*4+2];if(radius<=0)continue;active++;
  const px=x-center[0],py=y-center[1];let gap=world.geometry.radius-world.geometry.half_width-radius-Math.hypot(px,py),kind='rim';
  for(let segment=0;segment<world.geometry.segments.length;segment++){
   const [ax,ay,bx,by]=world.geometry.segments[segment],a=c*ax-s*ay,b=s*ax+c*ay,ex=c*(bx-ax)-s*(by-ay),ey=s*(bx-ax)+c*(by-ay),t=Math.max(0,Math.min(1,((px-a)*ex+(py-b)*ey)/(ex*ex+ey*ey))),barGap=Math.hypot(px-a-t*ex,py-b-t*ey)-radius-world.geometry.half_width;
   if(barGap<gap){gap=barGap;kind='baffle '+segment;}
  }
  if(gap<worstGap){worstGap=gap;worst={index:i,position:[x,y],radius,kind};}
 }
 return {active,worstGap,worst,angle};
}
async function checkWaterVolume(current){
 const world=current.world,n=current.physics.primaryCount,stride=12,bytes=n*stride*4;
 const read=auditDevice.createBuffer({size:bytes+n*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,bytes);e.copyBufferToBuffer(current.contact.resources.precisionOutput,0,read,bytes,n*8);auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 const positions=[],densities=[],sectorCounts=new Uint32Array(72);let minimumDensity=Infinity,maximumDensity=-Infinity,meanDensity=0,resolvedArea=0,physicalVolume=0,nearRimCount=0,radialSum=0;
 const particleArea=current.physics.seed.particleMass/current.physics.policy.restDensity;
 for(let i=0;i<n;i++){const x=raw[i*stride]+raw[n*stride+2*i],y=raw[i*stride+1]+raw[n*stride+2*i+1];positions.push([x,y]);const dx=x-world.geometry.center[0],dy=y-world.geometry.center[1],radius=Math.hypot(dx,dy);radialSum+=radius;if(radius>=.41)nearRimCount++;sectorCounts[Math.min(71,Math.floor((Math.atan2(dy,dx)+Math.PI)/(2*Math.PI)*72))]++;const density=raw[i*stride+7];densities.push(density);minimumDensity=Math.min(minimumDensity,density);maximumDensity=Math.max(maximumDensity,density);meanDensity+=density;resolvedArea+=current.physics.seed.particleMass/Math.max(density,current.physics.policy.restDensity);physicalVolume+=raw[i*stride+11];}
 meanDensity/=n;
 // The GPU stores density from the start of the last advection step. Rebuild
 // the material-only SPH density from the positions actually read above so
 // stale telemetry cannot be mistaken for compression or conservation.
 const support=current.physics.seed.supportRadius,mass=current.physics.seed.particleMass;
 const buckets=new Map();for(let i=0;i<n;i++){const p=positions[i],key=Math.floor(p[0]/support)+','+Math.floor(p[1]/support);if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(i);}
 const kernel=(dx,dy)=>{const q=Math.hypot(dx,dy)/support;if(q>=1)return 0;return 7/(Math.PI*support*support)*(1-q)**4*(1+4*q);};
 let freshResolvedArea=0,maximumDensityStaleness=0,freshMaximumDensity=0;
 for(let i=0;i<n;i++){const p=positions[i],cx=Math.floor(p[0]/support),cy=Math.floor(p[1]/support);let fresh=0;
  for(let y=cy-1;y<=cy+1;y++)for(let x=cx-1;x<=cx+1;x++)for(const j of buckets.get(x+','+y)||[])fresh+=mass*kernel(p[0]-positions[j][0],p[1]-positions[j][1]);
  freshResolvedArea+=mass/Math.max(fresh,current.physics.policy.restDensity);freshMaximumDensity=Math.max(freshMaximumDensity,fresh);maximumDensityStaleness=Math.max(maximumDensityStaleness,Math.abs(fresh-densities[i]));}
 densities.sort((a,b)=>a-b);const percentile=q=>densities[Math.min(n-1,Math.floor((n-1)*q))];
 const initial=[];for(let i=0;i<n;i++)initial.push([current.physics.seed.floats[i*stride],current.physics.seed.floats[i*stride+1]]);
 const radius=Math.sqrt(particleArea/Math.PI),cell=world.properties.spacing/4;
 function occupiedArea(points){let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const [x,y] of points){minX=Math.min(minX,x-radius);minY=Math.min(minY,y-radius);maxX=Math.max(maxX,x+radius);maxY=Math.max(maxY,y+radius);}const cells=new Set(),reach=Math.ceil(radius/cell);
  for(const [x,y] of points){const cx=Math.floor((x-minX)/cell),cy=Math.floor((y-minY)/cell);for(let yy=cy-reach;yy<=cy+reach;yy++)for(let xx=cx-reach;xx<=cx+reach;xx++){const px=minX+(xx+.5)*cell,py=minY+(yy+.5)*cell;if((px-x)**2+(py-y)**2<=radius**2)cells.add(xx+','+yy);}}
  return {area:cells.size*cell*cell,bounds:[minX,minY,maxX,maxY]};}
 const baseline=occupiedArea(initial),actual=occupiedArea(positions),occupiedAreaFraction=actual.area/baseline.area,totalArea=n*particleArea,retainedFraction=physicalVolume/totalArea,numericalDensityVolumeFraction=resolvedArea/totalArea;
 // Reproduce the embedding's additive isotropic splat and iso threshold. A
 // union of particle discs is not the rendered fluid surface when parcels
 // overlap, so keep the two measurements separate.
 function embeddedArea(points){const radius=support,step=world.properties.spacing/3,threshold=1.32;
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of points){minX=Math.min(minX,p[0]-radius);minY=Math.min(minY,p[1]-radius);maxX=Math.max(maxX,p[0]+radius);maxY=Math.max(maxY,p[1]+radius);}
  const width=Math.ceil((maxX-minX)/step),height=Math.ceil((maxY-minY)/step),field=new Float32Array(width*height);
  for(const p of points){const left=Math.max(0,Math.floor((p[0]-radius-minX)/step)),right=Math.min(width-1,Math.ceil((p[0]+radius-minX)/step)),bottom=Math.max(0,Math.floor((p[1]-radius-minY)/step)),top=Math.min(height-1,Math.ceil((p[1]+radius-minY)/step));
   for(let y=bottom;y<=top;y++)for(let x=left;x<=right;x++){const dx=(minX+(x+.5)*step-p[0])/radius,dy=(minY+(y+.5)*step-p[1])/radius,r2=dx*dx+dy*dy;if(r2<1)field[y*width+x]+=(1-r2)**3;}}
  let covered=0;for(const value of field)if(value>=threshold)covered++;return covered*step*step;}
 const initialEmbeddedArea=embeddedArea(initial),embeddedSurfaceArea=embeddedArea(positions);
 const telemetry=await current.physics.readTelemetry();
 return {vertices:n,totalArea,physicalVolume,retainedFraction,resolvedArea,numericalDensityVolumeFraction,freshDensityVolumeFraction:freshResolvedArea/totalArea,freshMaximumDensity,maximumDensityStaleness,baselineOccupiedArea:baseline.area,occupiedArea:actual.area,occupiedAreaFraction,initialEmbeddedArea,embeddedSurfaceArea,embeddedSurfaceAreaFraction:embeddedSurfaceArea/initialEmbeddedArea,meanRadius:radialSum/n,nearRimFraction:nearRimCount/n,occupiedSectors:[...sectorCounts].filter(count=>count>0).length,sectorCounts:[...sectorCounts],initialBounds:baseline.bounds,bounds:actual.bounds,minimumDensity,meanDensity,p90Density:percentile(.9),p99Density:percentile(.99),p999Density:percentile(.999),maximumDensity,telemetry};
}
async function measurePhysicsSteps(current){
 if(!auditDevice.features.has('timestamp-query'))throw Error('GPU timestamp query unavailable');
 current.paused=true;await auditDevice.queue.onSubmittedWorkDone();
 const query=auditDevice.createQuerySet({type:'timestamp',count:2});
 const resolved=auditDevice.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=auditDevice.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=auditDevice.createCommandEncoder();
 const start=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:0}});start.end();
 current.physics.stepMany(encoder,4);
 const end=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:1}});end.end();
 encoder.resolveQuerySet(query,0,2,resolved,0);encoder.copyBufferToBuffer(resolved,0,read,0,16);
 const wallStart=performance.now();auditDevice.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
 const values=new BigUint64Array(read.getMappedRange().slice(0));read.unmap();
 const gpuMs=Number(values[1]-values[0])/1e6,wallMs=performance.now()-wallStart;
 query.destroy();resolved.destroy();read.destroy();return {gpuMs,wallMs,steps:4};
}
async function measureEmbedding(current,app){
 const query=auditDevice.createQuerySet({type:'timestamp',count:3});
 const resolved=auditDevice.createBuffer({size:24,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=auditDevice.createBuffer({size:24,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const encoder=auditDevice.createCommandEncoder();
 const mark=i=>{const pass=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:i}});pass.end();};
 mark(0);current.embedding.render(encoder,{time:current.time,wheelAngle:current.angle,
  mode:current.world.kind==='granular'?'sand':'fluid'});mark(1);
 current.boundary.render(encoder,app.canvas.getContext('webgpu').getCurrentTexture().createView(),current.physics.policy,current.physics.wheel,current.angle);mark(2);
 encoder.resolveQuerySet(query,0,3,resolved,0);encoder.copyBufferToBuffer(resolved,0,read,0,24);
 const wallStart=performance.now();auditDevice.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
 const values=new BigUint64Array(read.getMappedRange().slice(0));read.unmap();
 const result={embeddingGpuMs:Number(values[1]-values[0])/1e6,boundaryGpuMs:Number(values[2]-values[1])/1e6,wallMs:performance.now()-wallStart};
 query.destroy();resolved.destroy();read.destroy();return result;
}
async function checkSandVisualMotion(current){
 const guides=current.embedding.particleCount,visuals=guides*8,bytes=(guides+visuals)*32;
 const read=auditDevice.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();
 e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,guides*32);
 e.copyBufferToBuffer(current.embedding.visualBuffer,0,read,guides*32,visuals*32);
 auditDevice.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
 const raw=read.getMappedRange().slice(0),f=new Float32Array(raw),u=new Uint32Array(raw);read.unmap();read.destroy();
 let freefallGuides=0,downward=0,relativeSpeed=0,spread=0,observations=0;
 for(let i=0;i<guides;i++){
  const g=i*8,guideSpeed=Math.hypot(f[g+2],f[g+3]);
  if(u[g+7]!==0||guideSpeed<.35)continue;
  freefallGuides++;
  for(let j=0;j<8;j++){
   const v=(guides+i*8+j)*8;
   downward+=f[v+3]<0?1:0;
   relativeSpeed+=Math.hypot(f[v+2]-f[g+2],f[v+3]-f[g+3]);
   spread+=Math.hypot(f[v]-f[g],f[v+1]-f[g+1]);observations++;
  }
 }
 return {freefallGuides,visualSamples:observations,downwardFraction:downward/Math.max(1,observations),meanRelativeSpeed:relativeSpeed/Math.max(1,observations),meanGuideSpread:spread/Math.max(1,observations)};
}
const rejectCaches=${process.argv.includes('--reject-caches')};
const legacyOverrides=${process.argv.includes('--legacy-overrides')};
if(legacyOverrides){const create=GPUDevice.prototype.createComputePipelineAsync;
 GPUDevice.prototype.createComputePipelineAsync=function(descriptor){
  if(descriptor.label?.startsWith('VKF contact ')&&Object.keys(descriptor.compute.constants??{}).length){
   const error=new Error('Compute library failed creation');error.name='GPUPipelineError';error.reason='validation';error.stack='_';return Promise.reject(error);
  }return create.call(this,descriptor);
 };
}
if(rejectCaches){const create=GPUDevice.prototype.createComputePipelineAsync;
 GPUDevice.prototype.createComputePipelineAsync=function(descriptor){
  if(['primal_cached_component','preventive_cache_pressure_geometry'].includes(descriptor.compute.entryPoint)){
   const error=new Error('Injected WebKit optional-cache compilation rejection');error.name='GPUPipelineError';error.reason='validation';error.stack='_';return Promise.reject(error);
  }return create.call(this,descriptor);
 };
}
const fastSpin=${process.argv.includes('--water-fast-spin')},wallSpin=${process.argv.includes('--water-wall-spin')},steadySpin=${process.argv.includes('--water-steady-spin')||process.argv.includes('--water-wall-spin')},steadyOmega=${process.argv.includes('--water-wall-spin')?6:2},waterRelax=${process.argv.includes('--water-relax')||process.argv.includes('--water-relax-long')||process.argv.includes('--water-fast-spin')},waterTurnGoal=${process.argv.includes('--water-relax-long')?32:8},waterTurnIncrement=fastSpin?1.6:.4,sandSteps=${process.argv.includes('--sand-step-diagnose')},sandMotion=${process.argv.includes('--sand-motion')},sandEquilibrium=${process.argv.includes('--sand-equilibrium')},sandDropcastle=${process.argv.includes('--sand-dropcastle-dry')||process.argv.includes('--sand-dropcastle-wet')},sandVisual=${process.argv.includes('--sand-visual')},sandDropcastleWet=${process.argv.includes('--sand-dropcastle-wet')},sandGpuProfile=${process.argv.includes('--sand-gpu-profile')},checkSand=${process.argv.includes('--sand-paused')||process.argv.includes('--sand-motion')||process.argv.includes('--sand-equilibrium')||process.argv.includes('--sand-dropcastle-dry')||process.argv.includes('--sand-dropcastle-wet')||process.argv.includes('--sand-visual')||process.argv.includes('--sand-step-diagnose')||process.argv.includes('--sand-gpu-profile')},checkDrag=${process.argv.includes('--paused-drag')},checkVolume=${process.argv.includes('--water-volume')},profileFps=${process.argv.includes('--fps-profile')},gpuProfile=${process.argv.includes('--gpu-profile')},sustained=${process.argv.includes('--sustained')},runningDrag=${process.argv.includes('--running-drag')},started=performance.now(), faults=[];let last='',played=false,pausedReceipt,sandStarted=false,sandPlayed=false,sandSample=0,sandPlayFrame,sandPlayElapsed,waterReceipt,sandFrame,dragTarget,dragStarted,sandReleaseStart,sandImmediate,pausedProfileStart,waterTurnCount=0,lastWaterTurnSample=0,waterReleaseTime,waterSnapshots=[],steadySpinSamples=[],sandEquilibriumSamples=[],sandFormationRequested=false,sandFormationInitial;
const waterTurnSamples=[];let steadySpinStartTime,steadySpinStartAngle;
let browserRafFrames=0;
function countBrowserRaf(){browserRafFrames++;requestAnimationFrame(countBrowserRaf);}
requestAnimationFrame(countBrowserRaf);
addEventListener('error',e=>faults.push(String(e.error||e.message)));
addEventListener('unhandledrejection',e=>faults.push(String(e.reason?.stack||e.reason)));
async function inspect(){
 const app=globalThis.__vfWorldLayerApplication;
 const selectedView=app?.program.views[app.program.active_view];
 const current=app?.applications?.find(a=>a.world.world_id===selectedView?.world_id&&a.world.layer_id===selectedView?.embedding?.layer);
 const state={elapsedMs:performance.now()-started,ready:document.body?.dataset.vfWorldLayerReady,
  worlds:app?.applications?.map(a=>a.world.kind),time:current?.time,angle:current?.angle,target:current?.targetAngle,logicalAngle:current?.logicalAngle,remainingTime:current?.remainingTime,peakWheelSurfaceSpeed:current?.peakWheelSurfaceSpeed,
  frames:Number(app?.canvas?.dataset.presentedFrames||0),status:document.querySelector('#vf-material-status')?.textContent,
  error:globalThis.__vfWorldLayerError||document.querySelector('[role=alert]:not([hidden])')?.textContent,faults};
 const signature=JSON.stringify([state.ready,state.worlds,state.status,state.error,checkDrag?state.angle:null,faults]);
 if(signature!==last){last=signature;await fetch('/page-progress',{method:'POST',body:JSON.stringify(state)});}
 if(document.querySelector('#vf-material-stage')&&!state.status)state.error='Startup replaced its loading message with a blank frame';
 if(!sandStarted&&state.worlds?.includes('granular'))state.error='Inactive sand initialized before the selected water View';
 if(state.error||faults.length||state.elapsedMs>((waterRelax||steadySpin)?220000:60000)){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,...state})});return;}
 if(!played&&state.ready==='true'&&state.frames>=3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
  if(!current.paused||state.time!==0||!play){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Selected World did not start visibly paused with Play',...state})});return;}
  if(profileFps){
   if(!pausedProfileStart){pausedProfileStart={elapsedMs:state.elapsedMs,frames:state.frames,rafFrames:browserRafFrames};setTimeout(inspect,250);return;}
   if(state.elapsedMs-pausedProfileStart.elapsedMs<3000){setTimeout(inspect,250);return;}
  }
  const pausedTelemetry=await current.physics.readTelemetry();
  pausedReceipt={elapsedMs:state.elapsedMs,time:state.time,frames:state.frames,
    worlds:state.worlds,telemetry:pausedTelemetry,
    fps:profileFps?(state.frames-pausedProfileStart.frames)*1000/(state.elapsedMs-pausedProfileStart.elapsedMs):undefined,
    browserRafFps:profileFps?(browserRafFrames-pausedProfileStart.rafFrames)*1000/(state.elapsedMs-pausedProfileStart.elapsedMs):undefined,
    browserRafFrames:profileFps?browserRafFrames:undefined};
  if(pausedTelemetry.nonFinite){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Paused water initialized with non-finite state',pausedReceipt,...state})});return;}
  if(checkDrag){
   if(dragTarget===undefined){dragTarget=current.angle+1.2;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
   if(Math.abs(current.logicalAngle-dragTarget)<.005){const dragMs=performance.now()-dragStarted,exclusion=await checkExclusion(current);await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:current.time===0,pausedReceipt,dragMs,exclusion,...state})});return;}
   if(performance.now()-dragStarted>3000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Paused 1.2-radian wheel drag not accepted within 3 seconds',dragMs:performance.now()-dragStarted,...state})});return;}
   setTimeout(inspect,100);return;
  }
  played=true;play.click();
 }
 if(played&&checkVolume&&state.time>=(sustained?12:5)){
  // A conserved per-particle restVolume is not proof that the simulated
  // material stayed incompressible. Audit the volume implied by the measured
  // density field too; this catches visually collapsing water.
  const conservation=await checkWaterVolume(current),passed=conservation.retainedFraction>=.9999&&conservation.retainedFraction<=1.000001&&conservation.freshDensityVolumeFraction>=.99&&conservation.embeddedSurfaceAreaFraction>=.99&&conservation.minimumDensity>0&&conservation.p90Density<=current.physics.policy.restDensity*1.04&&conservation.p999Density<=current.physics.policy.restDensity*1.15&&!conservation.telemetry.nonFinite&&!conservation.telemetry.gridOverflow;
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,conservation,pausedReceipt,...state,error:passed?undefined:'Water density or reconstructed surface missed the 99% floor'})});return;
 }
 if(played&&gpuProfile&&state.time>=(sustained?12:5)){
  const telemetry=await current.physics.readTelemetry(),physics=await measurePhysicsSteps(current),embedding=await measureEmbedding(current,app);
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,physics,embedding,telemetry,pausedReceipt,...state})});return;
 }
 if(played&&steadySpin){
  // Follow simulation time, not wall time: a slow browser must not turn this
  // into an accidental high-speed impulse. Wall-spin is above the Froude=1
  // threshold (sqrt(g/r) ~= 4.43 rad/s) for a half-metre-radius drum.
  if(steadySpinStartTime===undefined){steadySpinStartTime=state.time;steadySpinStartAngle=current.logicalAngle;}
  const target=steadySpinStartAngle+steadyOmega*(state.time-steadySpinStartTime);
  if(target-current.targetAngle>.015)app.setLayer(current.world.boundary_ids[0],{rotation:target});
  for(const mark of [3,12,24])if(state.time>=mark&&!steadySpinSamples.some(s=>s.mark===mark)){
   const volume=await checkWaterVolume(current),shape=await checkExclusion(current,{assert:false});
   steadySpinSamples.push({mark,volume,shape,peakWheelSurfaceSpeed:state.peakWheelSurfaceSpeed});
   if(mark===24)await fetch('/page-screenshot?name=water-steady',{method:'POST',body:app.canvas.toDataURL('image/png')});
   await fetch('/page-progress',{method:'POST',body:JSON.stringify({steadySpinMark:mark,volume,shape})});
  }
  if(steadySpinSamples.length===3){const late=steadySpinSamples.at(-1),passed=steadySpinSamples.every(s=>s.shape.boundaryGap>=-1e-5&&!s.volume.telemetry.gridOverflow)&&late.volume.freshDensityVolumeFraction>=.99&&late.volume.embeddedSurfaceAreaFraction>=.99&&(!wallSpin||late.volume.nearRimFraction>=.8&&late.volume.occupiedSectors>=68);await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,steadyOmega,steadySpinSamples,...state,error:passed?undefined:'Late steady-spin volume, wall coverage, or exclusion failed'})});return;}
  setTimeout(inspect,16);return;
 }
 if(played&&waterRelax&&state.time>=.5){
  if(waterTurnCount>0&&(fastSpin||waterTurnCount%8===0)&&waterTurnCount!==lastWaterTurnSample&&Math.abs(current.logicalAngle-dragTarget)<.005){lastWaterTurnSample=waterTurnCount;const shape=await checkExclusion(current,{assert:false}),telemetry=await current.physics.readTelemetry(),diffuse=fastSpin?await checkDiffuseWheel(current):undefined,volume=fastSpin&&waterTurnCount===8?await checkWaterVolume(current):undefined;waterTurnSamples.push({turn:waterTurnCount,shape,telemetry,diffuse,volume});if(fastSpin&&waterTurnCount===2)await fetch('/page-screenshot?name=water-foam',{method:'POST',body:app.canvas.toDataURL('image/png')});await fetch('/page-progress',{method:'POST',body:JSON.stringify({waterTurnCount,shape,diffuse,volume,peakSpeed:telemetry.peakSpeed})});}
  if(waterTurnCount<waterTurnGoal&&((dragTarget===undefined)||Math.abs(current.logicalAngle-dragTarget)<.005)){
   dragTarget=(dragTarget??current.logicalAngle)+waterTurnIncrement;waterTurnCount++;app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});
  }else if(waterTurnCount===waterTurnGoal&&waterReleaseTime===undefined&&Math.abs(current.logicalAngle-dragTarget)<.005){waterReleaseTime=state.time;waterSnapshots.push({after:0,shape:await checkExclusion(current,{assert:false})});}
  if(waterReleaseTime!==undefined){for(const mark of [2,5,10])if(state.time-waterReleaseTime>=mark&&!waterSnapshots.some(s=>s.after===mark))waterSnapshots.push({after:mark,shape:await checkExclusion(current,{assert:false})});if(waterSnapshots.length===4){const release=waterSnapshots[0].shape,rest=waterSnapshots[3].shape,conservation=fastSpin?await checkWaterVolume(current):undefined;
   // A browser frame may request a different wheel angular speed on each run.
   // Compare stored water speed with measured rim speed plus the free-fall
   // speed across this one-metre vessel, not a fixed arbitrary speed cap.
   const mechanicalSpeedCeiling=(state.peakWheelSurfaceSpeed??0)+Math.sqrt(2*9.82*1)+1;
   const diffuse=fastSpin?await checkDiffuseWheel(current):undefined;
   const diffuseClear=!fastSpin||[...waterTurnSamples.map(s=>s.diffuse),diffuse].every(s=>s.active===0||s.worstGap>=-1e-4);
   const passed=[...waterTurnSamples,...waterSnapshots].every(s=>s.shape.boundaryGap>=-1e-5)&&rest.meanSpeedSquared<Math.min(.05*.05,release.meanSpeedSquared*.01)&&diffuseClear&&(!fastSpin||(conservation.freshDensityVolumeFraction>=.99&&conservation.embeddedSurfaceAreaFraction>=.99&&!conservation.telemetry.gridOverflow&&conservation.telemetry.peakSpeed<mechanicalSpeedCeiling));await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,mechanicalSpeedCeiling,waterTurnCount,waterTurnSamples,waterSnapshots,conservation,diffuse,...state,error:passed?undefined:'Water relaxation, volume, foam, peak speed, or wheel-boundary exclusion failed'})});return;}}
 }
 if(played&&runningDrag){
  if(dragTarget===undefined&&state.time>=5){const beforeDrag=await checkExclusion(current,{assert:false});await fetch('/page-progress',{method:'POST',body:JSON.stringify({beforeDrag})});dragTarget=current.angle+.4;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
  if(dragTarget!==undefined&&Math.abs(current.logicalAngle-dragTarget)<.005){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,dragMs:performance.now()-dragStarted,...state})});return;}
  if(dragTarget!==undefined&&performance.now()-dragStarted>7000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Running wheel drag not accepted within 7 seconds',pausedReceipt,...state})});return;}
 }
 if(played&&profileFps&&state.elapsedMs-pausedReceipt.elapsedMs>=(sustained?15000:5000)){
  const wallSeconds=(state.elapsedMs-pausedReceipt.elapsedMs)/1000;
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,playing:{fps:(state.frames-pausedReceipt.frames)/wallSeconds,browserRafFps:(browserRafFrames-pausedReceipt.browserRafFrames)/wallSeconds,realtimeFactor:(state.time-pausedReceipt.time)/wallSeconds},...state})});return;
 }
 if(played&&!checkVolume&&!profileFps&&!gpuProfile&&!waterRelax&&!runningDrag&&!sandStarted&&state.frames>=6&&state.time>(sustained?12:.02)){
  if(!checkSand){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,...state})});return;}
 waterReceipt={...state};sandFrame=state.frames;sandStarted=true;
  const particles=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Particles');
  particles.click();if(particles.getAttribute('aria-pressed')!=='true')throw Error('Raw particle embedding did not activate');
  particles.click();if(particles.getAttribute('aria-pressed')!=='false')throw Error('Material embedding did not reactivate');
  [...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Sand').click();
 }
 if(sandStarted&&current?.world.kind==='granular'&&state.frames>=sandFrame+3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
  if(sandSteps){const snapshots=[];for(let step=1;step<=16;step++){const encoder=auditDevice.createCommandEncoder();current.physics.stepMany(encoder,1);auditDevice.queue.submit([encoder.finish()]);await auditDevice.queue.onSubmittedWorkDone();if([1,2,4,8,16].includes(step))snapshots.push({step,shape:await checkExclusion(current,{assert:false}),telemetry:await current.physics.readTelemetry()});}await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,snapshots,...state})});return;}
  if(sandGpuProfile){const physics=await measurePhysicsSteps(current),embedding=await measureEmbedding(current,app),telemetry=await current.physics.readTelemetry();await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,physics,embedding,telemetry,...state})});return;}
  if(sandDropcastle||sandVisual){
   const formation=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Dropcastle'||b.textContent==='Level bed');
   if(!sandFormationRequested){
    const selection=sandVisual?[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Sandfall'):formation;
    if(!selection||(!sandVisual&&selection.textContent!=='Dropcastle'))throw Error('Sand formation control unavailable');
    if(sandDropcastleWet){const slider=document.querySelector('input[aria-label="Wetness"]');slider.value='33';slider.dispatchEvent(new Event('input',{bubbles:true}));}
    sandFormationRequested=true;selection.click();setTimeout(inspect,250);return;
   }
   if(formation?.textContent!=='Level bed'){setTimeout(inspect,250);return;}
   if(!sandPlayed){if(!current.paused||state.time!==0||!play)throw Error('Dropcastle did not start paused');sandFormationInitial=await checkExclusion(current,{assert:false});sandPlayed=true;play.click();}
   if(sandVisual&&state.time>=.3){
    const visual=await checkSandVisualMotion(current);
    await fetch('/page-screenshot',{method:'POST',body:app.canvas.toDataURL('image/png')});
    const passed=visual.freefallGuides>=5&&visual.downwardFraction>.5&&visual.meanRelativeSpeed>.005;
    await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,visual,...state,error:passed?undefined:'Fine grains did not visibly advect during free fall'})});return;
   }
   if(state.time>=10){const shape=await checkExclusion(current,{assert:false});await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:shape.diskAreaFraction>=.995&&shape.boundaryGap>=-1e-5,wetness:sandDropcastleWet?0.33:0,initial:sandFormationInitial,shape,...state})});return;}
   setTimeout(inspect,250);return;
  }
  if(sandEquilibrium){
   if(!sandPlayed){if(!current.paused||state.time!==0||!play)throw Error('Sand did not start paused');sandEquilibriumSamples.push({time:0,shape:await checkExclusion(current,{assert:false})});sandPlayed=true;sandPlayFrame=state.frames;sandPlayElapsed=state.elapsedMs;play.click();}
   for(const mark of [5,10,20])if(state.time>=mark&&!sandEquilibriumSamples.some(s=>s.time===mark))sandEquilibriumSamples.push({time:mark,shape:await checkExclusion(current,{assert:false})});
   if(sandEquilibriumSamples.length===4){const first=sandEquilibriumSamples[0].shape,last=sandEquilibriumSamples[3].shape,centralRise=Math.max(...last.surfaceTopByBin.slice(5,11).map((height,i)=>height-first.surfaceTopByBin[i+5])),wallSeconds=(state.elapsedMs-sandPlayElapsed)/1000,passed=centralRise<.025&&last.maximumSpeed<.15&&last.boundaryGap>=-1e-5;await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,centralRise,sandEquilibriumSamples,fps:(state.frames-sandPlayFrame)/wallSeconds,realtimeFactor:state.time/wallSeconds,...state,error:passed?undefined:'Dry bed piled, kept moving, or entered the wheel'})});return;}
   setTimeout(inspect,250);return;
  }
  if(sandMotion){
   if(!sandPlayed){if(!current.paused||state.time!==0||!play)throw Error('Sand did not start paused');const initial=await checkExclusion(current,{assert:false});await fetch('/page-progress',{method:'POST',body:JSON.stringify({sandInitial:initial})});sandPlayed=true;sandPlayFrame=state.frames;sandPlayElapsed=state.elapsedMs;play.click();}
   if(sandSample<4&&state.time>=[.2,.5,1,2][sandSample]){const shape=await checkExclusion(current,{assert:false}),telemetry=await current.physics.readTelemetry();await fetch('/page-progress',{method:'POST',body:JSON.stringify({sandSample:sandSample++,time:state.time,shape,telemetry})});}
   if(dragTarget===undefined&&state.time>=2){const beforeDrag=await checkExclusion(current,{assert:false}),telemetry=await current.physics.readTelemetry();await fetch('/page-progress',{method:'POST',body:JSON.stringify({sandBeforeDrag:beforeDrag,sandTelemetry:telemetry})});dragTarget=current.angle+.4;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
   if(dragTarget!==undefined&&Math.abs(current.logicalAngle-dragTarget)<.005){if(sandReleaseStart===undefined){sandReleaseStart=state.time;sandImmediate=await checkExclusion(current,{assert:false});}if(state.time-sandReleaseStart>=10){const shape=await checkExclusion(current,{assert:false}),passed=shape.vertices===sandImmediate.vertices&&shape.diskAreaFraction>=.995&&shape.pairGap>=-current.physics.policy.grainRadius*.3&&shape.boundaryGap>=-1e-5&&shape.maximumSpeed<.15,wallSeconds=(state.elapsedMs-sandPlayElapsed)/1000;await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,pausedReceipt,waterReceipt,sandMotion:{time:state.time,dragMs:performance.now()-dragStarted,immediate:sandImmediate,shape,fps:(state.frames-sandPlayFrame)/wallSeconds,realtimeFactor:state.time/wallSeconds},...state,error:passed?undefined:'Sand support coverage, guide penetration, wheel exclusion, or relaxation failed'})});return;}}
   if(dragTarget!==undefined&&sandReleaseStart===undefined&&performance.now()-dragStarted>7000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Running sand drag not accepted within 7 seconds',...state})});return;}
   setTimeout(inspect,250);return;
  }
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:current.paused&&state.time===0&&!!play,pausedReceipt,waterReceipt,sandScope:'Paused sand startup/render only; not sand motion acceptance',...state})});return;
 }
 setTimeout(inspect,250);
}
setTimeout(inspect,0);
</script>`;
const server=createServer(async(req,res)=>{
 try{
  const url=new URL(req.url,'http://localhost');
  if(req.method==='POST'&&['/page-progress','/page-result','/page-screenshot'].includes(url.pathname)){
   let body='';for await(const chunk of req){body+=chunk;if(body.length>(url.pathname==='/page-screenshot'?3000000:100000))throw Error('Oversize probe');}
   if(url.pathname==='/page-screenshot'){
    if(!body.startsWith('data:image/png;base64,'))throw Error('Invalid screenshot payload');
    const requestedName=url.searchParams.get('name');
    const name=['water-foam','water-steady'].includes(requestedName)?requestedName:'sand-visual';
    await writeFile(path.join(work,name+'.png'),Buffer.from(body.slice(22),'base64'));
    res.end('ok');return;
   }
   res.end('ok');const data=JSON.parse(body);if(url.pathname==='/page-result')complete(data);else console.log(body);return;
  }
  const route=url.pathname.endsWith('/')?url.pathname+'index.html':url.pathname;
  const target=path.resolve(root,'.'+route),relative=path.relative(root,target);
  if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(404).end();return;}
  let bytes=await readFile(target);
  if(route==='/previews/0.6.0/live/material-wheel/index.html')bytes=Buffer.from(bytes.toString().replace('<head>','<head>'+probe));
  res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json'})[path.extname(target)]||'application/octet-stream');
  res.end(bytes);
 }catch(error){res.writeHead(500).end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const args=['--headless=new','--enable-gpu','--no-first-run','--no-default-browser-check',
 '--window-size=591,1280',`--user-data-dir=${path.join(work,'chrome-profile')}`,
 `http://127.0.0.1:${server.address().port}/previews/0.6.0/live/material-wheel/`];
const chrome=spawn(process.env.VKF_GPU_BROWSER||'C:/Program Files/Google/Chrome/Application/chrome.exe',args,{windowsHide:true});
let stderr='';chrome.stdout.resume();chrome.stderr.on('data',b=>stderr=(stderr+b).slice(-12000));
chrome.on('error',error=>complete({passed:false,error:String(error)}));
chrome.on('exit',code=>complete({passed:false,error:`Chrome exited ${code}`}));
const timeout=setTimeout(()=>complete({passed:false,error:'Actual wheel page did not finish within its deadline',stderr}),process.argv.includes('--water-steady-spin')||process.argv.includes('--water-wall-spin')?240000:process.argv.includes('--water-relax')||process.argv.includes('--water-relax-long')?150000:90000);
try{
 const report=await result;report.scope='Actual HTML startup and GPU embedding; '+(process.argv.includes('--legacy-overrides')?'older WebKit unused-override rejection emulation; ':'')+(process.argv.includes('--reject-caches')?'injected optional-cache failures, portable path':'production defaults')+'; not a phone test';
 report.chromeArguments=args;await writeFile(path.join(work,'result.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({...report,result:path.join(work,'result.json')}));if(!report.passed)process.exitCode=1;
}finally{clearTimeout(timeout);chrome.kill();server.closeAllConnections();server.close();}
