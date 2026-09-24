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
 const read=auditDevice.createBuffer({size:n*(stride+2)*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,n*stride*4);e.copyBufferToBuffer(current.contact.resources.precisionOutput,0,read,n*stride*4,n*8);auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 const r=world.kind==='liquid'?world.properties.spacing*.46:world.properties.radius,d=2*r,grid=new Map(),c=Math.cos(current.angle),s=Math.sin(current.angle);let pairGap=Infinity,boundaryGap=Infinity;
 for(let i=0;i<n;i++){const x=raw[i*stride]+raw[n*stride+2*i]-world.geometry.center[0],y=raw[i*stride+1]+raw[n*stride+2*i+1]-world.geometry.center[1],cx=Math.floor(x/d),cy=Math.floor(y/d);
  for(let yy=cy-1;yy<=cy+1;yy++)for(let xx=cx-1;xx<=cx+1;xx++)for(const p of grid.get(xx+','+yy)||[])pairGap=Math.min(pairGap,Math.hypot(x-p[0],y-p[1])-d);
  const key=cx+','+cy;if(!grid.has(key))grid.set(key,[]);grid.get(key).push([x,y]);boundaryGap=Math.min(boundaryGap,world.geometry.radius-world.geometry.half_width-r-Math.hypot(x,y));
  for(const [ax,ay,bx,by] of world.geometry.segments){const a=c*ax-s*ay,b=s*ax+c*ay,ex=c*(bx-ax)-s*(by-ay),ey=s*(bx-ax)+c*(by-ay),t=Math.max(0,Math.min(1,((x-a)*ex+(y-b)*ey)/(ex*ex+ey*ey)));boundaryGap=Math.min(boundaryGap,Math.hypot(x-a-t*ex,y-b-t*ey)-r-world.geometry.half_width);}
 }
 if(assert&&!(boundaryGap>=0&&(world.kind==='liquid'||pairGap>=0)))throw Error('Paused configuration overlap: '+JSON.stringify({pairGap,boundaryGap}));return {pairGap,boundaryGap,vertices:n};
}
async function checkWaterVolume(current){
 const world=current.world,n=current.physics.primaryCount,stride=12,bytes=n*stride*4;
 const read=auditDevice.createBuffer({size:bytes+n*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,bytes);e.copyBufferToBuffer(current.contact.resources.precisionOutput,0,read,bytes,n*8);auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 const positions=[],densities=[];let minimumDensity=Infinity,maximumDensity=-Infinity,meanDensity=0,resolvedArea=0,physicalVolume=0;
 const particleArea=current.physics.seed.particleMass/current.physics.policy.restDensity;
 for(let i=0;i<n;i++){positions.push([raw[i*stride]+raw[n*stride+2*i],raw[i*stride+1]+raw[n*stride+2*i+1]]);const density=raw[i*stride+7];densities.push(density);minimumDensity=Math.min(minimumDensity,density);maximumDensity=Math.max(maximumDensity,density);meanDensity+=density;resolvedArea+=current.physics.seed.particleMass/Math.max(density,current.physics.policy.restDensity);physicalVolume+=raw[i*stride+11];}
 meanDensity/=n;
 densities.sort((a,b)=>a-b);const percentile=q=>densities[Math.min(n-1,Math.floor((n-1)*q))];
 const initial=[];for(let i=0;i<n;i++)initial.push([current.physics.seed.floats[i*stride],current.physics.seed.floats[i*stride+1]]);
 const radius=Math.sqrt(particleArea/Math.PI),cell=world.properties.spacing/4;
 function occupiedArea(points){let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const [x,y] of points){minX=Math.min(minX,x-radius);minY=Math.min(minY,y-radius);maxX=Math.max(maxX,x+radius);maxY=Math.max(maxY,y+radius);}const cells=new Set(),reach=Math.ceil(radius/cell);
  for(const [x,y] of points){const cx=Math.floor((x-minX)/cell),cy=Math.floor((y-minY)/cell);for(let yy=cy-reach;yy<=cy+reach;yy++)for(let xx=cx-reach;xx<=cx+reach;xx++){const px=minX+(xx+.5)*cell,py=minY+(yy+.5)*cell;if((px-x)**2+(py-y)**2<=radius**2)cells.add(xx+','+yy);}}
  return {area:cells.size*cell*cell,bounds:[minX,minY,maxX,maxY]};}
 const baseline=occupiedArea(initial),actual=occupiedArea(positions),occupiedAreaFraction=actual.area/baseline.area,totalArea=n*particleArea,retainedFraction=physicalVolume/totalArea,numericalDensityVolumeFraction=resolvedArea/totalArea;
 const telemetry=await current.physics.readTelemetry();
 return {vertices:n,totalArea,physicalVolume,retainedFraction,resolvedArea,numericalDensityVolumeFraction,baselineOccupiedArea:baseline.area,occupiedArea:actual.area,occupiedAreaFraction,initialBounds:baseline.bounds,bounds:actual.bounds,minimumDensity,meanDensity,p90Density:percentile(.9),p99Density:percentile(.99),p999Density:percentile(.999),maximumDensity,telemetry};
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
 mark(0);current.embedding.render(encoder,{time:current.time,mode:'fluid'});mark(1);
 current.boundary.render(encoder,app.canvas.getContext('webgpu').getCurrentTexture().createView(),current.physics.policy,current.physics.wheel,current.angle);mark(2);
 encoder.resolveQuerySet(query,0,3,resolved,0);encoder.copyBufferToBuffer(resolved,0,read,0,24);
 const wallStart=performance.now();auditDevice.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
 const values=new BigUint64Array(read.getMappedRange().slice(0));read.unmap();
 const result={embeddingGpuMs:Number(values[1]-values[0])/1e6,boundaryGpuMs:Number(values[2]-values[1])/1e6,wallMs:performance.now()-wallStart};
 query.destroy();resolved.destroy();read.destroy();return result;
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
const sandMotion=${process.argv.includes('--sand-motion')},checkSand=${process.argv.includes('--sand-paused')||process.argv.includes('--sand-motion')},checkDrag=${process.argv.includes('--paused-drag')},checkVolume=${process.argv.includes('--water-volume')},profileFps=${process.argv.includes('--fps-profile')},gpuProfile=${process.argv.includes('--gpu-profile')},sustained=${process.argv.includes('--sustained')},runningDrag=${process.argv.includes('--running-drag')},started=performance.now(), faults=[];let last='',played=false,pausedReceipt,sandStarted=false,sandPlayed=false,waterReceipt,sandFrame,dragTarget,dragStarted,pausedProfileStart;
addEventListener('error',e=>faults.push(String(e.error||e.message)));
addEventListener('unhandledrejection',e=>faults.push(String(e.reason?.stack||e.reason)));
async function inspect(){
 const app=globalThis.__vfWorldLayerApplication;
 const current=app?.applications?.find(a=>a.world.world_id===app.program.views[app.program.active_view].world_id);
 const state={elapsedMs:performance.now()-started,ready:document.body?.dataset.vfWorldLayerReady,
  worlds:app?.applications?.map(a=>a.world.kind),time:current?.time,angle:current?.angle,target:current?.targetAngle,logicalAngle:current?.logicalAngle,remainingTime:current?.remainingTime,
  frames:Number(app?.canvas?.dataset.presentedFrames||0),status:document.querySelector('#vf-material-status')?.textContent,
  error:globalThis.__vfWorldLayerError||document.querySelector('[role=alert]:not([hidden])')?.textContent,faults};
 const signature=JSON.stringify([state.ready,state.worlds,state.status,state.error,checkDrag?state.angle:null,faults]);
 if(signature!==last){last=signature;await fetch('/page-progress',{method:'POST',body:JSON.stringify(state)});}
 if(document.querySelector('#vf-material-stage')&&!state.status)state.error='Startup replaced its loading message with a blank frame';
 if(!sandStarted&&state.worlds?.includes('granular'))state.error='Inactive sand initialized before the selected water View';
 if(state.error||faults.length||state.elapsedMs>60000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,...state})});return;}
 if(!played&&state.ready==='true'&&state.frames>=3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
  if(!current.paused||state.time!==0||!play){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Selected World did not start visibly paused with Play',...state})});return;}
  if(profileFps){
   if(!pausedProfileStart){pausedProfileStart={elapsedMs:state.elapsedMs,frames:state.frames};setTimeout(inspect,250);return;}
   if(state.elapsedMs-pausedProfileStart.elapsedMs<3000){setTimeout(inspect,250);return;}
  }
  const pausedTelemetry=await current.physics.readTelemetry();
  pausedReceipt={elapsedMs:state.elapsedMs,time:state.time,frames:state.frames,
    worlds:state.worlds,telemetry:pausedTelemetry,
    fps:profileFps?(state.frames-pausedProfileStart.frames)*1000/(state.elapsedMs-pausedProfileStart.elapsedMs):undefined};
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
  const conservation=await checkWaterVolume(current),passed=conservation.retainedFraction>=.9999&&conservation.retainedFraction<=1.000001&&conservation.minimumDensity>0&&conservation.p90Density<=current.physics.policy.restDensity*1.04&&conservation.p999Density<=current.physics.policy.restDensity*1.15&&!conservation.telemetry.nonFinite;
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,conservation,pausedReceipt,...state,error:passed?undefined:'Water lost occupied volume'})});return;
 }
 if(played&&gpuProfile&&state.time>=(sustained?12:5)){
  const telemetry=await current.physics.readTelemetry(),physics=await measurePhysicsSteps(current),embedding=await measureEmbedding(current,app);
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,physics,embedding,telemetry,pausedReceipt,...state})});return;
 }
 if(played&&runningDrag){
  if(dragTarget===undefined&&state.time>=5){const beforeDrag=await checkExclusion(current,{assert:false});await fetch('/page-progress',{method:'POST',body:JSON.stringify({beforeDrag})});dragTarget=current.angle+.4;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
  if(dragTarget!==undefined&&Math.abs(current.logicalAngle-dragTarget)<.005){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,dragMs:performance.now()-dragStarted,...state})});return;}
  if(dragTarget!==undefined&&performance.now()-dragStarted>7000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Running wheel drag not accepted within 7 seconds',pausedReceipt,...state})});return;}
 }
 if(played&&profileFps&&state.elapsedMs-pausedReceipt.elapsedMs>=(sustained?15000:5000)){
  const wallSeconds=(state.elapsedMs-pausedReceipt.elapsedMs)/1000;
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,playing:{fps:(state.frames-pausedReceipt.frames)/wallSeconds,realtimeFactor:(state.time-pausedReceipt.time)/wallSeconds},...state})});return;
 }
 if(played&&!checkVolume&&!profileFps&&!gpuProfile&&!runningDrag&&!sandStarted&&state.frames>=6&&state.time>(sustained?12:.02)){
  if(!checkSand){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,...state})});return;}
 waterReceipt={...state};sandFrame=state.frames;sandStarted=true;
  const particles=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Particles');
  particles.click();if(particles.getAttribute('aria-pressed')!=='true')throw Error('Raw particle embedding did not activate');
  particles.click();if(particles.getAttribute('aria-pressed')!=='false')throw Error('Material embedding did not reactivate');
  [...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Sand').click();
 }
 if(sandStarted&&current?.world.kind==='granular'&&state.frames>=sandFrame+3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
  if(sandMotion){
   if(!sandPlayed){if(!current.paused||state.time!==0||!play)throw Error('Sand did not start paused');sandPlayed=true;play.click();}
   if(dragTarget===undefined&&state.time>=2){const beforeDrag=await checkExclusion(current,{assert:false});await fetch('/page-progress',{method:'POST',body:JSON.stringify({sandBeforeDrag:beforeDrag})});dragTarget=current.angle+.4;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
   if(dragTarget!==undefined&&Math.abs(current.logicalAngle-dragTarget)<.005){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,waterReceipt,sandMotion:{time:state.time,dragMs:performance.now()-dragStarted},...state})});return;}
   if(dragTarget!==undefined&&performance.now()-dragStarted>7000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Running sand drag not accepted within 7 seconds',...state})});return;}
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
  if(req.method==='POST'&&['/page-progress','/page-result'].includes(url.pathname)){
   let body='';for await(const chunk of req){body+=chunk;if(body.length>100000)throw Error('Oversize probe');}
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
const timeout=setTimeout(()=>complete({passed:false,error:'Actual wheel page did not start within 90 seconds',stderr}),90000);
try{
 const report=await result;report.scope='Actual HTML startup and GPU embedding; '+(process.argv.includes('--legacy-overrides')?'older WebKit unused-override rejection emulation; ':'')+(process.argv.includes('--reject-caches')?'injected optional-cache failures, portable path':'production defaults')+'; not a phone test';
 report.chromeArguments=args;await writeFile(path.join(work,'result.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({...report,result:path.join(work,'result.json')}));if(!report.passed)process.exitCode=1;
}finally{clearTimeout(timeout);chrome.kill();server.closeAllConnections();server.close();}
