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
async function checkExclusion(current){
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
 if(!(pairGap>=0&&boundaryGap>=0))throw Error('Paused configuration overlap: '+JSON.stringify({pairGap,boundaryGap}));return {pairGap,boundaryGap,vertices:n};
}
async function checkWaterVolume(current){
 const world=current.world,n=current.physics.primaryCount,stride=12,bytes=n*stride*4;
 const read=auditDevice.createBuffer({size:bytes+n*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=auditDevice.createCommandEncoder();e.copyBufferToBuffer(current.physics.particleBuffer,0,read,0,bytes);e.copyBufferToBuffer(current.contact.resources.precisionOutput,0,read,bytes,n*8);auditDevice.queue.submit([e.finish()]);
 await read.mapAsync(GPUMapMode.READ);const raw=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
 const positions=[];let minimumDensity=Infinity,maximumDensity=-Infinity,meanDensity=0;
 for(let i=0;i<n;i++){positions.push([raw[i*stride]+raw[n*stride+2*i],raw[i*stride+1]+raw[n*stride+2*i+1]]);const density=raw[i*stride+7];minimumDensity=Math.min(minimumDensity,density);maximumDensity=Math.max(maximumDensity,density);meanDensity+=density;}
 meanDensity/=n;
 const initial=[];for(let i=0;i<n;i++)initial.push([current.physics.seed.floats[i*stride],current.physics.seed.floats[i*stride+1]]);
 const particleArea=current.physics.seed.particleMass/current.physics.policy.restDensity,radius=Math.sqrt(particleArea/Math.PI),cell=world.properties.spacing/4;
 function occupiedArea(points){let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const [x,y] of points){minX=Math.min(minX,x-radius);minY=Math.min(minY,y-radius);maxX=Math.max(maxX,x+radius);maxY=Math.max(maxY,y+radius);}const cells=new Set(),reach=Math.ceil(radius/cell);
  for(const [x,y] of points){const cx=Math.floor((x-minX)/cell),cy=Math.floor((y-minY)/cell);for(let yy=cy-reach;yy<=cy+reach;yy++)for(let xx=cx-reach;xx<=cx+reach;xx++){const px=minX+(xx+.5)*cell,py=minY+(yy+.5)*cell;if((px-x)**2+(py-y)**2<=radius**2)cells.add(xx+','+yy);}}
  return {area:cells.size*cell*cell,bounds:[minX,minY,maxX,maxY]};}
 const baseline=occupiedArea(initial),actual=occupiedArea(positions),retainedFraction=actual.area/baseline.area,totalArea=n*particleArea;
 const telemetry=await current.physics.readTelemetry();
 return {vertices:n,totalArea,baselineOccupiedArea:baseline.area,occupiedArea:actual.area,retainedFraction,initialBounds:baseline.bounds,bounds:actual.bounds,minimumDensity,meanDensity,maximumDensity,telemetry};
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
const checkSand=${process.argv.includes('--sand-paused')},checkDrag=${process.argv.includes('--paused-drag')},checkVolume=${process.argv.includes('--water-volume')},started=performance.now(), faults=[];let last='',played=false,pausedReceipt,sandStarted=false,waterReceipt,sandFrame,dragTarget,dragStarted;
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
  pausedReceipt={elapsedMs:state.elapsedMs,time:state.time,frames:state.frames,worlds:state.worlds};
  if(checkDrag){
   if(dragTarget===undefined){dragTarget=current.angle+1.2;dragStarted=performance.now();app.setLayer(current.world.boundary_ids[0],{rotation:dragTarget});}
   if(Math.abs(current.logicalAngle-dragTarget)<.005){const dragMs=performance.now()-dragStarted,exclusion=await checkExclusion(current);await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:current.time===0,pausedReceipt,dragMs,exclusion,...state})});return;}
   if(performance.now()-dragStarted>3000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Paused 1.2-radian wheel drag not accepted within 3 seconds',dragMs:performance.now()-dragStarted,...state})});return;}
   setTimeout(inspect,100);return;
  }
  played=true;play.click();
 }
 if(played&&checkVolume&&state.time>=5){
  const conservation=await checkWaterVolume(current),passed=conservation.retainedFraction>=.9&&conservation.minimumDensity>0&&conservation.maximumDensity<=current.physics.policy.restDensity*4;
  await fetch('/page-result',{method:'POST',body:JSON.stringify({passed,conservation,pausedReceipt,...state,error:passed?undefined:'Water lost occupied volume'})});return;
 }
 if(played&&!checkVolume&&!sandStarted&&state.frames>=6&&state.time>.02){
  if(!checkSand){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:true,pausedReceipt,...state})});return;}
  waterReceipt={...state};sandFrame=state.frames;sandStarted=true;
  [...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Sand').click();
 }
 if(sandStarted&&current?.world.kind==='granular'&&state.frames>=sandFrame+3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
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
