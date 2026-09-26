// Isolated physical-GPU shader/render/drop test, not browser UI automation.
import {createServer} from 'node:http';
import {spawn} from 'node:child_process';
import {readFile,mkdir,mkdtemp,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../public');
const isTree=process.argv.includes('--tree'),closeup=process.argv.includes('--closeup');
const parent=path.resolve(root,'../.w');await mkdir(parent,{recursive:true});const work=await mkdtemp(path.join(parent,'stones-gpu-'));
let finish;const completed=new Promise(resolve=>finish=resolve);
const html=String.raw`<!doctype html><canvas width="800" height="650" style="width:800px;height:650px"></canvas>
<script src="/previews/0.6.0/compiled/runtime-stones-13/vf-compiled-runtime-bridge.js"></script>
<script type="module">
import {readMechanicalAsset,prepareMechanicalInitialState} from '/previews/0.6.0/compiled/runtime-stones-13/vf-world-mechanical-runtime.mjs';
import {createMechanicalWorldGpu} from '/previews/0.6.0/compiled/runtime-stones-13/vf-world-mechanical-gpu.mjs';
import {createWorldSceneEmbeddingGpu,WORLD_SCENE_WGSL} from '/previews/0.6.0/compiled/runtime-stones-13/vf-world-scene-embedding-gpu.mjs';
const check=(x,m)=>{if(!x)throw Error(m)};let device;
function countEmbeddedStoneVertices(state,initial,meshes){
 const rotate=(q,p)=>{const t=[2*(q[1]*p[2]-q[2]*p[1]),2*(q[2]*p[0]-q[0]*p[2]),2*(q[0]*p[1]-q[1]*p[0])];return p.map((v,a)=>v+q[3]*t[a]+[q[1]*t[2]-q[2]*t[1],q[2]*t[0]-q[0]*t[2],q[0]*t[1]-q[1]*t[0]][a]);};
 const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]);
 let count=0,worstDepth=0;const samples=initial.hullCount,pairs=[];
 for(let a=0;a<initial.bodyCount;a++)for(let b=0;b<initial.bodyCount;b++)if(a!==b){
  let pairCount=0,pairDepth=0;
  const pa=Array.from(state.subarray(a*20,a*20+3)),qa=Array.from(state.subarray(a*20+8,a*20+12)),pb=Array.from(state.subarray(b*20,b*20+3)),qb=Array.from(state.subarray(b*20+8,b*20+12)),inverse=[-qb[0],-qb[1],-qb[2],qb[3]];
  for(let index=0;index<meshes[a].vertices.length;index+=24){const local=rotate(inverse,sub(add(pa,rotate(qa,Array.from(meshes[a].vertices.subarray(index,index+3)))),pb));let depth=Infinity;
   for(let k=0;k<samples;k++){const z=1-2*k/(samples-1),theta=k*2.399963229728653,r=Math.sqrt(1-z*z),n=[r*Math.cos(theta),r*Math.sin(theta),z],h=initial.geometry.subarray((b*samples+k)*4,(b*samples+k)*4+3);depth=Math.min(depth,dot(n,h)-dot(n,local));if(depth<=.001)break;}
   if(depth>.001){count++;pairCount++;worstDepth=Math.max(worstDepth,depth);pairDepth=Math.max(pairDepth,depth);}
  }
  if(pairCount)pairs.push({a,b,count:pairCount,worstDepth:pairDepth});
 }
 return {count,worstDepth,pairs};
}
try{
 const base='/previews/0.6.0/compiled/stones-mixed-13/';
 const inputs={bytes:new Uint8Array(await(await fetch(base+'main.wasm')).arrayBuffer()),manifest:await(await fetch(base+'manifest.json')).json()};
 const runtime=await (VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync?VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync(inputs):VfCompiledRuntimeBridge.instantiateWasmRuntime(inputs));runtime.init();
 const world=runtime.worldProgram().gpu_worlds[0],asset=world.kind==='wind'?await new Promise((resolve,reject)=>{
  const worker=new Worker('/previews/0.6.0/live/tree/runtime/vf-tree-generation-worker.mjs',{type:'module'});
  worker.onmessage=event=>{worker.terminate();event.data.error?reject(Error(event.data.error)):resolve(event.data.meshes)};
  worker.onerror=event=>{worker.terminate();reject(Error(event.message))};worker.postMessage({species:'oak',distribution:'normal',height:8,splitFactor:.65,turnFactor:.5});
 }):await readMechanicalAsset(world.properties.asset);
 const isTree=world.kind==='wind';check(isTree||asset.length===5,'Expected five stones');
 const adapter=await navigator.gpu.requestAdapter();check(adapter&&!adapter.isFallbackAdapter,'Requires physical GPU');device=await adapter.requestDevice();
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
 const {initial,meshes}=prepareMechanicalInitialState(world,runtime.worldLayerViews(),asset),prefix='$world$gpu$'+world.world_id;
 const physics=await createMechanicalWorldGpu(device,world,initial,runtime.readBinding(prefix+'$physics'));
 const canvas=document.querySelector('canvas'),embedding=await createWorldSceneEmbeddingGpu(device,canvas,world,physics,meshes,isTree?runtime.readBinding(prefix+'$embedding'):WORLD_SCENE_WGSL);
 let encoder,z=null,expected=null,wind=null,coupling=null,stoneDynamics=null,treePerformance=null;
 if(isTree){
  coupling=[];
  // A free branch and its attached leaf must exchange internal momentum.
  // Exercise the compiled World kernel, not a host integration formula.
  for(const applied of [0,.01]){
   const count=12,index=8,dt=.01,scale=16777216;
   const data=new ArrayBuffer(128),f=new Float32Array(data),u=new Uint32Array(data);f[3]=dt;u.set([0,0,count,0],4);u.set([2,2,3,0],20);
   const state=new Float32Array(count*16),geometry=new Float32Array(count*16),impulses=new Int32Array(count*3);
   state[(count+index)*8]=.2;impulses[index*3]=Math.round(applied*scale);
   for(let i=0;i<count;i++){geometry.set([2,.2,.03,1],(count+i)*4);geometry[(count*3+i)*4+3]=i;}
   geometry.set([.04,.2,.03,0],(count*2+index)*4);geometry.set([.1,.05,.01,index],(count*3+index)*4);
   const make=(array,uniform=false)=>{const b=device.createBuffer({size:array.byteLength,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(b,0,array);return b;};
   const buffers=[make(new Uint8Array(data),true),make(state),make(impulses),make(geometry),make(state)];
   const module=device.createShaderModule({code:runtime.readBinding(prefix+'$physics')}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'elastic_step'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[0,3,4,5,6].map((binding,i)=>({binding,resource:{buffer:buffers[i]}}))});
   const output=device.createBuffer({size:state.byteLength,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(buffers[1],0,output,0,state.byteLength);device.queue.submit([encoder.finish()]);await output.mapAsync(GPUMapMode.READ);
   const actual=new Float32Array(output.getMappedRange()),branchVelocity=actual[index*8+4],leafVelocity=actual[(count+index)*8+4],momentum=2.1*branchVelocity+.05*leafVelocity,expectedImpulse=impulses[index*3]/scale;
   check(Math.abs(momentum-expectedImpulse)<2e-6,'Lost leaf/branch momentum: '+momentum+' / '+expectedImpulse);check(branchVelocity>0,'Leaf reaction did not reach branch: '+JSON.stringify({applied,branchVelocity,leafVelocity}));
   coupling.push({applied:expectedImpulse,momentum,branchVelocity,leafVelocity});output.unmap();output.destroy();buffers.forEach(b=>b.destroy());
  }
 }
 if(!isTree){const droppedBody=4,dropHeight=initial.sceneFrame.liftCeiling,masses=Array.from({length:initial.bodyCount},(_,body)=>initial.bodies[body*20+16]);check(initial.bodies[droppedBody*20+16]===Math.min(...masses),'Regression requires the smallest stone');check(Math.abs(Math.max(...masses)-30)<1e-4,'Largest stone must be 30 kg: '+JSON.stringify(masses));physics.setHeld(droppedBody,dropHeight);encoder=device.createCommandEncoder();physics.placeHeld(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physics.setHeld(-1,0);
 const dropSteps=20;encoder=device.createCommandEncoder();for(let i=0;i<dropSteps;i++)physics.step(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
 const state=await physics.readBodies();z=state[droppedBody*20+2];expected=dropHeight+world.gravity[2]*world.time_step**2*dropSteps*(dropSteps+1)/2;
 check(state.every(Number.isFinite),'Nonfinite rigid state');check(Math.abs(z-expected)<0.002,'Drop differs from gravity: '+z+' / '+expected);check(Math.abs(physics.time-world.time_step*dropSteps)<1e-6,'Physical clock differs');physics.reset();
 physics.setHeld(droppedBody,dropHeight);encoder=device.createCommandEncoder();physics.placeHeld(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physics.setHeld(-1,0);
 let minimumVerticalSpeed=0,maximumUpwardSpeed=0,maximumAngularSpeed=0,maximumTransferredSpeed=0,maximumBaseSpeed=0,maximumBaseDisplacement=0;
 for(let batch=0;batch<150;batch++){encoder=device.createCommandEncoder();for(let i=0;i<8;i++)physics.step(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();const bodies=await physics.readBodies();minimumVerticalSpeed=Math.min(minimumVerticalSpeed,bodies[droppedBody*20+6]);maximumUpwardSpeed=Math.max(maximumUpwardSpeed,bodies[droppedBody*20+6]);maximumAngularSpeed=Math.max(maximumAngularSpeed,Math.hypot(...bodies.subarray(droppedBody*20+12,droppedBody*20+15)));for(let body=0;body<initial.bodyCount;body++){if(body!==droppedBody)maximumTransferredSpeed=Math.max(maximumTransferredSpeed,Math.hypot(...bodies.subarray(body*20+4,body*20+7)));if(body<3){maximumBaseSpeed=Math.max(maximumBaseSpeed,Math.hypot(...bodies.subarray(body*20+4,body*20+7)));maximumBaseDisplacement=Math.max(maximumBaseDisplacement,Math.hypot(...[0,1,2].map(axis=>bodies[body*20+axis]-initial.bodies[body*20+axis])));}}}
 const settled=await physics.readBodies(),linearSpeed=Math.hypot(...settled.subarray(droppedBody*20+4,droppedBody*20+7)),angularSpeed=Math.hypot(...settled.subarray(droppedBody*20+12,droppedBody*20+15)),sleepingBodies=Array.from({length:initial.bodyCount},(_,body)=>settled[body*20+19]>.5).filter(Boolean).length;let minimumFloorClearance=Infinity;for(let body=0;body<initial.bodyCount;body++){const q=settled.subarray(body*20+8,body*20+12),pz=settled[body*20+2];for(let point=0;point<initial.hullCount;point++){const h=initial.geometry.subarray((body*initial.hullCount+point)*4,(body*initial.hullCount+point)*4+3),t=[2*(q[1]*h[2]-q[2]*h[1]),2*(q[2]*h[0]-q[0]*h[2]),2*(q[0]*h[1]-q[1]*h[0])],rz=h[2]+q[3]*t[2]+q[0]*t[1]-q[1]*t[0];minimumFloorClearance=Math.min(minimumFloorClearance,pz+rz);}}
 const finalBodies=Array.from({length:initial.bodyCount},(_,body)=>({linearSpeed:Math.hypot(...settled.subarray(body*20+4,body*20+7)),quietSeconds:settled[body*20+7],angularSpeed:Math.hypot(...settled.subarray(body*20+12,body*20+15)),contact:settled[body*20+15],sleeping:settled[body*20+19]>.5}));
 stoneDynamics={droppedBody,droppedMassKg:initial.bodies[droppedBody*20+16],minimumVerticalSpeed,maximumUpwardSpeed,maximumAngularSpeed,maximumTransferredSpeed,maximumBaseSpeed,maximumBaseDisplacement,finalLinearSpeed:linearSpeed,finalAngularSpeed:angularSpeed,sleepingBodies,minimumFloorClearance,finalBodies,initialEmbeddedVertices:countEmbeddedStoneVertices(initial.bodies,initial,meshes),embeddedVertices:countEmbeddedStoneVertices(settled,initial,meshes)};
 check(stoneDynamics.initialEmbeddedVertices.count===0&&stoneDynamics.embeddedVertices.count===0,'Rendered stones overlap their collision envelopes: '+JSON.stringify(stoneDynamics));
 check(minimumVerticalSpeed<-1,'Stone never reached impact speed: '+JSON.stringify(stoneDynamics));check(maximumUpwardSpeed>.03||maximumAngularSpeed>.03,'Irregular stone contact produced no bounce or rotation response: '+JSON.stringify(stoneDynamics));check(minimumFloorClearance>-.002,'Stone tunneled through ground: '+JSON.stringify(stoneDynamics));check(sleepingBodies===initial.bodyCount&&linearSpeed<.01&&angularSpeed<.01,'Dropped stone did not sleep after settling: '+JSON.stringify(stoneDynamics));
 check(maximumBaseSpeed<.25&&maximumBaseDisplacement<initial.sceneFrame.span*.05,'Small stone moved supported bases unrealistically: '+JSON.stringify(stoneDynamics));
 }else{physics.setSpeed(8);const samples=[];for(let batch=0;batch<30;batch++){encoder=device.createCommandEncoder();for(let step=0;step<8;step++)physics.step(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();if(batch===14||batch===29)samples.push(await physics.inspectLeafModes());}wind=await physics.inspect();wind.leaves=samples;check(wind.finite&&samples.every(s=>s.finite),'Nonfinite wind state');check(samples.every(s=>s.maxAngle>.02&&s.maxAngle<2.73),'Leaf angle outside elastic response: '+JSON.stringify(wind));check(samples[1].maxAngularVelocity>.01,'Leaves stopped moving');}
 const format=navigator.gpu.getPreferredCanvasFormat(),context=canvas.getContext('webgpu');context.configure({device,format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
 let camera=isTree?{pos:[8,-17,7],target:[0,0,3.7],fov:42}:initial.sceneFrame.camera;
 if(!isTree&&location.search.includes('closeup')){const target=Array.from(initial.bodies.subarray(0,3));camera={target,pos:target.map((v,a)=>v+[.12,-.42,.15][a]),fov:42};}
 if(isTree&&location.search.includes('closeup')){physics.reset();const leaf=meshes.find(m=>m.vertices[15]===1).vertices;
  const anchor=Array.from(leaf.subarray(16,19)),tip=Array.from(leaf.subarray(35*24,35*24+3)),normal=Array.from(leaf.subarray(3,6));
  const target=anchor.map((v,a)=>(v+tip[a])*.5);camera={target,pos:target.map((v,a)=>v+normal[a]*.72+[.08,-.10,.10][a]),fov:42};
 }
 if(!isTree){
  // Benchmark complete 60 Hz presentation frames while a stone falls into
  // and collides with the pile. Idle/sleeping steps are not a performance gate.
  const percentile=(values,p)=>values.slice().sort((a,b)=>a-b)[Math.floor(values.length*p)];
  const renderOnly=[];for(let frame=0;frame<60;frame++){encoder=device.createCommandEncoder();embedding.render(encoder,camera,{grass:false,particles:false});const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();renderOnly.push(performance.now()-started);}
  physics.reset();physics.setHeld(4,initial.sceneFrame.liftCeiling);encoder=device.createCommandEncoder();physics.placeHeld(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physics.setHeld(-1,0);
  const stepsPerFrame=Math.ceil((1/60)/world.time_step-1e-8),physicsOnly=[];
  for(let frame=0;frame<120;frame++){encoder=device.createCommandEncoder();for(let step=0;step<stepsPerFrame;step++)physics.step(encoder);const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physicsOnly.push(performance.now()-started);}
  physics.reset();physics.setHeld(4,initial.sceneFrame.liftCeiling);encoder=device.createCommandEncoder();physics.placeHeld(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physics.setHeld(-1,0);
  const frameTimings=[];
  for(let frame=0;frame<120;frame++){encoder=device.createCommandEncoder();for(let step=0;step<stepsPerFrame;step++)physics.step(encoder);embedding.render(encoder,camera,{grass:false,particles:false});const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();frameTimings.push(performance.now()-started);}
  frameTimings.sort((a,b)=>a-b);stoneDynamics.stepsPerDisplayFrame=stepsPerFrame;stoneDynamics.p95RenderOnlyMs=percentile(renderOnly,.95);stoneDynamics.p95ActivePhysicsOnlyMs=percentile(physicsOnly,.95);stoneDynamics.medianActiveRenderedFrameMs=frameTimings[Math.floor(frameTimings.length*.5)];stoneDynamics.p95ActiveRenderedFrameMs=frameTimings[Math.floor(frameTimings.length*.95)];stoneDynamics.maxActiveRenderedFrameMs=frameTimings.at(-1);
  check(stoneDynamics.p95ActiveRenderedFrameMs<16.67,'Active collision rendered frame misses 60 Hz: '+JSON.stringify(stoneDynamics));
 }else{
  const ms=[];for(let frame=0;frame<60;frame++){encoder=device.createCommandEncoder();for(let step=0;step<Math.ceil((1/60)/world.time_step-1e-8);step++)physics.step(encoder);embedding.render(encoder,camera,{grass:true,particles:false});const started=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();ms.push(performance.now()-started);}
  ms.sort((a,b)=>a-b);treePerformance={medianMs:ms[30],p95Ms:ms[Math.floor(ms.length*.95)],maxMs:ms.at(-1)};
 }
 encoder=device.createCommandEncoder();embedding.render(encoder,camera,{grass:isTree,particles:false});
 const row=Math.ceil(canvas.width*4/256)*256,readback=device.createBuffer({size:row*canvas.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 encoder.copyTextureToBuffer({texture:context.getCurrentTexture()},{buffer:readback,bytesPerRow:row},{width:canvas.width,height:canvas.height,depthOrArrayLayers:1});device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
 const bytes=new Uint8Array(readback.getMappedRange()),pixels=new Uint8ClampedArray(canvas.width*canvas.height*4);
 for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){const a=y*row+x*4,b=(y*canvas.width+x)*4;pixels[b]=bytes[a+(format.startsWith('bgra')?2:0)];pixels[b+1]=bytes[a+1];pixels[b+2]=bytes[a+(format.startsWith('bgra')?0:2)];pixels[b+3]=255;}
 readback.unmap();readback.destroy();check(new Set(pixels).size>20,'Blank rendered image');
 const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;copy.getContext('2d').putImageData(new ImageData(pixels,canvas.width,canvas.height),0,0);
 const png=copy.toDataURL('image/png');check(errors.length===0,errors.join('\n'));
 await fetch('/result',{method:'POST',body:JSON.stringify({passed:true,adapter:adapter.info,z,expected,wind,coupling,stoneDynamics,treePerformance,png})});embedding.destroy();physics.destroy();device.destroy();
}catch(error){await fetch('/result',{method:'POST',body:JSON.stringify({passed:false,error:String(error.stack??error)})});device?.destroy();}
</script>`;
const server=createServer(async(req,res)=>{try{
 const route=new URL(req.url,'http://localhost').pathname;
 if(route==='/result'&&req.method==='POST'){let body='';for await(const data of req){body+=data;if(body.length>8_000_000)throw Error('Oversize result');}res.end('ok');finish(JSON.parse(body));return;}
 if(route==='/test'){res.setHeader('Content-Type','text/html');let page=isTree?html.replaceAll('runtime-stones-13','runtime-tree-12').replaceAll('stones-mixed-13','tree-air-12'):html;if(isTree&&process.argv.includes('--wind20'))page=page.replace('physics.setSpeed(8)','physics.setSpeed(20)');res.end(page);return;}
 const target=path.resolve(root,'.'+route),relative=path.relative(root,target);if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(404).end();return;}
 res.setHeader('Content-Type',({'.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm'})[path.extname(target)]??'application/octet-stream');res.end(await readFile(target));
}catch(e){res.writeHead(500).end(String(e));}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const args=['--headless=new','--enable-gpu','--no-first-run','--no-default-browser-check','--window-size=900,800','--user-data-dir='+path.join(work,'profile'),'http://127.0.0.1:'+server.address().port+'/test'+(closeup?'?closeup':'')];
const browser=process.env.VF_TEST_BROWSER??'C:/Program Files/Google/Chrome/Application/chrome.exe';
const child=spawn(browser,args,{windowsHide:true});let stderr='';child.stderr.on('data',d=>stderr=(stderr+d).slice(-16000));child.stdout.resume();child.on('error',e=>finish({passed:false,error:String(e)}));child.on('exit',code=>finish({passed:false,error:'Browser exited '+code,stderr}));
const timeout=setTimeout(()=>finish({passed:false,error:'GPU test timed out',stderr}),60000);
try{const result=await completed;if(result.png){await writeFile(path.join(work,'stones.png'),Buffer.from(result.png.split(',')[1],'base64'));delete result.png;}result.chromeArguments=args;await writeFile(path.join(work,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({...result,work}));if(!result.passed)process.exitCode=1;}
finally{clearTimeout(timeout);child.kill();server.closeAllConnections();server.close();}
