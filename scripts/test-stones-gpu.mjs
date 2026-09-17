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
<script src="/previews/0.6.0/compiled/runtime-stones-8/vf-compiled-runtime-bridge.js"></script>
<script type="module">
import {readMechanicalAsset,prepareMechanicalInitialState} from '/previews/0.6.0/compiled/runtime-stones-8/vf-world-mechanical-runtime.mjs';
import {createMechanicalWorldGpu} from '/previews/0.6.0/compiled/runtime-stones-8/vf-world-mechanical-gpu.mjs';
import {createWorldSceneEmbeddingGpu} from '/previews/0.6.0/compiled/runtime-stones-8/vf-world-scene-embedding-gpu.mjs';
const check=(x,m)=>{if(!x)throw Error(m)};let device;
try{
 const base='/previews/0.6.0/compiled/stones-granite-8/';
 const inputs={bytes:new Uint8Array(await(await fetch(base+'main.wasm')).arrayBuffer()),manifest:await(await fetch(base+'manifest.json')).json()};
 const runtime=await (VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync?VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync(inputs):VfCompiledRuntimeBridge.instantiateWasmRuntime(inputs));runtime.init();
 const world=runtime.worldProgram().gpu_worlds[0],asset=await readMechanicalAsset((world.kind==='wind'?world.solid_properties:world.properties).asset);
 const isTree=world.kind==='wind';check(isTree||asset.length===5,'Expected five stones');
 const adapter=await navigator.gpu.requestAdapter();check(adapter&&!adapter.isFallbackAdapter,'Requires physical GPU');device=await adapter.requestDevice();
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
 const {initial,meshes}=prepareMechanicalInitialState(world,runtime.worldLayerViews(),asset),prefix='$world$gpu$'+world.world_id;
 const physics=await createMechanicalWorldGpu(device,world,initial,runtime.readBinding(prefix+'$physics'));
 const canvas=document.querySelector('canvas'),embedding=await createWorldSceneEmbeddingGpu(device,canvas,world,physics,meshes,runtime.readBinding(prefix+'$embedding'));
 let encoder,z=null,expected=null,wind=null,coupling=null;
 if(isTree){
  coupling=[];
  // A free branch and its attached leaf must exchange internal momentum.
  // Exercise the compiled World kernel, not a host integration formula.
  for(const applied of [0,.01]){
   const count=12,index=8,dt=.01,scale=16777216;
   const data=new ArrayBuffer(128),f=new Float32Array(data),u=new Uint32Array(data);f[3]=dt;u.set([0,0,count,0],4);u.set([2,2,3,0],20);
   const state=new Float32Array(count*16),geometry=new Float32Array(count*16),impulses=new Int32Array(count*3);
   state[(count+index)*8]=.2;impulses[index*3]=Math.round(applied*scale);
   for(let i=0;i<count;i++){geometry[(count+i)*4]=2;geometry[(count*3+i)*4+3]=i;}
   geometry.set([.04,.2,.03,0],(count*2+index)*4);geometry.set([.1,.05,.01,index],(count*3+index)*4);
   const make=(array,uniform=false)=>{const b=device.createBuffer({size:array.byteLength,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(b,0,array);return b;};
   const buffers=[make(new Uint8Array(data),true),make(state),make(impulses),make(geometry),make(state)];
   const module=device.createShaderModule({code:runtime.readBinding(prefix+'$physics')}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'elastic_step'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[0,3,4,5,6].map((binding,i)=>({binding,resource:{buffer:buffers[i]}}))});
   const output=device.createBuffer({size:state.byteLength,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(buffers[1],0,output,0,state.byteLength);device.queue.submit([encoder.finish()]);await output.mapAsync(GPUMapMode.READ);
   const actual=new Float32Array(output.getMappedRange()),branchVelocity=actual[index*8+4],leafVelocity=actual[(count+index)*8+4],momentum=2.1*branchVelocity+.05*leafVelocity,expectedImpulse=impulses[index*3]/scale;
   check(Math.abs(momentum-expectedImpulse)<1e-7,'Lost leaf/branch momentum: '+momentum+' / '+expectedImpulse);check(branchVelocity>0,'Leaf reaction did not reach branch');
   coupling.push({applied:expectedImpulse,momentum,branchVelocity,leafVelocity});output.unmap();output.destroy();buffers.forEach(b=>b.destroy());
  }
 }
 if(!isTree){physics.setHeld(3,4);encoder=device.createCommandEncoder();physics.placeHeld(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();physics.setHeld(-1,0);
 encoder=device.createCommandEncoder();for(let i=0;i<48;i++)physics.step(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
 const state=await physics.readBodies();z=state[3*20+2];expected=4+world.gravity[2]*world.time_step**2*48*49/2;
 check(state.every(Number.isFinite),'Nonfinite rigid state');check(Math.abs(z-expected)<0.002,'Drop differs from gravity: '+z+' / '+expected);check(Math.abs(physics.time-.2)<1e-6,'Physical clock differs');physics.reset();
 }else{physics.setSpeed(8);const samples=[];for(let batch=0;batch<30;batch++){encoder=device.createCommandEncoder();for(let step=0;step<8;step++)physics.step(encoder);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();if(batch===14||batch===29)samples.push(await physics.inspectLeafModes());}wind=await physics.inspect();wind.leaves=samples;check(wind.finite&&samples.every(s=>s.finite),'Nonfinite wind state');check(samples.every(s=>s.maxAngle>.02&&s.maxAngle<2.73),'Leaf angle outside elastic response: '+JSON.stringify(wind));check(samples[1].maxAngularVelocity>.01,'Leaves stopped moving');}
 const format=navigator.gpu.getPreferredCanvasFormat(),context=canvas.getContext('webgpu');context.configure({device,format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
 let camera=isTree?{pos:[8,-17,7],target:[0,0,3.7],fov:42}:{pos:[3.4,-5.6,3.2],target:[0,0,.95],fov:42};
 if(isTree&&location.search.includes('closeup')){physics.reset();const leaf=meshes.find(m=>m.vertices[15]===1).vertices;
  const anchor=Array.from(leaf.subarray(16,19)),tip=Array.from(leaf.subarray(35*24,35*24+3)),normal=Array.from(leaf.subarray(3,6));
  const target=anchor.map((v,a)=>(v+tip[a])*.5);camera={target,pos:target.map((v,a)=>v+normal[a]*.72+[.08,-.10,.10][a]),fov:42};
 }
 encoder=device.createCommandEncoder();embedding.render(encoder,camera,{grass:isTree,particles:false});
 const row=Math.ceil(canvas.width*4/256)*256,readback=device.createBuffer({size:row*canvas.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 encoder.copyTextureToBuffer({texture:context.getCurrentTexture()},{buffer:readback,bytesPerRow:row},{width:canvas.width,height:canvas.height,depthOrArrayLayers:1});device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
 const bytes=new Uint8Array(readback.getMappedRange()),pixels=new Uint8ClampedArray(canvas.width*canvas.height*4);
 for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){const a=y*row+x*4,b=(y*canvas.width+x)*4;pixels[b]=bytes[a+(format.startsWith('bgra')?2:0)];pixels[b+1]=bytes[a+1];pixels[b+2]=bytes[a+(format.startsWith('bgra')?0:2)];pixels[b+3]=255;}
 readback.unmap();readback.destroy();check(new Set(pixels).size>20,'Blank rendered image');
 const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;copy.getContext('2d').putImageData(new ImageData(pixels,canvas.width,canvas.height),0,0);
 const png=copy.toDataURL('image/png');check(errors.length===0,errors.join('\n'));
 await fetch('/result',{method:'POST',body:JSON.stringify({passed:true,adapter:adapter.info,z,expected,wind,coupling,png})});embedding.destroy();physics.destroy();device.destroy();
}catch(error){await fetch('/result',{method:'POST',body:JSON.stringify({passed:false,error:String(error.stack??error)})});device?.destroy();}
</script>`;
const server=createServer(async(req,res)=>{try{
 const route=new URL(req.url,'http://localhost').pathname;
 if(route==='/result'&&req.method==='POST'){let body='';for await(const data of req){body+=data;if(body.length>8_000_000)throw Error('Oversize result');}res.end('ok');finish(JSON.parse(body));return;}
 if(route==='/test'){res.setHeader('Content-Type','text/html');res.end(isTree?html.replaceAll('runtime-stones-8','runtime-tree-9').replaceAll('stones-granite-8','tree-wind-9'):html);return;}
 const target=path.resolve(root,'.'+route),relative=path.relative(root,target);if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(404).end();return;}
 res.setHeader('Content-Type',({'.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm'})[path.extname(target)]??'application/octet-stream');res.end(await readFile(target));
}catch(e){res.writeHead(500).end(String(e));}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const args=['--headless=new','--enable-gpu','--no-first-run','--no-default-browser-check','--window-size=900,800','--user-data-dir='+path.join(work,'profile'),'http://127.0.0.1:'+server.address().port+'/test'+(closeup?'?closeup':'')];
const child=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',args,{windowsHide:true});let stderr='';child.stderr.on('data',d=>stderr=(stderr+d).slice(-16000));child.stdout.resume();child.on('error',e=>finish({passed:false,error:String(e)}));child.on('exit',code=>finish({passed:false,error:'Chrome exited '+code,stderr}));
const timeout=setTimeout(()=>finish({passed:false,error:'GPU test timed out',stderr}),60000);
try{const result=await completed;if(result.png){await writeFile(path.join(work,'stones.png'),Buffer.from(result.png.split(',')[1],'base64'));delete result.png;}result.chromeArguments=args;await writeFile(path.join(work,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({...result,work}));if(!result.passed)process.exitCode=1;}
finally{clearTimeout(timeout);child.kill();server.closeAllConnections();server.close();}
