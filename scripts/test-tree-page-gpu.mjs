// Run the published tree entry in isolated physical-GPU Chrome.
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {mkdir,mkdtemp,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const root=path.join(site,'public');
const stress=process.argv.includes('--stress')||process.argv.includes('--calm');
const targetFrames=process.argv.includes('--long')?600:stress?180:6;
const windTarget=process.argv.includes('--calm')?0:20;
const workParent=path.join(site,'.work');
await mkdir(workParent,{recursive:true});
const work=await mkdtemp(path.join(workParent,'tree-page-gpu-'));
const probe=`<script>
let sent=false;
const stress=${JSON.stringify(stress)};
const targetFrames=${JSON.stringify(targetFrames)};
const windTarget=${JSON.stringify(windTarget)};
async function report(result){
  if(sent)return;sent=true;
  await fetch('/_tree_gpu_result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(result)});
}
const start=performance.now();let windRaised=false,peakCandidates=0,nextSample=100,sampling=false;
let sampledFault=0;
setInterval(async()=>{
  const app=globalThis.__vfWorldLayerApplication;
  const error=globalThis.__vfWorldLayerError??document.getElementById('world-error')?.textContent;
  if(error){void report({passed:false,error:String(error),stage:globalThis.__vfWorldLayerStage});return;}
  if(stress&&app&&!windRaised){app.physics.setSpeed(windTarget);windRaised=true;}
  const canvas=app?.canvas,frames=Number(canvas?.dataset.renderedFrames??0),time=app?.physics?.time??0;
  if(stress&&app?.surface&&frames>=nextSample&&!sampling&&!sent){
    sampling=true;nextSample+=100;
    try{const sample=await app.surface.inspectStatus();
      peakCandidates=Math.max(peakCandidates,sample.candidates??0);
      sampledFault|=(sample.broadphaseFault??0)|(sample.sweepFault??0)|(sample.stateFault??0);
    }
    finally{sampling=false;}
  }
  if(frames>=targetFrames&&time>(stress?3:0.02)&&!sent){
    const world=app.program.gpu_worlds.find(world=>world.kind==='wind');
    const status=stress?await app.surface?.inspectStatus():null;
    const candidateKinds=stress?await app.surface?.inspectCandidateKinds():null;
    const worldState=stress?await app.physics.inspect():null;
    void report({passed:document.body.dataset.vfWorldLayerReady==='true'&&!!app.surface&&
      world?.surface_contact?.kind==='surface_contact'&&(!stress||
        status?.broadphaseFault===0&&status?.sweepFault===0&&status?.stateFault===0&&sampledFault===0&&
        status?.faultHistory?.broadphase===0&&status?.faultHistory?.sweep===0&&status?.faultHistory?.state===0),frames,time,
      physicalSurface:!!app.surface,law:world?.surface_contact?.kind,
      asset:world?.solid_properties?.asset,readyMs:canvas.dataset.readyMs,status,
      candidateKinds,peakCandidates:Math.max(peakCandidates,status?.candidates??0),sampledFault,
      worldState,
      windMetersPerSecond:stress?windTarget:null,elapsedMs:performance.now()-start});
  }else if(performance.now()-start>120000){
    void report({passed:false,error:'Tree did not advance six GPU frames',frames,time,
      stage:globalThis.__vfWorldLayerStage});
  }
},250);
</script>`;

let complete;
const completed=new Promise(resolve=>complete=resolve);
const server=createServer(async(req,res)=>{
  try{
    const route=new URL(req.url,'http://localhost').pathname;
    if(route==='/_tree_gpu_result'&&req.method==='POST'){
      let body='';for await(const chunk of req){body+=chunk;if(body.length>100000)throw Error('Oversize report');}
      res.end('ok');complete(JSON.parse(body));return;
    }
    const relativeRoute=route.endsWith('/')?route+'index.html':route;
    const target=path.resolve(root,'.'+relativeRoute),relative=path.relative(root,target);
    if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(404).end();return;}
    let bytes=await readFile(target);
    if(relativeRoute==='/previews/0.6.0/live/tree/index.html')
      bytes=Buffer.from(bytes.toString().replace('</body>',probe+'</body>'));
    const type={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript',
      '.json':'application/json','.wasm':'application/wasm','.gz':'application/octet-stream'}[path.extname(target)];
    res.setHeader('Content-Type',type??'application/octet-stream');res.end(bytes);
  }catch(error){res.writeHead(500).end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=process.env.VKF_GPU_BROWSER??'C:/Program Files/Google/Chrome/Application/chrome.exe';
const url=`http://127.0.0.1:${server.address().port}/previews/0.6.0/live/tree/index.html`;
const args=['--headless=new','--enable-gpu','--no-first-run','--no-default-browser-check',
  '--user-data-dir='+path.join(work,'profile'),url];
const child=spawn(browser,args,{windowsHide:true});
let stderr='';child.stderr.on('data',bytes=>stderr=(stderr+bytes).slice(-16000));child.stdout.resume();
child.on('error',error=>complete({passed:false,error:String(error)}));
child.on('exit',code=>complete({passed:false,error:`Chrome exited ${code} before result`}));
const timer=setTimeout(()=>complete({passed:false,error:'Tree page GPU acceptance timed out'}),150000);
try{
  const result=await completed;
  const report={...result,url,chromeArguments:args,chromeStderr:stderr};
  const resultPath=path.join(work,'result.json');
  await writeFile(resultPath,JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:report.passed,frames:report.frames,time:report.time,
    physicalSurface:report.physicalSurface,status:report.status,error:report.error,result:resultPath}));
  if(!report.passed)process.exitCode=1;
}finally{
  clearTimeout(timer);child.kill();server.closeAllConnections();server.close();
}
