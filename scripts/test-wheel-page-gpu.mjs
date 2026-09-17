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
const checkSand=${process.argv.includes('--sand-paused')},started=performance.now(), faults=[];let last='',played=false,pausedReceipt,sandStarted=false,waterReceipt,sandFrame;
addEventListener('error',e=>faults.push(String(e.error||e.message)));
addEventListener('unhandledrejection',e=>faults.push(String(e.reason?.stack||e.reason)));
async function inspect(){
 const app=globalThis.__vfWorldLayerApplication;
 const current=app?.applications?.find(a=>a.world.world_id===app.program.views[app.program.active_view].world_id);
 const state={elapsedMs:performance.now()-started,ready:document.body?.dataset.vfWorldLayerReady,
  worlds:app?.applications?.map(a=>a.world.kind),time:current?.time,
  frames:Number(app?.canvas?.dataset.presentedFrames||0),status:document.querySelector('#vf-material-status')?.textContent,
  error:globalThis.__vfWorldLayerError||document.querySelector('[role=alert]:not([hidden])')?.textContent,faults};
 const signature=JSON.stringify([state.ready,state.worlds,state.status,state.error,faults]);
 if(signature!==last){last=signature;await fetch('/page-progress',{method:'POST',body:JSON.stringify(state)});}
 if(document.querySelector('#vf-material-stage')&&!state.status)state.error='Startup replaced its loading message with a blank frame';
 if(!sandStarted&&state.worlds?.includes('granular'))state.error='Inactive sand initialized before the selected water View';
 if(state.error||faults.length||state.elapsedMs>60000){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,...state})});return;}
 if(!played&&state.ready==='true'&&state.frames>=3){
  const play=[...document.querySelectorAll('#vf-material-controls button')].find(b=>b.textContent==='Play');
  if(!current.paused||state.time!==0||!play){await fetch('/page-result',{method:'POST',body:JSON.stringify({passed:false,error:'Selected World did not start visibly paused with Play',...state})});return;}
  pausedReceipt={elapsedMs:state.elapsedMs,time:state.time,frames:state.frames,worlds:state.worlds};
  played=true;play.click();
 }
 if(played&&!sandStarted&&state.frames>=6&&state.time>.02){
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
const timeout=setTimeout(()=>complete({passed:false,error:'Actual wheel page did not start within 75 seconds',stderr}),75000);
try{
 const report=await result;report.scope='Actual HTML startup, production defaults and GPU embedding; not a phone test';
 report.chromeArguments=args;await writeFile(path.join(work,'result.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({...report,result:path.join(work,'result.json')}));if(!report.passed)process.exitCode=1;
}finally{clearTimeout(timeout);chrome.kill();server.closeAllConnections();server.close();}
