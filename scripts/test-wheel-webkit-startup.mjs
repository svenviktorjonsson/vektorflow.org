// Isolated WebKit development fixture for our own local application, not UI
// automation of a personal browser. This engine may lack platform WebGPU.
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import {readFile,mkdir,mkdtemp,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const require=createRequire(import.meta.url);
const {webkit}=require(process.env.VKF_WEBKIT_TEST_PACKAGE??'playwright');
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),root=path.join(site,'public');
await mkdir(path.join(site,'.work'),{recursive:true});const work=await mkdtemp(path.join(site,'.work/webkit-wheel-'));
let finish;const done=new Promise(resolve=>finish=resolve);
const bundle=JSON.parse(await readFile(path.join(root,'previews/0.6.0/compiled/bundle.json'))),wheel=bundle.applications.wheel;
const html=`<!doctype html><script src="/previews/0.6.0/compiled/${wheel.runtime_directory}/vf-compiled-runtime-bridge.js"></script><script>
(async()=>{const report={stage:'fetch',gpu:!!navigator.gpu};try{
const bytes=await(await fetch('/previews/0.6.0/compiled/${wheel.directory}/main.wasm')).arrayBuffer(),manifest=await(await fetch('/previews/0.6.0/compiled/${wheel.directory}/manifest.json')).json();
report.stage='instantiate';const runtime=await VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync({bytes,manifest});
report.stage='init';runtime.init();report.stage='program';const p=runtime.worldProgram();report.worlds=p.gpu_worlds.map(w=>w.kind);report.seeds=runtime.worldLayerViews().map(a=>a.layer.count);
report.passed=true;report.stage='complete';
}catch(e){report.passed=false;report.name=e.name;report.message=e.message;report.stack=e.stack;report.string=String(e);}
await fetch('/result',{method:'POST',body:JSON.stringify(report)});})();</script>`;
const server=createServer(async(req,res)=>{try{const route=new URL(req.url,'http://localhost').pathname;
if(route==='/result'&&req.method==='POST'){let body='';for await(const chunk of req)body+=chunk;res.end('ok');finish(JSON.parse(body));return;}
if(route==='/'){res.setHeader('Content-Type','text/html');res.end(html);return;}
const file=path.resolve(root,'.'+route),relative=path.relative(root,file);if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(404).end();return;}
res.setHeader('Content-Type',({'.js':'text/javascript','.json':'application/json','.wasm':'application/wasm'})[path.extname(file)]||'application/octet-stream');res.end(await readFile(file));
}catch(e){res.writeHead(500).end(String(e));}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;const timer=setTimeout(()=>finish({passed:false,error:'WebKit startup fixture timed out'}),60000);
try{browser=await webkit.launch({headless:true,...(process.env.VKF_WEBKIT_TEST_EXECUTABLE?{executablePath:process.env.VKF_WEBKIT_TEST_EXECUTABLE}:{}),timeout:20000});const page=await browser.newPage();page.on('pageerror',e=>finish({passed:false,error:String(e)}));await page.goto('http://127.0.0.1:'+server.address().port+'/',{timeout:20000});const report=await done;
report.scope='WebKit WASM compile and authored-state initialization only; not physical iPhone/WebGPU acceptance';const file=path.join(work,'result.json');await writeFile(file,JSON.stringify(report,null,2));console.log(JSON.stringify({...report,result:file}));if(!report.passed)process.exitCode=1;
}finally{clearTimeout(timer);await browser?.close();server.closeAllConnections();server.close();}
