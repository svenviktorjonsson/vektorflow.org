const query=new URLSearchParams(location.search);
export const maxParticles=query.get('particles')==='1000000'?1_000_000:10_000;
const status=document.getElementById('status'),progress=document.getElementById('progress'),result=document.getElementById('result'),button=document.getElementById('start'),select=document.getElementById('particles');
select.value=String(maxParticles);
let started=false;
export function emitProgress(sample){
 if(!started){progress.textContent='';started=true;}
 status.textContent=sample.stage?`Preparing ${sample.particles.toLocaleString()} particles…`:`Passed ${sample.particles.toLocaleString()} particle contacts; continuing checks…`;
 progress.textContent+=JSON.stringify(sample,null,2)+'\n';
}
export function emitResult(report){
 report.sourceRevision='a712cf70';report.scope='Contact components; not full application or arbitrary swept-motion certification';report.requestedMaxParticles=maxParticles;
 window.__contactCoreResult=report;
 result.textContent=JSON.stringify(report,null,2);
 status.textContent=report.passed?`PASS · ${report.cases.length} checks completed. This does not certify the unfinished wheel backend.`:`FAIL · ${report.error??'See full result.'}`;
 button.disabled=false;select.disabled=false;button.textContent='Run GPU tests again';
}
if(query.get('run')==='1'){
 button.disabled=true;select.disabled=true;button.textContent='Running…';status.textContent='Starting physical GPU tests…';
 import('./run.mjs').catch(error=>emitResult({passed:false,error:String(error.stack??error)}));
}
