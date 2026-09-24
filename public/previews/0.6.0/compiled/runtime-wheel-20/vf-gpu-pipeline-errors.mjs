export function gpuErrorMessage(error,context='GPU operation') {
  const headline=error?.message?`${error.name||'Error'}: ${error.message}`:String(error);
  const reason=error?.reason?`\nReason: ${error.reason}`:'';
  const stack=error?.stack&& !['_','-',headline].includes(String(error.stack))?`\n${error.stack}`:'';
  return `${context}\n${headline}${reason}${stack}`;
}
export async function createCheckedGpuPipeline(device,kind,descriptor) {
  const entries=kind==='compute'?descriptor.compute.entryPoint:`${descriptor.vertex.entryPoint} / ${descriptor.fragment?.entryPoint||'depth'}`;
  try{return await device[kind==='compute'?'createComputePipelineAsync':'createRenderPipelineAsync'](descriptor);}
  catch(cause){
    const modules=kind==='compute'?[descriptor.compute.module]:[descriptor.vertex.module,descriptor.fragment?.module];
    const messages=[];
    for(const module of new Set(modules.filter(Boolean))){
      let timer;
      try{const info=await Promise.race([module.getCompilationInfo?.(),new Promise(resolve=>{timer=setTimeout(resolve,750);})]);for(const message of info?.messages??[])if(message.type==='error')messages.push(`WGSL ${message.lineNum}:${message.linePos}: ${message.message}`);}catch{}finally{clearTimeout(timer);}
    }
    const detail=messages.length?'\n'+messages.join('\n'):'';
    const error=new Error(gpuErrorMessage(cause,`${descriptor.label||'VKF'}: ${kind} pipeline ${entries}`)+detail,{cause});error.name='GPUStartupError';error.reason=cause?.reason;error.entryPoint=entries;throw error;
  }
}
export async function createOptionalGpuPipeline(device,kind,descriptor,onWarning=console.warn) {
  try{return await createCheckedGpuPipeline(device,kind,descriptor);}
  catch(error){if(error.cause?.name!=='GPUPipelineError')throw error;onWarning(gpuErrorMessage(error,'Optional acceleration unavailable; using the portable GPU path'));return null;}
}
