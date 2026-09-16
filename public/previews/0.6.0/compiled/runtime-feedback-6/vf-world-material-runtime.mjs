import { normalizeLiquidContainedWorldPolicy, createLiquidParticleWorldGpuRuntime }
  from './vf-liquid-contained-world-gpu.mjs';
import { calibrateUniformLocalLiquidParticleMassReference } from './vf-physics-liquid-local-particle-reference.mjs';
import { normalizeGranularParticleWorldGpuPolicy, createGranularParticleWorldGpuRuntime }
  from './vf-granular-particle-world-gpu.mjs';
import { createLiquidParticleEmbeddingGpu } from './vf-liquid-particle-embedding-gpu.mjs';
import { createGranularParticleEmbeddingGpu } from './vf-granular-particle-embedding-gpu.mjs';
import { createWheelEmbeddingGpu } from './vf-contained-boundary-embedding-gpu.mjs';
import { createPreventiveContactResources, createPreventiveParticleContactGpu } from './vf-preventive-particle-contact-gpu.mjs';

// Arena/GPU adapter only: authored data, properties, laws and shader
// specializations come from the compiler. No host material integration.
export function materialInitialState(world, arena) {
  const p=world.properties, count=arena.layer.count;
  const liquid=world.kind==='liquid', stride=liquid?12:8;
  const c=world.geometry.center,r=world.geometry.radius;
  const bounds={worldMinimum:[c[0]-r*1.4,c[1]-r*1.4],worldMaximum:[c[0]+r*1.4,c[1]+r*1.4],
    viewMinimum:[c[0]-r*1.24,c[1]-r*1.24],viewMaximum:[c[0]+r*1.24,c[1]+r*1.24]};
  const bytes=new ArrayBuffer(count*stride*4), floats=new Float32Array(bytes), integers=new Uint32Array(bytes);
  for(let i=0;i<count;i++) {
    const o=i*stride,s=i*6;
    floats[o]=arena.state[s];floats[o+1]=arena.state[s+3];
    floats[o+2]=arena.state[s+1];floats[o+3]=arena.state[s+4];
    floats[o+4]=liquid?floats[o+2]:floats[o];floats[o+5]=liquid?floats[o+3]:floats[o+1];
    if(liquid)floats[o+7]=p.density??1000;else integers[o+6]=i;
  }
  if(liquid) {
    const policy=normalizeLiquidContainedWorldPolicy({...bounds,columns:1,rows:1,
      seedMaximumX:c[0],seedMinimumY:c[1]-r*0.7,bedAtStone:c[1]-r*1.3,
      particleSpacing:p.spacing??0.009,restDensity:p.density??1000,
      viscosity:p.viscosity??1.8,timeStep:world.time_step,gravity:world.gravity,
      diffuseCapacity:Math.max(count,256)});
    const supportRadius=policy.particleSpacing*policy.supportScale;
    const particleMass=calibrateUniformLocalLiquidParticleMassReference({dimension:2,
      spacing:policy.particleSpacing,supportRadius,restDensity:policy.restDensity}).particleMass;
    return {policy,count,stride,bytes,floats,supportRadius,particleMass,
      boundaryPacket:{segmentCount:0,minimumX:bounds.worldMinimum[0],uniformStepX:1,maximumUniformXError:0}};
  }
  const policy=normalizeGranularParticleWorldGpuPolicy({...bounds,columns:1,rows:1,seedMinimum:c,
    grainRadius:p.radius??0.004,particleDensity:p.density??1600,
    friction:p.friction??0.9,boundaryFriction:p.wall_friction??0.78,
    restitution:p.restitution??0.005,rollingResistance:p.rolling_resistance??0.2,
    gravity:world.gravity,timeStep:world.time_step,contactIterations:32,contactSlop:0});
  return {policy,count,stride,bytes,floats,integers,
    particleMass:policy.particleDensity*Math.PI*policy.grainRadius**2,
    contactLaw:{friction:policy.friction,rollingResistance:policy.rollingResistance,restitution:policy.restitution}};
}

export async function createMaterialWorld(device,canvas,compiled,world,arena) {
  const prefix=`$world$gpu$${world.world_id}`;
  const initialState=materialInitialState(world,arena);
  const physicsSource=compiled.readBinding(`${prefix}$physics`),boundarySource=compiled.readBinding(`${prefix}$boundary`);
  if(typeof physicsSource!=='string'||!physicsSource||typeof boundarySource!=='string'||!boundarySource)throw new Error('compiled GPU shaders are missing');
  let solidPacket;
  if(world.kind==='liquid') {
    const policy=initialState.policy,h=initialState.supportRadius;
    const cells=Math.ceil((policy.worldMaximum[0]-policy.worldMinimum[0])/h)*Math.ceil((policy.worldMaximum[1]-policy.worldMinimum[1])/h);
    solidPacket={bytes:new ArrayBuffer((4+cells)*4),boundaryParticleCount:0,maximumCellOccupancy:0,stateHash:'analytic-added-boundaries'};
  }
  const factory=world.kind==='liquid'?createLiquidParticleWorldGpuRuntime:createGranularParticleWorldGpuRuntime;
  const contactResources=createPreventiveContactResources(device,world,initialState.count);
  const physics=await factory(device,{initialState,solidPacket,geometry:world.geometry,
    shaderSource:physicsSource,preventiveContact:contactResources});
  const contact=await createPreventiveParticleContactGpu(device,world,physics,physicsSource,contactResources);
  const embeddingFactory=world.kind==='liquid'?createLiquidParticleEmbeddingGpu:createGranularParticleEmbeddingGpu;
  const embedding=await embeddingFactory(device,canvas,physics,{maximumPixelRatio:1.5});
  const boundary=await createWheelEmbeddingGpu(device,canvas,navigator.gpu.getPreferredCanvasFormat(),{
    shaderSource:boundarySource});
  physics.setWheel({angle:world.geometry.rotation,angularVelocity:0});
  return {world,physics,embedding,boundary,contact,time:0,accumulator:0,paused:false,revision:0,angle:world.geometry.rotation,targetAngle:world.geometry.rotation,logicalAngle:world.geometry.rotation,
    reset(){this.revision++;physics.reset();contact.reset();this.time=0;this.accumulator=0;this.angle=world.geometry.rotation;this.targetAngle=this.angle;this.logicalAngle=this.angle;
      physics.setWheel({angle:world.geometry.rotation,angularVelocity:0});},
    advance(encoder,elapsed,targetAngle){
      this.targetAngle=targetAngle;
      if(!this.paused)this.accumulator=Math.min(this.accumulator+elapsed,world.time_step*8);
      contact.beginFrame(encoder,{requestedDelta:targetAngle-this.logicalAngle,windowDuration:Math.max(elapsed,world.time_step),timeLimit:this.time+this.accumulator,paused:this.paused});
      const budget=contact.eventBudget();
      for(let consumed=0;consumed<budget;consumed+=8){if(!this.paused)physics.predictForces(encoder);contact.advance(encoder,{events:Math.min(8,budget-consumed)});}
      if(physics.publishParticles)physics.publishParticles(encoder);
      contact.finishFrame(encoder);
    },accept(receipt){const advanced=receipt.time-this.time;this.accumulator=Math.max(0,this.accumulator-advanced);this.time=receipt.time;this.angle=receipt.angle;this.logicalAngle+=receipt.angularDelta;this.remainingTime=receipt.remainingTime;},
    destroy(){contact.destroy();physics.destroy();embedding.destroy();boundary.destroy();}};
}

export async function bootMaterialWorlds(compiled) {
  if(!navigator.gpu)throw new Error('The authored material laws require WebGPU.');
  const program=compiled.worldProgram(), arenas=compiled.worldLayerViews();
  const style=document.createElement('style');
  style.textContent=`html,body{margin:0;background:#080e10;color:#c9dcdb;font:16px system-ui;height:100%;overflow:hidden}
    #vf-material-controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:12px;background:#162227}
    #vf-material-controls button{font:inherit;color:inherit;background:#1c2c32;border:1px solid #638080;border-radius:9px;padding:10px 16px}
    #vf-material-controls button[aria-pressed=true]{background:#d3e7e5;color:#122126}
    #vf-material-status{padding:7px 12px;font-size:13px;color:#a5b7b7}
    body{display:flex;flex-direction:column}
    #vf-material-stage{display:block;width:100%;min-height:0;flex:1;touch-action:none;overscroll-behavior:none;user-select:none}
    #vf-material-error{color:#ffd0c0;white-space:pre-wrap;padding:12px}`;
  document.head.append(style);document.body.replaceChildren();
  const controls=document.createElement('div');controls.id='vf-material-controls';
  const status=document.createElement('div');status.id='vf-material-status';status.setAttribute('role','status');
  const canvas=document.createElement('canvas');canvas.id='vf-material-stage';canvas.setAttribute('aria-label','Drag the wheel to rotate its boundaries');
  const errorBox=document.createElement('pre');errorBox.id='vf-material-error';errorBox.hidden=true;errorBox.setAttribute('role','alert');
  document.body.append(controls,status,canvas,errorBox);
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw new Error('No WebGPU adapter is available.');
  const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[]});
  let stopped=false,visible=true,parentVisible=true,active=program.active_view,particles=program.views[program.active_view].embedding?.kind==='particles',angle=0,omega=0,drag=null,previous=null,pending=false;
  const fail=error=>{stopped=true;errorBox.hidden=false;errorBox.textContent=String(error.stack||error.message||error);console.error(error);};
  device.lost.then(info=>fail(new Error(`WebGPU device lost: ${info.message}`)));
  device.addEventListener('uncapturederror',event=>fail(event.error));
  const applications=[];
  for(const world of program.gpu_worlds) {
    const arena=arenas.find(item=>item.layer.id===world.layer_id);
    applications.push(await createMaterialWorld(device,canvas,compiled,world,arena));
  }
  const current=()=>applications.find(app=>app.world.world_id===program.views[active].world_id);
  if(!current())throw new Error('The active View has no material-law application.');
  angle=current().world.geometry.rotation;
  const button=(label,pressed,handler)=>{const b=document.createElement('button');b.textContent=label;b.setAttribute('aria-pressed',String(pressed));b.addEventListener('click',()=>handler(b));controls.append(b);return b;};
  const materialButtons=[];
  let particleButton,pauseButton,resetButton;
  const refreshStatus=()=>{const app=current();status.textContent=`${app.world.kind==='liquid'?'Water':'Sand'} · ${app.physics.primaryCount} vertices · Ø ${(app.world.geometry.radius*2).toFixed(1)} m · ${particles?'raw particles':'material embedding'} · time ${app.time.toFixed(2)} s · wheel ${angle.toFixed(2)} rad`;};
  const refreshControls=()=>{
    const settings=program.views[active].controls??{};
    controls.setAttribute('aria-label',settings.title??'Material World controls');
    for(const entry of materialButtons)entry.b.setAttribute('aria-pressed',String(entry.i===active));
    if(particleButton){particleButton.hidden=settings.particles!==true;particleButton.setAttribute('aria-pressed',String(particles));}
    if(pauseButton){pauseButton.hidden=settings.pause!==true;pauseButton.textContent=current().paused?'Play':'Pause';pauseButton.setAttribute('aria-pressed',String(current().paused));}
    if(resetButton)resetButton.hidden=settings.reset!==true;
    refreshStatus();
  };
  const flip=view=>{
    if(!Number.isInteger(view)||!program.views[view]||!applications.some(app=>app.world.world_id===program.views[view].world_id))throw new Error('invalid material View');
    current().targetAngle=angle;active=view;program.active_view=view;angle=current().targetAngle;previous=null;drag=null;omega=0;
    particles=program.views[view].embedding?.kind==='particles';refreshControls();
  };
  for(let i=0;i<program.views.length;i++) {
    const world=applications.find(app=>app.world.world_id===program.views[i].world_id);
    if(!world)continue;
    materialButtons.push({i,b:button(world.world.kind==='liquid'?'Water':'Sand',i===active,()=>{
      flip(i);
    })});
  }
  particleButton=button('Particles',particles,()=>{particles=!particles;refreshControls();});
  pauseButton=button('Pause',false,()=>{current().paused=!current().paused;previous=null;refreshControls();});
  resetButton=button('Reset',false,()=>{current().reset();angle=current().world.geometry.rotation;omega=0;previous=null;refreshControls();});
  refreshControls();
  const normalize=x=>Math.atan2(Math.sin(x),Math.cos(x));
  const pointerAngle=event=>{const app=current(),p=app.boundary.screenToWorld(event,app.physics.policy);return Math.atan2(p[1]-app.world.geometry.center[1],p[0]-app.world.geometry.center[0]);};
  canvas.addEventListener('pointerdown',event=>{event.preventDefault();if(program.views[active].controls?.rotation!==true)return;drag={id:event.pointerId,last:pointerAngle(event),time:event.timeStamp};omega=0;canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{event.preventDefault();if(!drag||drag.id!==event.pointerId)return;
    const next=pointerAngle(event),dt=Math.max(1/240,Math.min(.05,(event.timeStamp-drag.time)/1000));
    const delta=normalize(next-drag.last);
    angle+=delta;omega=delta/dt;drag.last=next;drag.time=event.timeStamp;
    refreshStatus();
  });
  const release=event=>{event.preventDefault();if(!drag||drag.id!==event.pointerId)return;drag=null;omega=0;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);};
  canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
  for(const type of ['touchstart','touchmove','touchend','touchcancel'])canvas.addEventListener(type,event=>event.preventDefault(),{passive:false});
  const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;previous=null;});observer.observe(canvas);
  window.addEventListener('message',event=>{if(event.origin===location.origin&&event.source===window.parent&&event.data?.type==='vf-preview-visibility'){parentVisible=event.data.active===true;previous=null;}});
  if(window.parent!==window)window.parent.postMessage({type:'vf-preview-ready'},location.origin);
  let frames=0;
  function frame(timestamp) {
    if(stopped)return;
    requestAnimationFrame(frame);
    if(document.hidden||!visible||!parentVisible){previous=null;return;}
    if(pending)return;
    const elapsed=previous==null?0:Math.max(0,Math.min(.05,(timestamp-previous)/1000));previous=timestamp;
    try {
      const app=current();
      const revision=app.revision;
      const encoder=device.createCommandEncoder({label:'VKF authored material World frame'});
      app.advance(encoder,elapsed,angle);
      device.queue.submit([encoder.finish()]);pending=true;
      app.contact.inspect().then(receipt=>{
        if(stopped||revision!==app.revision){pending=false;return;}
        app.accept(receipt);
        if(app!==current()){pending=false;return;}
        const display=device.createCommandEncoder({label:'VKF certified material World embedding'});
        app.embedding.render(display,{time:app.time,mode:particles?'particles':app.world.kind==='liquid'?'fluid':'sand'});
        app.boundary.render(display,canvas.getContext('webgpu').getCurrentTexture().createView(),app.physics.policy,app.physics.wheel,app.angle);
        device.queue.submit([display.finish()]);device.queue.onSubmittedWorkDone().then(()=>{pending=false;},fail);
      },error=>{pending=false;if(revision!==app.revision)return;fail(error);});
      canvas.dataset.renderedFrames=String(++frames);
      if(frames%12===0)refreshStatus();
    }catch(error){fail(error);}
  }
  const application={compiled,program,applications,canvas,
    flip,
    setLayer(id,properties){const layer=program.layers.find(layer=>layer.id===id);if(!layer)throw new Error('unknown retained Layer');Object.assign(layer.properties,properties);
      const app=applications.find(app=>app.world.boundary_ids.includes(id));
      if(properties.rotation!==undefined&&app){if(!Number.isFinite(properties.rotation))throw new Error('rotation must be finite');app.targetAngle=properties.rotation;if(app===current()){angle=app.targetAngle;omega=0;refreshStatus();}}},
    destroy(){stopped=true;observer.disconnect();for(const app of applications)app.destroy();device.destroy();}};
  globalThis.__vfWorldLayerApplication=application;document.body.dataset.vfWorldLayerReady='true';
  requestAnimationFrame(frame);return application;
}
