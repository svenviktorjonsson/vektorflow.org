import { normalizeLiquidContainedWorldPolicy, createLiquidParticleWorldGpuRuntime }
  from './vf-liquid-contained-world-gpu.mjs';
import { calibrateUniformLocalLiquidParticleMassReference } from './vf-physics-liquid-local-particle-reference.mjs';
import { normalizeGranularParticleWorldGpuPolicy, createGranularParticleWorldGpuRuntime }
  from './vf-granular-particle-world-gpu.mjs';
import { createLiquidParticleEmbeddingGpu } from './vf-liquid-particle-embedding-gpu.mjs?v=sand-lane-density-1';
import { createGranularParticleEmbeddingGpu } from './vf-granular-particle-embedding-gpu.mjs?v=sand-lane-density-1';
import { createWheelEmbeddingGpu } from './vf-contained-boundary-embedding-gpu.mjs';
import { PREVENTIVE_PARTICLE_CONTACT_WGSL, createPreventiveContactResources,
  createPreventiveParticleContactGpu } from './vf-preventive-particle-contact-gpu.mjs';
import { createRetainedWorldLoader } from './vf-retained-world-loader.mjs';
import {gpuErrorMessage} from './vf-gpu-pipeline-errors.mjs';
import {granularFormationSeed} from './vf-granular-formation-seed.mjs';

// Arena/GPU adapter only: authored data, properties, laws and shader
// specializations come from the compiler. No host material integration.
export function materialInitialState(world, arena, grainRadiusOverride,
  formation='bed', liquidViscosityOverride, liquidFrictionOverride) {
  const p=world.properties;
  const liquidPile=world.kind==='liquid'&&formation==='dropcastle';
  const seedRadius=liquidPile?(p.spacing??0.009)*0.5:grainRadiusOverride??p.radius??0.004;
  const sandSeed=(world.kind==='granular'
    &&(grainRadiusOverride!==undefined||formation!=='bed'))||liquidPile
    ?granularFormationSeed(seedRadius,
      arena.layer.count*Math.PI*(liquidPile?seedRadius:p.radius??0.004)**2,
      world.geometry,formation):null;
  const count=sandSeed?.positions.length??arena.layer.count;
  const liquid=world.kind==='liquid', stride=liquid?12:8;
  const c=world.geometry.center,r=world.geometry.radius;
  const bounds={worldMinimum:[c[0]-r*1.4,c[1]-r*1.4],worldMaximum:[c[0]+r*1.4,c[1]+r*1.4],
    viewMinimum:[c[0]-r*1.24,c[1]-r*1.24],viewMaximum:[c[0]+r*1.24,c[1]+r*1.24]};
  const bytes=new ArrayBuffer(count*stride*4), floats=new Float32Array(bytes), integers=new Uint32Array(bytes);
  for(let i=0;i<count;i++) {
    const o=i*stride,s=i*6;
    floats[o]=sandSeed?sandSeed.positions[i][0]:arena.state[s];
    floats[o+1]=sandSeed?sandSeed.positions[i][1]:arena.state[s+3];
    floats[o+2]=sandSeed?0:arena.state[s+1];
    floats[o+3]=sandSeed?0:arena.state[s+4];
    floats[o+4]=liquid?floats[o+2]:floats[o];floats[o+5]=liquid?floats[o+3]:floats[o+1];
    if(liquid)floats[o+7]=p.density??1000;else integers[o+6]=i;
  }
  if(liquid) {
    const policy=normalizeLiquidContainedWorldPolicy({...bounds,columns:1,rows:1,
      seedMaximumX:c[0],seedMinimumY:c[1]-r*0.7,bedAtStone:c[1]-r*1.3,
      particleSpacing:p.spacing??0.009,restDensity:p.density??1000,
      viscosity:liquidViscosityOverride??p.viscosity??1.8,
      friction:liquidFrictionOverride??0,
      timeStep:world.time_step,gravity:world.gravity,
      densityIterations:20,maximumParticlesPerCell:192,
      diffuseCapacity:Math.max(count,256)});
    const supportRadius=policy.particleSpacing*policy.supportScale;
    const particleMass=calibrateUniformLocalLiquidParticleMassReference({dimension:2,
      spacing:policy.particleSpacing,supportRadius,restDensity:policy.restDensity,lattice:p.lattice??'cartesian'}).particleMass;
    const restVolume=particleMass/policy.restDensity;
    for(let i=0;i<count;i++)floats[i*stride+11]=restVolume;
    return {policy,count,stride,bytes,floats,supportRadius,particleMass,
      boundaryPacket:{segmentCount:0,minimumX:bounds.worldMinimum[0],uniformStepX:1,maximumUniformXError:0}};
  }
  const policy=normalizeGranularParticleWorldGpuPolicy({...bounds,columns:1,rows:1,seedMinimum:c,
    grainRadius:grainRadiusOverride??p.radius??0.004,particleDensity:p.density??1600,
    friction:p.friction??0.9,boundaryFriction:p.wall_friction??0.78,
    restitution:p.restitution??0.005,rollingResistance:p.rolling_resistance??0.2,
    gravity:world.gravity,timeStep:p.time_step??world.time_step,contactIterations:32,
    gridRebuildInterval:2,
    contactSlop:0});
  return {policy,count,stride,bytes,floats,integers,
    particleMass:policy.particleDensity*Math.PI*policy.grainRadius**2,
    contactLaw:{friction:policy.friction,rollingResistance:policy.rollingResistance,restitution:policy.restitution}};
}

export async function createMaterialWorld(device,canvas,compiled,world,arena,engineeringOptions={}) {
  const prefix=world.binding_prefix??`$world$gpu$${world.world_id}`;
  const initialState=materialInitialState(world,arena,
    engineeringOptions.grainRadius,engineeringOptions.formation,
    engineeringOptions.liquidViscosity,engineeringOptions.liquidFriction);
  const physicsSource=compiled.readBinding(`${prefix}$physics`),boundarySource=compiled.readBinding(`${prefix}$boundary`);
  if(typeof physicsSource!=='string'||!physicsSource||typeof boundarySource!=='string'||!boundarySource)throw new Error('compiled GPU shaders are missing');
  const contactSeam=physicsSource.indexOf('struct MotionParams');
  if(contactSeam<0)throw new Error('compiled material shader is missing its contact seam');
  // Compiler owns material-law specialization. Runtime owns shared transport
  // adapters, so one current contact implementation replaces embedded copies.
  const runtimePhysicsSource=physicsSource.slice(0,contactSeam)+PREVENTIVE_PARTICLE_CONTACT_WGSL;
  let solidPacket;
  if(world.kind==='liquid') {
    const policy=initialState.policy,h=initialState.supportRadius;
    const cells=Math.ceil((policy.worldMaximum[0]-policy.worldMinimum[0])/h)*Math.ceil((policy.worldMaximum[1]-policy.worldMinimum[1])/h);
    solidPacket={bytes:new ArrayBuffer((4+cells)*4),boundaryParticleCount:0,maximumCellOccupancy:0,stateHash:'analytic-added-boundaries'};
  }
  const factory=world.kind==='liquid'?createLiquidParticleWorldGpuRuntime:createGranularParticleWorldGpuRuntime;
  const contactResources=createPreventiveContactResources(device,world,initialState.count);
  // Liquid pressure and granular friction/cohesion are particle-grid Laws.
  // Preventive contact owns only rigid wheel boundaries in this shared pipeline;
  // duplicating particle pairs here caused quadratic work and zero-TOI stalls.
  contactResources.particlePairContact=false;
  contactResources.persistentSkin=engineeringOptions.contactPersistentSkin===true;
  contactResources.predictedContacts=engineeringOptions.predictedContacts===true;
  contactResources.starCG=engineeringOptions.starCG===true;
  contactResources.sandBarrier=engineeringOptions.sandBarrier===true;
  contactResources.fullWindow=engineeringOptions.fullWindow===true;
  contactResources.blockPreconditioner=engineeringOptions.blockPreconditioner===true;
  contactResources.rotatingTrajectory=engineeringOptions.rotatingTrajectory??runtimePhysicsSource.includes('fn motion_enter_rotating_frame');
  contactResources.starNonlinearRim=engineeringOptions.starNonlinearRim===true;
  contactResources.starSweeps=engineeringOptions.starSweeps??1;
  contactResources.experimentalSolve=engineeringOptions.liquidSolve??'barrier';
  contactResources.experimentalDispatch=engineeringOptions.liquidDispatch??'auto';
  contactResources.cacheEpoch=engineeringOptions.cacheEpoch===true;
  const physics=await factory(device,{initialState,solidPacket,geometry:world.geometry,
    shaderSource:runtimePhysicsSource,preventiveContact:contactResources,
    forceGeometryCaching:engineeringOptions.forceGeometryCaching??contactResources.rotatingTrajectory});
  engineeringOptions.onStage?.('particle laws ready');
  const contact=await createPreventiveParticleContactGpu(device,world,physics,runtimePhysicsSource,contactResources);
  engineeringOptions.onStage?.('wheel contact ready');
  const embeddingFactory=world.kind==='liquid'?createLiquidParticleEmbeddingGpu:createGranularParticleEmbeddingGpu;
  const embedding=await embeddingFactory(device,canvas,physics,{maximumPixelRatio:1.5,
    wheelGeometry:world.kind==='liquid'?world.geometry:undefined});
  if(world.kind==='granular'){
    physics.setEffectiveGrainRadius(engineeringOptions.effectiveGrainRadius??initialState.policy.grainRadius);
    embedding.setEffectiveGrainRadius(engineeringOptions.effectiveGrainRadius??initialState.policy.grainRadius);
    physics.setWetness(engineeringOptions.wetness??0);
    embedding.setWetness(engineeringOptions.wetness??0);
    embedding.setLaneWidth(engineeringOptions.laneWidthPixels??1);
  }
  if(engineeringOptions.visualMode==='sand')
    embedding.setEffectiveGrainRadius(engineeringOptions.effectiveGrainRadius??0.0075);
  if(engineeringOptions.visualMode==='sand')
    embedding.setColors({water:[0.72,0.57,0.39,1]});
  engineeringOptions.onStage?.('material embedding ready');
  const boundary=await createWheelEmbeddingGpu(device,canvas,navigator.gpu.getPreferredCanvasFormat(),{
    shaderSource:boundarySource});
  physics.setWheel({angle:world.geometry.rotation,angularVelocity:0});
  if(world.kind==='granular')physics.setSweepReference(world.geometry.rotation);
  const startPaused=engineeringOptions.startPaused===true;
  return {world,physics,embedding,boundary,contact,
    visualMode:engineeringOptions.visualMode,
    time:0,accumulator:0,paused:startPaused,revision:0,angle:world.geometry.rotation,targetAngle:world.geometry.rotation,logicalAngle:world.geometry.rotation,wheelRate:0,
    reset(){this.revision++;physics.reset();contact.reset();embedding.reset?.();this.pausedReferenceAngle=undefined;this.published=false;this.time=0;this.accumulator=0;this.paused=startPaused;this.angle=world.geometry.rotation;this.targetAngle=this.angle;this.logicalAngle=this.angle;this.wheelRate=0;
      physics.setWheel({angle:world.geometry.rotation,angularVelocity:0});
      if(world.kind==='granular')physics.setSweepReference(world.geometry.rotation);},
    advance(encoder,elapsed,targetAngle,commandedOmega=0){
      this.targetAngle=targetAngle;
      if(this.paused){
        if(Math.abs(targetAngle-this.logicalAngle)<1e-7){
          if(this.published)return false;
          contact.prepareRepresentation(encoder);if(physics.publishParticles)physics.publishParticles(encoder);contact.finishFrame(encoder);this.published=true;return true;
        }
        const capture=this.pausedReferenceAngle===undefined;
        if(capture)this.pausedReferenceAngle=this.logicalAngle;
        contact.rotatePaused(encoder,{delta:targetAngle-this.pausedReferenceAngle,capture});
        contact.prepareRepresentation(encoder);if(physics.publishParticles)physics.publishParticles(encoder);contact.finishFrame(encoder);return true;
      }
      this.pausedReferenceAngle=undefined;
      if(world.kind==='granular'){
        const stepTime=physics.policy.timeStep;
        this.accumulator=Math.min(this.accumulator+elapsed,stepTime*8);
        const horizon=Math.max(elapsed,stepTime);
        const requested=targetAngle-this.logicalAngle;
        // Swept contact subdivides the authored rigid path on the GPU. Bound
        // angular speed, not angle per rendered frame, so a low frame rate
        // cannot make pointer rotation crawl.
        const angularBudget=Math.min(0.12,4*horizon);
        const delta=Math.max(-angularBudget,Math.min(angularBudget,requested));
        const nextAngle=this.logicalAngle+delta;
        physics.setWheel({angle:nextAngle,angularVelocity:delta/horizon});
        if(Math.abs(delta)>1e-8)physics.sweepWheel(encoder,horizon);
        const steps=Math.min(8,Math.floor((this.accumulator+stepTime*1e-6)/stepTime));
        if(steps>0){physics.stepMany(encoder,steps);const advanced=steps*stepTime;
          this.accumulator=Math.max(0,this.accumulator-advanced);this.time+=advanced;}
        this.angle=nextAngle;this.logicalAngle=nextAngle;this.remainingTime=0;
        this.peakWheelSurfaceSpeed=Math.max(this.peakWheelSurfaceSpeed??0,Math.abs(delta/horizon)*world.geometry.radius);
        device.queue.writeBuffer(contact.resources.control,4,new Float32Array([nextAngle]));
        contact.synchronizeExternalStep(this.time);
        return false;
      }
      // Keep real elapsed time as debt. Clamping it to eight substeps made
      // simulation time freeze whenever a rotating frame took >33 ms.
      this.accumulator+=elapsed;
      const stepTime=world.time_step;
      const steps=Math.min(16,Math.floor((this.accumulator+stepTime*1e-6)/stepTime));
      const requested=targetAngle-this.logicalAngle;
      if(Math.abs(commandedOmega)>1e-6&&Math.sign(commandedOmega)===Math.sign(requested))this.wheelRate=commandedOmega;
      const rate=Math.min(6,Math.abs(this.wheelRate)||Math.abs(requested)/Math.max(elapsed,stepTime));
      const duration=Math.max(steps*stepTime,stepTime);
      const delta=Math.sign(requested)*Math.min(Math.abs(requested),rate*duration);
      const nextAngle=this.logicalAngle+delta;
      if(Math.abs(delta)>1e-8){
        physics.setWheel({angle:nextAngle,angularVelocity:delta/duration});
        if(steps>0)physics.stepMovingWheel(encoder,steps);
        else physics.sweepWheel(encoder,Math.max(elapsed,stepTime));
        this.angle=nextAngle;this.logicalAngle=nextAngle;
        this.peakWheelSurfaceSpeed=Math.max(this.peakWheelSurfaceSpeed??0,
          Math.abs(delta/duration)*world.geometry.radius);
        device.queue.writeBuffer(contact.resources.control,4,new Float32Array([nextAngle]));
      }else physics.setWheel({angle:this.logicalAngle,angularVelocity:0});
      if(Math.abs(targetAngle-this.logicalAngle)<1e-7)this.wheelRate=0;
      if(steps>0){
        if(Math.abs(delta)<=1e-8)physics.stepMany(encoder,steps);
        const advanced=steps*stepTime;
        this.accumulator=Math.max(0,this.accumulator-advanced);this.time+=advanced;
      }
      contact.synchronizeExternalStep(this.time);
      return false;
    },accept(receipt){const advanced=receipt.time-this.time;this.accumulator=Math.max(0,this.accumulator-advanced);this.time=receipt.time;this.angle=receipt.angle;this.logicalAngle+=receipt.angularDelta;this.remainingTime=receipt.remainingTime;this.peakWheelSurfaceSpeed=receipt.peakWheelSurfaceSpeed;physics.setWheel({angle:this.angle,angularVelocity:0});},
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
    #vf-material-controls label{display:flex;align-items:center;gap:7px;font-size:13px;white-space:nowrap}
    #vf-material-controls label[hidden]{display:none}
    #vf-material-controls input[type=range]{width:clamp(64px,18vw,110px);accent-color:#b6d4c6}
    #vf-material-controls output{min-width:48px;text-align:right;font-variant-numeric:tabular-nums}
    #vf-material-status{padding:7px 12px;font-size:13px;color:#a5b7b7}
    body{display:flex;flex-direction:column}
    #vf-material-stage{display:block;width:100%;min-height:0;flex:1;touch-action:none;overscroll-behavior:none;user-select:none}
    #vf-material-error{color:#ffd0c0;white-space:pre-wrap;padding:12px;overflow:auto;max-height:45vh;flex-shrink:0}
    #vf-material-controls,#vf-material-status{flex-shrink:0}`;
  document.head.append(style);document.body.replaceChildren();
  const controls=document.createElement('div');controls.id='vf-material-controls';
  const status=document.createElement('div');status.id='vf-material-status';status.setAttribute('role','status');
  status.textContent='Starting compiled material World: requesting GPU…';
  const canvas=document.createElement('canvas');canvas.id='vf-material-stage';canvas.setAttribute('aria-label','Drag the wheel to rotate its boundaries');
  const errorBox=document.createElement('pre');errorBox.id='vf-material-error';errorBox.hidden=true;errorBox.setAttribute('role','alert');
  document.body.append(controls,status,canvas,errorBox);
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw new Error('No WebGPU adapter is available.');
  status.textContent='Starting compiled material World: creating GPU device…';
  const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[]});
  let stopped=false,visible=true,parentVisible=true,active=program.active_view,particles=program.views[program.active_view].embedding?.kind==='particles',angle=0,omega=0,drag=null,previous=null,programmaticInputTime=0,pending=false,sandFramesInFlight=0,liquidFramesInFlight=0,switching=false;
  const sandWorld=program.gpu_worlds.find(world=>world.kind==='granular');
  const liquidWorld=program.gpu_worlds.find(world=>world.kind==='liquid');
  let sandWetness=0,sandLaneWidth=1,
    sandRadius=sandWorld?.properties.radius??0.0075,
    sandGuideRadius=sandRadius,
    sandFormation='bed';
  const fail=error=>{stopped=true;errorBox.hidden=false;errorBox.textContent=gpuErrorMessage(error,'Material World stopped');console.error(error);};
  device.lost.then(info=>fail(new Error(`WebGPU device lost: ${info.message}`)));
  device.addEventListener('uncapturederror',event=>fail(event.error));
  const materialLayerId=view=>view.embedding?.layer;
  const layerForView=view=>program.gpu_worlds.find(world=>world.world_id===view.world_id&&world.layer_id===materialLayerId(view));
  const loader=createRetainedWorldLoader(program.gpu_worlds,async world=>{
    status.textContent=`Starting ${world.kind==='liquid'?'water':'sand'}: compiling its GPU laws and embedding…`;
    const arena=arenas.find(item=>item.layer.id===world.layer_id);
    const view=program.views.find(view=>view.world_id===world.world_id&&materialLayerId(view)===world.layer_id);
    return createMaterialWorld(device,canvas,compiled,world,arena,{startPaused:view?.controls?.start_paused===true,
      wetness:world.kind==='granular'?sandWetness:undefined,
      laneWidthPixels:world.kind==='granular'?sandLaneWidth:undefined,
      effectiveGrainRadius:world.kind==='granular'?sandRadius:undefined,
      formation:world.kind==='granular'?sandFormation:'bed',
      onStage:stage=>{status.textContent=`Starting ${world.kind==='liquid'?'water':'sand'}: ${stage}…`;}});
  },world=>world.layer_id);
  const applications=loader.applications;
  await loader.ensure(materialLayerId(program.views[active]));
  let fluidSandApplication=null,comparingFluidSand=false;
  const current=()=>comparingFluidSand?fluidSandApplication:
    applications.find(app=>app.world.layer_id===materialLayerId(program.views[active]));
  if(!current())throw new Error('The active View has no material-law application.');
  angle=current().world.geometry.rotation;
  const button=(label,pressed,handler)=>{const b=document.createElement('button');b.textContent=label;b.setAttribute('aria-pressed',String(pressed));b.addEventListener('click',()=>handler(b));controls.append(b);return b;};
  const materialButtons=[];
  let fluidSandButton;
  let particleButton,pauseButton,resetButton,formationButton,rainButton,
    wetnessControl,grainControl,laneWidthControl;
  const refreshStatus=()=>{const app=current();status.textContent=`${comparingFluidSand?'Liquid-sand test':app.world.kind==='liquid'?'Water':'Sand'} · ${app.physics.primaryCount} vertices · Ø ${(app.world.geometry.radius*2).toFixed(1)} m · ${particles?'raw particles':'material embedding'} · time ${app.time.toFixed(2)} s · wheel ${app.angle.toFixed(2)} rad${comparingFluidSand?` · visual grain Ø ${(sandRadius*2000).toFixed(1)} mm · pressure + friction μ=0.8`:app.world.kind==='granular'?` · wet ${Math.round(sandWetness*100)}% · effective grain Ø ${(sandRadius*2000).toFixed(1)} mm`:''}`;};
  const refreshControls=()=>{
    const settings=program.views[active].controls??{};
    controls.setAttribute('aria-label',settings.title??'Material World controls');
    for(const entry of materialButtons)entry.b.setAttribute('aria-pressed',String(entry.i===active&&!comparingFluidSand));
    if(fluidSandButton)fluidSandButton.setAttribute('aria-pressed',String(comparingFluidSand));
    if(particleButton){particleButton.hidden=settings.particles!==true;particleButton.setAttribute('aria-pressed',String(particles));}
    if(pauseButton){pauseButton.hidden=settings.pause!==true;pauseButton.textContent=current().paused?'Play':'Pause';pauseButton.setAttribute('aria-pressed',String(current().paused));}
    if(resetButton)resetButton.hidden=settings.reset!==true;
    if(formationButton){formationButton.hidden=current().world.kind!=='granular';
      formationButton.textContent=sandFormation==='bed'?'Dropcastle':'Level bed';}
    if(rainButton)rainButton.hidden=current().world.kind!=='granular';
    if(wetnessControl)wetnessControl.hidden=current().world.kind!=='granular';
    if(grainControl){grainControl.hidden=current().world.kind!=='granular'&&!comparingFluidSand;
      const label=comparingFluidSand?'Visual grain Ø':'Effective grain Ø';
      grainControl.querySelector('span').textContent=label;
      grainControl.querySelector('input').setAttribute('aria-label',label);}
    if(laneWidthControl)laneWidthControl.hidden=current().world.kind!=='granular'||particles;
    refreshStatus();
  };
  const flip=async view=>{
    if(!Number.isInteger(view)||!program.views[view]||!layerForView(program.views[view]))throw new Error('invalid material View');
    if(switching||(view===active&&!comparingFluidSand))return;
    switching=true;drag=null;omega=0;
    for(const b of controls.querySelectorAll('button'))b.disabled=true;
    try{
      if(pending||sandFramesInFlight>0||liquidFramesInFlight>0)await device.queue.onSubmittedWorkDone();
      await loader.ensure(materialLayerId(program.views[view]));
      current().targetAngle=angle;comparingFluidSand=false;
      active=view;program.active_view=view;angle=current().targetAngle;previous=null;
      particles=program.views[view].embedding?.kind==='particles';errorBox.hidden=true;refreshControls();
    }catch(error){errorBox.hidden=false;errorBox.textContent=`This World could not start: ${error.message||error}`;refreshControls();throw error;}
    finally{switching=false;previous=null;for(const b of controls.querySelectorAll('button'))b.disabled=false;}
  };
  for(let i=0;i<program.views.length;i++) {
    const world=layerForView(program.views[i]);
    if(world?.kind!=='liquid')continue;
    materialButtons.push({i,b:button('Water',i===active,()=>{
      flip(i).catch(error=>console.error('VKF View startup',error));
    })});
  }
  for(let i=0;i<program.views.length;i++) {
    const world=layerForView(program.views[i]);
    if(world?.kind!=='granular')continue;
    materialButtons.push({i,b:button('Sand',false,()=>{
      flip(i).catch(error=>console.error('VKF View startup',error));
    })});
  }
  fluidSandButton=button('Liquid-sand test',false,async()=>{
    if(switching||comparingFluidSand)return;
    switching=true;drag=null;omega=0;
    for(const element of controls.querySelectorAll('button,input'))element.disabled=true;
    try{
      if(pending||sandFramesInFlight>0||liquidFramesInFlight>0)
        await device.queue.onSubmittedWorkDone();
      if(!fluidSandApplication){
        const arena=arenas.find(item=>item.layer.id===liquidWorld.layer_id);
        fluidSandApplication=await createMaterialWorld(device,canvas,compiled,
          liquidWorld,arena,{startPaused:true,formation:'dropcastle',liquidViscosity:3.6,liquidFriction:0.8,
            visualMode:'sand',effectiveGrainRadius:sandRadius,onStage:stage=>{
              status.textContent=`Starting sand: ${stage}…`;}});
        applications.push(fluidSandApplication);
      }
      current().targetAngle=angle;
      comparingFluidSand=true;
      active=program.views.findIndex(view=>view.world_id===liquidWorld.world_id);
      program.active_view=active;
      angle=fluidSandApplication.targetAngle;
      particles=false;previous=null;errorBox.hidden=true;refreshControls();
    }catch(error){errorBox.hidden=false;
      errorBox.textContent=`Sand could not start: ${error.message||error}`;
      refreshControls();}
    finally{switching=false;previous=null;
      for(const element of controls.querySelectorAll('button,input'))element.disabled=false;}
  });
  particleButton=button('Particles',particles,()=>{particles=!particles;refreshControls();});
  pauseButton=button('Pause',false,()=>{const app=current();app.paused=!app.paused;app.pausedReferenceAngle=undefined;
    if(!app.paused&&app.world.kind==='granular')app.physics.setSweepReference(app.angle);
    previous=null;refreshControls();});
  resetButton=button('Reset',false,()=>{current().reset();angle=current().world.geometry.rotation;omega=0;previous=null;refreshControls();});
  const rebuildSand=async(requestedFormation)=>{
    if(requestedFormation===sandFormation)return;
    switching=true;drag=null;omega=0;
    for(const element of controls.querySelectorAll('button,input'))element.disabled=true;
    status.textContent='Rebuilding the conserved sand formation…';
    try{
      if(pending||sandFramesInFlight>0||liquidFramesInFlight>0)await device.queue.onSubmittedWorkDone();
      const old=applications.find(app=>app.world.kind==='granular');
      const arena=arenas.find(item=>item.layer.id===sandWorld.layer_id);
      const view=program.views.find(item=>item.world_id===sandWorld.world_id&&materialLayerId(item)===sandWorld.layer_id);
      const replacement=await createMaterialWorld(device,canvas,compiled,sandWorld,arena,{
        startPaused:view?.controls?.start_paused===true,wetness:sandWetness,
        laneWidthPixels:sandLaneWidth,
        effectiveGrainRadius:sandRadius,formation:requestedFormation,
        onStage:stage=>{status.textContent=`Rebuilding sand: ${stage}…`;}});
      const index=applications.indexOf(old);
      if(index<0){replacement.destroy();throw new Error('Sand application disappeared during rebuild');}
      old.revision++;applications[index]=replacement;old.destroy();
      sandFormation=requestedFormation;
      if(current()===replacement){angle=replacement.angle;previous=null;}
      refreshControls();
    }catch(error){errorBox.hidden=false;
      errorBox.textContent=`Sand could not be rebuilt: ${error.message||error}`;
      refreshControls();}
    finally{switching=false;previous=null;
      for(const element of controls.querySelectorAll('button,input'))element.disabled=false;}
  };
  formationButton=button('Dropcastle',false,()=>{
    rebuildSand(sandFormation==='bed'?'dropcastle':'bed');
  });
  rainButton=button('Sandfall',false,()=>rebuildSand('rain'));
  const slider=(name,min,max,step,value,format)=>{
    const label=document.createElement('label');
    const title=document.createElement('span');title.textContent=name;
    const input=document.createElement('input');input.type='range';
    input.min=String(min);input.max=String(max);input.step=String(step);input.value=String(value);
    input.setAttribute('aria-label',name);
    const output=document.createElement('output');output.value=format(value);
    label.append(title,input,output);controls.append(label);
    return {label,input,output};
  };
  const wetnessSlider=slider('Wetness',0,100,1,0,value=>`${value}%`);
  wetnessControl=wetnessSlider.label;
  wetnessSlider.input.addEventListener('input',()=>{
    sandWetness=Number(wetnessSlider.input.value)/100;
    wetnessSlider.output.value=`${wetnessSlider.input.value}%`;
    for(const app of applications)if(app.world.kind==='granular'){
      app.physics.setWetness(sandWetness);app.embedding.setWetness(sandWetness);
    }
    refreshStatus();
  });
  const grainSlider=slider('Effective grain Ø',sandGuideRadius*200,sandGuideRadius*2000,0.1,sandRadius*2000,
    value=>`${Number(value).toFixed(1)} mm`);
  grainControl=grainSlider.label;
  grainSlider.input.addEventListener('input',()=>{
    sandRadius=Number(grainSlider.input.value)/2000;
    grainSlider.output.value=`${(sandRadius*2000).toFixed(1)} mm`;
    for(const app of applications)if(app.world.kind==='granular'){
      app.physics.setEffectiveGrainRadius(sandRadius);
      app.embedding.setEffectiveGrainRadius(sandRadius);
    }
    fluidSandApplication?.embedding.setEffectiveGrainRadius(sandRadius);
    refreshStatus();
  });
  const laneWidthSlider=slider('Flow width',0.75,5,0.25,sandLaneWidth,
    value=>`${Number(value).toFixed(2)} px`);
  laneWidthControl=laneWidthSlider.label;
  laneWidthSlider.input.addEventListener('input',()=>{
    sandLaneWidth=Number(laneWidthSlider.input.value);
    laneWidthSlider.output.value=`${sandLaneWidth.toFixed(2)} px`;
    for(const app of applications)if(app.world.kind==='granular')
      app.embedding.setLaneWidth(sandLaneWidth);
  });
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
    if(document.hidden||!visible||!parentVisible||switching){previous=null;return;}
    if(pending||(current().world.kind==='granular'?sandFramesInFlight>=2:liquidFramesInFlight>=3))return;
    const elapsed=previous==null?0:Math.max(0,(timestamp-previous)/1000);previous=timestamp;
    try {
      const app=current();
      const revision=app.revision;
      const encoder=device.createCommandEncoder({label:'VKF authored material World frame'});
      const updated=app.advance(encoder,elapsed,angle,omega);
      if(!updated){
        app.embedding.render(encoder,{time:app.time,wheelAngle:app.angle,mode:particles?'particles':app.visualMode??(app.world.kind==='liquid'?'fluid':'sand')});
        app.boundary.render(encoder,canvas.getContext('webgpu').getCurrentTexture().createView(),app.physics.policy,app.physics.wheel,app.angle);
        device.queue.submit([encoder.finish()]);
        if(app.world.kind==='granular')sandFramesInFlight++;else liquidFramesInFlight++;
        device.queue.onSubmittedWorkDone().then(()=>{
          if(app.world.kind==='granular')sandFramesInFlight--;else liquidFramesInFlight--;
          if(stopped||revision!==app.revision)return;
          refreshStatus();
          canvas.dataset.presentedFrames=String(Number(canvas.dataset.presentedFrames||0)+1);
        }).catch(error=>{
          if(app.world.kind==='granular')sandFramesInFlight--;else liquidFramesInFlight--;
          if(revision!==app.revision)return;fail(error);
        });
        canvas.dataset.renderedFrames=String(++frames);
        if(frames%12===0)refreshStatus();
        return;
      }
      device.queue.submit([encoder.finish()]);pending=true;
      (updated?app.contact.inspect():Promise.resolve(null)).then(receipt=>{
        if(stopped||revision!==app.revision){pending=false;return;}
        if(receipt)app.accept(receipt);
        refreshStatus();
        if(app!==current()){pending=false;return;}
        const display=device.createCommandEncoder({label:'VKF certified material World embedding'});
        app.embedding.render(display,{time:app.time,wheelAngle:app.angle,mode:particles?'particles':app.visualMode??(app.world.kind==='liquid'?'fluid':'sand')});
        app.boundary.render(display,canvas.getContext('webgpu').getCurrentTexture().createView(),app.physics.policy,app.physics.wheel,app.angle);
        device.queue.submit([display.finish()]);return device.queue.onSubmittedWorkDone().then(()=>{pending=false;canvas.dataset.presentedFrames=String(Number(canvas.dataset.presentedFrames||0)+1);});
      }).catch(error=>{pending=false;if(revision!==app.revision)return;fail(error);});
      canvas.dataset.renderedFrames=String(++frames);
      if(frames%12===0)refreshStatus();
    }catch(error){fail(error);}
  }
  const application={compiled,program,applications,canvas,
    flip,
    setLayer(id,properties){const layer=program.layers.find(layer=>layer.id===id);if(!layer)throw new Error('unknown retained Layer');Object.assign(layer.properties,properties);
      const affected=applications.filter(app=>app.world.boundary_ids.includes(id));
      if(properties.rotation!==undefined){if(!Number.isFinite(properties.rotation))throw new Error('rotation must be finite');for(const app of affected)app.targetAngle=properties.rotation;if(affected.includes(current())){const now=performance.now();const interval=programmaticInputTime>0?Math.max(1/240,(now-programmaticInputTime)/1000):1/60;omega=(properties.rotation-angle)/interval;programmaticInputTime=now;angle=properties.rotation;refreshStatus();}}},
    destroy(){stopped=true;observer.disconnect();loader.destroy();device.destroy();}};
  globalThis.__vfWorldLayerApplication=application;document.body.dataset.vfWorldLayerReady='true';
  requestAnimationFrame(frame);return application;
}
