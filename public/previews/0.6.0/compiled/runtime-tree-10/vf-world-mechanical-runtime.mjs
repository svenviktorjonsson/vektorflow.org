import { createMechanicalWorldGpu } from './vf-world-mechanical-gpu.mjs';
import { createWorldSceneEmbeddingGpu,projectPoint,sub } from './vf-world-scene-embedding-gpu.mjs';
import {cantileverParameters,zoomCamera,emissiveSphere} from './vf-world-physical-parameters.mjs';

export function mechanicalFrameSchedule(world,debt,elapsed){
  const dt=world.time_step,budget=world.kind==='rigid'?64:8;
  const total=world.kind==='rigid'?debt+elapsed:Math.min(debt+Math.min(.05,elapsed),dt*8);
  const steps=Math.min(budget,Math.floor((total+dt*1e-12)/dt));
  return {steps,debt:Math.max(0,total-steps*dt)};
}

export async function readMechanicalAsset(url){
  const response=await fetch(url);if(!response.ok)throw new Error(`Added geometry unavailable: ${response.status}`);
  const bytes=url.includes('.gz')?await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():await response.arrayBuffer();
  const view=new DataView(bytes),decode=new TextDecoder();if(decode.decode(new Uint8Array(bytes,0,8))!=='VFTREE02')throw new Error('Invalid added geometry asset');
  let offset=20;const meshes=[];
  for(let i=0;i<view.getUint32(8,true);i++){
    const sizes=Array.from({length:5},(_,j)=>view.getUint32(offset+j*4,true));offset+=20;
    const meta=JSON.parse(decode.decode(new Uint8Array(bytes,offset,sizes[0])));offset+=sizes[0]+(4-sizes[0]%4)%4;
    const vertices=new Float32Array(bytes,offset,sizes[1]);offset+=sizes[1]*4;
    const indices=new Uint32Array(bytes,offset,sizes[2]);offset+=sizes[2]*4;
    const uvs=new Float32Array(bytes,offset,sizes[3]);offset+=(sizes[3]+sizes[4])*4;
    if(vertices.length%10||indices.some(index=>index>=vertices.length/10)||vertices.some(v=>!Number.isFinite(v)))throw new Error('Invalid geometry vertices or indices');
    meshes.push({...meta,vertices,indices,uvs});
  }
  if(offset!==bytes.byteLength)throw new Error('Geometry asset length mismatch');return meshes;
}
export function prepareMechanicalInitialState(world,arenas,asset){
  const arena=arenas.find(a=>a.layer.id===world.layer_id),kind=world.kind,p=world.properties;
  const meshes=[];const initial={bodyCount:0,parcelCount:0,nodeCount:0,bodies:new Float32Array(20),parcels:new Float32Array(8),nodes:new Float32Array(8),geometry:new Float32Array(4)};
  const convert=(vertices,tag,compliance,transform)=>{
    const out=new Float32Array(vertices.length/10*24);
    for(let i=0;i<vertices.length/10;i++){const src=i*10,d=i*24;out.set(vertices.subarray(src,src+10),d);if(transform)transform(out,d);out[d+10]=tag;out[d+11]=compliance?compliance(out,d):0;}
    return out;
  };
  if(kind==='rigid'){
    if(asset.length!==arena.layer.count)throw new Error('Added bodies and stone asset disagree');
    initial.bodyCount=asset.length;initial.hullCount=96;initial.bodies=new Float32Array(asset.length*20);initial.geometry=new Float32Array(asset.length*96*4);
    for(let i=0;i<asset.length;i++){
      const item=asset[i],o=i*20,s=i*9;const position=[arena.state[s],arena.state[s+3],arena.state[s+6]];
      if(!item.collision||item.collision.center.some((v,a)=>Math.abs(v-position[a])>1e-5))throw new Error('Added rigid placement disagrees with geometry asset');
      const mass=p.mass??item.collision.mass*(p.density??2700)/2700;if(!(mass>0))throw new Error('Rigid mass must be positive');
      initial.bodies.set([...position,0,arena.state[s+1],arena.state[s+4],arena.state[s+7],0,0,0,0,1,0,0,0,0,mass,item.collision.inertia*mass/item.collision.mass,item.collision.radius,p.sleep===true?1:0],o);
      initial.geometry.set(item.collision.hull,i*96*4);
      meshes.push({vertices:convert(item.vertices,i,null),indices:item.indices});
    }
  }else{
    initial.parcelCount=arena.layer.count;initial.parcels=new Float32Array(initial.parcelCount*8);
    for(let i=0;i<initial.parcelCount;i++){const s=i*9;initial.parcels.set([arena.state[s],arena.state[s+3],arena.state[s+6],0,arena.state[s+1],arena.state[s+4],arena.state[s+7],0],i*8);}
    const nodeArena=arenas.find(a=>a.layer.id===world.nodes_layer_id);initial.nodeCount=nodeArena.layer.count;initial.nodes=new Float32Array(initial.nodeCount*16);initial.geometry=new Float32Array(initial.nodeCount*20);
    const beam=world.nodes_properties.young_modulus?cantileverParameters(world.nodes_properties):null;
    const grid=p.grid,min=p.domain_min,span=p.domain_span;
    const cell=(x,y,z)=>Math.min(grid[0]-1,Math.max(0,Math.floor((x-min[0])/span[0]*grid[0])))+grid[0]*(Math.min(grid[1]-1,Math.max(0,Math.floor((y-min[1])/span[1]*grid[1])))+grid[1]*Math.min(grid[2]-1,Math.max(0,Math.floor((z-min[2])/span[2]*grid[2]))));
    for(let i=0;i<initial.nodeCount;i++){
      const x=i%grid[0],y=Math.floor(i/grid[0])%grid[1],z=Math.floor(i/(grid[0]*grid[1]));
      const expected=[min[0]+(x+.5)*span[0]/grid[0],min[1]+(y+.5)*span[1]/grid[1],min[2]+(z+.5)*span[2]/grid[2]];
      for(let a=0;a<3;a++)if(Math.abs(nodeArena.state[i*9+a*3]-expected[a])>1e-6)throw new Error('Added node positions disagree with the declared regular grid');
      initial.geometry.set([...expected,z===0?2:0],i*4);
      initial.geometry[(initial.nodeCount*3+i)*4+3]=i;
    }
    let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const item of asset)for(let j=0;j<item.vertices.length;j+=10)for(let a=0;a<3;a++){lo[a]=Math.min(lo[a],item.vertices[j+a]);hi[a]=Math.max(hi[a],item.vertices[j+a]);}
    const scale=(world.solid_properties.height??8)/(hi[2]-lo[2]);const origin=arenas.find(a=>a.layer.id===world.solid_layer_id).state;
    const routeDistance=new Float64Array(initial.nodeCount).fill(Infinity);
    for(const item of asset){const collisionClass=item.id?.includes('foliage')?3:1;const vertices=convert(item.vertices,-1,(v,d)=>Math.pow(Math.max(0,v[d+2])/8,2)*0.65,(v,d)=>{
      v[d]=(v[d]-(lo[0]+hi[0])*.5)*scale+origin[0];v[d+1]=(v[d+1]-(lo[1]+hi[1])*.5)*scale+origin[3];v[d+2]=(v[d+2]-lo[2])*scale+origin[6];const occupied=cell(v[d],v[d+1],v[d+2]);initial.geometry[occupied*4+3]=Math.max(initial.geometry[occupied*4+3],collisionClass);
    });
      if(item.id?.includes('wood'))for(let d=0;d<vertices.length;d+=24)vertices[d+15]=2;
      if(item.id?.includes('foliage')){
        // Each accepted leaf carries its vertex count. Its first pair
        // straddles the petiole attachment and its last vertex is the apex.
        const leafCount=item.leaf_vertex_count??12;
        if(vertices.length%(leafCount*24))throw new Error('Invalid procedural leaf topology');
        for(let base=0;base<vertices.length;base+=leafCount*24){
          const anchor=[0,1,2].map(a=>(vertices[base+a]+vertices[base+24+a])*.5);
          const tip=Array.from(vertices.subarray(base+(leafCount-1)*24,base+(leafCount-1)*24+3));
          const axis=tip.map((v,a)=>v-anchor[a]),length=Math.hypot(...axis),normal=Array.from(vertices.subarray(base+3,base+6));
          const hinge=[axis[1]*normal[2]-axis[2]*normal[1],axis[2]*normal[0]-axis[0]*normal[2],axis[0]*normal[1]-axis[1]*normal[0]],norm=Math.hypot(...hinge);
          const c=cell(...anchor),o=(initial.nodeCount*2+c)*4;
          let halfWidth=0;const wide=hinge.map(v=>v/Math.max(norm,1e-8));
          for(let j=0;j<leafCount;j++){const d=base+j*24;halfWidth=Math.max(halfWidth,Math.abs(wide.reduce((sum,v,a)=>sum+v*(vertices[d+a]-anchor[a]),0)));}
          const lever=length*.45,area=length*halfWidth*1.1,flowArea=span[1]/grid[1]*span[2]/grid[2];
          // Angular leaf modes use actual blade scale. Coarse air parcels must
          // only transfer the fraction of a cell's flow intercepted by a leaf.
          initial.geometry[o]+=(world.solid_properties.leaf_mass??0.0003)*length*length/3;
          initial.geometry[o+1]+=(world.solid_properties.leaf_spring_constant??0.3)*length*length;
          initial.geometry[o+2]+=(world.solid_properties.leaf_damping??0.006)*length*length;
          initial.geometry[o+3]+=area*lever/flowArea;
          const attached=(initial.nodeCount*3+c)*4,leafMass=world.solid_properties.leaf_mass??0.0003;
          initial.geometry[attached]+=leafMass;initial.geometry[attached+1]+=leafMass*lever;initial.geometry[attached+2]+=area;
          // Leaf normals are two-sided. Align their equivalent orientation to
          // the prevailing flow before area aggregation so opposing triangle
          // windings cannot cancel the aerodynamic surface normal.
          const facing=normal[0]<0?normal.map(value=>-value):normal,normalOffset=(initial.nodeCount*4+c)*4;
          for(let axis=0;axis<3;axis++)initial.geometry[normalOffset+axis]+=facing[axis]*area;
          initial.geometry[normalOffset+3]+=area;
          // The View's vein field owns colour variation, not triangle vertices.
          const bladeColor=Array.from(vertices.subarray(base+5*24+6,base+5*24+10));
          for(let j=0;j<leafCount;j++){const d=base+j*24;vertices.set(anchor,d+16);vertices[d+19]=1;vertices[d+15]=1;vertices.set(bladeColor,d+6);vertices.set(normal,d+3);vertices.set(item.uvs?.subarray((d/24)*2,(d/24)*2+2)??[.5,0],d+12);vertices.set(hinge.map(v=>v/Math.max(norm,1e-8)),d+20);vertices[d+23]=length;vertices[d+11]=Math.pow(Math.max(0,anchor[2])/8,2)*0.65;
            const hit=cell(vertices[d],vertices[d+1],vertices[d+2]),distance=anchor.reduce((sum,v,a)=>sum+(v-initial.geometry[hit*4+a])**2,0);
            if(distance<routeDistance[hit]){routeDistance[hit]=distance;initial.geometry[(initial.nodeCount*3+hit)*4+3]=c;}
          }
        }
      }
      // The blade asset stores both windings. With two-sided rendering, one
      // triangle already covers both sides; remove exact reversed duplicates.
      const indices=item.id?.includes('foliage')?new Uint32Array(Array.from(item.indices).filter((_,i)=>i%6<3)):item.indices;
      meshes.push({vertices,indices});}
    // Material modes follow occupied physical state. Empty air cells no longer
    // form a hidden elastic solid, and trunk cells at lawn height retain wood
    // coefficients instead of accidentally receiving grass stiffness.
    const wood=beam??{mass:world.nodes_properties.mass??1,stiffness:world.nodes_properties.spring_constant??60,damping:world.nodes_properties.damping??12};
    const grass={mass:world.grass_properties.mass??0.08,stiffness:world.grass_properties.spring_constant??40,damping:world.grass_properties.damping??0.8};
    for(let i=0;i<initial.nodeCount;i++){
      const z=Math.floor(i/(grid[0]*grid[1])),collisionClass=initial.geometry[i*4+3],attachedMass=initial.geometry[(initial.nodeCount*3+i)*4];
      const woodMode=collisionClass===1||collisionClass===3||attachedMass>0,grassMode=!woodMode&&z<=1;
      const coefficients=woodMode?wood:grassMode?grass:null,mode=woodMode?1:grassMode?2:0;
      initial.geometry.set(coefficients?[coefficients.mass,coefficients.stiffness,coefficients.damping,mode]:[1,0,0,0],(initial.nodeCount+i)*4);
      const normalOffset=(initial.nodeCount*4+i)*4,totalArea=initial.geometry[normalOffset+3],normalLength=Math.hypot(...initial.geometry.subarray(normalOffset,normalOffset+3));
      if(totalArea>0&&normalLength>1e-8)for(let axis=0;axis<3;axis++)initial.geometry[normalOffset+axis]/=normalLength;
    }
    p.grass_count=world.grass_properties.count??81920;p.density??=1.225;p.turbulence_intensity??=.12;p.integral_scale??=2.4;p.drag_coefficient??=1.05;p.eddy_decay??=.75;p.parcel_mass=p.mass??p.density*span.reduce((v,n)=>v*n,1)/initial.parcelCount;p.node_mass=beam?.mass??world.nodes_properties.mass??1;p.spring_constant=beam?.stiffness??(world.nodes_properties.spring_constant??60)/(world.nodes_properties.elasticity??1);p.damping=beam?.damping??world.nodes_properties.damping??12;p.beam=beam;
    if(!(p.parcel_mass>0&&p.node_mass>0&&p.spring_constant>0&&p.damping>=0&&p.density>0&&p.turbulence_intensity>=0&&p.integral_scale>0&&p.drag_coefficient>=0&&p.eddy_decay>=0))throw new Error('Invalid wind/elastic World properties');
  }
  const radius=kind==='wind'?7:3.5,color=kind==='wind'?[.065,.14,.022,1]:[.12,.125,.12,1];
  meshes.push({shadow:false,vertices:new Float32Array([[-radius,-radius],[radius,-radius],[radius,radius],[-radius,radius]].flatMap(([x,y])=>[x,y,-.012,0,0,1,...color,-1,0,...Array(12).fill(0)])),indices:new Uint32Array([0,1,2,0,2,3])});
  p.lights=[];
  for(const layer of arenas.filter(a=>a.layer.world_id===world.world_id&&a.layer.properties?.shape==='sphere'&&a.layer.properties?.emissivity>0)){
    for(let i=0;i<layer.layer.count;i++){const center=[layer.state[i*9],layer.state[i*9+3],layer.state[i*9+6]],mesh=emissiveSphere(center,layer.layer.properties);meshes.push(mesh);p.lights.push(mesh.light);}
  }
  return {initial,meshes};
}

export async function bootMaterialWorlds(compiled){
  const started=performance.now(),program=compiled.worldProgram(),world=program.gpu_worlds.find(w=>w.world_id===program.views[program.active_view].world_id),settings=program.views[program.active_view].controls??{};
  if(!world||!navigator.gpu)throw new Error('This World requires WebGPU');
  document.head.insertAdjacentHTML('beforeend','<style>html,body{margin:0;height:100%;overflow:hidden;background:#0c1417;color:#d7e5df;font:15px system-ui}body{display:flex;flex-direction:column}#world-controls{display:flex;flex-wrap:wrap;align-items:center;gap:9px;padding:10px;background:#172a25}button,select{font:inherit;border:1px solid #73897c;border-radius:8px;padding:8px 12px;background:#20352b;color:inherit}button[aria-pressed=true]{background:#dcebd6;color:#16271d}#world-status{font-size:12px;padding:6px 10px}canvas{width:100%;min-height:0;flex:1;touch-action:none;overscroll-behavior:none;user-select:none}#world-error{white-space:pre-wrap;color:#ffb4ab}</style>');
  document.body.replaceChildren();const controls=document.createElement('div');controls.id='world-controls';controls.setAttribute('aria-label',settings.title??'World controls');
  const title=document.createElement('strong');title.textContent=settings.title??(world.kind==='rigid'?'Five 3D stones':'8 m tree, lawn and wind');controls.append(title);
  const status=document.createElement('div');status.id='world-status';status.setAttribute('role','status');status.textContent='Loading added geometry…';
  const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',world.kind==='rigid'?'Touch a stone, lift it vertically and release it':'Swipe to orbit the tree and lawn');
  const error=document.createElement('pre');error.id='world-error';error.hidden=true;error.setAttribute('role','alert');document.body.append(controls,status,canvas,error);
  let paused=false,particles=false,grass=true,active=true,pending=false,previous=null,accumulator=0,drag=null,stopped=false,bodyRead=null,snapshot=null,frames=0;
  const pointers=new Map();let pinchDistance=0;
  const source=world.kind==='rigid'?world.properties:world.solid_properties;
  const requested=new URLSearchParams(location.search).get('generation')??'original';
  const assetUrl=requested==='original'?source.asset:source.variants?.[requested];if(!assetUrl)throw new Error('Unknown procedural geometry distribution');
  const [asset,adapter]=await Promise.all([readMechanicalAsset(assetUrl),navigator.gpu.requestAdapter({powerPreference:'high-performance'})]);if(!adapter)throw new Error('No WebGPU adapter');const device=await adapter.requestDevice();
  const fail=e=>{stopped=true;error.hidden=false;error.textContent=String(e.stack??e.message??e);console.error(error.textContent);};device.addEventListener('uncapturederror',e=>fail(e.error));device.lost.then(info=>{if(!stopped)fail(new Error(info.message));});
  const {initial,meshes}=prepareMechanicalInitialState(world,compiled.worldLayerViews(),asset);
  const prefix=`$world$gpu$${world.world_id}`;const physics=await createMechanicalWorldGpu(device,world,initial,compiled.readBinding(`${prefix}$physics`));
  const embedding=await createWorldSceneEmbeddingGpu(device,canvas,world,physics,meshes,compiled.readBinding(`${prefix}$embedding`));
  const camera=world.kind==='rigid'?{pos:[3.4,-5.6,3.2],target:[0,0,.95],fov:42}:{pos:[8,-17,7],target:[0,0,3.7],fov:42};
  const report=()=>{status.textContent=world.kind==='rigid'?`5 stones · rigid contacts + friction · ${drag?'lifted':'touch, lift and drop'} · ${physics.time.toFixed(2)} s`:`8 m tree · ${world.properties.grass_count.toLocaleString()} grass blades · ${initial.parcelCount.toLocaleString()} coherent air parcels · ${particles?'contact parcels, α ≤ 0.5':'particles hidden'} · ${physics.time.toFixed(2)} s`;};
  const button=(label,pressed,handler)=>{const b=document.createElement('button');b.textContent=label;b.type='button';if(pressed!==null)b.setAttribute('aria-pressed',String(pressed));b.addEventListener('click',()=>handler(b));controls.append(b);return b;};
  if(settings.pause)button('Pause',false,b=>{paused=!paused;b.textContent=paused?'Play':'Pause';b.setAttribute('aria-pressed',String(paused));previous=null;report();});
  if(settings.reset)button('Reset',null,()=>{physics.reset();drag=null;snapshot=initial.bodies.slice();previous=null;accumulator=0;report();});
  if(new URLSearchParams(location.search).has('verify')){const output=document.createElement('pre');output.id='world-inspection';output.setAttribute('role','status');output.style.cssText='margin:0;font:11px monospace;white-space:pre-wrap';document.body.append(output);button('Inspect GPU state',null,async()=>{output.textContent=JSON.stringify(await physics.inspect());});
    if(world.kind==='wind')button('Verify pinch gesture',null,()=>{const rect=canvas.getBoundingClientRect(),x=rect.left+rect.width/2,y=rect.top+rect.height/2,before=camera.pos.slice(),orbit=Number(canvas.dataset.orbitRevision??0);
      const dispatch=(type,id,dx)=>canvas.dispatchEvent(new PointerEvent(type,{pointerId:id,pointerType:'touch',button:0,clientX:x+dx,clientY:y,bubbles:true,cancelable:true}));
      dispatch('pointerdown',901,-40);dispatch('pointerdown',902,40);dispatch('pointermove',902,100);dispatch('pointerup',902,100);dispatch('pointerup',901,-40);
      output.textContent=JSON.stringify({pinchZoom:Math.hypot(...sub(camera.pos,camera.target))<Math.hypot(...sub(before,camera.target)),orbitUnchanged:Number(canvas.dataset.orbitRevision??0)===orbit,position:camera.pos});
    });}
  if(settings.particles)button('Particles',false,b=>{particles=!particles;b.setAttribute('aria-pressed',String(particles));report();});
  if(settings.grass)button('Grass',true,b=>{grass=!grass;b.setAttribute('aria-pressed',String(grass));});
  if(settings.wind){const label=document.createElement('label');label.textContent='Wind ';const input=document.createElement('input');input.type='range';input.min='1';input.max='8';input.step='.1';input.value=String(world.properties.speed??3.5);input.setAttribute('aria-label','Wind speed');const output=document.createElement('output');output.value=`${input.value} m/s`;input.addEventListener('input',()=>{physics.setSpeed(Number(input.value));output.value=`${Number(input.value).toFixed(1)} m/s`;});label.append(input,output);controls.append(label);}
  if(source.variants){const select=document.createElement('select');select.setAttribute('aria-label','Branch and leaf distribution');for(const name of ['original',...Object.keys(source.variants)]){const option=document.createElement('option');option.value=name;option.textContent=name;select.append(option);}select.value=requested;select.addEventListener('change',()=>{const url=new URL(location.href);url.searchParams.set('generation',select.value);location.replace(url);});controls.append(select);}
  const read=()=>{if(bodyRead||world.kind!=='rigid')return;bodyRead=physics.readBodies().then(data=>{snapshot=data;},fail).finally(()=>{bodyRead=null;});};snapshot=initial.bodies.slice();
  canvas.addEventListener('pointerdown',event=>{event.preventDefault();if(event.button!==0)return;
    if(world.kind==='wind'){pointers.set(event.pointerId,[event.clientX,event.clientY]);if(pointers.size===2){const [a,b]=[...pointers.values()];pinchDistance=Math.hypot(a[0]-b[0],a[1]-b[1]);drag=null;if(event.isTrusted)canvas.setPointerCapture(event.pointerId);return;}}
    if(world.kind==='rigid'){
      const rect=canvas.getBoundingClientRect();const choices=Array.from({length:initial.bodyCount},(_,i)=>({i,p:projectPoint(Array.from(snapshot.subarray(i*20,i*20+3)),camera,rect)})).filter(({p})=>p.depth>0).sort((a,b)=>Math.hypot(a.p.x-event.clientX,a.p.y-event.clientY)-Math.hypot(b.p.x-event.clientX,b.p.y-event.clientY));
      const choice=choices[0];if(!choice||Math.hypot(choice.p.x-event.clientX,choice.p.y-event.clientY)>Math.max(30,snapshot[choice.i*20+18]/choice.p.depth*rect.height*1.2))return;
      drag={id:event.pointerId,body:choice.i,y:event.clientY,z:snapshot[choice.i*20+2],depth:choice.p.depth};physics.setHeld(choice.i,drag.z);
    }else if(settings.orbit){drag={id:event.pointerId,x:event.clientX,y:event.clientY};}else return;
    if(event.isTrusted)canvas.setPointerCapture(event.pointerId);report();
  });
  canvas.addEventListener('pointermove',event=>{event.preventDefault();
    if(world.kind==='wind'&&pointers.has(event.pointerId)){pointers.set(event.pointerId,[event.clientX,event.clientY]);if(pointers.size>=2){const [a,b]=[...pointers.values()],distance=Math.hypot(a[0]-b[0],a[1]-b[1]);if(distance>0&&pinchDistance>0)zoomCamera(camera,pinchDistance/distance);pinchDistance=distance;canvas.dataset.zoomRevision=String(Number(canvas.dataset.zoomRevision??0)+1);return;}}
    if(!drag||drag.id!==event.pointerId)return;
    if(world.kind==='rigid'){const metres=2*drag.depth*Math.tan(camera.fov*Math.PI/360)/canvas.getBoundingClientRect().height;physics.setHeld(drag.body,Math.max(drag.z,Math.min(4,drag.z+(drag.y-event.clientY)*metres)));}
    else{const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.x=event.clientX;drag.y=event.clientY;const v=sub(camera.pos,camera.target),r=Math.hypot(...v),yaw=Math.atan2(v[1],v[0])-dx*.008,pitch=Math.max(-.1,Math.min(1.25,Math.asin(v[2]/r)+dy*.006));camera.pos=[camera.target[0]+Math.cos(yaw)*Math.cos(pitch)*r,camera.target[1]+Math.sin(yaw)*Math.cos(pitch)*r,camera.target[2]+Math.sin(pitch)*r];canvas.dataset.orbitRevision=String(Number(canvas.dataset.orbitRevision??0)+1);}
  });
  const release=event=>{event.preventDefault();pointers.delete(event.pointerId);pinchDistance=0;if(world.kind==='wind'){const remaining=[...pointers.entries()][0];drag=remaining?{id:remaining[0],x:remaining[1][0],y:remaining[1][1]}:null;}else if(drag?.id===event.pointerId){physics.setHeld(-1,0);drag=null;}if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);report();};canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('lostpointercapture',event=>{pointers.delete(event.pointerId);if(drag?.id===event.pointerId){physics.setHeld(-1,0);drag=null;}});
  canvas.addEventListener('wheel',event=>{event.preventDefault();if(world.kind==='wind')zoomCamera(camera,Math.exp(event.deltaY*.001));},{passive:false});
  for(const type of ['touchstart','touchmove','touchend','touchcancel'])canvas.addEventListener(type,event=>event.preventDefault(),{passive:false});
  window.addEventListener('message',event=>{if(event.origin===location.origin&&event.source===window.parent&&event.data?.type==='vf-preview-visibility'){active=event.data.active;previous=null;}});
  if(window.parent!==window)window.parent.postMessage({type:'vf-preview-ready'},location.origin);
  function frame(timestamp){if(stopped)return;requestAnimationFrame(frame);if(document.hidden||!active){previous=null;return;}if(pending)return;
    // Rigid drops keep elapsed-time debt instead of silently slowing gravity
    // below 30 fps. The bounded submission budget does not change the law's dt.
    const elapsed=previous==null?0:(timestamp-previous)/1000;previous=timestamp;
    try{const encoder=device.createCommandEncoder();if(!paused){const schedule=mechanicalFrameSchedule(world,accumulator,elapsed);accumulator=schedule.debt;for(let step=0;step<schedule.steps;step++)physics.step(encoder);}
      physics.placeHeld(encoder);embedding.render(encoder,camera,{grass,particles});device.queue.submit([encoder.finish()]);pending=true;device.queue.onSubmittedWorkDone().then(()=>{pending=false;},fail);canvas.dataset.renderedFrames=String(++frames);if(frames%12===0){report();read();}
    }catch(e){fail(e);}
  }
  report();canvas.dataset.readyMs=String(Math.round(performance.now()-started));document.body.dataset.vfWorldLayerReady='true';requestAnimationFrame(frame);
  const application={compiled,program,physics,embedding,canvas,destroy(){stopped=true;physics.destroy();embedding.destroy();device.destroy();}};globalThis.__vfWorldLayerApplication=application;return application;
}
