// Asset decoding transfers offline-generated geometry into World and View.
// It never derives collision topology from render geometry at frame time.
const decoder=new TextDecoder();
const encoder=new TextEncoder();
const raw=value=>new Uint8Array(value.buffer,value.byteOffset,value.byteLength);

// Applied once when an authored asset enters World. Render and contact vertices
// share this transform; the frame loop only consumes their GPU-resident state.
export function mapPhysicalSurfaceToWorld(contact,{scale,offset}){
  if(!contact||!(contact.positions instanceof Float32Array)||
    !(contact.surfaceThickness instanceof Float32Array)||
    !Number.isFinite(scale)||scale<=0||!Array.isArray(offset)||offset.length!==3||
    offset.some(value=>!Number.isFinite(value)))
    throw new RangeError('Invalid physical surface World transform');
  const positions=new Float32Array(contact.positions);
  for(let i=0;i<positions.length;i+=4)for(let axis=0;axis<3;axis++)
    positions[i+axis]=positions[i+axis]*scale+offset[axis];
  const surfaceThickness=Float32Array.from(contact.surfaceThickness,value=>value*scale);
  let bendShapes=contact.bendShapes;
  if(bendShapes){
    bendShapes=new Float32Array(bendShapes);
    for(let shape=0;shape<bendShapes.length;shape+=16){
      for(let axis=0;axis<3;axis++)bendShapes[shape+axis]=bendShapes[shape+axis]*scale+offset[axis];
      bendShapes[shape+12]*=scale;
    }
  }
  return {...contact,positions,surfaceThickness,...(bendShapes?{bendShapes}:{})};
}

export function parseMechanicalAssetBytes(bytes){
  if(!(bytes instanceof ArrayBuffer)||bytes.byteLength<20)
    throw new RangeError('Invalid mechanical asset buffer');
  const view=new DataView(bytes),length=bytes.byteLength;
  const magic=decoder.decode(new Uint8Array(bytes,0,8));
  if(magic!=='VFTREE02'&&magic!=='VFWORLD3'&&magic!=='VFWORLD4')throw new Error('Invalid added geometry asset');
  const meshCount=view.getUint32(8,true),declaredVertices=view.getUint32(12,true);
  const declaredIndices=view.getUint32(16,true);
  let offset=20,vertexCount=0,indexCount=0;
  const meshes=[];
  for(let mesh=0;mesh<meshCount;mesh++){
    if(offset+20>length)throw new RangeError('Truncated mechanical mesh header');
    const sizes=Array.from({length:5},(_,i)=>view.getUint32(offset+i*4,true));offset+=20;
    const padded=(sizes[0]+3)&~3;
    const payload=padded+4*(sizes[1]+sizes[2]+sizes[3]+sizes[4]);
    if(!Number.isSafeInteger(payload)||offset+payload>length)
      throw new RangeError('Truncated mechanical mesh payload');
    const meta=JSON.parse(decoder.decode(new Uint8Array(bytes,offset,sizes[0])));
    if(!meta||typeof meta!=='object'||Array.isArray(meta))throw new RangeError('Invalid mechanical mesh metadata');
    offset+=padded;
    const vertices=new Float32Array(bytes,offset,sizes[1]);offset+=sizes[1]*4;
    const indices=new Uint32Array(bytes,offset,sizes[2]);offset+=sizes[2]*4;
    const uvs=new Float32Array(bytes,offset,sizes[3]);offset+=sizes[3]*4;
    const roughness=new Float32Array(bytes,offset,sizes[4]);offset+=sizes[4]*4;
    if(vertices.length%10||indices.length%3||indices.some(index=>index>=vertices.length/10)||
      vertices.some(value=>!Number.isFinite(value)))
      throw new RangeError('Invalid mechanical mesh vertices or indices');
    vertexCount+=vertices.length/10;indexCount+=indices.length;
    meshes.push({...meta,vertices,indices,uvs,roughness});
  }
  if(vertexCount!==declaredVertices||indexCount!==declaredIndices)
    throw new RangeError('Mechanical asset length or count mismatch');
  if(magic==='VFTREE02'){
    if(offset!==length)throw new RangeError('Mechanical asset length or count mismatch');
    return {meshes,contact:null};
  }
  if(offset+12>length)throw new RangeError('Truncated physical contact header');
  const physicalCount=view.getUint32(offset,true),triangleCount=view.getUint32(offset+4,true);
  const islandCount=view.getUint32(offset+8,true);offset+=12;
  if(physicalCount<vertexCount||triangleCount<1||islandCount<1)
    throw new RangeError('Invalid physical contact dimensions');
  const take=(count,kind)=>{
    const size=count*4;
    if(!Number.isSafeInteger(size)||offset+size>length)
      throw new RangeError('Truncated physical contact packet');
    const result=kind==='float'?new Float32Array(bytes,offset,count):new Uint32Array(bytes,offset,count);
    offset+=size;return result;
  };
  const positions=take(physicalCount*4,'float'),compliance=take(physicalCount,'float');
  const islandIds=take(physicalCount,'uint'),surfaceTopology=take(triangleCount*4,'uint');
  const surfaceThickness=take(triangleCount,'float');
  let bendShapeIds=null,bendShapes=null;
  if(magic==='VFWORLD4'){
    if(offset+4>length)throw new RangeError('Truncated physical motion header');
    const shapeCount=view.getUint32(offset,true);offset+=4;
    if(shapeCount<1)throw new RangeError('Invalid physical motion dimensions');
    bendShapeIds=take(physicalCount,'uint');bendShapes=take(shapeCount*16,'float');
    if(bendShapeIds.some(id=>id>shapeCount)||
      bendShapes.some(value=>!Number.isFinite(value)))
      throw new RangeError('Invalid physical motion state');
    for(let shape=0;shape<shapeCount;shape++){
      const at=shape*16;
      if(!(bendShapes[at+12]>0)||!Number.isSafeInteger(bendShapes[at+13])||
        bendShapes[at+13]<-1||bendShapes[at+14]!==1)
        throw new RangeError('Invalid physical motion shape');
    }
  }
  if(offset!==length||positions.some((value,i)=>!Number.isFinite(value)||i%4===3&&value<0)||
    compliance.some(value=>!Number.isFinite(value)||value<0)||
    islandIds.some(id=>id>=islandCount)||
    surfaceThickness.some(value=>!Number.isFinite(value)||value<0))
    throw new RangeError('Invalid physical contact state');
  let visual=0;
  for(const mesh of meshes)for(let i=0;i<mesh.vertices.length;i+=10){
    for(let axis=0;axis<3;axis++)if(Math.abs(positions[visual*4+axis]-mesh.vertices[i+axis])>1e-5)
      throw new RangeError('Physical positions disagree with visual vertex order');
    visual++;
  }
  const covered=new Set();
  for(let face=0;face<triangleCount;face++){
    const at=face*4,a=surfaceTopology[at],b=surfaceTopology[at+1],c=surfaceTopology[at+2];
    const patch=surfaceTopology[at+3];
    if(a>=physicalCount||b>=physicalCount||c>=physicalCount||a===b||a===c||b===c||
      patch<1||patch>islandCount||islandIds[a]+1!==patch||islandIds[b]+1!==patch||islandIds[c]+1!==patch)
      throw new RangeError('Invalid physical contact topology');
    covered.add(patch-1);
  }
  if(covered.size!==islandCount)throw new RangeError('Physical patch lacks a contact proxy');
  return {meshes,contact:{positions,compliance,islandIds,islandCount,surfaceTopology,surfaceThickness,
    ...(bendShapeIds?{bendShapeIds,bendShapes}:{})}};
}

export function encodeMechanicalAssetV3(meshes,contact){return encodeMechanicalAsset(meshes,contact,false);}
export function encodeMechanicalAssetV4(meshes,contact){return encodeMechanicalAsset(meshes,contact,true);}

function encodeMechanicalAsset(meshes,contact,withBend){
  if(!Array.isArray(meshes)||!meshes.length||!contact||
    !(contact.positions instanceof Float32Array)||!(contact.compliance instanceof Float32Array)||
    !(contact.islandIds instanceof Uint32Array)||!(contact.surfaceTopology instanceof Uint32Array)||
    !(contact.surfaceThickness instanceof Float32Array)||
    !Number.isInteger(contact.islandCount)||contact.islandCount<1)
    throw new RangeError('Offline mechanical asset requires a physical contact packet');
  const records=meshes.map(mesh=>{
    if(!(mesh.vertices instanceof Float32Array)||!(mesh.indices instanceof Uint32Array))
      throw new RangeError('Offline mechanical mesh requires vertices and indices');
    const uvs=mesh.uvs??new Float32Array(),roughness=mesh.roughness??new Float32Array();
    if(!(uvs instanceof Float32Array)||!(roughness instanceof Float32Array))
      throw new RangeError('Offline mechanical mesh channels must be Float32');
    const meta={};for(const [key,value] of Object.entries(mesh))
      if(!['vertices','indices','uvs','roughness'].includes(key)&&!ArrayBuffer.isView(value))meta[key]=value;
    const description=encoder.encode(JSON.stringify(meta)),padded=(description.length+3)&~3;
    return {mesh,uvs,roughness,description,padded,
      size:20+padded+mesh.vertices.byteLength+mesh.indices.byteLength+uvs.byteLength+roughness.byteLength};
  });
  const vertices=records.reduce((sum,record)=>sum+record.mesh.vertices.length/10,0);
  const indices=records.reduce((sum,record)=>sum+record.mesh.indices.length,0);
  const physicalCount=contact.positions.length/4,triangleCount=contact.surfaceTopology.length/4;
  if(withBend){
    if(!(contact.bendShapeIds instanceof Uint32Array)||
      !(contact.bendShapes instanceof Float32Array)||
      contact.bendShapeIds.length!==physicalCount||contact.bendShapes.length%16)
      throw new RangeError('Offline physical motion packet requires sparse shapes');
  }else if(contact.bendShapeIds||contact.bendShapes)
    throw new RangeError('V3 asset cannot discard physical motion state');
  const total=20+records.reduce((sum,record)=>sum+record.size,0)+12+
    contact.positions.byteLength+contact.compliance.byteLength+contact.islandIds.byteLength+
    contact.surfaceTopology.byteLength+contact.surfaceThickness.byteLength+
    (withBend?4+contact.bendShapeIds.byteLength+contact.bendShapes.byteLength:0);
  if(!Number.isSafeInteger(total)||total>0xffffffff||!Number.isInteger(vertices)||
    !Number.isInteger(physicalCount)||!Number.isInteger(triangleCount))
    throw new RangeError('Offline mechanical asset exceeds its binary format');
  const output=new Uint8Array(total),view=new DataView(output.buffer);
  output.set(encoder.encode(withBend?'VFWORLD4':'VFWORLD3'));
  const u32=(offset,value)=>view.setUint32(offset,value,true);
  u32(8,records.length);u32(12,vertices);u32(16,indices);
  let offset=20;
  for(const {mesh,uvs,roughness,description,padded} of records){
    for(const size of [description.length,mesh.vertices.length,mesh.indices.length,uvs.length,roughness.length]){
      u32(offset,size);offset+=4;
    }
    output.set(description,offset);offset+=padded;
    for(const part of [mesh.vertices,mesh.indices,uvs,roughness]){
      output.set(raw(part),offset);offset+=part.byteLength;
    }
  }
  for(const size of [physicalCount,triangleCount,contact.islandCount]){u32(offset,size);offset+=4;}
  for(const part of [contact.positions,contact.compliance,contact.islandIds,
    contact.surfaceTopology,contact.surfaceThickness]){
    output.set(raw(part),offset);offset+=part.byteLength;
  }
  if(withBend){
    u32(offset,contact.bendShapes.length/16);offset+=4;
    for(const part of [contact.bendShapeIds,contact.bendShapes]){
      output.set(raw(part),offset);offset+=part.byteLength;
    }
  }
  parseMechanicalAssetBytes(output.buffer);
  return output.buffer;
}
