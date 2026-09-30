import {createContactSurfaceFieldTargetGpu} from './vf-contact-surface-field-target-gpu.mjs';
import {createContactSurfaceWorldGpu} from './vf-contact-surface-world-gpu.mjs';

// A field is one possible target Law. Contact admission remains independent of
// the generated object's name and of the View consuming accepted positions.
export async function createFieldDrivenSurfaceWorldGpu(device,{contact,nodes,grid,domainMin,
  domainSpan,timeStep,broadphase,bendShapes=contact?.bendShapes??null,
  bendShapeIds=contact?.bendShapeIds??null,angularMode=1}){
  if(!contact||!(contact.positions instanceof Float32Array)||
    !(contact.compliance instanceof Float32Array)||
    !(contact.islandIds instanceof Uint32Array)||
    !(contact.surfaceTopology instanceof Uint32Array)||
    !(contact.surfaceThickness instanceof Float32Array)||
    !broadphase||!Number.isFinite(timeStep)||timeStep<=0)
    throw new RangeError('Field-driven World requires a physical surface packet');
  const vertexCount=contact.positions.length/4;
  if(!Number.isInteger(vertexCount)||contact.compliance.length!==vertexCount||
    contact.islandIds.length!==vertexCount||
    contact.surfaceThickness.length*4!==contact.surfaceTopology.length||
    bendShapes!==null&&(!(bendShapes instanceof Float32Array)||
      bendShapes.length%16!==0||bendShapeIds===null&&bendShapes.length!==vertexCount*16)||
    bendShapeIds!==null&&(!(bendShapeIds instanceof Uint32Array)||
      bendShapeIds.length!==vertexCount||bendShapes===null||bendShapes.length<16||
      bendShapes.length%16!==0||bendShapeIds.some(id=>id>bendShapes.length/16)))
    throw new RangeError('Field-driven World physical channels disagree');
  const storage=GPUBufferUsage.STORAGE,write=GPUBufferUsage.COPY_DST;
  const rest=device.createBuffer({size:contact.positions.byteLength,usage:storage|write});
  const compliance=device.createBuffer({size:contact.compliance.byteLength,usage:storage|write});
  const targets=device.createBuffer({size:contact.positions.byteLength,usage:storage});
  const bendBuffer=bendShapes===null?null:device.createBuffer({size:bendShapes.byteLength,usage:storage|write});
  const bendIdsBuffer=bendShapeIds===null?null:device.createBuffer({size:bendShapeIds.byteLength,usage:storage|write});
  device.queue.writeBuffer(rest,0,contact.positions);
  device.queue.writeBuffer(compliance,0,contact.compliance);
  if(bendBuffer)device.queue.writeBuffer(bendBuffer,0,bendShapes);
  if(bendIdsBuffer)device.queue.writeBuffer(bendIdsBuffer,0,bendShapeIds);
  let field,world;
  try{
    field=await createContactSurfaceFieldTargetGpu(device,{vertexCount,restPositions:rest,
      compliance,nodes,targets,grid,domainMin,domainSpan,bendShapes:bendBuffer,
      bendShapeIds:bendIdsBuffer,angularMode});
    world=await createContactSurfaceWorldGpu(device,{positions:contact.positions,
      velocities:new Float32Array(contact.positions.length),islandIds:contact.islandIds,
      islandCount:contact.islandCount,surfaceTopology:contact.surfaceTopology,
      surfaceThickness:contact.surfaceThickness,targetPositions:targets,dt:timeStep,
      pairCapacity:broadphase.pairCapacity,broadphaseCellSize:broadphase.cellSize,
      broadphaseCellSlots:broadphase.cellSlots,broadphaseNodeCapacity:broadphase.nodeCapacity??null,
      broadphaseBucketCount:broadphase.bucketCount});
  }catch(error){
    field?.destroy();world?.destroy();rest.destroy();compliance.destroy();targets.destroy();bendBuffer?.destroy();bendIdsBuffer?.destroy();
    throw error;
  }
  return {positionBuffer:world.positionBuffer,
    encode(encoder){field.encode(encoder);world.encode(encoder);},
    reset(){world.reset();},inspect(){return world.inspect();},inspectStatus(){return world.inspectStatus();},
    inspectCandidateKinds(){return world.inspectCandidateKinds();},
    destroy(){world.destroy();field.destroy();rest.destroy();compliance.destroy();targets.destroy();bendBuffer?.destroy();bendIdsBuffer?.destroy();}};
}
