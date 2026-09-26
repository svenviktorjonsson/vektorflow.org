import {generateTreeMeshes} from './vf-tree-live-generation.mjs';

self.onmessage=event=>{
  try{
    const result=generateTreeMeshes(event.data);
    const buffers=result.meshes.flatMap(mesh=>[
      mesh.vertices.buffer,mesh.indices.buffer,mesh.uvs.buffer,mesh.roughness.buffer,
    ]);
    self.postMessage({meshes:result.meshes,primitiveCount:result.primitiveCount,
      vertexCount:result.vertexCount,indexCount:result.indexCount},buffers);
  }catch(error){self.postMessage({error:String(error?.stack??error)});}
};
