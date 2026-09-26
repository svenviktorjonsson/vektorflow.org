import {createForestPopulationReference,realizeForestPatchesReference} from './vf-forest-population.mjs';
import {createTreeGeometryPlannerReference,planTreeGeometryReference} from './vf-tree-geometry-plan.mjs';
import {createTreeMaterialFieldReference,realizeTreeMaterialsReference} from './vf-tree-material-field.mjs';
import {adaptTreeWorkingSetsToRetainedPacketsReference} from './vf-tree-renderer-packets.mjs';
import {adaptTreeRenderPacketToWebGpuMeshesReference} from './vf-tree-webgpu-packets.mjs';
import {treeProGenPresets} from './vf-tree-pro-gen-presets.mjs';
import {proGen} from './vf-pro-gen-distribution-reference.mjs';

export const TREE_SPECIES=Object.freeze({
  oak:Object.freeze({name:'English oak',index:0,foliage:[.18,.31,.09,1],bark:[.26,.19,.12],leafOutline:'oak',
    leafShape:Object.freeze({widthRatio:proGen.normal(.55,.035),roundness:proGen.normal(.58,.05)})}),
  birch:Object.freeze({name:'Silver birch',index:1,foliage:[.27,.46,.12,1],bark:[.70,.69,.61],leafOutline:'birch',
    leafShape:Object.freeze({widthRatio:proGen.normal(.37,.03),roundness:proGen.normal(.78,.04)})}),
  beech:Object.freeze({name:'European beech',index:2,foliage:[.20,.37,.11,1],bark:[.37,.37,.34],leafOutline:'beech',
    leafShape:Object.freeze({widthRatio:proGen.normal(.46,.025),roundness:proGen.normal(.53,.035)})}),
});

export function treeGenerationParameters(input={}){
  const finite=(name,fallback,lo,hi)=>{
    const value=Number(input[name]??fallback);
    if(!Number.isFinite(value)||value<lo||value>hi)throw new RangeError(`${name} must be in [${lo}, ${hi}]`);
    return value;
  };
  const species=input.species??'oak',distribution=input.distribution??'normal';
  if(!Object.hasOwn(TREE_SPECIES,species))throw new RangeError('Unknown tree species');
  if(!Object.hasOwn(treeProGenPresets,distribution))throw new RangeError('Unknown tree distribution');
  return Object.freeze({species,distribution,height:finite('height',8,3,8),
    splitFactor:finite('splitFactor',.65,0,1),turnFactor:finite('turnFactor',.5,0,1)});
}

export function generateTreeMeshes(input={}){
  const controls=treeGenerationParameters(input),species=TREE_SPECIES[controls.species];
  const identity={generator:'vkf.conditioned',version:1,seed:[0x1f83d9ab,269],domain:'material',
    hierarchy:['world:boreal','tree:webgpu-demo'],lod:0,channel:'population'};
  const base=realizeForestPatchesReference(createForestPopulationReference(identity),
    {patches:[[0,0]],treeBudget:1});
  const growth=base.growth.slice(),scale=controls.height/growth[1];
  growth[1]=controls.height;growth[2]*=scale;growth[3]*=scale;
  const speciesIndices=Uint32Array.of(species.index);
  const forest={...base,growth,speciesIndices,species:[Object.freeze({
    ...base.species[0],id:`tree:species:${species.index}`,index:species.index,
    foliageColor:species.foliage,barkColor:species.bark,
  })]};
  const preset=treeProGenPresets[controls.distribution];
  const planner=createTreeGeometryPlannerReference(identity,{
    splitDepth:6,lateralShoots:controls.splitFactor>0,trunkShoots:false,
    foliageDensity:.42,scaffoldBranches:Math.round(2*controls.splitFactor),
    branching:preset.branching,splitFactor:controls.splitFactor,turnFactor:controls.turnFactor,
  });
  const geometry=planTreeGeometryReference(planner,forest,
    {treeIndices:[0],detailLevels:[2],primitiveBudget:2400});
  const materials=realizeTreeMaterialsReference(createTreeMaterialFieldReference(identity),
    forest,geometry,{materialBudget:2400});
  const retained=adaptTreeWorkingSetsToRetainedPacketsReference(geometry,materials);
  const leafShape={...preset.leafShape,...species.leafShape};
  const tree=adaptTreeRenderPacketToWebGpuMeshesReference(retained.packets[0],
    {vertexBudget:180000,indexBudget:1080000,leafContact:true,leafShape,leafOutline:species.leafOutline,
      allowSparse:true,barkDetail:false});
  return {controls,meshes:tree.meshes.map(mesh=>({...mesh,use_vertex_bark:mesh.id.includes('wood')})),primitiveCount:geometry.primitiveCount,
    vertexCount:tree.vertexCount,indexCount:tree.indexCount};
}
