import { mkdir, writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { createForestPopulationReference, realizeForestPatchesReference } from '../public/previews/0.6.0/live/tree/runtime/vf-forest-population.mjs';
import { createTreeGeometryPlannerReference, planTreeGeometryReference } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-geometry-plan.mjs';
import { createTreeMaterialFieldReference, realizeTreeMaterialsReference } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-material-field.mjs';
import { adaptTreeWorkingSetsToRetainedPacketsReference } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-renderer-packets.mjs';
import { adaptTreeRenderPacketToWebGpuMeshesReference } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-webgpu-packets.mjs';
import { treeProGenPresets, treeProGenAsset } from '../public/previews/0.6.0/live/tree/runtime/vf-tree-pro-gen-presets.mjs';

const variant = process.argv[2] ?? 'original';
const asset = treeProGenAsset(variant);
const preset = treeProGenPresets[variant];

const identity={generator:'vkf.conditioned',version:1,seed:[0x1f83d9ab,269],domain:'material',hierarchy:['world:boreal','tree:webgpu-demo'],lod:0,channel:'population'};
const forest=realizeForestPatchesReference(createForestPopulationReference(identity),{patches:[[0,0]],treeBudget:1});
const geometry=planTreeGeometryReference(createTreeGeometryPlannerReference(identity,{splitDepth:7,lateralShoots:true,trunkShoots:false,foliageDensity:.42,scaffoldBranches:2,branching:preset.branching}),forest,{treeIndices:[0],detailLevels:[2],primitiveBudget:2400});
const materials=realizeTreeMaterialsReference(createTreeMaterialFieldReference(identity),forest,geometry,{materialBudget:2400});
const retained=adaptTreeWorkingSetsToRetainedPacketsReference(geometry,materials);
const tree=adaptTreeRenderPacketToWebGpuMeshesReference(retained.packets[0],{vertexBudget:393216,indexBudget:2359296,leafContact:true,leafShape:preset.leafShape});
const encoder=new TextEncoder();
const chunks=[];
const pushU32=(value)=>{const bytes=new Uint8Array(4);new DataView(bytes.buffer).setUint32(0,value,true);chunks.push(bytes);};
chunks.push(encoder.encode('VFTREE02'));pushU32(tree.meshes.length);pushU32(tree.vertexCount);pushU32(tree.indexCount);
for(const mesh of tree.meshes){
  const meta={};for(const[key,value]of Object.entries(mesh)){if(key==='vertices'||key==='indices'||ArrayBuffer.isView(value))continue;meta[key]=value;}
  const uvs=ArrayBuffer.isView(mesh.uvs)?mesh.uvs:new Float32Array(),roughness=ArrayBuffer.isView(mesh.roughness)?mesh.roughness:new Float32Array();
  const bytes=encoder.encode(JSON.stringify(meta));pushU32(bytes.length);pushU32(mesh.vertices.length);pushU32(mesh.indices.length);pushU32(uvs.length);pushU32(roughness.length);chunks.push(bytes);const pad=(4-(bytes.length%4))%4;if(pad)chunks.push(new Uint8Array(pad));chunks.push(new Uint8Array(mesh.vertices.buffer,mesh.vertices.byteOffset,mesh.vertices.byteLength));chunks.push(new Uint8Array(mesh.indices.buffer,mesh.indices.byteOffset,mesh.indices.byteLength));chunks.push(new Uint8Array(uvs.buffer,uvs.byteOffset,uvs.byteLength));chunks.push(new Uint8Array(roughness.buffer,roughness.byteOffset,roughness.byteLength));
}
const total=chunks.reduce((sum,part)=>sum+part.byteLength,0),output=new Uint8Array(total);let offset=0;for(const part of chunks){output.set(part,offset);offset+=part.byteLength;}
const target=new URL('../public/previews/0.6.0/live/tree/assets/',import.meta.url);await mkdir(target,{recursive:true});await writeFile(new URL(asset,target),gzipSync(output,{level:9}));
console.log(JSON.stringify({meshes:tree.meshes.map((mesh)=>({id:mesh.id,keys:Object.keys(mesh),vertices:mesh.vertices.length/10,indices:mesh.indices.length})),rawBytes:output.byteLength,gzipBytes:gzipSync(output,{level:9}).byteLength}));
