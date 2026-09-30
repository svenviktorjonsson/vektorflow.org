// Generic physical field-to-surface target Law. It samples World displacement
// into GPU target positions; the contact World decides whether to accept them.
// A View must read accepted positions, not re-evaluate this deformation.
import {ELASTIC_BLADE_POSE_WGSL} from './vf-elastic-blade-pose-wgsl.mjs';
export const CONTACT_SURFACE_FIELD_TARGET_WGSL=/* wgsl */`
struct Params { grid:vec4<u32>,mode_pad:vec4<u32>,minimum:vec4<f32>,span:vec4<f32> };
struct Node { d:vec4<f32>,v:vec4<f32> };
struct BendShape { anchor:vec4<f32>,normal:vec4<f32>,hinge:vec4<f32>,params:vec4<f32> };
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> rest:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> compliance:array<f32>;
@group(0) @binding(3) var<storage,read> nodes:array<Node>;
@group(0) @binding(4) var<storage,read_write> targets:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> bend_shapes:array<BendShape>;
@group(0) @binding(6) var<storage,read> bend_shape_ids:array<u32>;
@compute @workgroup_size(64) fn field_target(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=params.grid.w){return;}
 let p=rest[i].xyz;
 let coordinate=clamp((p-params.minimum.xyz)/params.span.xyz*vec3<f32>(params.grid.xyz)-0.5,
  vec3<f32>(0.0),vec3<f32>(params.grid.xyz)-1.001);
 let low=vec3<u32>(floor(coordinate));let fraction=fract(coordinate);
 var displacement=vec3<f32>(0.0);
 for(var z=0u;z<2u;z++){for(var y=0u;y<2u;y++){for(var x=0u;x<2u;x++){
  let at=low+vec3<u32>(x,y,z);
  let w=select(1.0-fraction,fraction,vec3<bool>(x==1u,y==1u,z==1u));
  let index=params.mode_pad.x*params.grid.x*params.grid.y*params.grid.z+
   at.x+params.grid.x*(at.y+params.grid.y*at.z);
  displacement+=nodes[index].d.xyz*w.x*w.y*w.z;
 }}}
 var next_position=p+displacement*compliance[i];
 if(params.mode_pad.z!=0u){
  var shape_index=i;
  var has_shape=true;
  if(params.mode_pad.w!=0u){
   let id=bend_shape_ids[i];
   has_shape=id!=0u;
   shape_index=id-1u;
  }
  if(has_shape){
   let shape=bend_shapes[shape_index];
   if(shape.params.z>0.5){
    let cells=params.grid.x*params.grid.y*params.grid.z;
    var local_node=0u;
    if(shape.params.y<0.0){
     let coordinate=clamp(floor((shape.anchor.xyz-params.minimum.xyz)/params.span.xyz*
      vec3<f32>(params.grid.xyz)),vec3<f32>(0.0),vec3<f32>(params.grid.xyz)-1.0);
     let at=vec3<u32>(coordinate);
     local_node=at.x+params.grid.x*(at.y+params.grid.y*at.z);
    }else{local_node=u32(shape.params.y);}
    let node=params.mode_pad.y*cells+local_node;
    let bend=nodes[node].d.x;
    next_position=elastic_blade_pose(p,shape.normal.xyz,shape.anchor.xyz,shape.hinge.xyz,
     shape.params.x,bend,displacement*compliance[i]).p;
   }
  }
 }
 targets[i]=vec4<f32>(next_position,0.0);
}
`+ELASTIC_BLADE_POSE_WGSL;

export async function createContactSurfaceFieldTargetGpu(device,{vertexCount,restPositions,compliance,
 nodes,targets,grid,domainMin,domainSpan,mode=0,bendShapes=null,bendShapeIds=null,angularMode=1}){
 if(!Number.isSafeInteger(vertexCount)||vertexCount<1||!Array.isArray(grid)||grid.length!==3||
  !grid.every(value=>Number.isSafeInteger(value)&&value>=2)||!Number.isSafeInteger(mode)||mode<0||
  !Number.isSafeInteger(angularMode)||angularMode<0||bendShapeIds!==null&&bendShapes===null||
  !Array.isArray(domainMin)||domainMin.length!==3||!domainMin.every(Number.isFinite)||
  !Array.isArray(domainSpan)||domainSpan.length!==3||!domainSpan.every(value=>Number.isFinite(value)&&value>0))
  throw new RangeError('Invalid World surface field dimensions');
 const gridCount=grid[0]*grid[1]*grid[2];
 if(!Number.isSafeInteger(gridCount)||!Number.isSafeInteger((mode+1)*gridCount)||
  restPositions.size<vertexCount*16||compliance.size<vertexCount*4||
  nodes.size<Math.max(mode+1,bendShapes?(angularMode+1):0)*gridCount*32||
  targets.size<vertexCount*16||bendShapes&&bendShapes.size<(bendShapeIds?64:vertexCount*64)||
  bendShapeIds&&bendShapeIds.size<vertexCount*4)
  throw new RangeError('World surface field buffers are undersized');
 const limit=Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize);
 if(Math.max(vertexCount*16,bendShapes?.size??0,bendShapeIds?.size??0,(mode+1)*gridCount*32)>limit||
  Math.ceil(vertexCount/64)>device.limits.maxComputeWorkgroupsPerDimension)
  throw new RangeError('World surface field requires GPU batching');
 const uniform=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const emptyBend=bendShapes??device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE});
 const emptyIds=bendShapeIds??device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE});
 const bytes=new ArrayBuffer(64),u=new Uint32Array(bytes),f=new Float32Array(bytes);
 u.set([...grid,vertexCount,mode,angularMode,bendShapes?1:0,bendShapeIds?1:0],0);f.set([...domainMin,0],8);f.set([...domainSpan,0],12);
 device.queue.writeBuffer(uniform,0,bytes);
 const shader=device.createShaderModule({label:'VKF World surface field target',code:CONTACT_SURFACE_FIELD_TARGET_WGSL});
 const errors=(await shader.getCompilationInfo()).messages.filter(message=>message.type==='error');
 if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
  ...[1,2,3,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}})),
  {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
 const group=device.createBindGroup({layout,entries:[uniform,restPositions,compliance,nodes,targets,emptyBend,emptyIds].map((buffer,binding)=>
  ({binding,resource:{buffer}}))});
 const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),
  compute:{module:shader,entryPoint:'field_target'}});
 return {encode(encoder){const pass=encoder.beginComputePass({label:'VKF physical surface field target'});
  pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(vertexCount/64));pass.end();},
  destroy(){uniform.destroy();if(!bendShapes)emptyBend.destroy();if(!bendShapeIds)emptyIds.destroy();}};
}
