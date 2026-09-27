import {createTriangleSurfaceAdmissionReference} from './vf-stone-triangle-contact.mjs';

// Test-only readback: run the production View vertex map as a compute entry,
// then compare the resulting leaf blades against the deformed wood surface.
export async function auditDynamicTreeContact(device, world, physics, meshes, asset, sceneWgsl) {
  const [nx,ny,nz]=world.properties.grid,nodeCount=nx*ny*nz;
  const nodeRead=device.createBuffer({size:nodeCount*32,
    usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const nodeEncoder=device.createCommandEncoder();
  nodeEncoder.copyBufferToBuffer(physics.buffers[2],0,nodeRead,0,nodeCount*32);
  device.queue.submit([nodeEncoder.finish()]);await nodeRead.mapAsync(GPUMapMode.READ);
  const nodeState=new Float32Array(nodeRead.getMappedRange()).slice();
  nodeRead.unmap();nodeRead.destroy();
  const step=world.properties.domain_span.map((span,axis)=>span/world.properties.grid[axis]);
  const edge=[0,0,0];let maximumDisplacement=0;
  for(let i=0;i<nodeCount;i++){
    const x=i%nx,y=Math.floor(i/nx)%ny,z=Math.floor(i/(nx*ny)),at=i*8;
    maximumDisplacement=Math.max(maximumDisplacement,
      Math.hypot(nodeState[at],nodeState[at+1],nodeState[at+2]));
    for(const [axis,next,available] of [[0,i+1,x+1<nx],[1,i+nx,y+1<ny],[2,i+nx*ny,z+1<nz]]){
      if(!available)continue;
      const other=next*8;
      edge[axis]=Math.max(edge[axis],Math.hypot(
        nodeState[at]-nodeState[other],
        nodeState[at+1]-nodeState[other+1],
        nodeState[at+2]-nodeState[other+2])/step[axis]);
    }
  }
  const top=Math.max(0,world.properties.domain_min[2]+world.properties.domain_span[2]);
  const complianceMaximum=.65*(top/8)**2;
  const complianceGradientMaximum=.65*2*top/64;
  const deformationLipschitzUpperBound=
    complianceMaximum*Math.hypot(...edge)+maximumDisplacement*complianceGradientMaximum;
  const shader = `${sceneWgsl}
@group(1) @binding(0) var<storage,read> packed_vertices:array<vec4<f32>>;
@group(1) @binding(1) var<storage,read_write> posed_positions:array<vec4<f32>>;
@compute @workgroup_size(64) fn audit_pose(@builtin(global_invocation_id) id:vec3<u32>) {
  let vertex=id.x;
  if(vertex>=arrayLength(&posed_positions)){return;}
  let a=packed_vertices[vertex*6u];let b=packed_vertices[vertex*6u+1u];
  let c=packed_vertices[vertex*6u+2u];let d=packed_vertices[vertex*6u+3u];
  let e=packed_vertices[vertex*6u+4u];let f=packed_vertices[vertex*6u+5u];
  let v=Vertex(a.xyz,vec3<f32>(a.w,b.xy),vec4<f32>(b.zw,c.xy),c.z,c.w,
    d.xyz,e.xyz,e.w,f.xyz,f.w,d.w);
  posed_positions[vertex]=vec4<f32>(posed(v).p,1.0);
}`;
  const module = device.createShaderModule({code: shader});
  const pipeline = await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'audit_pose'}});
  const scene = new ArrayBuffer(256),f = new Float32Array(scene),u = new Uint32Array(scene);
  f[0]=f[5]=f[10]=f[15]=1;
  f[16]=f[21]=f[26]=f[31]=1;
  f[35]=physics.time;
  f.set([...world.properties.domain_min,0],40);
  f.set([...world.properties.domain_span,0],44);
  u.set([...world.properties.grid,0],48);
  f[52]=1;
  const uniform=device.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(uniform,0,scene);
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:uniform}},
    {binding:1,resource:{buffer:physics.buffers[0]}},
    {binding:3,resource:{buffer:physics.buffers[2]}},
  ]});
  const posed=[];
  for(const mesh of meshes){
    const count=mesh.vertices.length/24;
    const source=device.createBuffer({size:mesh.vertices.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const target=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:count*16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(source,0,mesh.vertices);
    const pair=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[
      {binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:target}},
    ]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.setBindGroup(1,pair);
    pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
    encoder.copyBufferToBuffer(target,0,read,0,count*16);
    device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const positions=new Float32Array(read.getMappedRange()).slice();
    read.unmap();read.destroy();target.destroy();source.destroy();
    const vertices=new Float32Array(count*10);
    for(let i=0;i<count;i++)vertices.set(positions.subarray(i*4,i*4+3),i*10);
    posed.push({...mesh,vertices});
  }
  uniform.destroy();
  const woodSurface=createTriangleSurfaceAdmissionReference(posed[0]);
  const foliage=posed[1],sourceFoliage=asset[1];
  const stride=sourceFoliage.leaf_vertex_count,total=foliage.vertices.length/(stride*10);
  const blades=Array.from({length:total},()=>[]);
  for(let offset=0;offset<foliage.indices.length;offset+=3){
    const triangle=foliage.indices.subarray(offset,offset+3),leaf=Math.floor(triangle[0]/stride);
    if(triangle.every(index=>sourceFoliage.uvs[index*2+1]>=.16))
      blades[leaf].push(...Array.from(triangle,index=>index-leaf*stride));
  }
  let penetrations=0;
  for(let leaf=0;leaf<total;leaf++){
    const first=leaf*stride*10,packet={vertices:foliage.vertices.subarray(first,first+stride*10),indices:new Uint32Array(blades[leaf])};
    if(woodSurface.intersects(packet))penetrations++;
  }
  return {leaves:total,penetrations,time:physics.time,deformationLipschitzUpperBound};
}
