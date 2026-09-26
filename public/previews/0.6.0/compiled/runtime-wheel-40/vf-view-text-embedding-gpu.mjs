import {createCheckedGpuPipeline} from './vf-gpu-pipeline-errors.mjs';

// Generic retained View text embedding. VKF owns strings, positions and style;
// this adapter only builds a glyph atlas and submits quads to the same canvas.
const FIRST=32, LAST=126, COLUMNS=16, CELL_WIDTH=24, CELL_HEIGHT=32;
const SHADER=/* wgsl */`
struct Canvas { size: vec2<f32>, pad: vec2<f32> };
struct Glyph { rect: vec4<f32>, uv: vec4<f32>, color: vec4<f32> };
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32> };
@group(0) @binding(0) var<uniform> canvas: Canvas;
@group(0) @binding(1) var<storage,read> glyphs: array<Glyph>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var atlasSampler: sampler;
@vertex fn vertex_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Out {
  let point=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),
    vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1))[vertex];
  let glyph=glyphs[instance];
  let pixel=glyph.rect.xy+point*glyph.rect.zw;
  var output:Out;
  output.position=vec4<f32>(pixel.x/canvas.size.x*2.0-1.0,1.0-pixel.y/canvas.size.y*2.0,0.0,1.0);
  output.uv=glyph.uv.xy+point*glyph.uv.zw;
  output.color=glyph.color;
  return output;
}
@fragment fn fragment_main(input:Out)->@location(0) vec4<f32> {
  let alpha=textureSample(atlas,atlasSampler,input.uv).a*input.color.a;
  return vec4<f32>(input.color.rgb*alpha,alpha);
}`;

export function fillViewTextTemplate(template,bindings={}) {
  return String(template).replace(/%([A-Z_]+)%/g,(match,key)=>
    Object.hasOwn(bindings,key)?String(bindings[key]):match);
}

export function layoutViewTextGlyphs(layers,bindings,width,height,pixelRatio=1,dataToPixel) {
  const data=[];
  for(const layer of layers) {
    const p=layer.properties??{},x=layer.position?.[0]??0,y=layer.position?.[1]??0;
    const space=p.space??'data';
    const anchor=space==='relative'?[x*width,y*height]
      :space==='pixel'?[x*pixelRatio,y*pixelRatio]
        :space==='data'&&typeof dataToPixel==='function'?dataToPixel([x,y])
          :null;
    if(!anchor)throw new Error(`View text has no projection for ${space} coordinates`);
    const text=fillViewTextTemplate(p.text??'',bindings);
    const fontSize=Math.max(1,Number(p.font_size??13))*pixelRatio;
    const advance=fontSize*0.64,lineHeight=fontSize*1.42;
    const color=Array.isArray(p.color)&&p.color.length===4?p.color:[.65,.77,.77,1];
    const lines=text.split('\n');
    for(let line=0;line<lines.length;line++) {
      const alignment=p.ha==='right'?1:p.ha==='center'?.5:0;
      const originX=anchor[0]-lines[line].length*advance*alignment;
      const originY=anchor[1]+line*lineHeight;
      for(let i=0;i<lines[line].length;i++) {
        const code=lines[line].charCodeAt(i);
        if(code===32)continue;
        const glyph=Math.min(LAST,Math.max(FIRST,code))-FIRST;
        const column=glyph%COLUMNS,row=Math.floor(glyph/COLUMNS);
        // Atlas cells include blank side bearings; quad is wider than advance
        // so the visible ink reaches the authored CSS font size.
        data.push(originX+i*advance,originY,fontSize*1.1,fontSize*1.8,
          (column*CELL_WIDTH+.5)/(COLUMNS*CELL_WIDTH),
          (row*CELL_HEIGHT+.5)/(6*CELL_HEIGHT),
          (CELL_WIDTH-1)/(COLUMNS*CELL_WIDTH),
          (CELL_HEIGHT-1)/(6*CELL_HEIGHT),
          color[0],color[1],color[2],color[3]);
      }
    }
  }
  return new Float32Array(data);
}

export async function createViewTextEmbeddingGpu(device,canvas,format,{capacity=2048}={}) {
  const surface=typeof OffscreenCanvas==='function'
    ?new OffscreenCanvas(COLUMNS*CELL_WIDTH,6*CELL_HEIGHT)
    :document.createElement('canvas');
  surface.width=COLUMNS*CELL_WIDTH;surface.height=6*CELL_HEIGHT;
  const context=surface.getContext('2d');
  if(!context)throw new Error('View text glyph atlas needs a 2D rasterizer');
  context.clearRect(0,0,surface.width,surface.height);
  context.font='24px ui-monospace, SFMono-Regular, Consolas, monospace';
  context.textBaseline='top';context.fillStyle='#fff';
  for(let code=FIRST;code<=LAST;code++) {
    const index=code-FIRST;
    context.fillText(String.fromCharCode(code),(index%COLUMNS)*CELL_WIDTH+1,
      Math.floor(index/COLUMNS)*CELL_HEIGHT+2);
  }
  const atlas=device.createTexture({label:'VKF View text glyph atlas',
    size:[surface.width,surface.height],format:'rgba8unorm',
    usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});
  device.queue.copyExternalImageToTexture({source:surface},{texture:atlas},
    [surface.width,surface.height]);
  const module=device.createShaderModule({label:'VKF View text embedding',code:SHADER});
  const pipeline=await createCheckedGpuPipeline(device,'render',{
    label:'VKF View text pass',layout:'auto',
    vertex:{module,entryPoint:'vertex_main'},
    fragment:{module,entryPoint:'fragment_main',targets:[{format,blend:{
      color:{operation:'add',srcFactor:'one',dstFactor:'one-minus-src-alpha'},
      alpha:{operation:'add',srcFactor:'one',dstFactor:'one-minus-src-alpha'},
    }}]},primitive:{topology:'triangle-list'}});
  const glyphBuffer=device.createBuffer({label:'VKF View text glyph instances',size:capacity*48,
    usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const canvasBuffer=device.createBuffer({label:'VKF View text canvas size',size:16,
    usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const sampler=device.createSampler({magFilter:'linear',minFilter:'linear'});
  const bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:canvasBuffer}},
    {binding:1,resource:{buffer:glyphBuffer}},
    {binding:2,resource:atlas.createView()},
    {binding:3,resource:sampler},
  ]});
  const render=(encoder,target,layers,bindings={},dataToPixel)=>{
    const pixelRatio=canvas.width/Math.max(canvas.clientWidth||canvas.width,1);
    const glyphs=layoutViewTextGlyphs(layers,bindings,canvas.width,canvas.height,pixelRatio,dataToPixel);
    const count=glyphs.length/12;
    if(count>capacity)throw new Error(`View text glyph budget exceeded: ${count}/${capacity}`);
    if(!count)return;
    device.queue.writeBuffer(glyphBuffer,0,glyphs);
    device.queue.writeBuffer(canvasBuffer,0,new Float32Array([canvas.width,canvas.height,0,0]));
    const pass=encoder.beginRenderPass({label:'VKF retained View text',
      colorAttachments:[{view:target,loadOp:'load',storeOp:'store'}]});
    pass.setPipeline(pipeline);pass.setBindGroup(0,bindGroup);pass.draw(6,count);pass.end();
  };
  return Object.freeze({render,destroy(){glyphBuffer.destroy();canvasBuffer.destroy();atlas.destroy();}});
}
