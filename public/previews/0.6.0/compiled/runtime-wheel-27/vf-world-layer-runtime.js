(function(root, factory) {
  if (typeof module === 'object' && module.exports) { module.exports = factory(); return; }
  var api = factory(); root.VfWorldLayerRuntime = api;
  var script = document.currentScript;
  if (script && script.dataset.vfWorldWasm) api.boot({
    wasmUrl: script.dataset.vfWorldWasm, manifestUrl: script.dataset.vfWorldManifest
  }).catch(function(error) {
    root.__vfWorldLayerError = api.formatError(error,root.__vfWorldLayerStage);
    console.error('VKF World layer application',error);
    var box = document.getElementById('vf-material-error') || document.createElement('pre');
    box.setAttribute('role','alert');box.hidden=false;
    box.style.cssText='position:fixed;inset:12px;z-index:1000;overflow:auto;margin:0;padding:12px;background:#162227;color:#ffd0c0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:none';
    box.textContent = root.__vfWorldLayerError;
    if(!box.parentNode)document.body.appendChild(box);
  });
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';

  function formatError(error,stage) {
    var headline = error && error.message ? String(error.name || 'Error') + ': ' + error.message : String(error);
    var reason = error && error.reason ? '\nReason: ' + error.reason : '';
    var stack = error && error.stack ? String(error.stack) : '';
    if (stack === '_' || stack === '-' || stack === headline) stack = '';
    return (stage ? 'Startup stage: ' + stage + '\n' : '') + headline + reason + (stack ? '\n' + stack : '');
  }

  function createPresentation(runtime, packets) {
    var program = runtime.worldProgram();
    if (!program) throw new Error('compiled World authoring metadata is unavailable');
    var views = runtime.worldLayerViews();
    var display = packets.find(function(packet) { return packet.kind === 'display.replace'; }).payload.display;
    var view = program.views[program.active_view];
    var geometry = display.geom[view.frame_id];
    if (!geometry) throw new Error('compiled active View is unavailable');
    geometry.unified_renderer = true;
    var transfers = [];
    geometry.meshes.forEach(function(mesh) {
      var arena = views.find(function(item) { return item.layer.id === mesh.layer_id; });
      if (!arena) return; // Plot axes are presentation-only.
      transfers.push({ mesh:mesh, arena:arena });
    });
    return { frameId:view.frame_id, geometry:geometry, refresh:function() {
      // Only placement transfer. No derivatives, forces, material inference,
      // time sampling or per-frame JSON serialization belongs in the host.
      transfers.forEach(function(item) {
        var layer = item.arena.layer, state = item.arena.state, vertices = item.mesh.vertices;
        for (var i=0; i<layer.count; ++i) for (var c=0; c<layer.dimension; ++c) {
          vertices[i*10+c] = state[(i*layer.dimension+c)*3];
        }
      });
      return geometry;
    }};
  }

  async function boot(options) {
    globalThis.__vfWorldLayerStage='fetch compiled application';
    var responses = await Promise.all([fetch(options.wasmUrl),fetch(options.manifestUrl)]);
    if (!responses.every(function(response) { return response.ok; })) throw new Error('compiled World artifacts failed to load');
    var bytes = new Uint8Array(await responses[0].arrayBuffer()), manifest = await responses[1].json();
    globalThis.__vfWorldLayerStage='instantiate compiled WASM';
    var runtime = await globalThis.VfCompiledRuntimeBridge.instantiateWasmRuntimeAsync({bytes:bytes,manifest:manifest});
    globalThis.__vfWorldLayerStage='initialize compiled authored data';
    runtime.init();
    var authored = runtime.worldProgram();
    if (authored.gpu_worlds && authored.gpu_worlds.length) {
      var script = document.querySelector('script[src*="vf-world-layer-runtime.js"]');
      var mechanical = authored.gpu_worlds.every(function(world){return world.kind==='rigid'||world.kind==='wind';});
      var moduleUrl = new URL(mechanical?'vf-world-mechanical-runtime.mjs':'vf-world-material-runtime.mjs',script.src);
      moduleUrl.searchParams.set('v','sand-transport-4');
      globalThis.__vfWorldLayerStage='import World GPU adapter';
      var adapter = await import(moduleUrl.href);
      globalThis.__vfWorldLayerStage='compile World GPU laws and embedding';
      return adapter.bootMaterialWorlds(runtime,options);
    }
    // A full scene embedding replaces default plotting. Do not decode the
    // unused per-parcel plot packet before starting a GPU-owned World.
    var packets = JSON.parse(runtime.readBinding('$ui$compiled$packets'));
    await globalThis.VfRuntimeShell.ensureSceneDependencies();
    globalThis.VfRuntimeShell.ensureSceneDocumentMeta();
    globalThis.VfRuntimeShell.ensureSceneHostStyles();
    globalThis.VfRuntimeShell.ensureShellDom('layer','vf-screen-canvas');
    globalThis.VfRuntimeShell.boot({pollPackets:false});
    packets.filter(function(packet) { return packet.kind !== 'display.replace'; }).forEach(function(packet) {
      globalThis.VfRuntimeShell.applyRuntimePacket(packet);
    });
    var presentation = createPresentation(runtime,packets);
    globalThis.VfDisplay.mountDynamicGeomFrame(presentation.frameId,presentation.refresh);
    var program = runtime.worldProgram();
    var worldId = program.views[program.active_view].world_id;
    var steps = program.layers.filter(function(layer) { return layer.world_id === worldId; }).map(function(layer) { return layer.time_step; });
    var dt = steps.length ? Math.min.apply(Math,steps) : 1/120;
    if (!steps.every(function(step) { return step === dt; })) throw new Error('browser World clock currently requires a common fixed timestep');
    var previous = null, accumulator = 0;
    function frame(timestamp) {
      if (document.hidden) {
        previous = null; accumulator = 0; requestAnimationFrame(frame); return;
      }
      if (previous !== null) accumulator = Math.min(dt*8,accumulator + Math.max(0,Math.min(0.05,(timestamp-previous)/1000)));
      previous = timestamp;
      if (!globalThis.VfDisplay.dynamicGeomFrameCanAcceptUpdate(presentation.frameId) ||
          globalThis.VfDisplay.dynamicGeomFrameHasRenderBackpressure(presentation.frameId)) {
        requestAnimationFrame(frame); return;
      }
      var count = Math.min(8,Math.floor(accumulator/dt));
      if (count) {
        for (var i=0; i<count; ++i) runtime.stepWorld(worldId);
        accumulator -= count*dt;
        globalThis.VfDisplay.requestDynamicGeomFrameUpdate(presentation.frameId);
      }
      requestAnimationFrame(frame);
    }
    globalThis.__vfWorldLayerApplication = {runtime:runtime,presentation:presentation};
    document.body.dataset.vfWorldLayerReady = 'true';
    requestAnimationFrame(frame);
    return globalThis.__vfWorldLayerApplication;
  }
  return {createPresentation:createPresentation,boot:boot,formatError:formatError};
});
