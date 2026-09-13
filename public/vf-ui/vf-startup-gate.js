/** Atomic application startup: reveal only fully interactive, presented UI. */
(function (global) {
  "use strict";
  if (global.VfStartupGate) { return; }

  var frames = Object.create(null);
  var revealRaf = 0;
  var revealed = false;
  var firstGpuPublished = false;
  var timeline = global.__vfStartupTimeline = Array.isArray(global.__vfStartupTimeline)
    ? global.__vfStartupTimeline
    : [];

  function nowMs() {
    return global.performance && typeof global.performance.now === "function"
      ? global.performance.now()
      : Date.now();
  }
  function mark(name, detail) {
    timeline.push({ name: String(name), t: nowMs(), detail: detail || null });
  }
  function recordFor(id) {
    var key = String(id || "");
    if (!key) { return null; }
    if (!frames[key]) {
      frames[key] = {
        id: key, expected: false, presentation: false,
        interaction: true, interactive: false, submitted: false, presented: false
      };
    }
    return frames[key];
  }
  function frameElement(id) {
    if (!global.document || typeof global.document.querySelector !== "function") { return null; }
    return global.document.querySelector('.vf-frame[data-vf-frame-id="' + String(id) + '"]');
  }
  function allReady() {
    var ids = Object.keys(frames).filter(function (id) { return frames[id].expected; });
    if (!ids.length) { return false; }
    return ids.every(function (id) {
      var frame = frames[id];
      return (!frame.interaction || frame.interactive) &&
        (!frame.presentation || frame.presented);
    });
  }
  function expectsGpuPresentation() {
    return Object.keys(frames).some(function (id) {
      return frames[id].expected && frames[id].presentation;
    });
  }
  function publishReady() {
    try {
      if (global.chrome && global.chrome.webview && typeof global.chrome.webview.postMessage === "function") {
        global.chrome.webview.postMessage({
          type: "vf-ui-ready", source: "atomic-startup-gate", timeline: timeline.slice()
        });
      }
    } catch (_) {}
  }
  function commitReveal() {
    revealRaf = 0;
    if (revealed || !allReady()) { return; }
    revealed = true;
    var doc = global.document;
    var root = doc && doc.documentElement;
    if (root) {
      root.removeAttribute("data-vf-startup-pending");
      root.setAttribute("data-vf-startup-ready", "1");
    }
    mark("ui:revealed");
    try {
      if (doc && typeof doc.dispatchEvent === "function" && typeof global.CustomEvent === "function") {
        doc.dispatchEvent(new global.CustomEvent("vf-ui-interactive", {
          detail: { timeline: timeline.slice() }
        }));
      }
    } catch (_) {}
    publishReady();
    if (global.VfFrame && typeof global.VfFrame.postNativeHostLayout === "function") {
      try {
        var layer = doc && doc.getElementById ? doc.getElementById("layer") : null;
        global.VfFrame.postNativeHostLayout(layer || (doc && doc.body), {
          stageAlpha: 0,
          contentReady: true
        });
      } catch (_) {}
    }
  }
  function scheduleReveal() {
    if (revealed || revealRaf || !allReady()) { return; }
    if (!firstGpuPublished && expectsGpuPresentation()) {
      firstGpuPublished = true;
      try {
        if (global.chrome && global.chrome.webview && typeof global.chrome.webview.postMessage === "function") {
          global.chrome.webview.postMessage({
            type: "vf_startup_stage_v1",
            stage: "first_gpu_frame_ready"
          });
        }
      } catch (_) {}
    }
    mark("ui:ready-to-reveal");
    revealRaf = global.requestAnimationFrame(commitReveal);
  }

  global.VfStartupGate = {
    isArmed: function () {
      var root = global.document && global.document.documentElement;
      return !!(root && root.hasAttribute && root.hasAttribute("data-vf-startup-pending"));
    },
    expectFrames: function (specs) {
      var list = Array.isArray(specs) ? specs : [];
      list.forEach(function (value) {
        var spec = typeof value === "string" ? { id: value } : (value || {});
        var record = recordFor(spec.id);
        if (!record) { return; }
        record.expected = true;
        record.presentation = spec.presentation !== false;
        record.interaction = spec.interaction !== false;
        var element = frameElement(record.id);
        if (record.interaction && element && element.__vfPanel) { record.interactive = true; }
      });
      mark("frames:expected", list.map(function (value) {
        return typeof value === "string" ? value : String(value && value.id || "");
      }));
      scheduleReveal();
    },
    markInteractive: function (id, root) {
      var record = recordFor(id);
      if (!record) { return; }
      record.interactive = true;
      if (root && root.setAttribute) { root.setAttribute("data-vf-startup-interactive", "1"); }
      mark("frame:interactive", { frame_id: record.id });
      scheduleReveal();
    },
    markGpuSubmitted: function (id) {
      var record = recordFor(id);
      if (!record) { return; }
      record.submitted = true;
      mark("gpu:first-submit", { frame_id: record.id });
    },
    invalidate: function (id, detail) {
      var record = recordFor(id);
      if (!record || !record.presentation || revealed) { return; }
      record.presented = false;
      mark("frame:composite-invalidated", {
        frame_id: record.id,
        dependency: String(detail && detail.dependency || "")
      });
    },
    markPresented: function (id) {
      var record = recordFor(id);
      if (!record) { return; }
      record.presented = true;
      mark("gpu:first-work-done", { frame_id: record.id });
      scheduleReveal();
    },
    fail: function (error) {
      mark("startup:failed", { message: String(error && error.message ? error.message : error) });
      var root = global.document && global.document.documentElement;
      if (root) {
        root.removeAttribute("data-vf-startup-pending");
        root.setAttribute("data-vf-startup-error", "1");
      }
    },
    mark: mark,
    snapshot: function () {
      var copy = Object.create(null);
      Object.keys(frames).forEach(function (id) { copy[id] = Object.assign({}, frames[id]); });
      return { revealed: revealed, frames: copy, timeline: timeline.slice() };
    }
  };
  mark("gate:installed");
})(typeof window !== "undefined" ? window : this);
