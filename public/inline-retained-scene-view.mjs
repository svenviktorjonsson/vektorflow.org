function retainedPacket(packet) {
  return packet
    && packet.schema === "vektor-flow/retained-scene-arena"
    && packet.version === 1
    && packet.metadata?.schema === "vektor-flow/retained-scene-arena"
    && packet.metadata?.version === 1
    && typeof packet.metadata?.scene?.frame === "string"
    && packet.arena instanceof Uint8Array;
}

let nextRetainedSceneScope = 0;

function scopeFrameReferences(value, frameIds, property = "") {
  if (Array.isArray(value)) {
    return value.map((entry) => scopeFrameReferences(entry, frameIds));
  }
  if (!value || typeof value !== "object") {
    const isFrameReference = property === "frame"
      || property === "frame_id"
      || property.endsWith("_frame_id")
      || property === "frame_ref"
      || property.endsWith("_frame_ref");
    return isFrameReference && typeof value === "string" && frameIds.has(value)
      ? frameIds.get(value)
      : value;
  }
  return Object.fromEntries(Object.entries(value).map(([key, entry]) =>
    [key, scopeFrameReferences(entry, frameIds, key)]));
}

function ownRetainedScenePackets(packets) {
  const scope = `vf-inline-result-${++nextRetainedSceneScope}--`;
  const frameIds = new Map(packets.map((packet) => [
    packet.metadata.scene.frame,
    `${scope}${packet.metadata.scene.frame}`,
  ]));
  return packets.map((packet) => ({
    ...packet,
    metadata: scopeFrameReferences(packet.metadata, frameIds),
  }));
}

export function mountRetainedSceneResult(container, packets, {
  document = globalThis.document,
  VfFrame = globalThis.VfFrame,
  VfDisplay = globalThis.VfDisplay,
  requestAnimationFrame = globalThis.requestAnimationFrame ?? ((callback) => { callback(0); return 0; }),
  cancelAnimationFrame = globalThis.cancelAnimationFrame ?? (() => {}),
} = {}) {
  if (!Array.isArray(packets) || packets.length === 0 ||
      !packets.every(retainedPacket)) {
    throw new TypeError("retained scene arena schema is missing or unsupported");
  }
  if (!container || typeof container.append !== "function" ||
      !document || typeof document.createElement !== "function" ||
      !VfFrame || typeof VfFrame.mount !== "function" ||
      !VfDisplay || typeof VfDisplay.renderRetainedSceneArena !== "function") {
    throw new Error("native VfFrame/VfDisplay retained renderer is unavailable");
  }
  const ownedPackets = ownRetainedScenePackets(packets);
  const layer = document.createElement("div");
  layer.className = "readme-example-retained-layer";
  layer.dataset.vfRenderState = "pending";
  const frames = [];
  const showFrame = (frame, visible) => {
    if (!frame?.root) return;
    frame.root.hidden = !visible;
    frame.root.setAttribute?.("aria-hidden", visible ? "false" : "true");
    if (frame.root.style) frame.root.style.display = visible ? "" : "none";
  };
  for (const [index, packet] of ownedPackets.entries()) {
    const frame = VfFrame.mount(layer, {
      id: packet.metadata.scene.frame,
      title: packet.metadata.scene.title ?? "",
      frameless: true,
      draggable: false,
      dockable: false,
      resizable: false,
      closable: false,
      alpha: 1,
      toolbar: packet.metadata.scene.toolbar === null ? null : undefined,
    });
    frame?.body?.classList?.add("vf-frame__body--transparent");
    showFrame(frame, index === 0);
    frames.push(frame);
  }
  const selectors = ownedPackets.map((packet) => packet.metadata.scene.view_selector);
  const hasViewSelector = ownedPackets.length > 1 && selectors.every((selector, index) =>
    selector?.axis === "n" && selector.index === index && selector.count === ownedPackets.length);
  let selector = null;
  let viewport = layer;
  if (hasViewSelector) {
    viewport = document.createElement("div");
    viewport.className = "readme-example-view-shell";
    selector = document.createElement("nav");
    selector.className = "readme-example-view-selector";
    selector.setAttribute("aria-label", "Views");
    const header = document.createElement("div");
    header.className = "readme-example-view-selector__label";
    header.textContent = String(selectors[0].header ?? "n");
    selector.append(header);
    const buttons = [];
    selectors.forEach((entry, selected) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = String(entry.label ?? entry.value);
      button.setAttribute("aria-pressed", selected === 0 ? "true" : "false");
      button.addEventListener("click", () => {
        frames.forEach((frame, index) => {
          showFrame(frame, index === selected);
          buttons[index]?.setAttribute?.(
            "aria-pressed", index === selected ? "true" : "false");
        });
      });
      buttons.push(button);
      selector.append(button);
    });
    viewport.append(selector, layer);
  }
  container.append(viewport);
  let disposed = false;
  let presentFrame = 0;
  let animationStart = null;
  const animated = ownedPackets.some((packet) =>
    packet.metadata.scene.meshes?.some((mesh) => mesh._layer_time));
  const animate = (timestamp) => {
    if (disposed) return;
    if (animationStart == null) animationStart = timestamp;
    for (const packet of ownedPackets) {
      updateRetainedSceneTime(packet, (timestamp - animationStart) / 1000);
      VfDisplay.renderRetainedSceneArena(packet);
    }
    presentFrame = requestAnimationFrame(animate);
  };
  presentFrame = requestAnimationFrame(() => {
    presentFrame = 0;
    if (disposed) return;
    for (const packet of ownedPackets) VfDisplay.renderRetainedSceneArena(packet);
    layer.dataset.vfRenderState = "presenting";
    presentFrame = requestAnimationFrame(() => {
      presentFrame = 0;
      if (disposed) return;
      layer.dataset.vfRenderState = "visible";
      layer.dispatchEvent?.(new Event("vf-inline-result-visible"));
      if (animated) presentFrame = requestAnimationFrame(animate);
    });
  });
  const enlarge = document.createElement("button");
  enlarge.type = "button";
  enlarge.className = "readme-example-enlarge";
  enlarge.textContent = "Enlarge";
  enlarge.setAttribute("aria-label", "Enlarge interactive result");
  enlarge.setAttribute("aria-haspopup", "dialog");
  container.append(enlarge);
  let dialog = null;
  let returnFocus = null;
  const restore = () => {
    if (!dialog) return;
    container.append(viewport, enlarge);
    dialog.remove();
    dialog = null;
    returnFocus?.focus();
  };
  enlarge.addEventListener("click", () => {
    if (dialog) return;
    returnFocus = document.activeElement;
    dialog = document.createElement("dialog");
    dialog.className = "readme-result-dialog";
    dialog.setAttribute("aria-label", "Interactive result");
    const close = document.createElement("button");
    close.type = "button";
    close.className = "readme-result-close";
    close.textContent = "Close";
    close.setAttribute("aria-label", "Close enlarged result");
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); dialog.close(); });
    const openedDialog = dialog;
    dialog.addEventListener("close", () => { if (dialog === openedDialog) restore(); });
    dialog.append(close, viewport);
    document.body.append(dialog);
    dialog.showModal();
    close.focus();
  });
  return () => {
    disposed = true;
    if (presentFrame) cancelAnimationFrame(presentFrame);
    if (dialog) { dialog.close(); restore(); }
    enlarge.remove();
    for (const frame of frames) frame?.root?.remove?.();
    viewport.remove?.();
  };
}

function temporalSample(descriptor, elapsed) {
  const coordinates = descriptor?.coordinates;
  if (!Array.isArray(coordinates) || coordinates.length < 2) return null;
  const start = Number(coordinates[0]);
  const end = Number(coordinates.at(-1));
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  let time = start + Math.max(0, elapsed);
  const duration = end - start;
  if (descriptor.mode === "repeat") time = start + ((time - start) % duration);
  else if (descriptor.mode === "mirror") {
    const phase = ((time - start) % (duration * 2)) / duration;
    time = phase <= 1 ? start + phase * duration : end - (phase - 1) * duration;
  } else if (descriptor.mode === "reset" && time >= end) time = start;
  else time = Math.min(time, end);
  let upper = coordinates.findIndex((coordinate) => Number(coordinate) >= time);
  if (upper <= 0) upper = 1;
  const lower = upper - 1;
  const span = Number(coordinates[upper]) - Number(coordinates[lower]);
  return { lower, upper, alpha: span > 0 ? (time - Number(coordinates[lower])) / span : 0 };
}

export function updateRetainedSceneTime(packet, elapsedSeconds) {
  let changed = false;
  for (const mesh of packet?.metadata?.scene?.meshes ?? []) {
    const descriptor = mesh?._layer_time;
    const sample = temporalSample(descriptor, elapsedSeconds);
    if (!sample || !mesh.vertices || !(packet.arena instanceof Uint8Array)) continue;
    const view = new DataView(packet.arena.buffer,
      packet.arena.byteOffset + mesh.vertices.byte_offset,
      mesh.vertices.length * Float32Array.BYTES_PER_ELEMENT);
    for (const channel of descriptor.channels ?? []) {
      const lower = channel.value?.[sample.lower];
      const upper = channel.value?.[sample.upper];
      if (lower == null || upper == null) continue;
      const values = Array.isArray(lower) ? lower.map((value, index) =>
        Number(value) + (Number(upper[index]) - Number(value)) * sample.alpha) :
        [Number(lower) + (Number(upper) - Number(lower)) * sample.alpha];
      const offset = channel.name === "p" ? 0 : channel.name === "c" ? 6 : -1;
      if (offset >= 0) values.forEach((value, index) =>
        view.setFloat32((offset + index) * 4, value, true));
      if (channel.name === "s") mesh.vertex_size = values[0];
      changed = true;
    }
  }
  return changed;
}
