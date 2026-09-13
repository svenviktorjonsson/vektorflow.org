function retainedPacket(packet) {
  return packet
    && packet.schema === "vektor-flow/retained-scene-arena"
    && packet.version === 1
    && packet.metadata?.schema === "vektor-flow/retained-scene-arena"
    && packet.metadata?.version === 1
    && typeof packet.metadata?.scene?.frame === "string"
    && packet.arena instanceof Uint8Array;
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
  const layer = document.createElement("div");
  layer.className = "readme-example-retained-layer";
  layer.dataset.vfRenderState = "pending";
  container.append(layer);
  const frames = [];
  for (const packet of packets) {
    const frame = VfFrame.mount(layer, {
      id: packet.metadata.scene.frame,
      title: "",
      frameless: true,
      draggable: false,
      dockable: false,
      resizable: false,
      closable: false,
      alpha: 1,
      toolbar: packet.metadata.scene.toolbar === null ? null : undefined,
    });
    frame?.body?.classList?.add("vf-frame__body--transparent");
    frames.push(frame);
  }
  let disposed = false;
  let presentFrame = 0;
  presentFrame = requestAnimationFrame(() => {
    presentFrame = 0;
    if (disposed) return;
    for (const packet of packets) VfDisplay.renderRetainedSceneArena(packet);
    layer.dataset.vfRenderState = "presenting";
    presentFrame = requestAnimationFrame(() => {
      presentFrame = 0;
      if (disposed) return;
      layer.dataset.vfRenderState = "visible";
      layer.dispatchEvent?.(new Event("vf-inline-result-visible"));
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
    container.append(layer, enlarge);
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
    dialog.append(close, layer);
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
    layer.remove?.();
  };
}
