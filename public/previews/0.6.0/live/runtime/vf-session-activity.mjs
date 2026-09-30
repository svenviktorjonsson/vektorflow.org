// Suspend off-screen work without resetting World state or the user's pause choice.
let active = true;
window.addEventListener('message', (event) => {
  if (window.parent === window || event.source !== window.parent || event.origin !== location.origin
    || event.data?.type !== 'vf-preview-visibility' || typeof event.data.active !== 'boolean') return;
  active = event.data.active;
});
export const isLiveSessionActive = () => active && !document.hidden;
