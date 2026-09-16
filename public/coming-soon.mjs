export const applications = [
  { id: 'wheel', files: ['main.vkf', 'geometry.vkf', 'materials.vkf', 'particles.vkf'] },
  { id: 'stones', files: ['main.vkf', 'geometry.vkf', 'materials.vkf'] },
  { id: 'tree', files: ['main.vkf', 'geometry.vkf', 'materials.vkf'] },
];

function mountSourceTabs({ id, files }) {
  const host = document.querySelector(`[data-source-app="${id}"]`);
  const tabs = host.querySelector('[role="tablist"]');
  const panel = host.querySelector('[role="tabpanel"]');
  const code = panel.querySelector('code');
  const link = host.querySelector('[data-source-download]');
  const cache = new Map(); let request = 0;
  async function select(index, focus = false) {
    const current = ++request; const name = files[index];
    const url = `./sources/coming-soon/${id}/${name}`;
    for (const [i, button] of [...tabs.children].entries()) {
      button.setAttribute('aria-selected', String(i === index)); button.tabIndex = i === index ? 0 : -1;
    }
    panel.setAttribute('aria-labelledby', `${id}-source-${index}`);
    panel.setAttribute('aria-busy', 'true');
    link.href = url; link.textContent = `Download ${name}`;
    if (focus) tabs.children[index].focus();
    code.textContent = `Loading ${name}…`;
    try {
      if (!cache.has(name)) cache.set(name, fetch(url).then((response) => {
        if (!response.ok) throw new Error(`Source unavailable (${response.status})`); return response.text();
      }).catch((error) => { cache.delete(name); throw error; }));
      const source = await cache.get(name);
      if (request === current) code.textContent = source;
    } catch (error) { if (request === current) code.textContent = String(error.message); }
    finally { if (request === current) panel.setAttribute('aria-busy', 'false'); }
  }
  files.forEach((name, index) => {
    const button = document.createElement('button');
    button.type = 'button'; button.id = `${id}-source-${index}`; button.textContent = name;
    button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', panel.id);
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % files.length;
      if (event.key === 'ArrowLeft') next = (index + files.length - 1) % files.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = files.length - 1;
      if (next !== undefined) { event.preventDefault(); select(next, true); }
    });
    tabs.append(button);
  });
  select(0);
}

function mountPreviewActivity() {
  const frames = [...document.querySelectorAll('.session-shell iframe')];
  const visible = new Map(frames.map((frame) => [frame, false]));
  const notify = (frame) => frame.contentWindow?.postMessage({
    type: 'vf-preview-visibility', active: visible.get(frame) && !document.hidden,
  }, location.origin);
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) { visible.set(entry.target, entry.isIntersecting); notify(entry.target); }
  }, { threshold: 0 });
  for (const frame of frames) { observer.observe(frame); frame.addEventListener('load', () => notify(frame)); }
  document.addEventListener('visibilitychange', () => frames.forEach(notify));
  window.addEventListener('message',event=>{if(event.origin!==location.origin||event.data?.type!=='vf-preview-ready')return;const frame=frames.find(frame=>frame.contentWindow===event.source);if(frame)notify(frame);});
}

if (typeof document !== 'undefined') { applications.forEach(mountSourceTabs); mountPreviewActivity(); }
