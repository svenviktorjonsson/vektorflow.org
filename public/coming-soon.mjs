import './vendor/prism/prism-core-1.30.0.min.js';
import {registerVektorFlowPrism} from './editor/prism-vektorflow.mjs';
registerVektorFlowPrism(globalThis.Prism);

export async function verifyPreviewBytes(bytes,expected){
  const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(digest!==expected)throw Error('Source/build mismatch: reload before running');
}
let bundlePromise;
const previewBundle=()=>bundlePromise??=fetch('./previews/0.6.0/compiled/bundle.json?v=preview-release-17').then(r=>{if(!r.ok)throw Error('Build receipt unavailable');return r.json();});

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
  code.classList.add('language-vkf');
  const state=document.createElement('span');state.setAttribute('role','status');
  const run=document.createElement('button');run.type='button';run.textContent='Run compiled application';run.className='source-run';
  host.querySelector('.source-footer').append(' · ',run,' · ',state);
  const cache = new Map(); let request = 0;
  const sourceUrl=name=>`./sources/coming-soon/${id}/${name}?v=${id==='stones'?'granite-8':id==='tree'?'air-10':'wheel-performance-16'}`;
  async function source(name){
    if(!cache.has(name))cache.set(name,(async()=>{
      const record=(await previewBundle()).applications[id],response=await fetch(sourceUrl(name));
      if(!response.ok)throw Error(`Source unavailable (${response.status})`);
      const bytes=await response.arrayBuffer();await verifyPreviewBytes(bytes,record.sources[name]);return new TextDecoder().decode(bytes);
    })().catch(error=>{cache.delete(name);throw error;}));
    return cache.get(name);
  }
  run.addEventListener('click',async()=>{
    run.disabled=true;state.textContent='Checking source and executable…';
    try{
      const record=(await previewBundle()).applications[id];await Promise.all(files.map(source));
      for(const [name,digest] of [['main.wasm',record.wasm],['manifest.json',record.manifest]]){
        const response=await fetch(`./previews/0.6.0/compiled/${record.directory??id}/${name}`);if(!response.ok)throw Error('Executable unavailable');await verifyPreviewBytes(await response.arrayBuffer(),digest);
      }
      const frame=host.closest('section').querySelector('iframe'),url=new URL(frame.src,location.href);url.searchParams.set('build',record.wasm);url.searchParams.set('run',String(Date.now()));frame.src=url.href;
      state.textContent=`${record.build?'Source-compiled':'Published legacy'} build ${record.wasm.slice(0,12)} · restarted`;
    }catch(error){state.textContent=error.message;}finally{run.disabled=false;}
  });
  async function select(index, focus = false) {
    const current = ++request; const name = files[index];
    const url = sourceUrl(name);
    for (const [i, button] of [...tabs.children].entries()) {
      button.setAttribute('aria-selected', String(i === index)); button.tabIndex = i === index ? 0 : -1;
    }
    panel.setAttribute('aria-labelledby', `${id}-source-${index}`);
    panel.setAttribute('aria-busy', 'true');
    link.href = url; link.textContent = `Download ${name}`;
    if (focus) tabs.children[index].focus();
    code.textContent = `Loading ${name}…`;
    try {
      const text=await source(name);
      if (request === current) {code.textContent=text;globalThis.Prism.highlightElement(code);
        const record=(await previewBundle()).applications[id];state.textContent=`Read-only · ${record.build?'source-compiled':'legacy published'} build ${record.wasm.slice(0,12)}`;
      }
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
