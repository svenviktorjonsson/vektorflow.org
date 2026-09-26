// Loads the actual hourglass page in GPU Chrome without CDP or a software adapter.
import {createServer} from 'node:http';
import {spawn} from 'node:child_process';
import {readFile, mkdtemp, mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.join(site, 'public');
await mkdir(path.join(site, '.work'), {recursive: true});
const work = await mkdtemp(path.join(site, '.work', 'hourglass-gpu-'));
let finish;
const result = new Promise(resolve => finish = resolve);
const probe = `<script>
(() => {
  const streamOnly = ${process.env.VKF_STREAM_CAPTURE_ONLY === '1'};
  const start = performance.now();
  let frames = 0, early, first, second, third, fourth, near, texture;
  let streamContinuity;
  let beforeTurn, turned, afterTurn;
  let flipped = false, aperturePhase = 0;
  let done = false, lastProgress = start, started = false;
  const send = report => { if (done) return; done = true;
    fetch('/page-result', {method:'POST', body: JSON.stringify(report)}); };
  addEventListener('error', event => send({passed:false,error:String(event.message)}));
  addEventListener('unhandledrejection', event =>
    send({passed:false,error:String(event.reason)}));
  const slope = state => {
    const degrees = [];
    const dx = 0.5 / state.pile.length;
    // Measure the free surface, not the segment clipped by the glass wall.
    for (let i = 65; i < 106; i++) {
      if (state.pile[i] < 0.02 || state.pile[i+1] < 0.02) continue;
      degrees.push(Math.atan2(state.pile[i] - state.pile[i+1],dx)*180/Math.PI);
    }
    return degrees.reduce((sum, angle) => sum+angle,0)/degrees.length;
  };
  const buriedStreamArea = state => {
    let buried = 0;
    for (let i = 0; i < state.stream.length; i++) {
      if (!state.stream[i]) continue;
      const [x,y] = state.streamMotion[i];
      const index = Math.max(0,Math.min(127,Math.floor((x+.25)/.5*128)));
      if (state.direction > 0 && y+.485 >= .91-state.pile[index])
        buried += state.stream[i];
      if (state.direction < 0 && y+.485 <= .06+state.pile[index])
        buried += state.stream[i];
    }
    return buried;
  };
  const compactTextureVariation = async () => {
    const source = document.querySelector('#scene');
    const copy = document.createElement('canvas');
    copy.width = source.width; copy.height = source.height;
    const context = copy.getContext('2d', {willReadFrequently:true});
    const image = new Image();
    image.src = source.toDataURL('image/png');
    await image.decode();
    context.drawImage(image,0,0);
    const pixels = context.getImageData(0,0,copy.width,copy.height).data;
    const variation = (start,end) => {
      let total = 0;
      for (let row = .82; row <= .88; row += .01) {
        const y = Math.floor(copy.height * row);
        const values = [];
        for (let fraction = start; fraction <= end; fraction += .002) {
          const x = Math.floor(copy.width * fraction);
          values.push(pixels[(y*copy.width+x)*4]);
        }
        const mean = values.reduce((sum,value)=>sum+value,0)/values.length;
        total += Math.sqrt(values.reduce((sum,value)=>sum+(value-mean)**2,0)/values.length);
      }
      return total / 7;
    };
    return {left:variation(.40,.46),right:variation(.54,.60)};
  };
  const measureStreamContinuity = async () => {
    const source = document.querySelector('#scene');
    const image = new Image();
    image.src = source.toDataURL('image/png');
    await image.decode();
    const copy = document.createElement('canvas');
    copy.width = source.width; copy.height = source.height;
    const context = copy.getContext('2d', {willReadFrequently:true});
    context.drawImage(image,0,0);
    const data = context.getImageData(0,0,copy.width,copy.height).data;
    const center = Math.floor(copy.width / 2);
    const values = [];
    for (let y = Math.floor(copy.height*.53); y <= Math.floor(copy.height*.79); y++)
      values.push(data[(y*copy.width+center)*4]);
    const jumps = values.slice(1).map((value,i) => Math.abs(value-values[i]));
    return {maximumJump:Math.max(...jumps),
      meanJump:jumps.reduce((sum,value)=>sum+value,0)/jumps.length,
      darkPixels:values.filter(value=>value<14).length};
  };
  const watch = async () => {
    frames++;
    const app = window.__hourglass;
    const message = document.querySelector('#status')?.textContent || '';
    if (message.includes('could not start') || message.includes('GPU device lost'))
      return send({passed:false,error:message});
    if (!app) return requestAnimationFrame(watch);
    if (!started) {
      started = true;
      if (document.querySelector('#play').textContent === 'Play')
        document.querySelector('#play').click();
    }
    if (performance.now()-lastProgress > 5000) {
      lastProgress = performance.now();
      fetch('/page-progress', {method:'POST', body:JSON.stringify({
        wallSeconds:(lastProgress-start)/1000, simulated:app.world.simulatedSeconds,
        frames, status:message})});
    }
    if (app.world.simulatedSeconds >= 1 && !early) {
      early = await app.audit();
      await fetch('/page-screenshot-stream', {method:'POST',
        body:document.querySelector('#scene').toDataURL('image/png')});
      streamContinuity = await measureStreamContinuity();
      if (streamOnly) return send({passed:early.retainedFraction>=.9999,
        time:early.simulatedSeconds,streamArea:early.streamArea,
        retained:early.retainedFraction,streamContinuity});
    }
    if (app.world.simulatedSeconds >= 30 && !first) {
      first = await app.audit();
      document.querySelector('#width').value = '2';
      document.querySelector('#width').dispatchEvent(new Event('input'));
      document.querySelector('#angle').value = '35';
      document.querySelector('#angle').dispatchEvent(new Event('input'));
    }
    if (app.world.simulatedSeconds >= 60 && first && !second)
      second = await app.audit();
    if (app.world.simulatedSeconds >= 130 && second && !flipped) {
      third = await app.audit();
      texture = await compactTextureVariation();
      document.querySelector('#flip').click();
      flipped = true;
    }
    if (app.world.simulatedSeconds >= 160 && flipped && aperturePhase === 0) {
      fourth = await app.audit();
      await fetch('/page-screenshot', {method:'POST',
        body:document.querySelector('#scene').toDataURL('image/png')});
      app.world.setPoseAngle(0);
      document.querySelector('#throat').value = '1.6';
      document.querySelector('#throat').dispatchEvent(new Event('input'));
      document.querySelector('#reset').click();
      aperturePhase = 1;
      return requestAnimationFrame(watch);
    }
    if (aperturePhase === 1 && app.world.simulatedSeconds >= 5) {
      near = await app.audit();
      document.querySelector('#throat').value = '1.5';
      document.querySelector('#throat').dispatchEvent(new Event('input'));
      document.querySelector('#reset').click();
      aperturePhase = 2;
      return requestAnimationFrame(watch);
    }
    if (aperturePhase === 2 && app.world.simulatedSeconds >= 5) {
      const blocked = await app.audit();
      document.querySelector('#throat').value = '8';
      document.querySelector('#throat').dispatchEvent(new Event('input'));
      document.querySelector('#reset').click();
      aperturePhase = 3;
      window.__hourglassBlocked = blocked;
      return requestAnimationFrame(watch);
    }
    if (aperturePhase === 3 && app.world.simulatedSeconds >= 2) {
      beforeTurn = await app.audit();
      document.querySelector('#play').click();
      app.world.setPoseAngle(.35);
      turned = await app.audit();
      document.querySelector('#play').click();
      aperturePhase = 4;
      return requestAnimationFrame(watch);
    }
    if (aperturePhase === 4 && app.world.simulatedSeconds >= 2.05) {
      afterTurn = await app.audit();
      const blocked = window.__hourglassBlocked;
      const survivors = beforeTurn.stream.map((mass,i) => ({mass,i}))
        .filter(({mass,i}) => mass > 0 && afterTurn.stream[i] > 0
          && beforeTurn.streamMotion[i][1] > .08
          && beforeTurn.streamMotion[i][1] < .22);
      const inertial = survivors.some(({i}) => {
        const [x,y,vx] = afterTurn.streamMotion[i];
        const localX = Math.cos(.35)*x + Math.sin(.35)*y;
        return Math.abs(x) < .003 && Math.abs(vx) < .003
          && localX > .02 && y > beforeTurn.streamMotion[i][1];
      });
      const firstSlope = slope(first), secondSlope = slope(second);
      const fps = frames / ((performance.now()-start)/1000);
      const passed = early.retainedFraction >= .9999
        && early.upperArea < .08 && early.streamArea > 0
        && first.retainedFraction >= .9999 && first.retainedFraction <= 1.0001
        && second.retainedFraction >= .9999 && second.retainedFraction <= 1.0001
        && third.retainedFraction >= .9999 && third.retainedFraction <= 1.0001
        && fourth.retainedFraction >= .9999 && fourth.retainedFraction <= 1.0001
        && Math.abs(first.pileProfileArea-first.pileArea)<1e-6
        && Math.abs(second.pileProfileArea-second.pileArea)<1e-6
        && Math.abs(third.pileProfileArea-third.pileArea)<1e-6
        && Math.abs(fourth.pileProfileArea-fourth.chamberAArea)<1e-6
        && Math.abs(firstSlope-32)<2 && Math.abs(secondSlope-35)<2
        && first.streamArea>0 && second.streamArea>0
        && buriedStreamArea(first)<1e-10 && buriedStreamArea(second)<1e-10
        && second.pileArea>first.pileArea && second.upperArea<first.upperArea
        && third.upperArea<1e-6 && third.streamArea<1e-5
        && third.pileArea>second.pileArea
        && texture.left > 4 && texture.right > 4
        && texture.left > texture.right * .35
        && fourth.direction < 0 && fourth.chamberBArea < third.chamberBArea
        && fourth.chamberAArea > 0 && fourth.streamArea > 0
        && near.upperArea < .079999 && near.streamArea > 0
        && near.retainedFraction >= .9999 && near.retainedFraction <= 1.0001
        && Math.abs(blocked.upperArea - .08) < 1e-7
        && blocked.streamArea === 0 && blocked.pileArea === 0
        && blocked.retainedFraction >= .9999 && blocked.retainedFraction <= 1.0001
        && beforeTurn.streamArea > 0 && inertial
        && Math.abs(turned.streamArea-beforeTurn.streamArea) < 1e-9
        && afterTurn.retainedFraction >= .9999
        && afterTurn.retainedFraction <= 1.0001
        && streamContinuity.maximumJump <= 10
        && streamContinuity.meanJump <= 2.2
        && streamContinuity.darkPixels === 0;
      return send({passed, fps, firstSlope, secondSlope,texture,
        streamContinuity,
        inertial, survivorCount:survivors.length,
        turnedStreamArea:turned.streamArea, afterTurnStreamArea:afterTurn.streamArea,
        early:{time:early.simulatedSeconds,upper:early.upperArea,
          stream:early.streamArea,retained:early.retainedFraction},
        first:{time:first.simulatedSeconds,upper:first.upperArea,
          stream:first.streamArea,pile:first.pileArea,
          pileProfile:first.pileProfileArea,retained:first.retainedFraction},
        second:{time:second.simulatedSeconds,upper:second.upperArea,
          stream:second.streamArea,pile:second.pileArea,
          pileProfile:second.pileProfileArea,retained:second.retainedFraction},
        third:{time:third.simulatedSeconds,upper:third.upperArea,
          stream:third.streamArea,pile:third.pileArea,
          pileProfile:third.pileProfileArea,retained:third.retainedFraction},
        fourth:{time:fourth.simulatedSeconds,chamberA:fourth.chamberAArea,
          chamberB:fourth.chamberBArea,stream:fourth.streamArea,
          pileProfile:fourth.pileProfileArea,retained:fourth.retainedFraction},
        near:{time:near.simulatedSeconds,upper:near.upperArea,
          stream:near.streamArea,remainder:near.releaseRemainder,
          retained:near.retainedFraction},
        blocked:{time:blocked.simulatedSeconds,upper:blocked.upperArea,
          stream:blocked.streamArea,remainder:blocked.releaseRemainder,
          retained:blocked.retainedFraction},
        error:passed?undefined:'GPU conservation, flow, or repose failed'});
    }
    requestAnimationFrame(watch);
  };
  requestAnimationFrame(watch);
})();
</script>`;

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (request.method === 'POST' && ['/page-result','/page-progress','/page-screenshot',
      '/page-screenshot-stream'].includes(url.pathname)) {
      let body = '';
      for await (const chunk of request) {
        body += chunk;
        if (body.length > (url.pathname.startsWith('/page-screenshot') ? 4000000 : 100000))
          throw new Error('Oversize probe');
      }
      response.end('ok');
      if (url.pathname.startsWith('/page-screenshot')) {
        if (body.startsWith('data:image/png;base64,'))
          await writeFile(path.join(work,url.pathname === '/page-screenshot-stream'
            ? 'hourglass-stream.png' : 'hourglass.png'),Buffer.from(body.slice(22),'base64'));
        return;
      }
      if (url.pathname === '/page-result') finish(JSON.parse(body));
      else console.log(body);
      return;
    }
    const route = url.pathname.endsWith('/') ? url.pathname + 'index.html' : url.pathname;
    const target = path.resolve(root, '.' + route);
    const relative = path.relative(root, target);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      response.writeHead(404).end(); return;
    }
    let bytes = await readFile(target);
    if (route === '/previews/0.6.0/live/hourglass/index.html')
      bytes = Buffer.from(bytes.toString().replace('</head>', `${probe}</head>`));
    response.setHeader('Content-Type', ({'.html':'text/html','.mjs':'text/javascript'})
      [path.extname(target)] || 'application/octet-stream');
    response.end(bytes);
  } catch (error) { response.writeHead(500).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const args = ['--headless=new', '--enable-gpu', '--no-first-run',
  '--no-default-browser-check', '--disable-background-timer-throttling',
  '--window-size=620,1000', `--user-data-dir=${path.join(work, 'chrome-profile')}`,
  `http://127.0.0.1:${server.address().port}/previews/0.6.0/live/hourglass/`];
const chrome = spawn(process.env.VKF_GPU_BROWSER
  || 'C:/Program Files/Google/Chrome/Application/chrome.exe', args, {windowsHide:true});
let stderr = '';
chrome.stdout.resume();
chrome.stderr.on('data', bytes => stderr = (stderr + bytes).slice(-12000));
chrome.on('error', error => finish({passed:false,error:String(error)}));
chrome.on('exit', code => finish({passed:false,error:`Chrome exited ${code}`}));
const timeout = setTimeout(() => finish({passed:false,
  error:'GPU hourglass did not finish within 120 seconds',stderr}), 120000);
try {
  const report = await result;
  await writeFile(path.join(work, 'result.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({...report, result:path.join(work,'result.json')}));
  if (!report.passed) process.exitCode = 1;
} finally {
  clearTimeout(timeout);
  chrome.kill();
  server.closeAllConnections();
  server.close();
}
