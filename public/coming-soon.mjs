const scenarioModules = Object.freeze({
  'stones-water': './vf-ui/vf-coming-soon-stones-water.mjs',
  'tree-wind': './vf-ui/vf-coming-soon-tree-wind.mjs',
  'sand-drum': './vf-ui/vf-coming-soon-sand-drum.mjs',
});

const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;

function pointerSample(canvas, event, phase) {
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(bounds.width, 1);
  const height = Math.max(bounds.height, 1);
  const x = finite(event.clientX) - bounds.left;
  const y = finite(event.clientY) - bounds.top;
  return Object.freeze({
    phase,
    pointerId: event.pointerId,
    pointerType: event.pointerType || 'mouse',
    buttons: event.buttons,
    pressure: finite(event.pressure),
    x,
    y,
    normalizedX: x / width,
    normalizedY: y / height,
    width,
    height,
    timeStamp: finite(event.timeStamp),
  });
}

function attachCapturedPointer(canvas, deliver) {
  let activePointer = null;
  const send = (event, phase) => {
    event.preventDefault();
    deliver(pointerSample(canvas, event, phase));
  };
  const down = (event) => {
    if (activePointer !== null) return;
    activePointer = event.pointerId;
    canvas.setPointerCapture?.(event.pointerId);
    send(event, 'start');
  };
  const move = (event) => {
    if (event.pointerId === activePointer) send(event, 'move');
  };
  const finish = (event, phase) => {
    if (event.pointerId !== activePointer) return;
    send(event, phase);
    if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    activePointer = null;
  };
  canvas.addEventListener('pointerdown', down, { passive: false });
  canvas.addEventListener('pointermove', move, { passive: false });
  canvas.addEventListener('pointerup', (event) => finish(event, 'end'), { passive: false });
  canvas.addEventListener('pointercancel', (event) => finish(event, 'cancel'), { passive: false });
  canvas.addEventListener('lostpointercapture', (event) => {
    if (event.pointerId !== activePointer) return;
    deliver(pointerSample(canvas, event, 'cancel'));
    activePointer = null;
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    deliver(Object.freeze({
      ...pointerSample(canvas, event, 'wheel'),
      deltaX: finite(event.deltaX),
      deltaY: finite(event.deltaY),
      deltaMode: event.deltaMode,
    }));
  }, { passive: false });
}

function validateScenario(kind, scenario) {
  if (!scenario || typeof scenario !== 'object') {
    throw new TypeError(`${kind} did not return a live scenario`);
  }
  for (const method of ['setActive', 'reset', 'pointer', 'dispose']) {
    if (typeof scenario[method] !== 'function') {
      throw new TypeError(`${kind} live scenario is missing ${method}()`);
    }
  }
  if (!scenario.authoritativeState || scenario.renderState !== scenario.authoritativeState) {
    throw new TypeError(`${kind} must render the same authoritative physics state it advances`);
  }
  if (scenario.execution !== 'gpu-realtime') {
    throw new TypeError(`${kind} must report measured real-time GPU execution`);
  }
  return scenario;
}

async function connectShowcase(root) {
  const kind = root.dataset.vkfLiveShowcase;
  const modulePath = scenarioModules[kind];
  if (!modulePath) throw new Error(`Unknown VKF showcase ${kind}`);
  const canvas = root.querySelector('[data-vkf-canvas]');
  const status = root.querySelector('[data-vkf-status]');
  const play = root.querySelector('[data-vkf-action="play"]');
  const reset = root.querySelector('[data-vkf-action="reset"]');
  let scenario = null;
  let userRunning = true;
  let inViewport = false;

  const setStatus = (message) => { status.textContent = String(message); };
  const reconcile = () => scenario?.setActive(userRunning && inViewport
    && document.visibilityState !== 'hidden');
  const fail = (error) => {
    root.dataset.vkfError = '';
    root.setAttribute('aria-busy', 'false');
    setStatus(`Live VKF scene unavailable: ${error?.message || error}`);
  };

  play.addEventListener('click', () => {
    userRunning = !userRunning;
    play.textContent = userRunning ? 'Pause' : 'Play';
    play.setAttribute('aria-pressed', String(userRunning));
    reconcile();
  });
  reset.addEventListener('click', () => {
    scenario?.reset();
    setStatus('Deterministic state reset');
  });
  document.addEventListener('visibilitychange', reconcile);

  const observer = new IntersectionObserver((entries) => {
    inViewport = entries.some((entry) => entry.isIntersecting);
    reconcile();
  }, { rootMargin: '160px 0px', threshold: 0.04 });
  observer.observe(root);

  try {
    const implementation = await import(modulePath);
    if (typeof implementation.createComingSoonScenario !== 'function') {
      throw new TypeError(`${kind} module does not export createComingSoonScenario()`);
    }
    scenario = validateScenario(kind, await implementation.createComingSoonScenario({
      canvas,
      onStatus: setStatus,
    }));
    attachCapturedPointer(canvas, (sample) => scenario.pointer(sample));

    if (kind === 'tree-wind') {
      const seedOutput = root.querySelector('[data-vkf-seed]');
      root.querySelector('[data-vkf-action="regenerate"]').addEventListener('click', () => {
        const seed = scenario.regenerate();
        seedOutput.textContent = `Seed ${seed}`;
      });
    }

    for (const input of root.querySelectorAll('[data-vkf-parameter]')) {
      const output = input.parentElement.querySelector('output');
      input.addEventListener('input', () => {
        const value = Number(input.value);
        output.value = value.toFixed(2).replace(/\.00$/u, '');
        scenario.setParameter(input.dataset.vkfParameter, value);
      });
    }

    if (kind === 'sand-drum') {
      const viewButtons = [...root.querySelectorAll('[data-vkf-view]')];
      for (const button of viewButtons) button.addEventListener('click', () => {
        for (const candidate of viewButtons) candidate.setAttribute('aria-pressed',
          String(candidate === button));
        scenario.setView(button.dataset.vkfView);
      });
    }

    root.setAttribute('aria-busy', 'false');
    reconcile();
  } catch (error) {
    fail(error);
  }

  return () => {
    observer.disconnect();
    scenario?.dispose();
  };
}

for (const root of document.querySelectorAll('[data-vkf-live-showcase]')) {
  connectShowcase(root);
}
