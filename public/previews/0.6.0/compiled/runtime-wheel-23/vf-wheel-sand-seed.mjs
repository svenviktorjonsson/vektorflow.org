// Rebuild the same amount of sand at a different computational grain scale.
// This is initial data only; the shared GPU World owns all subsequent motion.
const BAFFLES = Object.freeze([
  [.5, 0, .35, 0], [0, .5, 0, .35], [-.5, 0, -.35, 0],
  [0, -.5, 0, -.35], [-.22, .175, -.075, .135],
  [.075, -.055, .215, -.115], [-.105, -.25, .02, -.155],
]);

export function wheelSandSeed(radius, targetArea, wheelAngle = 0.22,
  formation = 'bed') {
  if (!Number.isFinite(radius) || radius < 0.0055 || radius > 0.01)
    throw new RangeError('Wheel sand radius must be 5.5–10 mm');
  if (!Number.isFinite(targetArea) || targetArea <= 0)
    throw new RangeError('Wheel sand target area must be positive');
  if (formation !== 'bed' && formation !== 'dropcastle' && formation !== 'rain')
    throw new RangeError('Unknown wheel sand formation');
  const spacing = radius * 2.004;
  const pitch = spacing * Math.sqrt(3) / 2;
  const targetCount = Math.round(targetArea / (Math.PI * radius * radius));
  const cosine = Math.cos(wheelAngle), sine = Math.sin(wheelAngle);
  const positions = [];
  for (let row = 0, y = formation === 'rain' ? 0.32 : -0.1;
    y < (formation === 'dropcastle' ? 0.75 : formation === 'rain' ? 0.82 : 0.32)
      && (formation === 'rain' || positions.length < targetCount);
    row++, y += pitch) {
    const candidates = [];
    for (let x = -0.49 + (row % 2) * spacing / 2; x <= 0.49; x += spacing) {
      if (formation === 'dropcastle') {
        const halfWidth = 0.18 - 0.055 * Math.min(1, (y + 0.1) / 0.85);
        if (Math.abs(x - 0.25) > halfWidth) continue;
      }
      const localY = y - 0.32;
      if (Math.hypot(x, localY) > 0.5 - 0.006 - radius - 1e-6) continue;
      const u = cosine * x + sine * localY;
      const v = -sine * x + cosine * localY;
      if (BAFFLES.some(([ax, ay, bx, by]) => {
        const dx = bx - ax, dy = by - ay;
        const t = Math.max(0, Math.min(1,
          ((u - ax) * dx + (v - ay) * dy) / (dx * dx + dy * dy)));
        return Math.hypot(u - ax - dx * t, v - ay - dy * t)
          < 0.006 + radius + 1e-6;
      })) continue;
      candidates.push([x, y]);
    }
    // Complete the last row about the active formation's centre rather than
    // leaving an artificial side ledge.
    const centerX = formation === 'dropcastle' ? 0.25 : 0;
    candidates.sort((a, b) => Math.abs(a[0] - centerX) - Math.abs(b[0] - centerX));
    positions.push(...(formation === 'rain' ? candidates
      : candidates.slice(0, targetCount - positions.length)));
  }
  if (formation === 'rain') {
    const rank = ([x, y]) => {
      let bits = Math.imul(Math.round((x + 1) / spacing), 0x9e3779b1)
        ^ Math.imul(Math.round((y + 1) / pitch), 0x85ebca77);
      bits ^= bits >>> 16;
      bits = Math.imul(bits, 0x7feb352d);
      return (bits ^ (bits >>> 15)) >>> 0;
    };
    positions.sort((a, b) => rank(a) - rank(b));
    positions.length = Math.min(positions.length, targetCount);
  }
  if (positions.length !== targetCount)
    throw new RangeError('Chosen grain size cannot fit the conserved sand area');
  return Object.freeze({positions, representedArea: targetCount * Math.PI * radius ** 2,
    relativeAreaError: targetCount * Math.PI * radius ** 2 / targetArea - 1});
}
