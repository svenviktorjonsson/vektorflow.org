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
  if (formation !== 'bed' && formation !== 'dropcastle')
    throw new RangeError('Unknown wheel sand formation');
  const spacing = radius * 2.004;
  const pitch = spacing * Math.sqrt(3) / 2;
  const targetCount = Math.round(targetArea / (Math.PI * radius * radius));
  const cosine = Math.cos(wheelAngle), sine = Math.sin(wheelAngle);
  const positions = [];
  for (let row = 0, y = -0.1;
    y < (formation === 'dropcastle' ? 0.42 : 0.32) && positions.length < targetCount;
    row++, y += pitch) {
    const candidates = [];
    for (let x = -0.49 + (row % 2) * spacing / 2; x <= 0.49; x += spacing) {
      if (formation === 'dropcastle' && y > 0.10) {
        const halfWidth = 0.145 - 0.05 * Math.min(1, (y - 0.10) / 0.32);
        if (Math.abs(x + 0.18) > halfWidth) continue;
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
    // A partially filled last row should form a central mound, not a ledge
    // filled left-to-right. Deterministic ordering keeps resets reproducible.
    candidates.sort((a, b) => Math.abs(a[0]) - Math.abs(b[0]));
    positions.push(...candidates.slice(0, targetCount - positions.length));
  }
  if (positions.length !== targetCount)
    throw new RangeError('Chosen grain size cannot fit the conserved sand area');
  return Object.freeze({positions, representedArea: targetCount * Math.PI * radius ** 2,
    relativeAreaError: targetCount * Math.PI * radius ** 2 / targetArea - 1});
}
