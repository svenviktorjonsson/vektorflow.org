// UI formation data only. Contact geometry comes from compiled authored Layers;
// the GPU Granular Law owns every subsequent physical update.
export function granularFormationSeed(radius, targetArea, geometry,
  formation = 'bed') {
  if (!Number.isFinite(radius) || radius <= 0)
    throw new RangeError('Granular guide radius must be positive');
  if (!Number.isFinite(targetArea) || targetArea <= 0)
    throw new RangeError('Granular target area must be positive');
  if (formation !== 'bed' && formation !== 'dropcastle' && formation !== 'rain')
    throw new RangeError('Unknown granular formation');
  const center = geometry?.center, chamberRadius = geometry?.radius;
  const halfWidth = geometry?.half_width, angle = geometry?.rotation;
  const segments = geometry?.segments;
  if (!Array.isArray(center) || center.length !== 2 ||
      !center.every(Number.isFinite) || !Number.isFinite(chamberRadius) ||
      chamberRadius <= 0 || !Number.isFinite(halfWidth) || halfWidth < 0 ||
      !Number.isFinite(angle) || !Array.isArray(segments))
    throw new TypeError('Granular formation requires compiled circle and segment layers');
  const [cx, cy] = center;
  const spacing = radius * 2.004;
  const pitch = spacing * Math.sqrt(3) / 2;
  const targetCount = Math.round(targetArea / (Math.PI * radius * radius));
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const bottom = cy - chamberRadius * 0.84;
  const top = cy + chamberRadius;
  const positions = [];
  for (let row = 0, y = formation === 'rain' ? cy : bottom;
    y < (formation === 'dropcastle' ? cy + chamberRadius * 0.86
      : formation === 'rain' ? top : cy)
      && (formation === 'rain' || positions.length < targetCount);
    row++, y += pitch) {
    const candidates = [];
    for (let x = cx - chamberRadius + (row % 2) * spacing / 2;
      x <= cx + chamberRadius; x += spacing) {
      if (formation === 'dropcastle') {
        const half = chamberRadius * (0.36 - 0.11 * Math.min(1,
          (y - bottom) / (chamberRadius * 1.7)));
        if (Math.abs(x - (cx + chamberRadius * 0.5)) > half) continue;
      }
      if (formation === 'rain') {
        // A pre-existing falling cloud widens below its release point.
        // The Embedding adds only sideways dispersion to actual free fall.
        const half = radius * 2 + (top - y) * 1.1;
        if (Math.abs(x - cx) > half) continue;
      }
      const dx = x - cx, dy = y - cy;
      if (Math.hypot(dx, dy) > chamberRadius - halfWidth - radius - 1e-6)
        continue;
      const u = cosine * dx + sine * dy;
      const v = -sine * dx + cosine * dy;
      if (segments.some(([ax, ay, bx, by]) => {
        const ex = bx - ax, ey = by - ay;
        const t = Math.max(0, Math.min(1,
          ((u - ax) * ex + (v - ay) * ey) / (ex * ex + ey * ey)));
        return Math.hypot(u - ax - ex * t, v - ay - ey * t)
          < halfWidth + radius + 1e-6;
      })) continue;
      candidates.push([x, y]);
    }
    const centerX = formation === 'dropcastle' ? cx + chamberRadius * 0.5 : cx;
    candidates.sort((a, b) => Math.abs(a[0] - centerX) - Math.abs(b[0] - centerX));
    positions.push(...(formation === 'rain' ? candidates
      : candidates.slice(0, targetCount - positions.length)));
  }
  if (formation === 'rain') {
    const rank = ([x, y]) => {
      let bits = Math.imul(Math.round((x - cx + chamberRadius) / spacing), 0x9e3779b1)
        ^ Math.imul(Math.round((y - bottom) / pitch), 0x85ebca77);
      bits ^= bits >>> 16;
      bits = Math.imul(bits, 0x7feb352d);
      return (bits ^ (bits >>> 15)) >>> 0;
    };
    positions.sort((a, b) => rank(a) - rank(b));
    positions.length = Math.min(positions.length, targetCount);
  }
  if (positions.length !== targetCount)
    throw new RangeError('Chosen guide size and formation cannot fit conserved area');
  return Object.freeze({positions, representedArea: targetCount * Math.PI * radius ** 2,
    relativeAreaError: targetCount * Math.PI * radius ** 2 / targetArea - 1});
}
