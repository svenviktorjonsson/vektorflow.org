# Compiled Coming Soon applications

These source files compile on the `pre-gen` branch of vektor-flow. Each live
application loads the resulting WASM retained World program and its emitted GPU
laws. The tabbed files are executable inputs, not illustrative API sketches.

Build each application's `main.vkf` with `vkf -b main.vkf` (or the branch's
`vkf-strict -b main.vkf`). Relative imports resolve the other source files.
The site's `scripts/stage-compiled-previews.mjs` stages those build artifacts and
the runtime's transitive module dependencies together.

- `wheel`: `main.vkf`, `geometry.vkf`, `materials.vkf`, `particles.vkf`.
  A one-metre wheel and seven baffles are added geometry. Water and sand have
  independent retained Worlds and material/contact laws; the Display flips Views.
- `stones`: `main.vkf`, `geometry.vkf`, `materials.vkf`. Five irregular solids
  use an immutable initial-geometry asset, GPU support contacts and friction.
  Dragging prescribes a held position; releasing restores dynamic gravity.
- `tree`: `main.vkf`, `geometry.vkf`, `materials.vkf`. An eight-metre cached
  generated tree, a dense lawn, elastic nodes and 8,192 wind parcels share a
  World. Impacts transfer momentum into a damped local spring field. This is a
  reduced-order elastic model, not a full aerodynamic or branch finite-element
  solver. Grass and branches read its displacement rather than animated gusts.

The asset URLs in the material files are initial data. They can be downloaded
from this site. Stone geometry is produced by `scripts/precompute-rigid-stones.mjs`;
tree variants are the cached outputs of seeded branch/leaf distributions. Assets
do not prescribe time-dependent motion. Distances use metres, time uses seconds.

Untimed `add` data is initial state. Named records provide properties; properties
without applicable World laws remain tags. Raw particle mode shows simulation
data only; material effects belong to the embedding. All interaction canvases
capture touch gestures and disable page scrolling within their bounds. Off-screen
and background applications suspend work independently of the Play/Pause switch.
