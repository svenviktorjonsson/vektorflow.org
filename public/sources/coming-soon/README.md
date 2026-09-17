# Compiled Coming Soon applications

These source files compile on the `pre-gen` branch of vektor-flow. Each live
application loads the resulting WASM retained World program and its emitted GPU
laws. The tabbed files are executable inputs, not illustrative API sketches.

Build each application's `main.vkf` with `vkf -b main.vkf` (or the branch's
`vkf-strict -b main.vkf`). Relative imports resolve the other source files.
For mechanical previews, `scripts/stage-stones-preview.mjs <compiler-root> tree`
(or `stones`) snapshots all three VKF files, recompiles `main.vkf`, then stages
those exact sources and the resulting WASM with compiler/emitter/typed-IR hashes.
Source changes during compilation reject publication. The read-only Prism tabs
validate source hashes. Run validates source, WASM and manifest before restarting
the corresponding compiled application; it never executes editable browser text.
The wheel retains its older published artifact and has no new compile receipt.

- `wheel`: `main.vkf`, `geometry.vkf`, `materials.vkf`, `particles.vkf`.
  A one-metre wheel and seven baffles are added geometry. Water and sand have
  independent retained Worlds and material/contact laws; the Display flips Views.
- `stones`: `main.vkf`, `geometry.vkf`, `materials.vkf`. Five irregular solids
  use an immutable initial-geometry asset, GPU support contacts and friction.
  Dragging prescribes a held position; releasing restores dynamic gravity.
- `tree`: `main.vkf`, `geometry.vkf`, `materials.vkf`. An eight-metre cached
  generated tree, a dense lawn, elastic nodes and 64,000 wind parcels share a
  World. Air density determines parcel mass. Wood density, elastic modulus and
  representative branch dimensions determine a damped cantilever-mode field.
  The sun is an added emissive sphere; illumination and shadows use its position.
  One finger orbits and two fingers zoom. This is a
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

Wheel input has no speed clamp. Analytic swept baffles and particle paths replace
endpoint-only contact testing; static sand friction includes positional arrest.
Sand now uses 14,120 non-overlapping initial grains, versus the former 3,849.
The wood-field model uses equivalent local modes, not the generated branches'
exact beam topology. Its isolated elastic response is numerically verified
against analytical equilibrium, not laboratory-validated full-tree aerodynamics.
See [branch cantilever research](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.00059/full)
and [wind-induced tree response](https://www.mdpi.com/2073-4433/14/6/1010).

The wheel's frozen-motion regression remains unresolved; its older code and
artifact are preserved, not rebuilt against an unverified contact candidate.
Tree leaf/branch reduced modes now exchange momentum in a coupled implicit solve.
Blade-cell impacts route to their attachment owner, and denser stratified air
sampling preserves density. GPU impulse/reaction checks pass; this does not claim
individual branch-beam topology or resolved aerodynamic accuracy. Phone behaviour
and performance still need device testing.
