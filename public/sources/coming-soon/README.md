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
The wheel is also staged by that script with the `wheel` argument. Its four
displayed files are recompiled, with the same source/executable build receipt.

- `wheel`: `main.vkf`, `geometry.vkf`, `materials.vkf`, `particles.vkf`.
  A one-metre wheel, seven baffles, water, and sand are physical layers of one
  retained World. Their constitutive laws differ; the Display flips Views.
  Status text is a nonphysical `View.add` annotation drawn in the same canvas.
  `space:"pixel"` uses CSS pixels from the top-left, `space:"relative"` uses
  top-left viewport fractions, and `space:"data"` uses View-projected coordinates.
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

The water wheel uses matched rotating-reference contact trajectories and input
timing. Its rebuilt WASM passed 60 settling and 60 rotation frames on a physical
Intel Gen9 GPU, with full World-time progress and no detected overlaps. This
is a functional acceptance run, not a 10-ms performance result: the measured
complete-update maximum was 118.29248 ms. The later request prioritised working
motion rather than further optimisation. Sand's separate contact/friction
solver still has a known frozen-motion regression; it is not declared fixed.
Tree leaf/branch reduced modes now exchange momentum in a coupled implicit solve.
Blade-cell impacts route to their attachment owner, and denser stratified air
sampling preserves density. GPU impulse/reaction checks pass; this does not claim
individual branch-beam topology or resolved aerodynamic accuracy. Phone behaviour
and performance still need device testing.
