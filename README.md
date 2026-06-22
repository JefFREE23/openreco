# OpenReco: Open-Source Particle Track Reconstruction Framework

OpenReco is a minimal open-source charged-particle track reconstruction prototype.

The project started as a v0 single-track Kalman-filter core and has now reached a v1 event-level reconstruction prototype.

The current v1 reconstruction chain is:

```text
multi-particle event generation
→ mixed cylindrical detector hits
→ triplet seed building
→ greedy track finding
→ EKF-style track fitting
→ RTS-style smoothing
→ truth matching
→ efficiency, fake-rate, duplicate-rate, chi-square, covariance, and momentum validation
```

OpenReco is not intended to reproduce a full experiment framework yet. The goal is to build the smallest serious reconstruction chain that exposes the same core problems found in real high-energy physics tracking software:

```text
state parameterization
surface measurements
magnetic-field propagation
track seeding
track finding
track fitting
smoothing
holes and missing measurements
fake and duplicate tracks
validation against truth
```

The implementation is deliberately compact and Python-based so that the mathematics and reconstruction logic can be studied, debugged, and extended before moving toward realistic detector descriptions, ACTS GenericDetector samples, ODD samples, or CMS Open Data.

---

## Current status: v1 complete

OpenReco v1 currently includes:

```text
multi-particle event generation
simple cylindrical barrel detector
truth-labeled and noise hits
mixed detector measurements grouped by layer
triplet seed construction
hole-aware seed construction from multiple layer triplets
greedy track finding
shared-hit rejection
EKF fitting using the v0 cylindrical Kalman core
RTS-style smoothing
truth matching
fake-track counting
duplicate-track counting
hole counting
track quality cuts
covariance validity checks
momentum residual validation
performance scan CSV output
2D event visualization
```

Current v1 validation tests pass:

```text
29 passed
```

The v0 core test suite previously passed:

```text
197 passed
```

---

## v1 final validation

The final v1 validation was run with:

```text
events:                200
truth particles/event: 5
hit efficiency:        0.95
noise hits/layer:      1
minimum hits/track:    5
seed mode:             hole-aware
magnetic field:        uniform Bz
detector:              6 cylindrical barrel layers
```

Final validation result:

```text
tracking efficiency:    0.956
fake rate:              0.000
duplicate rate:         0.000
mean holes/track:       0.21
mean chi2/ndof:         0.992
covariance valid rate:  1.000
momentum residual mean: 0.0011
runtime/event:          2.0938 s
```

Interpretation:

```text
tracking efficiency is above the v1 target
fake rate is zero in this toy validation
duplicate rate is zero in this toy validation
covariance validity is stable
momentum residual mean is centered near zero
chi2/ndof is close to one
hole-aware tracking recovers most 5/6-hit tracks
```

The v1 chain is still a simplified reconstruction prototype. It does not include full ambiguity resolution, realistic detector material, multiple scattering, energy loss, detector misalignment, or a full combinatorial Kalman filter. Those are intentionally deferred.

---

## v1 reconstruction chain

OpenReco v1 turns the v0 single-track core into a small event reconstruction chain.

### 1. Event generation

The event generator creates multiple charged particles and propagates them through a simple cylindrical barrel detector.

Each generated hit stores:

```text
hit id
layer index
layer name
radius
phi
z
measurement covariance
truth particle id
noise flag
```

This allows reconstructed tracks to be compared directly to the generated truth particles.

---

### 2. Triplet seeding

The first v1 seed builder creates track candidates from three detector hits.

The strict mode uses the first three barrel layers:

```text
barrel_0, barrel_1, barrel_2
```

The hole-aware mode builds seeds from multiple three-layer combinations. This allows the reconstruction to recover tracks when one of the early layers is missing.

This follows the usual track-reconstruction idea that seeding provides a coarse first estimate of candidate trajectories before track following and fitting.

---

### 3. Track finding

Track finding starts from each triplet seed and greedily searches for compatible hits on the other detector layers.

The current track finder includes:

```text
simple phi(r), z(r) prediction model
chi-square compatibility test
shared-hit rejection
minimum hit requirement
hole counting
missing-layer bookkeeping
basic track quality cuts
```

This is intentionally not a full combinatorial Kalman filter yet. It is a minimal local track-following prototype.

---

### 4. EKF fitting and smoothing

Accepted v1 track candidates are passed to the existing v0 cylindrical EKF-style fitting core.

The fit performs:

```text
surface-to-surface prediction
numerical transport Jacobian
covariance propagation
2D local measurement update [phi, z]
chi-square calculation
RTS-style backward smoothing
covariance validation
momentum extraction
```

Tracks with invalid covariance, non-finite momentum, or extreme fitted chi-square are rejected by final quality cuts.

---

### 5. Truth matching and metrics

Each reconstructed track is matched to truth using the majority truth label among its hits.

A reconstructed track is treated as matched if at least half of its hits come from one truth particle.

The validation reports:

```text
tracking efficiency
fake rate
duplicate rate
mean hits per track
mean holes per track
mean chi2/ndof
covariance valid rate
momentum residual mean/std
runtime per event
```

---

## Repository structure

```text
openreco/
  __init__.py
  diagnostics.py
  event_generation.py
  field.py
  geometry.py
  kalman.py
  measurements.py
  particle_gun.py
  propagation.py
  seeding.py
  smoothing.py
  state.py
  track_finding.py
  track_fitting.py
  truth_matching.py
  visualization.py

examples/
  multi_event_validation.py
  multi_track_reconstruction.py
  single_track_straight_line.py
  single_track_uniform_B.py
  v1_performance_scan.py

tests/
  test_diagnostics.py
  test_end_to_end.py
  test_event_generation.py
  test_field.py
  test_geometry.py
  test_hole_aware_tracking.py
  test_kalman.py
  test_measurements.py
  test_multi_track_reconstruction.py
  test_particle_gun.py
  test_propagation.py
  test_seeding.py
  test_smoothing.py
  test_state.py
  test_track_finding.py
  test_track_fitting.py
  test_truth_matching.py
  test_v1_performance_scan.py
  test_visualization.py
```

---

## Quick start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the v1 multi-track demo:

```powershell
python examples/multi_track_reconstruction.py
```

Run the v1 performance scan:

```powershell
python examples/v1_performance_scan.py
```

Run the final v1-style validation configuration:

```powershell
python examples/v1_performance_scan.py --n-events 200 --particle-counts 5 --noise-hits-per-layer 1 --hit-efficiencies 0.95 --min-hits 5 --seed-mode hole-aware
```

Run the v1 tests:

```powershell
python -m pytest tests/test_event_generation.py tests/test_seeding.py tests/test_truth_matching.py tests/test_track_finding.py tests/test_multi_track_reconstruction.py tests/test_v1_performance_scan.py tests/test_track_fitting.py tests/test_hole_aware_tracking.py -q
```

Expected v1 result:

```text
29 passed
```

---

## v1 demo

Run:

```powershell
python examples/multi_track_reconstruction.py
```

The demo:

```text
creates a multi-particle event
generates truth-labeled cylindrical hits
adds optional noise hits
builds triplet seeds
finds reconstructed track candidates
fits tracks with the EKF wrapper
smooths the fitted tracks
matches reconstructed tracks to truth
prints efficiency, fake rate, duplicate rate, chi-square, covariance validity, and momentum residuals
saves an x-y event display
```

Example v1 demo result after EKF integration:

```text
truth particles:        5
measurements:           36
real measurements:      30
noise measurements:     6
seeds built:            216
reconstructed tracks:   5
matched tracks:         5
fake tracks:            0
duplicate tracks:       0

tracking efficiency:    1.000
fake rate:              0.000
duplicate rate:         0.000
mean chi2/ndof:         0.861
covariance valid rate:  1.000
momentum rel error:     mean=-0.0177, std=0.0556
plot saved:             docs/images/v1_multi_track_event.png
```

---

## v1 demo plot

![OpenReco v1 multi-track event](docs/images/v1_multi_track_event.png)

---

# v0 tracking core

OpenReco v0 is the single-track Kalman-filter core that v1 builds on.

The v0 goal was to validate the smallest serious tracking loop:

```text
particle gun
→ uniform magnetic field
→ cylindrical detector layers
→ smeared surface measurements
→ truth-assisted seed
→ bound-state EKF-style prediction/update
→ RTS-style smoothing
→ residuals, pulls, chi-square, momentum estimate, uncertainty estimate
```

OpenReco v0 uses a simple cylindrical tracker and a homogeneous magnetic field along the beam axis. The main purpose was to debug the tracking mathematics before adding event-level pattern recognition.

---

## v0 core components

OpenReco v0 includes:

```text
5D surface-bound track state
5×5 covariance matrix
cylindrical detector layers
particle gun event source
uniform Bz magnetic field
smeared cylindrical measurements [phi, z]
truth-assisted seed
bound-state EKF-style prediction/update
2D local measurement update [phi, z]
RTS-style backward smoothing
residuals
Kalman pulls using residual covariance
chi-square summaries
momentum estimate
momentum uncertainty estimate
covariance validity checks
multi-event validation
3D and x-y visualization
```

---

## Core design

### Track state

OpenReco uses a five-parameter bound state on a reference surface:

```text
[loc0, loc1, dir0, dir1, q_over_p]
```

For cylindrical detector layers, this becomes:

```text
[phi, z, alpha, tan_lambda, q_over_p]
```

where:

```text
phi        = angular coordinate on the cylinder
z          = longitudinal coordinate
alpha      = transverse momentum direction angle
tan_lambda = pz / pt
q_over_p   = charge / momentum
```

Each state carries a 5×5 covariance matrix.

This follows the tracking idea that states and measurements live on detector surfaces, while propagation can use a free/global representation internally.

---

### Detector model

The current detector is a simple barrel tracker made from cylindrical layers:

```text
r = 10, 20, 30, 40, 50, 60
```

Each cylindrical layer can hold a local measurement:

```text
[phi, z]
```

The geometry is intentionally small so that the Kalman filter, covariance behavior, and event-level reconstruction logic can be debugged before adding detector complexity.

---

### Magnetic field, propagation, and prediction model

OpenReco uses a homogeneous magnetic field along z:

```text
B = [0, 0, Bz]
```

For a cylindrical bound state on layer `k`,

```text
x_k = [phi, z, alpha, tan_lambda, q_over_p]
```

OpenReco first converts the bound state into a free particle-like representation:

```text
p  = 1 / |q_over_p|
pt = p / sqrt(1 + tan_lambda²)
pz = pt * tan_lambda
px = pt * cos(alpha)
py = pt * sin(alpha)
```

The particle is then propagated in the transverse plane using a minimal helix-like model in a uniform `Bz` field:

```text
kappa  = curvature_scale * q * Bz / pt
phi(s) = phi0 + kappa * s
```

where `s` is transverse path length.

For nonzero curvature, the transverse position is propagated as:

```text
x(s) = x0 + [sin(phi(s)) - sin(phi0)] / kappa
y(s) = y0 - [cos(phi(s)) - cos(phi0)] / kappa
z(s) = z0 + (pz / pt) * s
```

The next cylindrical layer intersection is found by scanning and bisection until:

```text
sqrt(x(s)^2 + y(s)^2) = R_layer
```

The propagated free state is then converted back into a bound state on the next cylindrical surface:

```text
[phi, z, alpha, tan_lambda, q_over_p]
```

The transport Jacobian `F_k` used for covariance propagation is computed numerically by central finite differences:

```text
C_k^- = F_k C_{k-1} F_k^T + Q_k
```

This is why the filter is described as EKF-style: the local measurement update is linear in the chosen bound coordinates, but the surface-to-surface prediction model is nonlinear.

The current unit convention is simplified and toy-consistent. More realistic HEP unit handling is future work.

---

### Measurements

Cylindrical hits are generated by smearing the truth intersection with Gaussian noise:

```text
measurement = [phi, z]
covariance  = diag([sigma_phi², sigma_z²])
```

The Kalman update uses both local coordinates, `phi` and `z`.

---

### Kalman filter

The EKF-style loop performs:

```text
prediction to next cylindrical layer
covariance propagation using numerical transport Jacobian
2D local measurement update with [phi, z]
chi-square calculation
residual covariance calculation
```

The measurement update is linear in the selected cylindrical bound coordinates:

```text
h(x) = [phi, z]
```

The update uses the local measurement matrix:

```text
H =
[1 0 0 0 0
 0 1 0 0 0]
```

because the bound state is:

```text
[phi, z, alpha, tan_lambda, q_over_p]
```

The nonlinear part is the propagation between cylindrical surfaces. The transport Jacobian is computed numerically and used in the covariance prediction.

So “EKF-style” here means:

```text
nonlinear surface-to-surface prediction
linear local measurement update
numerical transport Jacobian
```

---

### Smoothing

OpenReco includes an RTS-style backward smoother.

For filtered state `k` and predicted state `k+1`, the smoother uses:

```text
A_k = C_k^f F_{k+1}ᵀ (C_{k+1}^-)⁻¹
```

and computes smoothed states and covariances by walking backward through the track.

The final smoothed state is equal to the final filtered state, which is expected because there is no later measurement after the last layer.

---

## v0 single-track demo

Run:

```powershell
python examples/single_track_uniform_B.py
```

This demo:

```text
creates a cylindrical detector
creates a uniform Bz field
generates one charged particle
propagates truth to cylindrical layers
creates smeared [phi, z] measurements
builds a truth-assisted seed
runs bound-state EKF-style filtering
runs backward smoothing
prints residuals, pulls, chi-square, covariance validity
prints momentum estimate and uncertainty
shows a 3D event display
shows an x-y top view
```

Example output:

```text
OpenReco v0 uniform-B cylindrical demo
--------------------------------------
Number of cylindrical layers: 6
Layer radii: [10. 20. 30. 40. 50. 60.]

Momentum estimate from final smoothed state:
  truth p  = 2.8395
  fitted p = 2.9712 ± 0.1582
  abs err  = 0.1317
  rel err  = 0.0464

Fit quality:
  total chi2       = 4.4334
  n updates        = 6
  pull mean [φ,z]  = [-0.1866, -0.3720]
  pull std  [φ,z]  = [0.2134, 0.7220]
  covariance valid = True
```

---

## v0 demo plots

### 3D cylindrical barrel view

![OpenReco 3D cylindrical barrel view](docs/images/openreco_3d_event.png)

### x-y top view of magnetic bending

![OpenReco x-y top view](docs/images/openreco_xy_view.png)

The single-event pull values are only a smoke test. Pull distributions need many events.

---

## v0 multi-event validation

Run:

```powershell
python examples/multi_event_validation.py
```

Current 200-event v0 validation result:

```text
events requested:       200
events successful:      200
success rate:           1.0000
covariance valid rate:  1.0000

Kalman pull summary:
  phi mean:              0.0041
  phi std:               0.6263
  z mean:               -0.0234
  z std:                 0.8287

Momentum error summary:
  abs error mean:        0.0074
  abs error std:         0.0882
  rel error mean:        0.0026
  rel error std:         0.0310
```

Interpretation:

```text
success rate is good
covariance validity is good
pull means are close to zero
pull widths are clearly below 1
momentum error is small
```

The pull widths below 1 are a known v0 calibration issue, especially for `phi`.

This suggests that the current residual covariance is conservative or not perfectly calibrated. Possible causes include:

```text
measurement noise is too large relative to the generated residuals
process noise is too large
seed covariance is too conservative
transport/covariance propagation is overestimating uncertainty
truth-assisted seeding makes the fit easier than a real seeded track
the toy setup has no material, no scattering, no misalignment
```

This is not a reconstruction failure. It is exactly the kind of issue that pull validation is supposed to reveal.

---

## Running tests

Run the v1 validation tests:

```powershell
python -m pytest tests/test_event_generation.py tests/test_seeding.py tests/test_truth_matching.py tests/test_track_finding.py tests/test_multi_track_reconstruction.py tests/test_v1_performance_scan.py tests/test_track_fitting.py tests/test_hole_aware_tracking.py -q
```

Current result:

```text
29 passed
```

Run the v0 core test suite:

```powershell
python -m pytest tests/test_state.py tests/test_geometry.py tests/test_measurements.py tests/test_field.py tests/test_particle_gun.py tests/test_propagation.py tests/test_kalman.py tests/test_diagnostics.py tests/test_visualization.py tests/test_smoothing.py tests/test_end_to_end.py
```

Expected v0 result:

```text
197 passed
```

---

## What is intentionally deferred

OpenReco v1 intentionally does not include:

```text
vertexing
full ambiguity resolution
full combinatorial Kalman filter
realistic detector material
multiple scattering model
energy loss model
misalignment and alignment corrections
non-Gaussian electron fitting
hadronic physics lists
advanced event generation
Pythia8
Geant4 detector simulation
DD4hep or TGeo geometry import
CMS I/O
ACTS GenericDetector validation
ODD validation
CMS Open Data validation
GPU acceleration
machine-learning-based tracking
```

These are deferred until the local tracking and event-reconstruction core is stable.

---

## Roadmap after v1

Recommended next steps:

```text
1. Improve uncertainty calibration so pull widths approach 1.
2. Add a more realistic ambiguity-resolution stage.
3. Add configurable material/process-noise studies.
4. Add multiple-scattering and energy-loss effects.
5. Compare with ACTS GenericDetector truth samples.
6. Move to ODD full-chain samples.
7. Later validate against CMS Open Data.
8. Eventually explore CKF-style branching, GPU acceleration, and ML-based tracking.
```

The next serious external validation target should be ACTS GenericDetector particle-gun or truth-tracking outputs, not CMS data immediately.

---

## What it is

OpenReco v1 is a minimal event-level tracking prototype. It is not a full detector framework.

The current implementation is useful because it already contains a compact but complete reconstruction loop:

```text
multi-particle generation
surface-bound state
surface measurements
triplet seeding
track following
prediction
update
smoothing
truth matching
holes
fake-rate validation
duplicate-rate validation
momentum estimate
uncertainty estimate
multi-event performance scan
```

The main remaining limitations are calibration, material realism, ambiguity resolution, and validation against external truth data.

---

## References and Inspiration

OpenReco is inspired by standard charged-particle track reconstruction theory and modern tracking software architecture.

* R. Frühwirth, *Track and Vertex Fitting*
  Theory reference for track states, covariance matrices, measurement errors, Kalman filtering, smoothing, residuals, pulls, and uncertainty validation.
  https://cds.cern.ch/record/340476/files/p217.pdf

* ACTS Collaboration, *A Common Tracking Software Project*
  Architecture reference for surface-based geometry, bound track parameters, propagation, triplet seeding, track fitting, hole handling, truth matching, fake-rate validation, duplicate-rate validation, and scalable event reconstruction workflows.
  https://doi.org/10.1007/s41781-021-00078-8

* ACTS GitHub Repository
  Open-source tracking software project used as an architectural reference.
  https://github.com/acts-project/acts
