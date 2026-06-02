# OpenReco: Open-Source Detector Reconstruction Framework

OpenReco is a minimal charged-particle track reconstruction prototype.

The current v0 goal is not to reproduce a full experiment framework. It is to build the smallest serious tracking loop:

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

OpenReco v0 uses a simple cylindrical tracker and a homogeneous magnetic field along the beam axis. The main purpose is to debug the tracking mathematics before moving to more realistic detector descriptions, ACTS GenericDetector truth samples, ODD samples, or CMS Open Data.

---

## Current v0 status

OpenReco v0 currently includes:

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

The current full test suite passes:

```text
197 passed
```

---

## Repository structure

```text
openreco/
  state.py
  geometry.py
  field.py
  particle_gun.py
  propagation.py
  measurements.py
  kalman.py
  smoothing.py
  diagnostics.py
  visualization.py

examples/
  single_track_straight_line.py
  single_track_uniform_B.py
  multi_event_validation.py

tests/
  test_state.py
  test_geometry.py
  test_measurements.py
  test_field.py
  test_particle_gun.py
  test_propagation.py
  test_kalman.py
  test_diagnostics.py
  test_visualization.py
  test_smoothing.py
  test_end_to_end.py
```

---

## Core design

### Track state

OpenReco v0 uses a five-parameter bound state on a reference surface:

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

The v0 detector is a simple barrel tracker made from cylindrical layers:

```text
r = 10, 20, 30, 40, 50, 60
```

Each cylindrical layer can hold a local measurement:

```text
[phi, z]
```

The geometry is intentionally small so that the Kalman filter and covariance behavior can be debugged before adding detector complexity.

---

### Magnetic field, propagation, and prediction model

OpenReco v0 uses a homogeneous magnetic field along z:

```text
B = [0, 0, Bz]
```

The Kalman prediction step is based on a simple surface-to-surface propagation model.

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

The propagation tests verify that:

```text
charge sign flips bending direction
Bz sign flips bending direction
larger Bz bends more
lower pt bends more than higher pt
momentum magnitude is conserved
pz is conserved
propagation reaches the requested cylinder radii
```

---

### Measurements

Cylindrical hits are generated by smearing the truth intersection with Gaussian noise:

```text
measurement = [phi, z]
covariance  = diag([sigma_phi², sigma_z²])
```

The Kalman update uses both local coordinates, `phi` and `z`.

---

### Seeding

The current v0 example uses a truth-assisted seed on the first cylindrical layer.

This is intentional. The first goal is to validate the tracking core. Real triplet seeding comes later.

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

OpenReco v0 includes an RTS-style backward smoother.

For filtered state `k` and predicted state `k+1`, the smoother uses:

```text
A_k = C_k^f F_{k+1}ᵀ (C_{k+1}^-)⁻¹
```

and computes smoothed states and covariances by walking backward through the track.

The final smoothed state is equal to the final filtered state, which is expected because there is no later measurement after the last layer.

---

## Main demo

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

The single-event pull values are only a smoke test. Pull distributions need many events.

---

## Multi-event validation

Run:

```powershell
python examples/multi_event_validation.py
```

Current 200-event validation result:

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

A future calibration step should tune the measurement noise, process noise, and seed covariance so that pull widths approach 1 in a statistically meaningful multi-event sample.

---

## Running tests

Run the full test suite:

```powershell
python -m pytest tests/test_state.py tests/test_geometry.py tests/test_measurements.py tests/test_field.py tests/test_particle_gun.py tests/test_propagation.py tests/test_kalman.py tests/test_diagnostics.py tests/test_visualization.py tests/test_smoothing.py tests/test_end_to_end.py
```

Current result:

```text
197 passed
```

---

## What is intentionally deferred

OpenReco v0 intentionally does not include:

```text
vertexing
full ambiguity resolution
realistic detector material
multiple scattering model
energy loss model
non-Gaussian electron fitting
hadronic physics lists
advanced event generation
Pythia8
Geant4 detector simulation
CMS I/O
ACTS GenericDetector validation
ODD validation
CMS Open Data validation
```

These are deferred until the local Kalman tracking core is stable.

---

## Roadmap after v0

Recommended next steps:

```text
1. Improve uncertainty calibration so pull widths approach 1.
2. Add simple triplet seeding instead of truth-assisted seeding.
3. Add material/process-noise studies.
4. Compare with ACTS GenericDetector truth samples.
5. Move to ODD full-chain samples.
6. Later validate against CMS Open Data.
```

The next serious external validation target should be ACTS GenericDetector particle-gun or truth-tracking outputs, not CMS data immediately.

---

## What it is:

OpenReco v0 is a minimal local tracking core. It is not a full detector framework.

The current implementation is useful because it already contains the smallest complete reconstruction loop:

```text
surface-bound state
surface measurements
prediction
update
smoothing
residuals
pulls
momentum estimate
uncertainty estimate
multi-event validation
```

The main remaining limitation is not whether the loop exists. It does. The next challenge is calibration and validation against external truth data.


## References and Inspiration

OpenReco v0 is inspired by standard charged-particle track reconstruction theory and modern tracking software architecture.

- R. Frühwirth, *Track and Vertex Fitting*  
  Theory reference for track states, covariance matrices, measurement errors, Kalman filtering, smoothing, residuals, pulls, and uncertainty validation.  
  https://cds.cern.ch/record/340476/files/p217.pdf

- ACTS Collaboration, *A Common Tracking Software Project*  
  Architecture reference for surface-based geometry, bound track parameters, propagation, seeding, track fitting, and validation workflows.  
  https://doi.org/10.1007/s41781-021-00078-8

- ACTS GitHub Repository  
  Open-source tracking software project used as an architectural reference
  https://github.com/acts-project/acts
