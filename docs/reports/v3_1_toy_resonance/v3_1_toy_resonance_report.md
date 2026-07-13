# OpenReco v3.1 Toy Resonance Report

## Purpose

OpenReco v3.1 adds a downstream physics-observable study on top of the v3.0 detector-effects framework.

The goal is to show that detector assumptions and reconstruction choices can propagate from track-level reconstruction into a reconstructed invariant-mass observable.

The controlled toy channel is:

```text
J/psi -> mu+ mu-
```

This is a simplified benchmark, not a realistic muon analysis.

## Baseline event

The baseline example generates a truth-level toy resonance, converts the two daughter muons into OpenReco truth particles, reconstructs the two tracks, selects the opposite-charge pair, and computes the reconstructed invariant mass.

Example baseline output:

```text
Truth resonance mass: 3.096900 GeV
Truth invariant mass from daughters: 3.096900 GeV
Truth daughters: 2
Reconstructed tracks: 2
Matched tracks: 2
Tracking efficiency: 1.000000
Opposite-charge candidates: 1
Best reconstructed mass: 3.067214 GeV
Mass residual: -0.029686 GeV
```

## Mass-resolution scan

The mass-resolution scan propagates detector-effect settings into the reconstructed invariant mass.

The scan writes:

```text
docs/reports/v3_1_toy_resonance/mass_resolution_scan.csv
```

The current scan uses matched toy event seeds across detector-effect settings so that differences are mainly due to the detector/reconstruction setting, not random event changes.

Current scan summary with EKF fitting enabled:

| Study | Value | Candidate efficiency | Mass mean [GeV] | Mass width [GeV] | Residual mean [GeV] |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0000 | 1.000 | 2.999820 | 0.028568 | -0.097080 |
| hit_resolution_scale | 0.5000 | 0.900 | 3.006355 | 0.015216 | -0.090545 |
| hit_resolution_scale | 1.0000 | 1.000 | 2.999820 | 0.028568 | -0.097080 |
| hit_resolution_scale | 2.0000 | 1.000 | 2.986188 | 0.055811 | -0.110712 |
| hit_resolution_scale | 5.0000 | 1.000 | 2.956153 | 0.135466 | -0.140747 |
| material_budget | 0.0050 | 1.000 | 3.007296 | 0.046728 | -0.089604 |
| material_budget | 0.0200 | 1.000 | 3.006787 | 0.050721 | -0.090113 |
| bfield_reco_scale | 0.9800 | 1.000 | 2.940177 | 0.028009 | -0.156723 |
| bfield_reco_scale | 1.0200 | 1.000 | 3.059468 | 0.029126 | -0.037432 |

## Main observations

- The baseline toy resonance is reconstructed with two matched daughter tracks.
- The downstream invariant-mass observable is successfully computed from reconstructed tracks.
- Degraded hit resolution broadens the mass distribution.
- Material budget changes the mass width through truth-side scattering.
- Reconstruction magnetic-field scale mismatch biases the reconstructed mass.
- This demonstrates propagation from detector/reconstruction assumptions to a physics-facing observable.

## Scope

This is a compact toy benchmark.

It intentionally excludes:

- Geant4,
- ODD,
- CMS or ATLAS data,
- realistic muon identification,
- background modelling,
- vertex fitting,
- trigger simulation,
- pileup,
- production resonance fitting.

## Final claim

OpenReco can propagate controlled detector and reconstruction assumptions from track-level performance into a downstream invariant-mass observable in a simplified detector environment.