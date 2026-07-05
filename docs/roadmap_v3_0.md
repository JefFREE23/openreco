# OpenReco v3.0 Roadmap — Detector Effects and Uncertainty Calibration

## Purpose

OpenReco v3.0 adds a controlled detector-effects and uncertainty-calibration layer.

The goal is not to become Geant4, ACTS, ODD, CMS, or a full detector simulation stack. The goal is to make detector assumptions explicit and study how they affect reconstruction performance.

## Scientific question

How do detector assumptions and reconstruction choices affect:

- residuals
- pulls
- chi2/ndof
- tracking efficiency
- fake rate
- duplicate rate
- holes
- momentum bias and resolution
- covariance/uncertainty calibration

## Included in v3.0

- Configurable hit resolution
- Hit inefficiency
- Dead layers
- Noise occupancy / fake hits
- Layer material budget
- Multiple-scattering process noise
- Simple deterministic energy loss
- Magnetic-field scale mismatch
- Uncertainty-calibration scans

## Excluded from v3.0

- Full Geant4 simulation
- CMS Open Data reconstruction
- ODD geometry translation
- Detailed detector conditions
- Full alignment system
- Hadronic interactions
- Electron bremsstrahlung modelling
- Full experiment-framework integration

## Milestone chunks

1. Detector effects configuration layer
2. Configurable hit resolution
3. Hit inefficiency and dead layers
4. Noise occupancy / fake hits
5. Layer material model
6. Multiple-scattering process noise
7. Uncertainty calibration tools
8. Simple energy-loss model
9. Magnetic-field scale scans
10. Integrated detector-effects benchmark
11. Documentation and release polish

## Definition of done

v3.0 is complete when OpenReco can run controlled detector-effect scans and produce reproducible summaries showing how detector assumptions change tracking performance and uncertainty calibration.