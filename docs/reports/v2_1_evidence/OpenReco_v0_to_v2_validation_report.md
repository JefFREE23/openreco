# OpenReco v0–v2 Validation Report

## Title

**OpenReco v0–v2: A Compact Python Testbed for Controlled Tracking Reconstruction Studies**

---

## 1. Purpose of this report

This report consolidates the validation evidence for OpenReco from v0 through v2.

OpenReco is a compact Python reconstruction-and-analysis testbed for controlled charged-particle tracking studies in a simplified detector environment. Its purpose is not to replace full experiment software such as ACTS, CMSSW, Geant4, or DD4hep. Instead, OpenReco is designed to make the core reconstruction chain readable, testable, and suitable for controlled studies.

The core question behind OpenReco is:

```text
How do detector assumptions and reconstruction choices affect tracking performance,
statistical validation metrics, and downstream reconstructed observables?
```

The project currently connects:

```text
detector geometry assumptions
measurement generation
surface-bound track states
magnetic-field propagation
track seeding
track finding
Kalman-style fitting
smoothing
truth matching
fake and duplicate track validation
covariance and chi-square checks
momentum-resolution studies
external-data validation
```

This report summarizes what each version validated and identifies the evidence that should be used going forward.

---

## 2. Version overview

OpenReco has developed in three main stages.

```text
v0: single-track cylindrical Kalman-filter core
v1: event-level multi-particle reconstruction prototype
v2: external ACTS-style and ACTS/Fatras CSV validation interface
```

The current v2.1 milestone is not a new reconstruction-algorithm milestone. It is an evidence-consolidation milestone. Its purpose is to organize results, clarify scope, document limitations, and prepare the project for controlled performance studies.

---

## 3. Evidence sources

The main evidence files used in this report are:

```text
docs/reports/v2_1_evidence/validation_summary_table.csv
docs/v1_performance_scan.csv
docs/reports/acts_openreco_generated/v2_external_validation_summary.csv
docs/reports/acts_openreco_generated/v2_external_validation_tracks.csv
docs/reports/acts_fatras_smoke_test/v2_external_validation_summary.csv
docs/reports/acts_fatras_smoke_test/v2_external_validation_tracks.csv
```

The main validation figures are:

```text
docs/images/openreco_3d_event.png
docs/images/openreco_xy_view.png
docs/images/v1_multi_track_event.png
docs/reports/acts_openreco_generated/images/v2_efficiency_summary.png
docs/reports/acts_openreco_generated/images/v2_momentum_residuals.png
docs/reports/acts_fatras_smoke_test/images/v2_efficiency_summary.png
docs/reports/acts_fatras_smoke_test/images/v2_momentum_residuals.png
```

---

## 4. v0 validation: single-track Kalman-filter core

### 4.1 Goal

OpenReco v0 validated the smallest serious tracking loop:

```text
particle gun
→ uniform magnetic field
→ cylindrical detector layers
→ smeared [phi, z] measurements
→ truth-assisted seed
→ EKF-style filtering
→ RTS-style smoothing
→ residuals, pulls, chi-square, covariance checks, and momentum validation
```

The aim of v0 was to prove that the basic tracking mathematics worked before adding event-level pattern recognition.

### 4.2 Detector and reconstruction model

The v0 detector model used a simplified cylindrical barrel tracker with six layers. The track state used a five-parameter cylindrical bound representation:

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

The measurement model used local cylindrical measurements:

```text
[phi, z]
```

The Kalman update used the corresponding local measurement projection.

### 4.3 Validation result

The reproducible v0 validation run used:

```text
events requested:       200
events successful:      200
success rate:           1.0000
covariance valid rate:  1.0000
```

Kalman pull summary:

```text
phi mean:               0.0041
phi std:                0.6263
z mean:                -0.0234
z std:                  0.8287
```

Momentum error summary:

```text
abs error mean:         0.0074
abs error std:          0.0882
rel error mean:         0.0026
rel error std:          0.0310
```

### 4.4 Interpretation

The v0 result shows that the single-track Kalman-style loop is stable over 200 generated events. The success rate and covariance-validity rate are both 1.0. The pull means are close to zero, and the relative momentum error mean is small.

The pull widths are below 1, especially for `phi`. This indicates that the uncertainty model is conservative or not fully calibrated. This is not a failure of v0; it is useful evidence that uncertainty calibration should become a later controlled study.

### 4.5 v0 figures

3D cylindrical detector view:

```text
docs/images/openreco_3d_event.png
```

x-y magnetic-bending view:

```text
docs/images/openreco_xy_view.png
```

---

## 5. v1 validation: event-level reconstruction

### 5.1 Goal

OpenReco v1 extended the v0 single-track core into a multi-particle event-level reconstruction prototype.

The v1 chain was:

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

The purpose of v1 was to move from truth-assisted single-track fitting to event-level reconstruction with multiple particles, noise hits, seed building, track finding, and truth-based metrics.

### 5.2 Evidence source

The current reproducible v1 evidence source is:

```text
docs/v1_performance_scan.csv
```

This CSV is preferred over older README-only summary numbers because it is reproducible and records the scan conditions explicitly.

### 5.3 Clean five-particle condition

The representative clean event-level condition is:

```text
n_particles:             5
noise hits/layer:        1
hit efficiency:          1.00
events:                  50
truth particles:         250
```

Result:

```text
tracking efficiency:     0.960
fake rate:               0.000
duplicate rate:          0.000
mean chi2/ndof:          0.955
covariance valid rate:   1.000
momentum residual mean:  0.0033
momentum residual std:   0.0659
runtime/event:           2.1228 s
```

### 5.4 Missing-hit stress condition

A second representative condition tests reconstruction with missing hits:

```text
n_particles:             5
noise hits/layer:        1
hit efficiency:          0.95
events:                  50
truth particles:         250
```

Result:

```text
tracking efficiency:     0.736
fake rate:               0.000
duplicate rate:          0.000
mean chi2/ndof:          0.937
covariance valid rate:   1.000
momentum residual mean:  0.0028
momentum residual std:   0.0624
runtime/event:           1.6337 s
```

### 5.5 Interpretation

The clean condition demonstrates that v1 can reconstruct multi-particle events with high efficiency under controlled ideal-hit conditions. Fake and duplicate rates remain zero in the tested configuration, and covariance validity remains stable.

The missing-hit stress condition shows an expected efficiency decrease when the hit efficiency is reduced to 0.95. This is scientifically useful: it demonstrates that OpenReco is sensitive to detector-efficiency assumptions and can be used to study how missing measurements affect reconstruction performance.

The zero fake and duplicate rates in this toy configuration should not be overinterpreted as production-grade ambiguity resolution. The event model is simplified, and the current track finder uses greedy logic and basic shared-hit rejection.

### 5.6 v1 figure

The v1 event display is:

```text
docs/images/v1_multi_track_event.png
```

---

## 6. v2 validation: external-format reconstruction

### 6.1 Goal

OpenReco v2 added file-based external validation.

The v2 goal was to prove that OpenReco can read external tracking data, adapt it into its own detector/measurement representation, run the reconstruction chain, and produce validation metrics.

v2 supports:

```text
simple ACTS-style CSV datasets
official ACTS/Fatras GenericDetector CSV output
```

The v2 chain is:

```text
external CSV files
→ OpenReco external loader
→ OpenReco adapter
→ triplet seeding
→ greedy track finding
→ EKF-style fitting
→ smoothing
→ truth matching
→ validation metrics
→ CSV reports and plots
```

OpenReco does not call ACTS internally. The v2 validation is a file-based interface, not a full ACTS C++ runtime integration.

---

## 7. v2 OpenReco-generated external-format sample

### 7.1 Purpose

The OpenReco-generated external-format sample is a larger compatibility-validation sample.

Evidence files:

```text
datasets/acts_openreco_generated/
docs/reports/acts_openreco_generated/v2_external_validation_summary.csv
docs/reports/acts_openreco_generated/v2_external_validation_tracks.csv
docs/reports/acts_openreco_generated/images/v2_efficiency_summary.png
docs/reports/acts_openreco_generated/images/v2_momentum_residuals.png
```

This sample is useful because it contains multiple events and multiple truth particles, making the validation plots more meaningful than a one-track smoke test.

### 7.2 Result

```text
events processed:             5
truth particles:              25
reconstructed tracks:         13
matched tracks:               13
unique matched truth:         13
fake tracks:                  0
duplicate tracks:             0

unique tracking efficiency:   0.520
raw matched-track efficiency: 0.520
fake rate:                    0.000
duplicate rate:               0.000
mean chi2/ndof:               1.492
covariance valid rate:        1.000
momentum rel residual mean:  -0.1011
momentum rel residual std:    0.1835
runtime/event:                2.1836 s
```

### 7.3 Interpretation

This result validates the v2 ACTS-style loader, adapter, external reconstruction runner, truth matching, report writing, and plotting path on a multi-event sample.

The efficiency is lower than the clean v1 generated-event condition. This is acceptable because this sample is being used as an external-format validation path rather than as the main optimized event-generation benchmark.

The result is useful for showing that OpenReco can run the full external-data chain and produce stable metrics and plots.

---

## 8. v2 official ACTS/Fatras smoke test

### 8.1 Purpose

The official ACTS/Fatras sample is the strongest external-data proof in v2 because it comes from official ACTS/Fatras GenericDetector CSV output.

Evidence files:

```text
datasets/acts_fatras_sample/
docs/reports/acts_fatras_smoke_test/v2_external_validation_summary.csv
docs/reports/acts_fatras_smoke_test/v2_external_validation_tracks.csv
docs/reports/acts_fatras_smoke_test/images/v2_efficiency_summary.png
docs/reports/acts_fatras_smoke_test/images/v2_momentum_residuals.png
```

This validation uses a simplified cylindrical radius-shell mapping and a length-scale calibration parameter to map ACTS/Fatras coordinates into OpenReco's toy detector scale.

### 8.2 Result

```text
events processed:             1
truth particles:              1
reconstructed tracks:         1
matched tracks:               1
unique matched truth:         1
fake tracks:                  0
duplicate tracks:             0

unique tracking efficiency:   1.000
raw matched-track efficiency: 1.000
fake rate:                    0.000
duplicate rate:               0.000
mean chi2/ndof:               0.077
covariance valid rate:        1.000
momentum rel residual mean:  -0.0031
momentum rel residual std:    0.0000
runtime/event:                0.7339 s
```

### 8.3 Interpretation

The official ACTS/Fatras result proves that OpenReco can ingest official ACTS/Fatras CSV output and run its own validation chain end to end.

However, this result must be interpreted as a smoke test, not as a statistical benchmark. The current official sample contains only one event and one reconstructed truth particle. Larger ACTS/Fatras samples are needed before making strong statistical claims about external-data reconstruction performance.

---

## 9. Consolidated validation table

The consolidated evidence table is stored at:

```text
docs/reports/v2_1_evidence/validation_summary_table.csv
```

Summary:

```text
v0:
single-track Kalman core validated over 200 events.

v1:
event-level reconstruction validated with reproducible performance-scan evidence.

v2 OpenReco-generated external-format sample:
external-data path validated on a larger multi-event compatibility sample.

v2 official ACTS/Fatras sample:
official external-data ingestion validated as a one-event smoke test.
```

---

## 10. Main limitations

OpenReco v2.1 is still a compact reconstruction prototype.

Current detector-model limitations:

```text
simple cylindrical barrel geometry
no realistic material map
no detailed multiple-scattering model
no detailed energy-loss model
no detector misalignment model
no dead-channel map
no full non-cylindrical surface support
no ODD geometry import
no DD4hep or TGeo geometry import
```

Current reconstruction limitations:

```text
triplet seeding is simple
track following is greedy
ambiguity resolution is basic
shared-hit rejection is simple
duplicate suppression is limited
track finding is not a full CKF
track quality cuts are simple
covariance calibration is not final
pull widths are not yet fully calibrated
```

Current validation limitations:

```text
v0 and v1 use simplified generated events
v2 OpenReco-generated external sample is compatible with OpenReco geometry
official ACTS/Fatras sample currently contains only one event and one reconstructed truth particle
large-statistics ACTS/Fatras validation has not yet been performed
ODD-style validation has not yet been performed
CMS Open Data validation has not yet been performed
```

---

## 11. What OpenReco can now study

OpenReco can now support controlled studies of reconstruction behavior under simplified detector assumptions.

Examples of study questions:

```text
How does hit efficiency affect tracking efficiency?
How does noise occupancy affect fake rate?
How does missing-hit recovery affect efficiency?
How do shared-hit cuts affect duplicate rate?
How do measurement uncertainties affect chi-square and pulls?
How does covariance calibration affect pull widths?
How do reconstruction choices affect momentum resolution?
How do detector assumptions propagate into downstream observable resolution?
```

This is the correct post-v2 direction. OpenReco should be treated as a compact reconstruction-and-analysis testbed, not as a miniature replacement for ACTS, CMSSW, or Geant4.

---

## 12. Recommended next milestone

The next recommended milestone after v2.1 is:

```text
v2.2 — Tracking performance analysis suite
```

The purpose of v2.2 should be to automate performance studies over controlled detector and reconstruction parameters.

Recommended v2.2 outputs:

```text
efficiency curves
fake-rate curves
duplicate-rate curves
chi2/ndof distributions
pull distributions
momentum residual distributions
parameter bias and resolution plots
truth-matching summaries
quality-cut scans
occupancy scans
hit-efficiency scans
reproducible CSV or parquet summaries
markdown or notebook validation reports
```


---

## 13. Conclusion

OpenReco v0–v2 has built a compact but complete reconstruction-validation loop.

v0 validated the single-track cylindrical Kalman-filter core.

v1 extended the project to event-level reconstruction with seeding, track finding, fitting, smoothing, truth matching, and fake/duplicate-rate validation.

v2 added external file-based validation for ACTS-style and official ACTS/Fatras CSV outputs.

v2.1 consolidates this evidence and clarifies the project direction: OpenReco is a compact Python testbed for controlled studies of detector assumptions, reconstruction behavior, statistical validation, and downstream physics-observable resolution in a simplified detector environment.
