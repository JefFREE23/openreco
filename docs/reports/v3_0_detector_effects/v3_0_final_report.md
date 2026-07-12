# OpenReco v3.0 Final Report — Detector Effects and Uncertainty Calibration

## Summary

OpenReco v3.0 extends the event-level reconstruction prototype into a controlled detector-effects and uncertainty-calibration study framework.

The main purpose of v3.0 is not to simulate a full experimental detector. Instead, it provides compact, readable, reproducible studies showing how detector assumptions affect reconstruction behavior in a simplified cylindrical tracking environment.

v3.0 studies the connection between:

- hit resolution,
- hit inefficiency and dead layers,
- random noise occupancy,
- material budget and multiple scattering,
- reconstruction-side process noise,
- deterministic energy loss,
- magnetic-field scale mismatch,
- tracking efficiency,
- fake and duplicate rates,
- holes,
- chi2/ndof,
- momentum bias and momentum resolution,
- covariance size and uncertainty calibration.

The final validation state for this branch is:

```text
python -m pytest
340 passed
```