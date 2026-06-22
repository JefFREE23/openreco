# acts_small

Tiny ACTS-style external validation dataset for OpenReco v2.

This dataset is intentionally simple and barrel-like. It is not a full ACTS export.
Its purpose is to test the OpenReco v2 file loader and adapter before using more complex external data.

Files:

- `truth_particles.csv`
- `measurements.csv`

Units/conventions:

- Length: same internal length convention used by OpenReco examples
- Momentum: GeV
- Angles: radians
- `r = sqrt(x^2 + y^2)`
- `phi = atan2(y, x)`
- Noise hits have an empty `particle_id` and `is_noise=True`