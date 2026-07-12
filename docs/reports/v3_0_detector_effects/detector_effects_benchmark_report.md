# OpenReco v3.0 Detector-Effects Benchmark

This report summarizes the v3.0 detector-effects scan CSV artifacts.

The baseline point is the first row of each scan. The comparison point is the final row, except for process-noise calibration, where the comparison point is the scale with mean chi2/ndof closest to 1.

## Summary

| Study | Points | Baseline | Comparison | chi2/ndof | Momentum residual std |
|---|---:|---|---|---:|---:|
| hit_resolution | 4 | sigma_phi=0.0005; sigma_z=0.05 | sigma_phi=0.005; sigma_z=0.5 | 1.133 -> 1.057 | 0.02798 -> 0.9282 |
| inefficiency_dead_layers | 12 | hit_efficiency=1.0; dead_layer_scenario=none; dead_layers=none | hit_efficiency=0.8; dead_layer_scenario=dead_1_4; dead_layers=1,4 | 1.195 -> 0.9842 | 0.06518 -> 0.06608 |
| noise_occupancy | 5 | mean_noise_hits_per_layer=0.0 | mean_noise_hits_per_layer=5.0 | 1.195 -> 1.036 | 0.06518 -> 0.05712 |
| material_budget | 6 | x_over_x0_per_layer=0.0 | x_over_x0_per_layer=0.05 | 1.195 -> 1.944 | 0.06518 -> 0.1185 |
| process_noise_calibration | 7 | process_noise_scale=0.0; x_over_x0_per_layer=0.02 | process_noise_scale=5.0; x_over_x0_per_layer=0.02 | 1.103 -> 1.045 | 0.0794 -> 0.07921 |
| energy_loss | 7 | energy_loss_mev_per_layer=0.0 | energy_loss_mev_per_layer=100.0 | 7.014 -> 3.148 | 0.07954 -> 0.09479 |
| bfield_scale | 5 | truth_scale=1.0; reco_scale=0.95; scale_mismatch=0.95 | truth_scale=1.0; reco_scale=1.05; scale_mismatch=1.05 | 7.014 -> 7.014 | 0.07565 -> 0.08343 |

## Headlines

- hit_resolution: sigma_phi=0.0005; sigma_z=0.05 -> sigma_phi=0.005; sigma_z=0.5; chi2/ndof 1.133 -> 1.057; momentum residual std 0.02798 -> 0.9282
- inefficiency_dead_layers: hit_efficiency=1.0; dead_layer_scenario=none; dead_layers=none -> hit_efficiency=0.8; dead_layer_scenario=dead_1_4; dead_layers=1,4; chi2/ndof 1.195 -> 0.9842; momentum residual std 0.06518 -> 0.06608
- noise_occupancy: mean_noise_hits_per_layer=0.0 -> mean_noise_hits_per_layer=5.0; chi2/ndof 1.195 -> 1.036; momentum residual std 0.06518 -> 0.05712
- material_budget: x_over_x0_per_layer=0.0 -> x_over_x0_per_layer=0.05; chi2/ndof 1.195 -> 1.944; momentum residual std 0.06518 -> 0.1185
- process_noise_calibration: process_noise_scale=0.0; x_over_x0_per_layer=0.02 -> process_noise_scale=5.0; x_over_x0_per_layer=0.02; chi2/ndof 1.103 -> 1.045; momentum residual std 0.0794 -> 0.07921
- energy_loss: energy_loss_mev_per_layer=0.0 -> energy_loss_mev_per_layer=100.0; chi2/ndof 7.014 -> 3.148; momentum residual std 0.07954 -> 0.09479
- bfield_scale: truth_scale=1.0; reco_scale=0.95; scale_mismatch=0.95 -> truth_scale=1.0; reco_scale=1.05; scale_mismatch=1.05; chi2/ndof 7.014 -> 7.014; momentum residual std 0.07565 -> 0.08343
