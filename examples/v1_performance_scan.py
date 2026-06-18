from **future** import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

PROJECT_ROOT = Path(**file**).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
sys.path.insert(0, str(PROJECT_ROOT))

from examples.multi_track_reconstruction import run_multi_track_reconstruction

@dataclass(frozen=True)
class ScanConfig:
n_particles: int
noise_hits_per_layer: int
hit_efficiency: float
n_events: int
random_seed: int

@dataclass(frozen=True)
class ScanResult:
n_particles: int
noise_hits_per_layer: int
hit_efficiency: float
events_processed: int

```
truth_particles_generated: int
measurements_mean: float
seeds_mean: float
reconstructed_tracks_mean: float

tracking_efficiency_mean: float
fake_rate_mean: float
duplicate_rate_mean: float

mean_hits_per_track: float
mean_chi2_ndof: float
covariance_valid_rate_mean: float

momentum_residual_mean: float
momentum_residual_std: float

runtime_total_s: float
runtime_per_event_s: float
```

def run_performance_scan(
*,
n_events: int = 50,
particle_counts: Iterable[int] = (1, 2, 5),
noise_hits_per_layer_values: Iterable[int] = (0, 1),
hit_efficiencies: Iterable[float] = (1.0, 0.95),
random_seed: int = 123,
) -> list[ScanResult]:
"""
Run the OpenReco v1 event-level reconstruction chain many times.

```
The scan varies:
- number of truth particles
- noise hits per layer
- hit efficiency

It reports event-level reconstruction performance.
"""

if n_events <= 0:
    raise ValueError("n_events must be positive.")

results: list[ScanResult] = []

for n_particles in particle_counts:
    for noise_hits_per_layer in noise_hits_per_layer_values:
        for hit_efficiency in hit_efficiencies:
            config = ScanConfig(
                n_particles=int(n_particles),
                noise_hits_per_layer=int(noise_hits_per_layer),
                hit_efficiency=float(hit_efficiency),
                n_events=int(n_events),
                random_seed=int(random_seed),
            )

            results.append(run_single_scan_point(config))

return results
```

def run_single_scan_point(config: ScanConfig) -> ScanResult:
measurements_counts: list[int] = []
seeds_counts: list[int] = []
track_counts: list[int] = []
tracking_efficiencies: list[float] = []
fake_rates: list[float] = []
duplicate_rates: list[float] = []
hits_per_track_values: list[float] = []
chi2_ndof_values: list[float] = []
covariance_valid_rates: list[float] = []
momentum_residuals: list[float] = []

```
start_time = time.perf_counter()

for event_index in range(config.n_events):
    result = run_multi_track_reconstruction(
        event_id=event_index,
        n_particles=config.n_particles,
        hit_efficiency=config.hit_efficiency,
        noise_hits_per_layer=config.noise_hits_per_layer,
        random_seed=config.random_seed + event_index,
        make_plot=False,
    )

    measurements_counts.append(len(result.event.measurements))
    seeds_counts.append(len(result.seeds))
    track_counts.append(len(result.tracks))

    tracking_efficiencies.append(result.validation.tracking_efficiency)
    fake_rates.append(result.validation.fake_rate)
    duplicate_rates.append(result.validation.duplicate_rate)

    if result.tracks:
        hits_per_track_values.append(
            mean(len(track.used_measurements) for track in result.tracks)
        )

        chi2_ndof_values.extend(
            track.chi2_ndof
            for track in result.tracks
        )

        covariance_valid_rates.append(
            mean(
                1.0 if getattr(track, "covariance_valid", False) else 0.0
                for track in result.tracks
            )
        )

    momentum_residuals.extend(result.momentum_relative_errors)

runtime_total_s = time.perf_counter() - start_time

return ScanResult(
    n_particles=config.n_particles,
    noise_hits_per_layer=config.noise_hits_per_layer,
    hit_efficiency=config.hit_efficiency,
    events_processed=config.n_events,
    truth_particles_generated=config.n_particles * config.n_events,
    measurements_mean=_safe_mean(measurements_counts),
    seeds_mean=_safe_mean(seeds_counts),
    reconstructed_tracks_mean=_safe_mean(track_counts),
    tracking_efficiency_mean=_safe_mean(tracking_efficiencies),
    fake_rate_mean=_safe_mean(fake_rates),
    duplicate_rate_mean=_safe_mean(duplicate_rates),
    mean_hits_per_track=_safe_mean(hits_per_track_values),
    mean_chi2_ndof=_safe_mean(chi2_ndof_values),
    covariance_valid_rate_mean=_safe_mean(covariance_valid_rates),
    momentum_residual_mean=_safe_mean(momentum_residuals),
    momentum_residual_std=_safe_pstdev(momentum_residuals),
    runtime_total_s=runtime_total_s,
    runtime_per_event_s=runtime_total_s / config.n_events,
)
```

def format_scan_results(results: list[ScanResult]) -> str:
lines = [
"OpenReco v1 performance scan",
"",
(
"n_particles  noise/layer  hit_eff  events  "
"eff     fake    dup     seeds/event  tracks/event  "
"chi2/ndof  cov_valid  mom_res_mean  runtime/event"
),
"-" * 130,
]

```
for result in results:
    lines.append(
        f"{result.n_particles:<12}"
        f"{result.noise_hits_per_layer:<13}"
        f"{result.hit_efficiency:<9.2f}"
        f"{result.events_processed:<8}"
        f"{result.tracking_efficiency_mean:<8.3f}"
        f"{result.fake_rate_mean:<8.3f}"
        f"{result.duplicate_rate_mean:<8.3f}"
        f"{result.seeds_mean:<13.2f}"
        f"{result.reconstructed_tracks_mean:<14.2f}"
        f"{result.mean_chi2_ndof:<10.3f}"
        f"{result.covariance_valid_rate_mean:<11.3f}"
        f"{result.momentum_residual_mean:<14.4f}"
        f"{result.runtime_per_event_s:.4f}s"
    )

return "\n".join(lines)
```

def save_scan_csv(results: list[ScanResult], output_path: str | Path) -> Path:
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

```
with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(asdict(results[0]).keys()) if results else [],
    )

    if results:
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

return output_path
```

def _safe_mean(values: list[float] | list[int]) -> float:
return float(mean(values)) if values else float("nan")

def _safe_pstdev(values: list[float]) -> float:
return float(pstdev(values)) if len(values) > 1 else 0.0

def _parse_int_list(raw: str) -> tuple[int, ...]:
return tuple(int(item.strip()) for item in raw.split(",") if item.strip())

def _parse_float_list(raw: str) -> tuple[float, ...]:
return tuple(float(item.strip()) for item in raw.split(",") if item.strip())

def main() -> None:
parser = argparse.ArgumentParser(
description="Run the OpenReco v1 performance scan."
)

```
parser.add_argument("--n-events", type=int, default=50)
parser.add_argument("--particle-counts", type=str, default="1,2,5")
parser.add_argument("--noise-hits-per-layer", type=str, default="0,1")
parser.add_argument("--hit-efficiencies", type=str, default="1.0,0.95")
parser.add_argument("--random-seed", type=int, default=123)
parser.add_argument(
    "--output-csv",
    type=str,
    default="docs/v1_performance_scan.csv",
)

args = parser.parse_args()

results = run_performance_scan(
    n_events=args.n_events,
    particle_counts=_parse_int_list(args.particle_counts),
    noise_hits_per_layer_values=_parse_int_list(args.noise_hits_per_layer),
    hit_efficiencies=_parse_float_list(args.hit_efficiencies),
    random_seed=args.random_seed,
)

print(format_scan_results(results))

csv_path = save_scan_csv(results, args.output_csv)
print()
print(f"CSV saved: {csv_path}")
```

if **name** == "**main**":
main()
