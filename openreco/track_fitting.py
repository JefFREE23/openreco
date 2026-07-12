from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from openreco.detector_effects import DetectorEffectsConfig
from openreco.process_noise import material_process_noise_matrix
from openreco.event_generation import EventHit
from openreco.field import UniformMagneticField
from openreco.kalman import (
    KalmanUpdateResult,
    filter_cylindrical_track,
    total_chi2,
)
from openreco.measurements import Measurement
from openreco.seeding import TripletSeed
from openreco.smoothing import SmoothingResult, smooth_track
from openreco.state import TrackState, make_cylindrical_state
from openreco.track_finding import ReconstructedTrack


@dataclass(frozen=True)
class TrackFitResult:
    """
    EKF fit result for one reconstructed v1 track candidate.
    """

    filtered_results: tuple[KalmanUpdateResult, ...]
    smoothing_results: tuple[SmoothingResult, ...]
    filtered_states: tuple[TrackState, ...]
    smoothed_states: tuple[TrackState, ...]
    final_state: TrackState
    final_covariance: np.ndarray

    chi2: float
    ndof: int
    chi2_ndof: float

    q_over_p: float
    pt_estimate: float
    p_estimate: float
    momentum_uncertainty: float

    covariance_valid: bool
    fit_status: str


@dataclass(frozen=True)
class _CylindricalLayerAdapter:
    """
    Minimal layer adapter used by the existing v0 Kalman filter.

    It has the attributes/methods needed by propagation:
    - name
    - radius
    - contains_z(z)
    """

    name: str
    radius: float

    @property
    def surface_type(self) -> str:
        return "cylinder"

    def contains_z(self, z: float) -> bool:
        return np.isfinite(z)


@dataclass(frozen=True)
class _BarrelDetectorAdapter:
    """
    Minimal detector adapter for filter_cylindrical_track().
    """

    layers: tuple[_CylindricalLayerAdapter, ...]

    def __len__(self) -> int:
        return len(self.layers)


def event_hit_to_measurement(hit: EventHit) -> Measurement:
    """
    Convert a v1 EventHit into a v0 cylindrical Measurement.
    """

    return Measurement(
        values=np.array([hit.phi, hit.z], dtype=float),
        covariance=np.asarray(hit.covariance, dtype=float),
        layer_name=hit.layer_name,
        surface_type="cylinder",
    )


def build_initial_state_from_seed(
    seed: TripletSeed,
    *,
    covariance: np.ndarray | None = None,
) -> TrackState:
    """
    Build an initial cylindrical TrackState from a triplet seed.

    The initial local position is taken from the innermost seed hit.
    The direction and q/p estimate come from the seed.
    """

    if covariance is None:
        covariance = default_initial_covariance()

    seed_hits = sorted(seed.hits, key=lambda hit: hit.layer_index)
    first_hit = seed_hits[0]

    return make_cylindrical_state(
        phi=first_hit.phi,
        z=first_hit.z,
        dir0=seed.alpha,
        dir1=seed.tan_lambda,
        q_over_p=seed.q_over_p,
        covariance=covariance,
        surface_radius=first_hit.radius,
        surface_name=first_hit.layer_name,
    )


def default_initial_covariance() -> np.ndarray:
    """
    Broad but finite initial covariance for seed-started EKF fits.
    """

    return np.diag(
        [
            1e-4,   # phi
            1.0,    # z
            1e-2,   # alpha
            1e-2,   # tan_lambda
            0.25,   # q_over_p
        ]
    )

def build_material_process_noise_for_track(
    track: ReconstructedTrack,
    *,
    detector_effects: DetectorEffectsConfig,
    process_noise_scale: float = 1.0,
) -> np.ndarray:
    """
    Build a simplified material process-noise matrix for one track candidate.

    The current v3.0 EKF accepts one fixed 5x5 process-noise matrix per fit.
    For this first reconstruction-side material model, we use the mean material
    thickness across the candidate's used layers and the seed momentum estimate.

    This is intentionally compact and deterministic. Later calibration scans can
    vary process_noise_scale to study under/over-estimated process noise.
    """

    if process_noise_scale < 0.0:
        raise ValueError("process_noise_scale must be non-negative")

    if not track.used_measurements:
        return np.zeros((5, 5), dtype=float)

    x_over_x0_values = [
        float(detector_effects.material_for_layer(hit.layer_index).x_over_x0)
        for hit in track.used_measurements
    ]

    mean_x_over_x0 = float(np.mean(x_over_x0_values))

    if mean_x_over_x0 <= 0.0 or process_noise_scale == 0.0:
        return np.zeros((5, 5), dtype=float)

    q_over_p = float(track.seed.q_over_p)
    p_gev = momentum_from_q_over_p(q_over_p)

    if not np.isfinite(p_gev):
        return np.zeros((5, 5), dtype=float)

    charge_abs = abs(float(np.sign(q_over_p)))

    if charge_abs == 0.0:
        charge_abs = 1.0

    return material_process_noise_matrix(
        p_gev=p_gev,
        x_over_x0=mean_x_over_x0,
        process_noise_scale=process_noise_scale,
        beta=1.0,
        charge_abs=charge_abs,
    )

def fit_track_candidate_with_ekf(
    track: ReconstructedTrack,
    *,
    field: UniformMagneticField | None = None,
    initial_covariance: np.ndarray | None = None,
    process_noise: np.ndarray | None = None,
    detector_effects: DetectorEffectsConfig | None = None,
    process_noise_scale: float = 0.0,
    curvature_scale: float = 0.003,
    max_s: float = 10000.0,
    n_scan: int = 1000,
    run_smoothing: bool = True,
) -> TrackFitResult:
    """
    Fit one v1 reconstructed track candidate using the existing v0 EKF.

    Phase B is intentionally strict: the candidate is fitted using exactly
    the hits selected by the v1 track finder.
    """

    if len(track.used_measurements) < 3:
        raise ValueError("EKF fitting requires at least three measurements")

    if field is None:
        field = UniformMagneticField(bz=2.0)
    
    if process_noise_scale < 0.0:
        raise ValueError("process_noise_scale must be non-negative")

    ordered_hits = tuple(
        sorted(track.used_measurements, key=lambda hit: hit.layer_index)
    )

    detector = _BarrelDetectorAdapter(
        layers=tuple(
            _CylindricalLayerAdapter(
                name=hit.layer_name,
                radius=hit.radius,
            )
            for hit in ordered_hits
        )
    )

    measurements = [
        event_hit_to_measurement(hit)
        for hit in ordered_hits
    ]

    initial_state = build_initial_state_from_seed(
        track.seed,
        covariance=initial_covariance,
    )

    if (
        process_noise is None
        and detector_effects is not None
        and process_noise_scale > 0.0
    ):
        process_noise = build_material_process_noise_for_track(
            track,
            detector_effects=detector_effects,
            process_noise_scale=process_noise_scale,
        )

    filtered_results = tuple(
        filter_cylindrical_track(
            initial_state=initial_state,
            measurements=measurements,
            detector=detector,
            field=field,
            process_noise=process_noise,
            curvature_scale=curvature_scale,
            max_s=max_s,
            n_scan=n_scan,
        )
    )

    if run_smoothing:
        smoothing_results = tuple(smooth_track(list(filtered_results)))
        smoothed_states = tuple(
            result.smoothed_state
            for result in smoothing_results
        )
        final_state = smoothed_states[-1]
    else:
        smoothing_results = tuple()
        smoothed_states = tuple()
        final_state = filtered_results[-1].filtered_state

    filtered_states = tuple(
        result.filtered_state
        for result in filtered_results
    )

    final_covariance = final_state.covariance.copy()

    chi2 = total_chi2(filtered_results)
    ndof = max(1, 2 * len(measurements) - 5)
    chi2_ndof = chi2 / ndof

    q_over_p = final_state.q_over_p
    tan_lambda = final_state.dir1

    p_estimate = momentum_from_q_over_p(q_over_p)
    pt_estimate = p_estimate / np.sqrt(1.0 + tan_lambda**2)
    momentum_uncertainty = momentum_uncertainty_from_q_over_p(
        q_over_p=q_over_p,
        covariance=final_covariance,
    )

    covariance_valid = is_covariance_valid(final_covariance)

    return TrackFitResult(
        filtered_results=filtered_results,
        smoothing_results=smoothing_results,
        filtered_states=filtered_states,
        smoothed_states=smoothed_states,
        final_state=final_state,
        final_covariance=final_covariance,
        chi2=float(chi2),
        ndof=int(ndof),
        chi2_ndof=float(chi2_ndof),
        q_over_p=float(q_over_p),
        pt_estimate=float(pt_estimate),
        p_estimate=float(p_estimate),
        momentum_uncertainty=float(momentum_uncertainty),
        covariance_valid=bool(covariance_valid),
        fit_status="accepted",
    )


def attach_ekf_fit_to_track(
    track: ReconstructedTrack,
    fit_result: TrackFitResult,
) -> ReconstructedTrack:
    """
    Return a copy of a ReconstructedTrack with EKF fit outputs attached.
    """

    final_state = fit_result.final_state

    return replace(
        track,
        chi2=fit_result.chi2,
        ndof=fit_result.ndof,
        chi2_ndof=fit_result.chi2_ndof,
        final_phi=final_state.phi,
        final_z=final_state.z,
        tan_lambda=final_state.dir1,
        q_over_p=fit_result.q_over_p,
        pt_estimate=fit_result.pt_estimate,
        p_estimate=fit_result.p_estimate,
        fit_status=fit_result.fit_status,
        filtered_states=fit_result.filtered_states,
        smoothed_states=fit_result.smoothed_states,
        final_covariance=fit_result.final_covariance,
        covariance_valid=fit_result.covariance_valid,
        momentum_uncertainty=fit_result.momentum_uncertainty,
    )


def fit_reconstructed_track_with_ekf(
    track: ReconstructedTrack,
    **kwargs,
) -> ReconstructedTrack:
    """
    Fit one ReconstructedTrack and return the EKF-enhanced track.
    """

    fit_result = fit_track_candidate_with_ekf(track, **kwargs)
    return attach_ekf_fit_to_track(track, fit_result)


def fit_reconstructed_tracks_with_ekf(
    tracks: Iterable[ReconstructedTrack],
    *,
    fail_safely: bool = True,
    **kwargs,
) -> list[ReconstructedTrack]:
    """
    Fit many reconstructed tracks with the v0 EKF.

    If fail_safely=True, failed fits are kept but marked as failed instead
    of crashing a whole performance scan.
    """

    fitted_tracks: list[ReconstructedTrack] = []

    for track in tracks:
        try:
            fitted_track = fit_reconstructed_track_with_ekf(track, **kwargs)
            fitted_tracks.append(fitted_track)
        except Exception as exc:
            if not fail_safely:
                raise

            fitted_tracks.append(
                replace(
                    track,
                    fit_status=f"ekf_failed:{type(exc).__name__}",
                    covariance_valid=False,
                )
            )

    return fitted_tracks


def momentum_from_q_over_p(q_over_p: float) -> float:
    """
    Convert q/p into momentum magnitude.
    """

    if np.isclose(q_over_p, 0.0):
        return float("inf")

    return float(1.0 / abs(q_over_p))


def momentum_uncertainty_from_q_over_p(
    *,
    q_over_p: float,
    covariance: np.ndarray,
) -> float:
    """
    Approximate p uncertainty from q/p uncertainty using error propagation.
    """

    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (5, 5):
        return float("nan")

    if np.isclose(q_over_p, 0.0):
        return float("inf")

    qop_variance = covariance[4, 4]

    if qop_variance < 0.0:
        return float("nan")

    qop_sigma = np.sqrt(qop_variance)

    return float(qop_sigma / (q_over_p**2))


def is_covariance_valid(
    covariance: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check whether a covariance matrix is finite, symmetric, and positive semi-definite.
    """

    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (5, 5):
        return False

    if not np.all(np.isfinite(covariance)):
        return False

    if not np.allclose(covariance, covariance.T):
        return False

    eigenvalues = np.linalg.eigvalsh(covariance)

    return bool(np.min(eigenvalues) >= -tolerance)
