"""
Measurement models for OpenReco.

This module defines simple smeared detector measurements.

Supported measurement types:
    - Planar 2D measurements: [x, y]
    - Cylindrical 2D measurements: [phi, z]

The planar measurement is useful for the straight-line checkpoint.
The cylindrical measurement is required for the final OpenReco v0 tracker.
"""

from dataclasses import dataclass

import numpy as np

from openreco.geometry import CylindricalLayer, DetectorLayer


@dataclass(frozen=True)
class Measurement:
    """
    A detector measurement attached to a detector surface.

    Parameters
    ----------
    values:
        Measurement vector.
        For planar layers: [x, y]
        For cylindrical layers: [phi, z]
    covariance:
        Measurement covariance matrix.
    layer_name:
        Name of the detector layer where the measurement was made.
    surface_type:
        Surface type, for example "plane" or "cylinder".
    """

    values: np.ndarray
    covariance: np.ndarray
    layer_name: str
    surface_type: str

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)

        if values.ndim != 1:
            raise ValueError("Measurement values must be a 1D vector")

        n_values = values.shape[0]

        if covariance.shape != (n_values, n_values):
            raise ValueError(
                "Measurement covariance shape must match measurement dimension"
            )

        if not np.allclose(covariance, covariance.T):
            raise ValueError("Measurement covariance matrix must be symmetric")

        if not isinstance(self.layer_name, str):
            raise TypeError("Measurement layer_name must be a string")

        if not isinstance(self.surface_type, str):
            raise TypeError("Measurement surface_type must be a string")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dimension(self) -> int:
        """
        Return the measurement dimension.
        """
        return self.values.shape[0]

    def copy(self) -> "Measurement":
        """
        Return a deep copy of the measurement.
        """
        return Measurement(
            values=self.values.copy(),
            covariance=self.covariance.copy(),
            layer_name=self.layer_name,
            surface_type=self.surface_type,
        )


def smear_position(
    true_values: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Smear true measurement values using a Gaussian covariance matrix.
    """

    true_values = np.asarray(true_values, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if true_values.ndim != 1:
        raise ValueError("true_values must be a 1D vector")

    if covariance.shape != (true_values.shape[0], true_values.shape[0]):
        raise ValueError("covariance shape must match true_values dimension")

    if not np.allclose(covariance, covariance.T):
        raise ValueError("covariance matrix must be symmetric")

    if rng is None:
        rng = np.random.default_rng()

    return rng.multivariate_normal(mean=true_values, cov=covariance)


def make_planar_measurement(
    layer: DetectorLayer,
    true_position: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> Measurement:
    """
    Create a smeared 2D planar measurement [x, y].

    Parameters
    ----------
    layer:
        Planar detector layer.
    true_position:
        True 3D position [x, y, z].
    sigma:
        Measurement resolution for x and y.
    rng:
        Optional NumPy random generator.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    true_position = np.asarray(true_position, dtype=float)

    if true_position.shape != (3,):
        raise ValueError("true_position must have shape (3,)")

    true_values = true_position[:2]
    covariance = np.diag([sigma**2, sigma**2])
    measured_values = smear_position(true_values, covariance, rng=rng)

    return Measurement(
        values=measured_values,
        covariance=covariance,
        layer_name=layer.name,
        surface_type=layer.surface_type,
    )


def make_cylindrical_measurement(
    layer: CylindricalLayer,
    true_position: np.ndarray,
    sigma_phi: float,
    sigma_z: float,
    rng: np.random.Generator | None = None,
) -> Measurement:
    """
    Create a smeared 2D cylindrical measurement [phi, z].

    Parameters
    ----------
    layer:
        Cylindrical detector layer.
    true_position:
        True 3D position [x, y, z].
    sigma_phi:
        Measurement resolution in phi.
    sigma_z:
        Measurement resolution in z.
    rng:
        Optional NumPy random generator.
    """

    if sigma_phi <= 0.0:
        raise ValueError("sigma_phi must be positive")

    if sigma_z <= 0.0:
        raise ValueError("sigma_z must be positive")

    true_position = np.asarray(true_position, dtype=float)

    if true_position.shape != (3,):
        raise ValueError("true_position must have shape (3,)")

    x, y, z = true_position

    true_radius = np.sqrt(x**2 + y**2)

    if not np.isclose(true_radius, layer.radius, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "true_position is not on the cylindrical layer radius"
        )

    if not layer.contains_z(z):
        raise ValueError("true_position z is outside the cylindrical layer length")

    phi = np.arctan2(y, x)

    true_values = np.array([phi, z])
    covariance = np.diag([sigma_phi**2, sigma_z**2])
    measured_values = smear_position(true_values, covariance, rng=rng)

    return Measurement(
        values=measured_values,
        covariance=covariance,
        layer_name=layer.name,
        surface_type=layer.surface_type,
    )
