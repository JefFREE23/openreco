"""
Track state representation for OpenReco.

A TrackState stores the 5D charged-particle track parameters
at a reference z position:

    [x, y, tx, ty, q_over_p]

where:
    x, y       = transverse position at reference z
    tx, ty     = slopes dx/dz and dy/dz
    q_over_p   = charge divided by momentum

It also stores the 5x5 covariance matrix of these parameters.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrackState:
    """
    5D track state at a reference z position.
    """

    parameters: np.ndarray
    covariance: np.ndarray
    z: float

    def __post_init__(self):
        self.parameters = np.asarray(self.parameters, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.z = float(self.z)

        if self.parameters.shape != (5,):
            raise ValueError(
                f"TrackState parameters must have shape (5,), got {self.parameters.shape}"
            )

        if self.covariance.shape != (5, 5):
            raise ValueError(
                f"TrackState covariance must have shape (5, 5), got {self.covariance.shape}"
            )

        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("TrackState covariance matrix must be symmetric")

    @property
    def x(self) -> float:
        return self.parameters[0]

    @property
    def y(self) -> float:
        return self.parameters[1]

    @property
    def tx(self) -> float:
        return self.parameters[2]

    @property
    def ty(self) -> float:
        return self.parameters[3]

    @property
    def q_over_p(self) -> float:
        return self.parameters[4]

    def copy(self) -> "TrackState":
        """
        Return a deep copy of the track state.
        """
        return TrackState(
            parameters=self.parameters.copy(),
            covariance=self.covariance.copy(),
            z=self.z,
        )

    def as_vector(self) -> np.ndarray:
        """
        Return the state parameters as a 5D vector.
        """
        return self.parameters.copy()

    def position(self) -> np.ndarray:
        """
        Return the 2D position [x, y].
        """
        return self.parameters[:2].copy()

    def slopes(self) -> np.ndarray:
        """
        Return the slopes [tx, ty].
        """
        return self.parameters[2:4].copy()
