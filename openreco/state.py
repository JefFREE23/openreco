"""
Track state representation for OpenReco.

This module defines a surface-bound 5D TrackState.

Roadmap intent:
    A charged track is represented by five parameters on a reference surface
    plus a 5x5 covariance matrix.

Generic bound parameter vector:
    [loc0, loc1, dir0, dir1, q_over_p]

For a cylindrical surface:
    loc0     = phi
    loc1     = z
    dir0     = transverse/local direction parameter
    dir1     = longitudinal direction parameter
    q_over_p = charge / momentum

For a planar checkpoint surface:
    loc0     = x
    loc1     = y
    dir0     = dx/dz
    dir1     = dy/dz
    q_over_p = charge / momentum

This is intentionally closer to the ACTS idea of bound parameters on
surfaces, while still staying minimal for OpenReco v0.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackState:
    """
    Five-parameter track state bound to a reference surface.

    Parameters
    ----------
    parameters:
        5D bound parameter vector:
            [loc0, loc1, dir0, dir1, q_over_p]
    covariance:
        5x5 covariance matrix.
    surface_type:
        Surface type, for example "cylinder" or "plane".
    surface_name:
        Name of the reference surface/layer.
    surface_radius:
        Radius for cylindrical surfaces. Required for surface_type="cylinder".
        Should be None for plane surfaces.
    """

    parameters: np.ndarray
    covariance: np.ndarray
    surface_type: str
    surface_name: str = ""
    surface_radius: float | None = None

    def __post_init__(self):
        parameters = np.asarray(self.parameters, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)

        if parameters.shape != (5,):
            raise ValueError(
                f"TrackState parameters must have shape (5,), got {parameters.shape}"
            )

        if covariance.shape != (5, 5):
            raise ValueError(
                f"TrackState covariance must have shape (5, 5), got {covariance.shape}"
            )

        if not np.allclose(covariance, covariance.T):
            raise ValueError("TrackState covariance matrix must be symmetric")

        if not isinstance(self.surface_type, str):
            raise TypeError("surface_type must be a string")

        if not isinstance(self.surface_name, str):
            raise TypeError("surface_name must be a string")

        if self.surface_type not in ("cylinder", "plane"):
            raise ValueError("surface_type must be either 'cylinder' or 'plane'")

        if self.surface_type == "cylinder":
            if self.surface_radius is None:
                raise ValueError("surface_radius is required for cylindrical states")

            radius = float(self.surface_radius)

            if radius <= 0.0:
                raise ValueError("surface_radius must be positive")

            object.__setattr__(self, "surface_radius", radius)

        if self.surface_type == "plane":
            if self.surface_radius is not None:
                raise ValueError("surface_radius must be None for planar states")

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "covariance", covariance)

    @property
    def loc0(self) -> float:
        return float(self.parameters[0])

    @property
    def loc1(self) -> float:
        return float(self.parameters[1])

    @property
    def dir0(self) -> float:
        return float(self.parameters[2])

    @property
    def dir1(self) -> float:
        return float(self.parameters[3])

    @property
    def q_over_p(self) -> float:
        return float(self.parameters[4])

    @property
    def phi(self) -> float:
        """
        Cylindrical angular coordinate.

        Only valid for cylindrical states.
        """
        if self.surface_type != "cylinder":
            raise AttributeError("phi is only defined for cylindrical states")

        return self.loc0

    @property
    def z(self) -> float:
        """
        Longitudinal coordinate.

        For cylindrical states, loc1 is z.
        """
        if self.surface_type != "cylinder":
            raise AttributeError("z is only defined for cylindrical states")

        return self.loc1

    @property
    def x(self) -> float:
        """
        Global x position implied by the bound state.

        For cylindrical states:
            x = R cos(phi)

        For planar states:
            x = loc0
        """
        if self.surface_type == "cylinder":
            return float(self.surface_radius * np.cos(self.phi))

        return self.loc0

    @property
    def y(self) -> float:
        """
        Global y position implied by the bound state.

        For cylindrical states:
            y = R sin(phi)

        For planar states:
            y = loc1
        """
        if self.surface_type == "cylinder":
            return float(self.surface_radius * np.sin(self.phi))

        return self.loc1

    @property
    def radius(self) -> float:
        """
        Surface radius for cylindrical states.
        """
        if self.surface_type != "cylinder":
            raise AttributeError("radius is only defined for cylindrical states")

        return float(self.surface_radius)

    def local_position(self) -> np.ndarray:
        """
        Return local surface position [loc0, loc1].
        """
        return self.parameters[:2].copy()

    def direction_parameters(self) -> np.ndarray:
        """
        Return direction-like parameters [dir0, dir1].
        """
        return self.parameters[2:4].copy()

    def global_position(self) -> np.ndarray:
        """
        Return approximate global position [x, y, z].

        For planar states, z is not defined by the state itself, so this returns
        [x, y, 0]. Planar states are only a checkpoint path.
        """
        if self.surface_type == "cylinder":
            return np.array([self.x, self.y, self.z], dtype=float)

        return np.array([self.x, self.y, 0.0], dtype=float)

    def as_vector(self) -> np.ndarray:
        """
        Return a copy of the 5D bound parameter vector.
        """
        return self.parameters.copy()

    def copy(self) -> "TrackState":
        """
        Return a deep copy of the state.
        """
        return TrackState(
            parameters=self.parameters.copy(),
            covariance=self.covariance.copy(),
            surface_type=self.surface_type,
            surface_name=self.surface_name,
            surface_radius=self.surface_radius,
        )


def make_cylindrical_state(
    phi: float,
    z: float,
    dir0: float,
    dir1: float,
    q_over_p: float,
    covariance: np.ndarray,
    surface_radius: float,
    surface_name: str = "",
) -> TrackState:
    """
    Convenience constructor for a cylindrical bound TrackState.
    """

    parameters = np.array([phi, z, dir0, dir1, q_over_p], dtype=float)

    return TrackState(
        parameters=parameters,
        covariance=covariance,
        surface_type="cylinder",
        surface_name=surface_name,
        surface_radius=surface_radius,
    )


def make_planar_state(
    x: float,
    y: float,
    tx: float,
    ty: float,
    q_over_p: float,
    covariance: np.ndarray,
    surface_name: str = "",
) -> TrackState:
    """
    Convenience constructor for a planar checkpoint TrackState.
    """

    parameters = np.array([x, y, tx, ty, q_over_p], dtype=float)

    return TrackState(
        parameters=parameters,
        covariance=covariance,
        surface_type="plane",
        surface_name=surface_name,
        surface_radius=None,
    )
