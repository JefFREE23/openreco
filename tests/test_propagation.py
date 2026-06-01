import numpy as np
import pytest

from openreco.field import UniformMagneticField
from openreco.geometry import CylindricalLayer, make_barrel_detector
from openreco.particle_gun import Particle
from openreco.propagation import (
    PropagationResult,
    curvature_from_particle,
    find_cylinder_intersection_s,
    helix_momentum_at_s,
    helix_position_at_s,
    propagate_to_barrel_detector,
    propagate_to_cylindrical_layer,
    radial_distance,
    transverse_momentum,
)


def test_propagation_result_creation():
    result = PropagationResult(
        layer_name="barrel_0",
        position=np.array([10.0, 0.0, 1.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        path_length_xy=10.0,
    )

    assert result.layer_name == "barrel_0"
    np.testing.assert_allclose(result.position, np.array([10.0, 0.0, 1.0]))
    np.testing.assert_allclose(result.momentum, np.array([1.0, 0.0, 1.0]))
    assert result.path_length_xy == 10.0


def test_propagation_result_rejects_wrong_position_shape():
    with pytest.raises(ValueError):
        PropagationResult(
            layer_name="barrel_0",
            position=np.array([10.0, 0.0]),
            momentum=np.array([1.0, 0.0, 1.0]),
            path_length_xy=10.0,
        )


def test_propagation_result_rejects_wrong_momentum_shape():
    with pytest.raises(ValueError):
        PropagationResult(
            layer_name="barrel_0",
            position=np.array([10.0, 0.0, 1.0]),
            momentum=np.array([1.0, 0.0]),
            path_length_xy=10.0,
        )


def test_propagation_result_rejects_negative_path_length():
    with pytest.raises(ValueError):
        PropagationResult(
            layer_name="barrel_0",
            position=np.array([10.0, 0.0, 1.0]),
            momentum=np.array([1.0, 0.0, 1.0]),
            path_length_xy=-1.0,
        )


def test_transverse_momentum():
    momentum = np.array([3.0, 4.0, 12.0])

    assert transverse_momentum(momentum) == 5.0


def test_transverse_momentum_rejects_wrong_shape():
    with pytest.raises(ValueError):
        transverse_momentum(np.array([1.0, 2.0]))


def test_transverse_momentum_rejects_zero_pt():
    with pytest.raises(ValueError):
        transverse_momentum(np.array([0.0, 0.0, 1.0]))


def test_curvature_from_particle():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)

    curvature = curvature_from_particle(
        particle=particle,
        field=field,
        curvature_scale=0.003,
    )

    assert curvature == pytest.approx(0.003)


def test_curvature_changes_sign_with_charge():
    field = UniformMagneticField(bz=2.0)

    positive = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )
    negative = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=-1.0,
    )

    assert curvature_from_particle(positive, field) > 0.0
    assert curvature_from_particle(negative, field) < 0.0


def test_curvature_rejects_non_z_field():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bx=1.0, by=0.0, bz=2.0)

    with pytest.raises(ValueError):
        curvature_from_particle(particle, field)


def test_helix_position_straight_limit():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)

    position = helix_position_at_s(
        particle=particle,
        field=field,
        s_xy=10.0,
        curvature_scale=0.0,
    )

    np.testing.assert_allclose(position, np.array([10.0, 0.0, 10.0]))


def test_helix_position_rejects_negative_s():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField()

    with pytest.raises(ValueError):
        helix_position_at_s(particle, field, s_xy=-1.0)


def test_helix_momentum_preserves_magnitude():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)

    new_momentum = helix_momentum_at_s(
        particle=particle,
        field=field,
        s_xy=20.0,
    )

    assert np.linalg.norm(new_momentum) == pytest.approx(particle.p)


def test_radial_distance():
    position = np.array([3.0, 4.0, 12.0])

    assert radial_distance(position) == 5.0


def test_radial_distance_rejects_wrong_shape():
    with pytest.raises(ValueError):
        radial_distance(np.array([3.0, 4.0]))


def test_find_cylinder_intersection_straight_limit():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    s = find_cylinder_intersection_s(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=0.0,
    )

    assert s == pytest.approx(10.0)


def test_find_cylinder_intersection_rejects_bad_scan_settings():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField()
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    with pytest.raises(ValueError):
        find_cylinder_intersection_s(particle, field, layer, max_s=0.0)

    with pytest.raises(ValueError):
        find_cylinder_intersection_s(particle, field, layer, n_scan=1)


def test_propagate_to_cylindrical_layer_straight_limit():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=100.0)

    result = propagate_to_cylindrical_layer(
        particle=particle,
        field=field,
        layer=layer,
        curvature_scale=0.0,
    )

    assert result.layer_name == "barrel_0"
    np.testing.assert_allclose(result.position, np.array([10.0, 0.0, 10.0]))
    np.testing.assert_allclose(result.momentum, np.array([1.0, 0.0, 1.0]))
    assert result.path_length_xy == pytest.approx(10.0)


def test_propagate_to_cylindrical_layer_curved_hits_radius():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=20.0, half_length=100.0)

    result = propagate_to_cylindrical_layer(
        particle=particle,
        field=field,
        layer=layer,
    )

    assert radial_distance(result.position) == pytest.approx(20.0)
    assert layer.contains_z(result.position[2])


def test_propagate_to_cylindrical_layer_rejects_outside_z_length():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 10.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)
    layer = CylindricalLayer(name="barrel_0", radius=10.0, half_length=5.0)

    with pytest.raises(RuntimeError):
        propagate_to_cylindrical_layer(
            particle=particle,
            field=field,
            layer=layer,
            curvature_scale=0.0,
        )


def test_propagate_to_barrel_detector():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )
    field = UniformMagneticField(bz=2.0)
    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0],
        half_length=100.0,
    )

    results = propagate_to_barrel_detector(
        particle=particle,
        field=field,
        detector=detector,
        curvature_scale=0.0,
    )

    assert len(results) == 3
    np.testing.assert_allclose(
        [radial_distance(result.position) for result in results],
        np.array([10.0, 20.0, 30.0]),
    )

def azimuth(position):
    """
    Return azimuthal angle of a 3D position.
    """
    return float(np.arctan2(position[1], position[0]))


def test_charge_sign_flips_bending_direction():
    field = UniformMagneticField(bz=2.0)

    positive = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )

    negative = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=-1.0,
    )

    s_xy = 30.0

    positive_position = helix_position_at_s(
        particle=positive,
        field=field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    negative_position = helix_position_at_s(
        particle=negative,
        field=field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    positive_phi = azimuth(positive_position)
    negative_phi = azimuth(negative_position)

    assert positive_phi > 0.0
    assert negative_phi < 0.0
    assert abs(positive_phi) == pytest.approx(abs(negative_phi), rel=1e-6)


def test_bz_sign_flips_bending_direction():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )

    positive_field = UniformMagneticField(bz=2.0)
    negative_field = UniformMagneticField(bz=-2.0)

    s_xy = 30.0

    positive_b_position = helix_position_at_s(
        particle=particle,
        field=positive_field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    negative_b_position = helix_position_at_s(
        particle=particle,
        field=negative_field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    positive_b_phi = azimuth(positive_b_position)
    negative_b_phi = azimuth(negative_b_position)

    assert positive_b_phi > 0.0
    assert negative_b_phi < 0.0
    assert abs(positive_b_phi) == pytest.approx(abs(negative_b_phi), rel=1e-6)


def test_larger_bz_bends_more():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.0, 1.0]),
        charge=1.0,
    )

    weak_field = UniformMagneticField(bz=1.0)
    strong_field = UniformMagneticField(bz=3.0)

    s_xy = 30.0

    weak_position = helix_position_at_s(
        particle=particle,
        field=weak_field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    strong_position = helix_position_at_s(
        particle=particle,
        field=strong_field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    assert abs(azimuth(strong_position)) > abs(azimuth(weak_position))


def test_lower_pt_bends_more_than_higher_pt():
    field = UniformMagneticField(bz=2.0)

    low_pt_particle = Particle(
        position=np.zeros(3),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )

    high_pt_particle = Particle(
        position=np.zeros(3),
        momentum=np.array([5.0, 0.0, 1.0]),
        charge=1.0,
    )

    s_xy = 30.0

    low_pt_position = helix_position_at_s(
        particle=low_pt_particle,
        field=field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    high_pt_position = helix_position_at_s(
        particle=high_pt_particle,
        field=field,
        s_xy=s_xy,
        curvature_scale=0.03,
    )

    assert abs(azimuth(low_pt_position)) > abs(azimuth(high_pt_position))


def test_momentum_magnitude_is_conserved_in_uniform_bz():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.5, 1.0]),
        charge=1.0,
    )

    field = UniformMagneticField(bz=2.0)

    momentum_after = helix_momentum_at_s(
        particle=particle,
        field=field,
        s_xy=50.0,
        curvature_scale=0.03,
    )

    assert np.linalg.norm(momentum_after) == pytest.approx(particle.p)


def test_pz_is_conserved_in_uniform_bz():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.5, 1.25]),
        charge=1.0,
    )

    field = UniformMagneticField(bz=2.0)

    momentum_after = helix_momentum_at_s(
        particle=particle,
        field=field,
        s_xy=50.0,
        curvature_scale=0.03,
    )

    assert momentum_after[2] == pytest.approx(particle.momentum[2])


def test_curved_propagation_hits_requested_cylinder_radii():
    particle = Particle(
        position=np.zeros(3),
        momentum=np.array([2.0, 0.25, 1.0]),
        charge=1.0,
    )

    field = UniformMagneticField(bz=2.0)

    detector = make_barrel_detector(
        radii=[10.0, 20.0, 30.0, 40.0],
        half_length=200.0,
    )

    results = propagate_to_barrel_detector(
        particle=particle,
        field=field,
        detector=detector,
        curvature_scale=0.03,
    )

    for result, layer in zip(results, detector.layers):
        assert radial_distance(result.position) == pytest.approx(layer.radius)
