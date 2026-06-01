import numpy as np
import pytest

from openreco.particle_gun import Particle, ParticleGun, make_fixed_particle


def test_particle_creation():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 2.0, 2.0]),
        charge=1.0,
    )

    np.testing.assert_allclose(particle.position, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(particle.momentum, np.array([1.0, 2.0, 2.0]))
    assert particle.charge == 1.0


def test_particle_rejects_wrong_position_shape():
    with pytest.raises(ValueError):
        Particle(
            position=np.array([0.0, 0.0]),
            momentum=np.array([1.0, 0.0, 0.0]),
            charge=1.0,
        )


def test_particle_rejects_wrong_momentum_shape():
    with pytest.raises(ValueError):
        Particle(
            position=np.array([0.0, 0.0, 0.0]),
            momentum=np.array([1.0, 0.0]),
            charge=1.0,
        )


def test_particle_rejects_zero_momentum():
    with pytest.raises(ValueError):
        Particle(
            position=np.array([0.0, 0.0, 0.0]),
            momentum=np.array([0.0, 0.0, 0.0]),
            charge=1.0,
        )


def test_particle_rejects_zero_charge():
    with pytest.raises(ValueError):
        Particle(
            position=np.array([0.0, 0.0, 0.0]),
            momentum=np.array([1.0, 0.0, 0.0]),
            charge=0.0,
        )


def test_particle_momentum_properties():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([3.0, 4.0, 12.0]),
        charge=1.0,
    )

    assert particle.pt == 5.0
    assert particle.p == 13.0
    assert particle.q_over_p == 1.0 / 13.0
    np.testing.assert_allclose(
        particle.direction,
        np.array([3.0, 4.0, 12.0]) / 13.0,
    )


def test_particle_copy_is_deep_copy():
    particle = Particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 2.0, 3.0]),
        charge=-1.0,
    )

    copied = particle.copy()

    copied.position[0] = 99.0
    copied.momentum[0] = 99.0

    assert particle.position[0] == 0.0
    assert particle.momentum[0] == 1.0


def test_particle_gun_creation_default():
    gun = ParticleGun(rng=np.random.default_rng(123))

    np.testing.assert_allclose(gun.position, np.zeros(3))
    assert gun.momentum_range == (1.0, 5.0)
    assert gun.charge_choices == (-1.0, 1.0)


def test_particle_gun_rejects_wrong_position_shape():
    with pytest.raises(ValueError):
        ParticleGun(position=np.array([0.0, 0.0]))


def test_particle_gun_rejects_non_positive_momentum_range():
    with pytest.raises(ValueError):
        ParticleGun(momentum_range=(0.0, 5.0))

    with pytest.raises(ValueError):
        ParticleGun(momentum_range=(-1.0, 5.0))


def test_particle_gun_rejects_reversed_momentum_range():
    with pytest.raises(ValueError):
        ParticleGun(momentum_range=(5.0, 1.0))


def test_particle_gun_rejects_reversed_theta_range():
    with pytest.raises(ValueError):
        ParticleGun(theta_range=(1.0, 0.5))


def test_particle_gun_rejects_reversed_phi_range():
    with pytest.raises(ValueError):
        ParticleGun(phi_range=(1.0, 0.5))


def test_particle_gun_rejects_empty_charge_choices():
    with pytest.raises(ValueError):
        ParticleGun(charge_choices=())


def test_particle_gun_rejects_zero_charge_choice():
    with pytest.raises(ValueError):
        ParticleGun(charge_choices=(-1.0, 0.0, 1.0))


def test_particle_gun_shoot_returns_particle():
    gun = ParticleGun(rng=np.random.default_rng(123))

    particle = gun.shoot()

    assert isinstance(particle, Particle)
    assert particle.p >= 1.0
    assert particle.p <= 5.0
    assert particle.charge in (-1.0, 1.0)


def test_particle_gun_shoot_is_reproducible_with_seed():
    gun_1 = ParticleGun(rng=np.random.default_rng(123))
    gun_2 = ParticleGun(rng=np.random.default_rng(123))

    particle_1 = gun_1.shoot()
    particle_2 = gun_2.shoot()

    np.testing.assert_allclose(particle_1.position, particle_2.position)
    np.testing.assert_allclose(particle_1.momentum, particle_2.momentum)
    assert particle_1.charge == particle_2.charge


def test_particle_gun_fixed_ranges():
    gun = ParticleGun(
        momentum_range=(2.0, 2.0),
        theta_range=(np.pi / 2.0, np.pi / 2.0),
        phi_range=(0.0, 0.0),
        charge_choices=(1.0,),
        rng=np.random.default_rng(123),
    )

    particle = gun.shoot()

    np.testing.assert_allclose(particle.momentum, np.array([2.0, 0.0, 0.0]), atol=1e-12)
    assert particle.charge == 1.0


def test_make_fixed_particle():
    particle = make_fixed_particle(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 1.0]),
        charge=1.0,
    )

    assert isinstance(particle, Particle)
    np.testing.assert_allclose(particle.position, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(particle.momentum, np.array([1.0, 0.0, 1.0]))
    assert particle.charge == 1.0
