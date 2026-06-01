import numpy as np
import pytest

from openreco.field import UniformMagneticField


def test_uniform_magnetic_field_default_creation():
    field = UniformMagneticField()

    assert field.bx == 0.0
    assert field.by == 0.0
    assert field.bz == 2.0

    np.testing.assert_allclose(field.vector, np.array([0.0, 0.0, 2.0]))


def test_uniform_magnetic_field_custom_creation():
    field = UniformMagneticField(bx=1.0, by=2.0, bz=3.0)

    assert field.bx == 1.0
    assert field.by == 2.0
    assert field.bz == 3.0

    np.testing.assert_allclose(field.vector, np.array([1.0, 2.0, 3.0]))


def test_uniform_magnetic_field_converts_values_to_float():
    field = UniformMagneticField(bx=1, by=2, bz=3)

    assert isinstance(field.bx, float)
    assert isinstance(field.by, float)
    assert isinstance(field.bz, float)


def test_uniform_magnetic_field_magnitude():
    field = UniformMagneticField(bx=3.0, by=4.0, bz=0.0)

    assert field.magnitude == 5.0


def test_uniform_magnetic_field_at_position():
    field = UniformMagneticField(bx=0.0, by=0.0, bz=2.0)

    b_at_origin = field.at(np.array([0.0, 0.0, 0.0]))
    b_at_other = field.at(np.array([10.0, -5.0, 3.0]))

    np.testing.assert_allclose(b_at_origin, np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(b_at_other, np.array([0.0, 0.0, 2.0]))


def test_uniform_magnetic_field_at_rejects_wrong_position_shape():
    field = UniformMagneticField()

    with pytest.raises(ValueError):
        field.at(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        field.at(np.zeros((3, 1)))


def test_uniform_magnetic_field_is_along_z_true():
    field = UniformMagneticField(bx=0.0, by=0.0, bz=2.0)

    assert field.is_along_z()


def test_uniform_magnetic_field_is_along_z_false():
    field = UniformMagneticField(bx=0.1, by=0.0, bz=2.0)

    assert not field.is_along_z()
