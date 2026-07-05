import pytest

from openreco.detector_effects import (
    DeadLayerModel,
    DetectorEffectsConfig,
    HitResolutionModel,
    InefficiencyModel,
    LayerMaterial,
    MagneticFieldScale,
    NoiseOccupancyModel,
)


def test_default_config_is_clean_detector():
    config = DetectorEffectsConfig.default()

    assert config.hit_resolution.sigma_phi > 0.0
    assert config.hit_resolution.sigma_z > 0.0
    assert config.layer_materials == ()
    assert config.inefficiency.hit_efficiency == 1.0
    assert config.dead_layers.dead_layers == frozenset()
    assert config.noise_occupancy.mean_noise_hits_per_layer == 0.0
    assert config.b_field_scale.truth_scale == 1.0
    assert config.b_field_scale.reco_scale == 1.0


def test_hit_resolution_covariance_diagonal():
    model = HitResolutionModel(sigma_phi=0.002, sigma_z=0.3)

    assert model.covariance_diagonal == pytest.approx((0.002**2, 0.3**2))


@pytest.mark.parametrize("sigma_phi,sigma_z", [(-1.0, 0.1), (0.001, -0.1)])
def test_hit_resolution_rejects_negative_values(sigma_phi, sigma_z):
    with pytest.raises(ValueError):
        HitResolutionModel(sigma_phi=sigma_phi, sigma_z=sigma_z)


@pytest.mark.parametrize("hit_efficiency", [-0.1, 1.1])
def test_inefficiency_rejects_invalid_probability(hit_efficiency):
    with pytest.raises(ValueError):
        InefficiencyModel(hit_efficiency=hit_efficiency)


def test_inefficiency_drop_probability():
    model = InefficiencyModel(hit_efficiency=0.8)

    assert model.drop_probability == pytest.approx(0.2)


@pytest.mark.parametrize("x_over_x0", [-0.001, -1.0])
def test_layer_material_rejects_negative_material(x_over_x0):
    with pytest.raises(ValueError):
        LayerMaterial(layer_id=0, x_over_x0=x_over_x0)


def test_layer_material_rejects_negative_energy_loss():
    with pytest.raises(ValueError):
        LayerMaterial(layer_id=0, x_over_x0=0.01, energy_loss_mev=-1.0)


def test_dead_layer_model_stores_layers_and_checks_membership():
    model = DeadLayerModel(dead_layers=[1, 3, 5])

    assert model.dead_layers == frozenset({1, 3, 5})
    assert model.is_dead(3)
    assert not model.is_dead(4)


def test_dead_layer_rejects_negative_layer_id():
    with pytest.raises(ValueError):
        DeadLayerModel(dead_layers=[-1])


def test_noise_occupancy_rejects_negative_rate():
    with pytest.raises(ValueError):
        NoiseOccupancyModel(mean_noise_hits_per_layer=-0.5)


def test_magnetic_field_scale_rejects_non_positive_values():
    with pytest.raises(ValueError):
        MagneticFieldScale(truth_scale=0.0, reco_scale=1.0)

    with pytest.raises(ValueError):
        MagneticFieldScale(truth_scale=1.0, reco_scale=-1.0)


def test_magnetic_field_mismatch_ratio():
    scale = MagneticFieldScale(truth_scale=1.0, reco_scale=1.02)

    assert scale.mismatch == pytest.approx(1.02)


def test_config_accepts_layer_materials_and_finds_material_by_layer():
    config = DetectorEffectsConfig(
        layer_materials=(
            LayerMaterial(layer_id=0, x_over_x0=0.001),
            LayerMaterial(layer_id=1, x_over_x0=0.002),
        )
    )

    assert config.material_for_layer(0).x_over_x0 == pytest.approx(0.001)
    assert config.material_for_layer(1).x_over_x0 == pytest.approx(0.002)


def test_config_returns_vacuum_material_for_missing_layer():
    config = DetectorEffectsConfig(
        layer_materials=(LayerMaterial(layer_id=0, x_over_x0=0.001),)
    )

    missing = config.material_for_layer(5)

    assert missing.layer_id == 5
    assert missing.x_over_x0 == 0.0
    assert missing.energy_loss_mev == 0.0


def test_config_rejects_duplicate_layer_materials():
    with pytest.raises(ValueError):
        DetectorEffectsConfig(
            layer_materials=(
                LayerMaterial(layer_id=2, x_over_x0=0.001),
                LayerMaterial(layer_id=2, x_over_x0=0.002),
            )
        )


def test_uniform_material_helper():
    config = DetectorEffectsConfig.with_uniform_material(
        layer_ids=[0, 1, 2],
        x_over_x0=0.005,
        energy_loss_mev=0.1,
    )

    assert len(config.layer_materials) == 3
    assert config.material_for_layer(0).x_over_x0 == pytest.approx(0.005)
    assert config.material_for_layer(1).energy_loss_mev == pytest.approx(0.1)
    assert config.material_for_layer(2).x_over_x0 == pytest.approx(0.005)