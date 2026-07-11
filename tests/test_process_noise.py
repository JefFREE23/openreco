import numpy as np

from openreco.detector_effects import DetectorEffectsConfig, LayerMaterial
from openreco.process_noise import (
    ALPHA_INDEX,
    Q_OVER_P_INDEX,
    TAN_LAMBDA_INDEX,
    angular_process_noise_variance,
    material_process_noise_for_layer,
    material_process_noise_matrix,
)


def test_angular_process_noise_variance_zero_material_is_zero():
    variance = angular_process_noise_variance(
        p_gev=1.0,
        x_over_x0=0.0,
    )

    assert variance == 0.0


def test_angular_process_noise_variance_increases_with_material():
    thin = angular_process_noise_variance(
        p_gev=2.0,
        x_over_x0=0.001,
    )
    thick = angular_process_noise_variance(
        p_gev=2.0,
        x_over_x0=0.01,
    )

    assert thick > thin


def test_angular_process_noise_variance_scales_quadratically():
    base = angular_process_noise_variance(
        p_gev=2.0,
        x_over_x0=0.01,
        process_noise_scale=1.0,
    )
    scaled = angular_process_noise_variance(
        p_gev=2.0,
        x_over_x0=0.01,
        process_noise_scale=2.0,
    )

    assert np.isclose(scaled, 4.0 * base)


def test_material_process_noise_matrix_has_expected_shape_and_entries():
    process_noise = material_process_noise_matrix(
        p_gev=2.0,
        x_over_x0=0.01,
    )

    assert process_noise.shape == (5, 5)
    assert np.allclose(process_noise, process_noise.T)

    assert process_noise[ALPHA_INDEX, ALPHA_INDEX] > 0.0
    assert process_noise[TAN_LAMBDA_INDEX, TAN_LAMBDA_INDEX] > 0.0
    assert process_noise[Q_OVER_P_INDEX, Q_OVER_P_INDEX] == 0.0

    eigenvalues = np.linalg.eigvalsh(process_noise)
    assert np.all(eigenvalues >= 0.0)


def test_material_process_noise_for_layer_uses_detector_effects_material():
    config = DetectorEffectsConfig(
        layer_materials=(
            LayerMaterial(layer_id=0, x_over_x0=0.0),
            LayerMaterial(layer_id=1, x_over_x0=0.01),
        )
    )

    layer_0_noise = material_process_noise_for_layer(
        detector_effects=config,
        layer_id=0,
        p_gev=2.0,
    )
    layer_1_noise = material_process_noise_for_layer(
        detector_effects=config,
        layer_id=1,
        p_gev=2.0,
    )

    assert np.allclose(layer_0_noise, np.zeros((5, 5)))
    assert layer_1_noise[ALPHA_INDEX, ALPHA_INDEX] > 0.0
    assert layer_1_noise[TAN_LAMBDA_INDEX, TAN_LAMBDA_INDEX] > 0.0


def test_process_noise_rejects_invalid_inputs():
    try:
        angular_process_noise_variance(
            p_gev=1.0,
            x_over_x0=0.01,
            process_noise_scale=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative process_noise_scale")

    try:
        material_process_noise_matrix(
            p_gev=1.0,
            x_over_x0=0.01,
            state_size=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for too-small state_size")