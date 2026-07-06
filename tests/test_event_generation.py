import numpy as np

from openreco.event_generation import (
    count_noise_hits,
    count_real_hits,
    generate_event,
    make_default_barrel,
)

from openreco.detector_effects import (
    DeadLayerModel,
    DetectorEffectsConfig,
    HitResolutionModel,
    InefficiencyModel,
    LayerMaterial,
    NoiseOccupancyModel,
)

def test_generate_event_detector_effects_noise_occupancy_overrides_legacy_argument():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        noise_occupancy=NoiseOccupancyModel(mean_noise_hits_per_layer=0.0)
    )

    event = generate_event(
        event_id=0,
        n_particles=0,
        noise_hits_per_layer=5,
        detector_effects=config,
        rng=rng,
    )

    assert count_real_hits(event) == 0
    assert count_noise_hits(event) == 0


def test_generate_event_detector_effects_noise_occupancy_can_create_noise_hits():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        noise_occupancy=NoiseOccupancyModel(mean_noise_hits_per_layer=2.0)
    )

    event = generate_event(
        event_id=0,
        n_particles=0,
        detector_effects=config,
        rng=rng,
    )

    assert count_real_hits(event) == 0
    assert count_noise_hits(event) > 0

    for hit in event.measurements:
        assert hit.is_noise is True
        assert hit.truth_particle_id is None
        

def test_generate_event_with_five_particles_has_truth_ids():
    rng = np.random.default_rng(123)
    detector = make_default_barrel()

    event = generate_event(
        event_id=1,
        n_particles=5,
        detector=detector,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=rng,
    )

    assert event.event_id == 1
    assert len(event.truth_particles) == 5
    assert len(event.measurements_by_layer) == 6

    for layer_name, hits in event.measurements_by_layer.items():
        assert layer_name.startswith("barrel_")
        assert len(hits) == 5

        for hit in hits:
            assert hit.truth_particle_id is not None
            assert hit.is_noise is False
            assert hit.layer_name == layer_name
            assert hit.radius > 0.0
            assert hit.covariance.shape == (2, 2)

    assert count_real_hits(event) == 30
    assert count_noise_hits(event) == 0


def test_noise_hits_have_no_truth_particle_id():
    rng = np.random.default_rng(123)
    detector = make_default_barrel()

    event = generate_event(
        n_particles=2,
        detector=detector,
        hit_efficiency=1.0,
        noise_hits_per_layer=3,
        rng=rng,
    )

    assert count_real_hits(event) == 12
    assert count_noise_hits(event) == 18

    noise_hits = [hit for hit in event.measurements if hit.is_noise]
    assert len(noise_hits) == 18

    for hit in noise_hits:
        assert hit.truth_particle_id is None


def test_zero_hit_efficiency_produces_only_noise_when_noise_enabled():
    rng = np.random.default_rng(123)
    detector = make_default_barrel()

    event = generate_event(
        n_particles=5,
        detector=detector,
        hit_efficiency=0.0,
        noise_hits_per_layer=1,
        rng=rng,
    )

    assert count_real_hits(event) == 0
    assert count_noise_hits(event) == 6


def test_hit_global_position_is_consistent_with_radius():
    rng = np.random.default_rng(123)

    event = generate_event(
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=rng,
    )

    for hit in event.measurements:
        x, y, z = hit.global_position
        reconstructed_radius = np.sqrt(x * x + y * y)

        assert np.isclose(reconstructed_radius, hit.radius)
        assert np.isclose(z, hit.z)


def test_generate_event_uses_detector_effects_hit_resolution_in_covariance():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        hit_resolution=HitResolutionModel(sigma_phi=0.002, sigma_z=0.25)
    )

    event = generate_event(
        event_id=0,
        n_particles=1,
        detector_effects=config,
        rng=rng,
    )

    real_hits = [hit for hit in event.measurements if not hit.is_noise]
    assert real_hits

    for hit in real_hits:
        np.testing.assert_allclose(
            hit.covariance,
            np.diag([0.002**2, 0.25**2]),
        )


def test_generate_event_detector_effects_overrides_legacy_sigma_arguments():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        hit_resolution=HitResolutionModel(sigma_phi=0.003, sigma_z=0.30)
    )

    event = generate_event(
        event_id=0,
        n_particles=1,
        measurement_sigma_phi=0.001,
        measurement_sigma_z=0.10,
        detector_effects=config,
        rng=rng,
    )

    real_hits = [hit for hit in event.measurements if not hit.is_noise]
    assert real_hits

    for hit in real_hits:
        np.testing.assert_allclose(
            hit.covariance,
            np.diag([0.003**2, 0.30**2]),
        )
        

def test_generate_event_uses_detector_effects_hit_efficiency():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        inefficiency=InefficiencyModel(hit_efficiency=0.0)
    )

    event = generate_event(
        event_id=0,
        n_particles=3,
        detector_effects=config,
        noise_hits_per_layer=0,
        rng=rng,
    )

    assert count_real_hits(event) == 0
    assert count_noise_hits(event) == 0


def test_generate_event_detector_effects_hit_efficiency_overrides_legacy_argument():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        inefficiency=InefficiencyModel(hit_efficiency=0.0)
    )

    event = generate_event(
        event_id=0,
        n_particles=3,
        hit_efficiency=1.0,
        detector_effects=config,
        noise_hits_per_layer=0,
        rng=rng,
    )

    assert count_real_hits(event) == 0


def test_generate_event_dead_layers_produce_no_measurements_on_selected_layers():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        dead_layers=DeadLayerModel(dead_layers=[1, 4])
    )

    event = generate_event(
        event_id=0,
        n_particles=3,
        detector_effects=config,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=rng,
    )

    assert len(event.measurements_by_layer["barrel_0"]) == 3
    assert len(event.measurements_by_layer["barrel_1"]) == 0
    assert len(event.measurements_by_layer["barrel_2"]) == 3
    assert len(event.measurements_by_layer["barrel_3"]) == 3
    assert len(event.measurements_by_layer["barrel_4"]) == 0
    assert len(event.measurements_by_layer["barrel_5"]) == 3

    assert count_real_hits(event) == 12


def test_generate_event_dead_layers_suppress_noise_hits_too():
    rng = np.random.default_rng(123)

    config = DetectorEffectsConfig(
        inefficiency=InefficiencyModel(hit_efficiency=0.0),
        dead_layers=DeadLayerModel(dead_layers=[2]),
        noise_occupancy=NoiseOccupancyModel(mean_noise_hits_per_layer=2.0),
    )

    event = generate_event(
        event_id=0,
        n_particles=0,
        detector_effects=config,
        rng=rng,
    )

    assert len(event.measurements_by_layer["barrel_2"]) == 0
    assert count_real_hits(event) == 0
    assert count_noise_hits(event) > 0

    for hit in event.measurements:
        assert hit.is_noise is True
        assert hit.truth_particle_id is None
        assert hit.layer_index != 2

def test_generate_event_zero_material_matches_clean_detector_for_same_seed():
    seed = 123

    clean_event = generate_event(
        event_id=0,
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=np.random.default_rng(seed),
    )

    material_event = generate_event(
        event_id=0,
        n_particles=1,
        detector_effects=DetectorEffectsConfig.with_uniform_material(
            layer_ids=range(6),
            x_over_x0=0.0,
        ),
        noise_hits_per_layer=0,
        rng=np.random.default_rng(seed),
    )

    clean_hits = [hit for hit in clean_event.measurements if not hit.is_noise]
    material_hits = [hit for hit in material_event.measurements if not hit.is_noise]

    assert len(clean_hits) == len(material_hits)

    for clean_hit, material_hit in zip(clean_hits, material_hits):
        assert clean_hit.layer_index == material_hit.layer_index
        assert clean_hit.truth_particle_id == material_hit.truth_particle_id
        assert np.isclose(clean_hit.phi, material_hit.phi)
        assert np.isclose(clean_hit.z, material_hit.z)


def test_generate_event_material_scattering_changes_downstream_hit_positions():
    seed = 123

    clean_event = generate_event(
        event_id=0,
        n_particles=1,
        hit_efficiency=1.0,
        noise_hits_per_layer=0,
        rng=np.random.default_rng(seed),
    )

    scattering_event = generate_event(
        event_id=0,
        n_particles=1,
        detector_effects=DetectorEffectsConfig(
            layer_materials=tuple(
                LayerMaterial(layer_id=i, x_over_x0=0.05)
                for i in range(6)
            )
        ),
        noise_hits_per_layer=0,
        rng=np.random.default_rng(seed),
    )

    clean_phis = np.array(
        [hit.phi for hit in clean_event.measurements if not hit.is_noise]
    )
    scattering_phis = np.array(
        [hit.phi for hit in scattering_event.measurements if not hit.is_noise]
    )

    assert clean_phis.shape == scattering_phis.shape
    assert not np.allclose(clean_phis, scattering_phis)