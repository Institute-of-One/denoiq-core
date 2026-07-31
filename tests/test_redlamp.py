"""The operational floor, the excess-response and erasure measures, and the alert logic."""

from __future__ import annotations

import numpy as np
import pytest

from denoiq_core.redlamp import (
    AMBER,
    GREEN,
    RED,
    RedLampCriteria,
    amplitude_map,
    assess,
    classify,
    classify_grid,
    contrast_recovery,
    false_structure_rate,
    information_floor,
)
from denoiq_core.redlamp import _analytic_d_prime as analytic_d_prime

KV = np.linspace(80.0, 140.0, 4)
MAS = np.geomspace(4.0, 300.0, 5)


@pytest.fixture(scope="module")
def atlas():
    return information_floor(KV, MAS, threshold=5.0)


def test_atlas_shape_and_axes(atlas):
    assert atlas.d_prime.shape == (MAS.size, KV.size)
    assert atlas.floor_mas.shape == KV.shape


def test_detectability_rises_with_dose_and_falls_with_kv(atlas):
    assert np.all(np.diff(atlas.d_prime, axis=0) > 0.0)  # more mAs, more d'
    assert np.all(np.diff(atlas.d_prime, axis=1) < 0.0)  # more kV, less contrast


def test_the_floor_contour_is_exactly_the_threshold(atlas):
    """The contour is closed-form, so it should be exact, not merely close."""
    for kv, mas in zip(atlas.kv, atlas.floor_mas, strict=True):
        from denoiq_core.physics import DEFAULT_MODEL, DEFAULT_PHANTOM

        d = analytic_d_prime(kv, mas, model=DEFAULT_MODEL, phantom=DEFAULT_PHANTOM)
        assert d == pytest.approx(atlas.threshold, rel=1e-9)


def test_the_floor_needs_more_mas_at_higher_kv(atlas):
    """The floor rises with kV.

    Contrast falls with kV in this model; if the floor did not rise, the atlas would have no
    shape to read.
    """
    assert np.all(np.diff(atlas.floor_mas) > 0.0)


def test_grid_classification_matches_the_contour(atlas):
    below = atlas.d_prime < atlas.threshold
    assert np.array_equal(atlas.level == 2, below)


def test_classify_grid_bands():
    d = np.array([[1.0, 5.5, 12.0]])
    codes = classify_grid(d, threshold=5.0, amber_factor=1.2)
    assert codes.tolist() == [[2, 1, 0]]


# --------------------------------------------------------------------------------------
# lesion-like responses and erasure
# --------------------------------------------------------------------------------------


def test_amplitude_map_reads_one_lesion_as_one(trials_high_dose):
    """The scale of the map is the point: a true lesion must read as 1.0."""
    trials = trials_high_dose
    clean_present = trials.signal.copy()
    amp = amplitude_map(clean_present, trials.signal)
    centre = tuple(s // 2 for s in amp.shape)
    assert amp[centre] == pytest.approx(1.0, rel=1e-6)


def test_no_false_structure_in_a_noise_free_flat_image(trials_high_dose):
    flat = np.zeros((4, *trials_high_dose.shape))
    result = false_structure_rate(flat, trials_high_dose.signal)
    assert result["rate"] == 0.0
    assert result["mean_max_amplitude"] == pytest.approx(0.0, abs=1e-9)


def test_fabricated_lesions_are_detected(trials_high_dose):
    """Plant a lesion where the truth is flat: the measure must find all of them."""
    trials = trials_high_dose
    fabricated = np.stack([trials.signal.copy() for _ in range(4)])
    result = false_structure_rate(fabricated, trials.signal)
    assert result["rate"] == 1.0


def test_noise_alone_produces_more_false_structure_at_lower_dose(trials_high_dose, trials_low_dose):
    high = false_structure_rate(trials_high_dose.absent, trials_high_dose.signal)
    low = false_structure_rate(trials_low_dose.absent, trials_low_dose.signal)
    assert low["mean_max_amplitude"] > high["mean_max_amplitude"]


def test_contrast_recovery_of_untouched_images_is_one(trials_high_dose):
    trials = trials_high_dose
    assert contrast_recovery(trials.present, trials.absent, trials.signal) == pytest.approx(
        1.0, abs=0.05
    )


def test_contrast_recovery_tracks_an_attenuated_lesion(trials_high_dose):
    trials = trials_high_dose
    halved = trials.absent + 0.5 * trials.signal
    assert contrast_recovery(halved, trials.absent, trials.signal) == pytest.approx(0.5, abs=1e-9)
    erased = trials.absent.copy()
    assert contrast_recovery(erased, trials.absent, trials.signal) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# the alert logic
# --------------------------------------------------------------------------------------


GOOD = {
    "ceiling_d_prime": 9.0,
    "task_d_prime": 8.0,
    "raw_task_d_prime": 8.5,
    "ssim": 0.8,
    "false_rate": 0.05,
    "raw_false_rate": 0.05,
    "recovery": 0.95,
}


def test_green_when_everything_is_in_order():
    assert classify(**GOOD).level == GREEN


def test_red_below_the_floor():
    lamp = classify(**{**GOOD, "ceiling_d_prime": 3.0})
    assert lamp.level == RED
    assert "floor" in lamp.reasons[0]
    # The reason must say what is true — the requirement is not met and processing cannot
    # restore it — and must not claim the information is absent.
    assert "does not meet the detectability requirement" in lamp.reasons[0]
    for overclaim in ("plausibility, not information", "impossible", "no information"):
        assert overclaim not in lamp.reasons[0]


def test_red_when_fidelity_is_high_but_the_task_is_inadequate():
    lamp = classify(**{**GOOD, "task_d_prime": 0.4, "ssim": 0.9})
    assert lamp.level == RED
    assert any("task-inadequate" in reason for reason in lamp.reasons)
    # The threshold is adequacy, not chance: it must not be described as chance performance.
    assert not any("chance" in reason for reason in lamp.reasons)


def test_red_on_excess_lesion_like_responses():
    lamp = classify(**{**GOOD, "false_rate": 0.6, "raw_false_rate": 0.1})
    assert lamp.level == RED
    assert any("excess false structure" in reason for reason in lamp.reasons)


def test_excess_responses_are_not_blamed_on_the_denoiser_when_the_input_does_it_too():
    """Noise alone produces lesion-like responses; only an *excess* over the input counts."""
    lamp = classify(**{**GOOD, "false_rate": 0.6, "raw_false_rate": 0.7})
    assert lamp.level == GREEN


def test_red_on_erasure():
    lamp = classify(**{**GOOD, "recovery": 0.2})
    assert lamp.level == RED
    assert any("erasure" in reason for reason in lamp.reasons)


def test_amber_near_the_floor():
    lamp = classify(
        **{**GOOD, "ceiling_d_prime": 5.5, "task_d_prime": 5.2, "raw_task_d_prime": 5.3}
    )
    assert lamp.level == AMBER


def test_amber_when_the_task_falls_but_the_picture_does_not():
    lamp = classify(**{**GOOD, "task_d_prime": 5.0, "raw_task_d_prime": 8.5})
    assert lamp.level == AMBER
    assert any("fidelity-task discordance" in reason for reason in lamp.reasons)


def test_criteria_are_configurable():
    strict = RedLampCriteria(d_prime_threshold=10.0)
    assert classify(**GOOD, criteria=strict).level == RED


def test_every_verdict_carries_a_reason():
    for override in ({}, {"ceiling_d_prime": 2.0}, {"task_d_prime": 5.0, "raw_task_d_prime": 8.5}):
        lamp = classify(**{**GOOD, **override})
        assert lamp.reasons and lamp.reasons[0].strip()


def test_assess_runs_end_to_end_and_reports_its_evidence():
    lamp = assess(120.0, 6.25, "gaussian", {"sigma": 3.0}, n_trials=120, seed=17)
    assert lamp.level == RED
    assert lamp.metrics["ceiling_d_prime"] < 5.0
    for key in ("task_d_prime", "ssim", "false_structure_rate", "contrast_recovery", "criteria"):
        assert key in lamp.metrics
