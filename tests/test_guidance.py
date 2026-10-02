"""The guidance is the part a reader acts on, so it is the part that must not be wrong.

Every conversion here has a closed form, so the tests state the closed form rather than a
remembered number. The ones that matter most are the square in :func:`achievable_floor` -- an
efficiency that sounds like a quarter off costs most of a doubling in exposure, and getting the
power wrong would understate every protocol it is applied to -- and the warnings, which are the
only thing standing between a reader and an extrapolation the dose axis does not support.
"""

from __future__ import annotations

import numpy as np
import pytest

from denoiq_core.guidance import (
    EXTRAPOLATION_LIMIT,
    achievable_floor,
    apparent_exposure,
    decide,
    exposure_for_requirement,
)


# --------------------------------------------------------------------------------------
# exposure_for_requirement
# --------------------------------------------------------------------------------------


def test_meeting_the_requirement_exactly_needs_the_exposure_in_use():
    assert exposure_for_requirement(100.0, 5.0, 5.0) == pytest.approx(100.0)


def test_exposure_goes_as_the_square_of_the_detectability_ratio():
    """Halving d' costs four times the exposure, which is the whole reason dose is expensive."""
    assert exposure_for_requirement(100.0, 2.5, 5.0) == pytest.approx(400.0)
    assert exposure_for_requirement(100.0, 10.0, 5.0) == pytest.approx(25.0)


def test_the_answer_is_in_the_units_it_was_asked_in():
    # Measured twice the requirement, so a quarter of the exposure would meet it.
    for unit in (1.0, 100.0, 0.25):
        assert exposure_for_requirement(unit, 6.0, 3.0) == pytest.approx(unit * 0.25)


def test_it_inverts_the_square_root_law():
    """Scale the exposure to the answer and the detectability must be the requirement."""
    current, measured, requirement = 120.0, 6.025, 5.0
    target = exposure_for_requirement(current, measured, requirement)
    assert measured * np.sqrt(target / current) == pytest.approx(requirement)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_it_refuses_impossible_inputs(bad):
    with pytest.raises(ValueError):
        exposure_for_requirement(bad, 5.0, 5.0)
    with pytest.raises(ValueError):
        exposure_for_requirement(100.0, bad, 5.0)
    with pytest.raises(ValueError):
        exposure_for_requirement(100.0, 5.0, bad)


# --------------------------------------------------------------------------------------
# achievable_floor
# --------------------------------------------------------------------------------------


def test_a_perfect_observer_reaches_the_ideal_floor():
    assert achievable_floor(100.0, 1.0) == pytest.approx(100.0)


def test_the_correction_is_the_inverse_square_of_the_efficiency():
    """The square, not the fraction: this is the defect the whole module exists to prevent."""
    assert achievable_floor(100.0, 0.5) == pytest.approx(400.0)
    assert achievable_floor(100.0, 0.75) == pytest.approx(100.0 / 0.5625)
    # An efficiency that sounds like a quarter off costs most of a doubling.
    assert achievable_floor(1.0, 0.75) == pytest.approx(1.7778, rel=1e-4)


def test_a_worse_observer_always_needs_more_exposure():
    floors = [achievable_floor(100.0, eta) for eta in (1.0, 0.8, 0.6, 0.4)]
    assert floors == sorted(floors)


def test_it_corrects_a_whole_atlas_contour_at_once():
    ideal = np.array([50.0, 80.0, 120.0])
    corrected = achievable_floor(ideal, 0.5)
    assert isinstance(corrected, np.ndarray)
    assert np.allclose(corrected, ideal * 4.0)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_it_refuses_an_efficiency_outside_the_unit_interval(bad):
    with pytest.raises(ValueError):
        achievable_floor(100.0, bad)


def test_it_refuses_a_floor_that_is_not_an_exposure():
    with pytest.raises(ValueError):
        achievable_floor(0.0, 0.75)
    with pytest.raises(ValueError):
        achievable_floor(np.array([50.0, -1.0]), 0.75)


# --------------------------------------------------------------------------------------
# apparent_exposure
# --------------------------------------------------------------------------------------


def test_no_fidelity_gain_claims_nothing():
    assert apparent_exposure(100.0, 0.0) == pytest.approx(100.0)


def test_three_decibels_claims_twice_the_exposure():
    assert apparent_exposure(100.0, 10.0 * np.log10(2.0)) == pytest.approx(200.0)


def test_the_claim_is_the_exposure_with_that_much_less_noise_power():
    for gain in (1.0, 4.26, 6.5):
        claimed = apparent_exposure(100.0, gain)
        assert claimed / 100.0 == pytest.approx(10.0 ** (gain / 10.0))


def test_losing_fidelity_claims_less_exposure():
    assert apparent_exposure(100.0, -3.0) < 100.0


# --------------------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------------------


def test_headroom_above_the_requirement_says_concede():
    d = decide(100.0, 8.0, 5.0)
    assert d.verdict == "concede"
    assert d.minimum_exposure == pytest.approx(100.0 * (5.0 / 8.0) ** 2)
    assert d.headroom > 1.0


def test_being_at_the_requirement_says_hold():
    assert decide(100.0, 5.0, 5.0).verdict == "hold"
    assert decide(100.0, 5.05, 5.0).verdict == "hold"


def test_falling_short_says_raise():
    d = decide(100.0, 3.0, 5.0)
    assert d.verdict == "raise"
    assert d.minimum_exposure > 100.0


def test_a_flattering_image_is_reported_as_a_factor_and_a_warning():
    d = decide(100.0, 6.0, 5.0, fidelity_gain_db=4.26)
    assert d.overstatement == pytest.approx(10.0 ** 0.426)
    assert d.claimed_exposure == pytest.approx(100.0 * 10.0**0.426)
    assert any("looks like" in w for w in d.warnings)


def test_a_small_fidelity_gain_raises_no_alarm():
    assert decide(100.0, 6.0, 5.0, fidelity_gain_db=0.5).warnings == []


def test_failing_the_requirement_with_processing_says_processing_cannot_fix_it():
    d = decide(100.0, 3.0, 5.0, fidelity_gain_db=5.0)
    assert any("cannot add task information" in w for w in d.warnings)


def test_an_efficiency_carried_too_far_is_flagged():
    near = decide(100.0, 6.0, 5.0, efficiency=0.75, efficiency_measured_at=100.0)
    far = decide(
        100.0, 6.0, 5.0, efficiency=0.75, efficiency_measured_at=100.0 * EXTRAPOLATION_LIMIT * 2
    )
    assert not any("measured" in w and "away" in w for w in near.warnings)
    assert any("measured" in w and "away" in w for w in far.warnings)


def test_a_large_concession_is_flagged_as_an_extrapolation():
    """Conceding more than the limit is exactly where the efficiency stops being local."""
    d = decide(100.0, 20.0, 5.0, efficiency=0.75)
    assert d.verdict == "concede"
    assert any("extrapolation" in w for w in d.warnings)


def test_the_record_round_trips_and_carries_the_warnings():
    d = decide(100.0, 3.0, 5.0, efficiency=0.6, fidelity_gain_db=6.0)
    record = d.to_dict()
    assert record["verdict"] == "raise"
    assert record["efficiency"] == pytest.approx(0.6)
    assert len(record["warnings"]) == len(d.warnings) >= 2
    assert "raise" in d.explain().lower()
    assert "not detectability" in d.explain()


def test_an_impossible_efficiency_is_refused_by_decide_too():
    with pytest.raises(ValueError):
        decide(100.0, 6.0, 5.0, efficiency=1.5)


def test_the_study_s_own_numbers_reproduce_its_reported_decision():
    """The module has to agree with the measurement it was built from.

    On the held-out real liver split the unprocessed input reaches d' 6.025 at a quarter of the
    routine protocol against a requirement of 5. Under the total-noise law that is a concession
    to 0.172 of routine; the study's own dose axis, which counts only the noise the reduction
    inserted, puts the crossing at 0.187. The two conventions are supposed to differ, and in
    this direction: counting only the inserted noise is the more optimistic of the two about
    what the image still contains, so it permits the *less* aggressive cut.
    """
    d = decide(0.25, 6.025, 5.0)
    assert d.verdict == "concede"
    assert d.minimum_exposure == pytest.approx(0.1722, rel=1e-3)
    assert d.minimum_exposure < 0.187
