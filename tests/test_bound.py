r"""The central test: no processing raises the ideal-observer AUC above the raw ceiling.

.. math::  \mathrm{AUC}_{\mathrm{ideal}}(g(G)) \le \mathrm{AUC}_{\mathrm{ideal}}(G) + \mathrm{CI}

for every denoiser, at every condition. A failure here is not a discovery about information
theory; it means a denoiser saw something it should not have, or an observer estimate is
biased high, or the confidence margin is wrong. The last two tests in this file exist to
show that the check has teeth: it must *fail* on a processor that cheats, and the estimator
must come close enough to the ceiling that passing is not automatic.
"""

from __future__ import annotations

import numpy as np
import pytest

from denoiq_core.bound import check_denoiser, check_no_gain, ideal_ceiling, raw_estimator_check
from denoiq_core.evaluate import empirical_ideal
from denoiq_core.experiment import DEFAULT_DENOISERS

DENOISERS = [(spec.method, spec) for spec in DEFAULT_DENOISERS]


@pytest.mark.parametrize("method,spec", DENOISERS, ids=[s.name for _, s in DENOISERS])
@pytest.mark.parametrize("fixture", ["trials_high_dose", "trials_low_dose", "trials_correlated"])
def test_no_denoiser_exceeds_the_ceiling(method, spec, fixture, request):
    trials = request.getfixturevalue(fixture)
    result = check_denoiser(trials, method, spec.resolve(trials.noise_sd))
    assert result.ok, result.reason()


@pytest.mark.parametrize("fixture", ["trials_high_dose", "trials_low_dose", "trials_correlated"])
def test_repeated_denoising_cannot_recover_information(fixture, request):
    """Applying a second filter after the first can only lose more. The chain is a chain."""
    from denoiq_core.denoisers import denoise

    trials = request.getfixturevalue(fixture)
    once_present = denoise(trials.present, "gaussian", sigma=1.0)
    once_absent = denoise(trials.absent, "gaussian", sigma=1.0)
    twice_present = denoise(once_present, "tv", weight=0.4 * trials.noise_sd)
    twice_absent = denoise(once_absent, "tv", weight=0.4 * trials.noise_sd)

    first = check_no_gain(trials, once_present, once_absent, label="gaussian")
    second = check_no_gain(trials, twice_present, twice_absent, label="gaussian+tv")
    assert first.ok, first.reason()
    assert second.ok, second.reason()


def test_the_check_catches_a_processor_that_peeks_at_the_truth(trials_low_dose):
    """A "denoiser" that uses the label must be caught — otherwise the test proves nothing.

    This is the failure mode the ceiling test exists for: a pipeline that, by accident or by
    construction, lets information about the answer into the processing. Here the present
    stack is given a little extra signal that the absent stack does not get, which no
    function of a single image could do. The estimated ideal AUC then rises above a ceiling
    that is, correctly, unmoved.
    """
    trials = trials_low_dose
    cheat_present = trials.present + 2.0 * trials.signal
    result = check_no_gain(trials, cheat_present, trials.absent, label="cheating")
    assert not result.ok, result.reason()
    assert result.excess > result.margin


def test_the_estimator_is_not_trivially_below_the_ceiling(trials_high_dose):
    """The held-out estimator must recover most of the analytic ceiling on raw data.

    An estimator that landed far below the ceiling would satisfy the inequality for reasons
    of statistics rather than of information, and every "within the ceiling" verdict in the
    study would be uninformative.
    """
    check = raw_estimator_check(trials_high_dose)
    assert check["d_prime_ratio"] > 0.8, check


def test_identity_processing_reaches_the_ceiling(trials_high_dose):
    """An invertible ``g`` — here the identity — must sit at the ceiling, not below it."""
    trials = trials_high_dose
    result = check_no_gain(trials, trials.present.copy(), trials.absent.copy(), label="identity")
    assert result.ok
    assert result.efficiency > 0.8


def test_efficiency_is_reported_and_ordered(trials_low_dose):
    """Stronger smoothing at a low dose must not be recorded as *more* information."""
    trials = trials_low_dose
    light = check_denoiser(trials, "gaussian", {"sigma": 1.0})
    heavy = check_denoiser(trials, "gaussian", {"sigma": 4.0})
    assert light.ok and heavy.ok
    assert heavy.efficiency <= light.efficiency + 0.05


def test_ceiling_is_analytic_and_matches_the_observer(trials_high_dose):
    ceiling = ideal_ceiling(trials_high_dose)
    assert ceiling.source == "analytic"
    expected = float(np.sqrt(np.sum(trials_high_dose.signal**2)) / trials_high_dose.noise_sd)
    assert ceiling.d_prime == pytest.approx(expected, rel=1e-9)


def test_margin_covers_estimator_noise_but_not_a_real_gain(trials_low_dose):
    """The margin must be small: a few AUC points, not a licence."""
    trials = trials_low_dose
    est = empirical_ideal(trials.present, trials.absent, trials.spacing)
    result = check_no_gain(trials, trials.present, trials.absent, estimate=est)
    assert 0.0 < result.margin < 0.1


def test_bootstrap_margin_agrees_with_the_closed_form(trials_low_dose):
    """Two ways of pricing the sampling error should not disagree by much."""
    trials = trials_low_dose
    closed = check_denoiser(trials, "gaussian", {"sigma": 1.5})
    booted = check_denoiser(trials, "gaussian", {"sigma": 1.5}, bootstrap=True)
    assert booted.ok
    assert booted.margin == pytest.approx(closed.margin, rel=0.6)
