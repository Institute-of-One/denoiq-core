r"""The ideal observer against its closed form.

In white Gaussian noise the prewhitening observer's detectability is exactly
:math:`d' = \|s\|_2/\sigma`. There is no fitting, no estimation and no free parameter in
that statement, which makes it the one place where the whole measurement chain can be
checked against an answer known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest
from taskiq_core import auc_from_scores, d_prime_from_scores, pc_from_d_prime, score_images

from denoiq_core.bound import ideal_ceiling
from denoiq_core.evaluate import analytic_ideal
from denoiq_core.experiment import closed_form_validation
from denoiq_core.physics import PhantomSpec, make_trials

#: The acceptance criterion from the study design.
MAX_RELATIVE_ERROR = 0.01


@pytest.mark.parametrize("noise_sd", [15.0, 30.0, 120.0])
@pytest.mark.parametrize("radius_mm", [1.5, 3.0, 5.0])
def test_white_noise_identity(noise_sd, radius_mm):
    """``d' = ||s||/sigma``, to floating-point accuracy, not merely to 1 %."""
    from dataclasses import replace

    from denoiq_core.physics import DEFAULT_MODEL

    model = replace(DEFAULT_MODEL, sigma_ref=noise_sd)
    phantom = PhantomSpec(radius_mm=radius_mm, correlation_sigma_mm=0.0)
    trials = make_trials(
        model.kv_ref, model.mas_ref, n_trials=2, seed=1, model=model, phantom=phantom
    )

    expected = float(np.sqrt(np.sum(trials.signal**2)) / trials.noise_sd)
    observed = ideal_ceiling(trials).d_prime
    assert abs(observed - expected) / expected < 1e-9


def test_closed_form_table_meets_the_acceptance_criterion():
    """Table 1 as it will be published: every entry inside 1 %."""
    table = closed_form_validation()
    assert table["max_relative_error"] < MAX_RELATIVE_ERROR
    assert len(table["rows"]) >= 6


def test_auc_follows_from_d_prime():
    """AUC = Phi(d'/sqrt2) for a linear observer in Gaussian noise."""
    trials = make_trials(120.0, 25.0, n_trials=2, seed=7)
    ceiling = ideal_ceiling(trials)
    assert ceiling.auc == pytest.approx(float(pc_from_d_prime(ceiling.d_prime)), rel=1e-12)


def test_monte_carlo_agrees_with_the_closed_form(trials_high_dose):
    """Score real images with the analytic template: the measured ``d'`` must match.

    This is the step that closes the loop. The closed form could be right and the images
    wrong (or vice versa); only scoring the actual trials with the actual template tests
    both at once. The tolerance is the sampling error of ``d'`` on this many trials, not a
    fudge: with n per class, SE(d') is about sqrt(2/n + d'^2/(4n)).
    """
    trials = trials_high_dose
    template = analytic_ideal(trials)
    from taskiq_core import ideal_linear

    observer = ideal_linear(trials.signal, trials.nps, trials.spacing, nps_layout="centered")
    present = score_images(trials.present, observer.template)
    absent = score_images(trials.absent, observer.template)

    measured = d_prime_from_scores(present, absent)
    n = trials.n_trials
    se = np.sqrt(2.0 / n + template.d_prime**2 / (4.0 * n))
    assert abs(measured - template.d_prime) < 4.0 * se

    measured_auc = auc_from_scores(present, absent)
    assert measured_auc == pytest.approx(template.auc, abs=0.02)


def test_correlated_noise_costs_the_non_prewhitening_observer(trials_correlated):
    """Sanity of the physics, not of an estimator: prewhitening must be worth something.

    With correlated noise the ideal (prewhitening) observer sees more than the eye-filtered
    non-prewhitening one. If it did not, the correlated-noise conditions in the study would
    be testing nothing.
    """
    from denoiq_core.evaluate import empirical_ideal, empirical_npwe

    trials = trials_correlated
    ideal = empirical_ideal(trials.present, trials.absent, trials.spacing)
    npwe_est = empirical_npwe(trials.present, trials.absent, trials.spacing)
    assert ideal.d_prime > npwe_est.d_prime
