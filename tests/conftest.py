"""Shared fixtures: small, seeded experiments the whole suite can reuse."""

from __future__ import annotations

import pytest

from denoiq_core.evaluate import EvalConfig
from denoiq_core.physics import PhantomSpec, make_trials

#: Fast estimator settings for the tests: the same estimator as the study, fewer trials.
TEST_CONFIG = EvalConfig(roi_size=24)


@pytest.fixture(scope="session")
def trials_high_dose():
    """A comfortably-above-the-floor experiment (white noise)."""
    return make_trials(120.0, 100.0, n_trials=300, seed=101)


@pytest.fixture(scope="session")
def trials_low_dose():
    """A below-the-floor experiment (white noise)."""
    return make_trials(120.0, 6.25, n_trials=300, seed=202)


@pytest.fixture(scope="session")
def trials_correlated():
    """Correlated noise with a white floor — where prewhitening matters."""
    return make_trials(
        120.0,
        25.0,
        n_trials=300,
        seed=303,
        phantom=PhantomSpec(correlation_sigma_mm=0.5, white_floor_fraction=0.1),
    )
