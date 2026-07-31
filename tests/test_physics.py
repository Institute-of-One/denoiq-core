"""The acquisition model, checked against the proportionalities it claims to encode."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from denoiq_core.physics import (
    DEFAULT_MODEL,
    AcquisitionModel,
    PhantomSpec,
    acquisition_params,
    make_trials,
    relative_dose,
    relative_photons,
)


def test_reference_setting_returns_the_reference_values():
    acq = acquisition_params(DEFAULT_MODEL.kv_ref, DEFAULT_MODEL.mas_ref)
    assert acq.noise_sd == pytest.approx(DEFAULT_MODEL.sigma_ref)
    assert acq.contrast == pytest.approx(DEFAULT_MODEL.c_ref)
    assert acq.relative_dose == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [0.25, 0.5, 2.0, 4.0])
def test_noise_scales_as_one_over_sqrt_mas(factor):
    base = acquisition_params(120.0, 100.0)
    scaled = acquisition_params(120.0, 100.0 * factor)
    assert scaled.noise_sd == pytest.approx(base.noise_sd / np.sqrt(factor))


@pytest.mark.parametrize("kv", [80.0, 100.0, 140.0])
def test_noise_scales_as_one_over_kv(kv):
    """``N ∝ mAs kV^2`` and ``sigma ∝ 1/sqrt(N)`` together give ``sigma ∝ 1/(kV sqrt(mAs))``."""
    base = acquisition_params(120.0, 100.0)
    other = acquisition_params(kv, 100.0)
    assert other.noise_sd == pytest.approx(base.noise_sd * (120.0 / kv))


def test_photons_scale_with_mas_and_kv_squared():
    assert relative_photons(120.0, 200.0) == pytest.approx(2.0 * relative_photons(120.0, 100.0))
    assert relative_photons(240.0, 100.0) == pytest.approx(4.0 * relative_photons(120.0, 100.0))


def test_dose_is_proportional_to_mas():
    assert relative_dose(120.0, 50.0) == pytest.approx(0.5)
    assert relative_dose(120.0, 200.0) == pytest.approx(2.0)
    # ...and, by the documented modelling choice, does not depend on kV.
    assert relative_dose(80.0, 100.0) == relative_dose(140.0, 100.0)


@pytest.mark.parametrize("exponent", [1.0, 1.5, 2.0])
def test_contrast_falls_with_kv_at_the_stated_exponent(exponent):
    model = replace(DEFAULT_MODEL, contrast_exponent=exponent)
    kvs = np.array([80.0, 100.0, 120.0, 140.0])
    contrasts = np.array([acquisition_params(kv, 100.0, model=model).contrast for kv in kvs])
    assert np.all(np.diff(contrasts) < 0.0)
    expected = model.c_ref * (model.kv_ref / kvs) ** exponent
    assert contrasts == pytest.approx(expected)


def test_contrast_does_not_depend_on_mas():
    assert acquisition_params(120.0, 5.0).contrast == pytest.approx(
        acquisition_params(120.0, 500.0).contrast
    )


@pytest.mark.parametrize(
    "kv,mas", [(0.0, 100.0), (-120.0, 100.0), (120.0, 0.0), (120.0, -5.0), (np.nan, 100.0)]
)
def test_invalid_settings_raise(kv, mas):
    with pytest.raises(ValueError):
        acquisition_params(kv, mas)


@pytest.mark.parametrize("field", ["kv_ref", "mas_ref", "sigma_ref", "c_ref"])
def test_invalid_model_coefficients_raise(field):
    with pytest.raises(ValueError):
        AcquisitionModel(**{field: 0.0})


def test_make_trials_uses_the_model_numbers():
    """The bridge from the model to the images must not quietly rescale anything."""
    acq = acquisition_params(100.0, 40.0)
    trials = make_trials(100.0, 40.0, n_trials=4, seed=3)
    assert trials.noise_sd == pytest.approx(acq.noise_sd)
    assert float(np.max(np.abs(trials.signal))) == pytest.approx(acq.contrast, rel=1e-6)


def test_correlated_noise_gets_a_white_floor_and_white_noise_does_not():
    """Correlated noise gets a white floor; white noise does not.

    A prewhitening observer needs a floor under correlated noise; without correlation, adding
    one would only put noise into the model that the model does not claim.
    """
    white = make_trials(120.0, 25.0, n_trials=4, seed=4, phantom=PhantomSpec())
    correlated = make_trials(
        120.0,
        25.0,
        n_trials=4,
        seed=4,
        phantom=PhantomSpec(correlation_sigma_mm=0.5, white_floor_fraction=0.1),
    )
    assert white.white_floor_sd == 0.0
    assert correlated.white_floor_sd == pytest.approx(0.1 * correlated.noise_sd)


def test_detectability_scales_as_sqrt_dose():
    """The model's central consequence: ``d' ∝ sqrt(mAs)`` at fixed kV.

    The information floor contour is computed from exactly this scaling, so if it failed the
    atlas would be wrong everywhere between grid points.
    """
    from denoiq_core.bound import ideal_ceiling

    d_low = ideal_ceiling(make_trials(120.0, 25.0, n_trials=2, seed=9)).d_prime
    d_high = ideal_ceiling(make_trials(120.0, 100.0, n_trials=2, seed=9)).d_prime
    assert d_high == pytest.approx(2.0 * d_low, rel=1e-9)
