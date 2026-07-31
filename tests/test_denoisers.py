"""The unified denoiser interface: shape, determinism, and independence from the truth."""

from __future__ import annotations

import numpy as np
import pytest

from denoiq_core.denoisers import (
    DENOISER_DEFAULTS,
    available_methods,
    denoise,
    denoiser_label,
)

CLASSICAL = [m for m in available_methods() if m != "cnn"]


@pytest.fixture
def image(trials_high_dose):
    return trials_high_dose.present[0]


@pytest.fixture
def stack(trials_high_dose):
    return trials_high_dose.present[:6]


@pytest.mark.parametrize("method", CLASSICAL)
def test_shape_is_preserved(method, image, stack):
    assert denoise(image, method).shape == image.shape
    assert denoise(stack, method).shape == stack.shape


@pytest.mark.parametrize("method", CLASSICAL)
def test_output_is_finite_float64(method, stack):
    out = denoise(stack, method)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", CLASSICAL)
def test_deterministic(method, stack):
    assert np.array_equal(denoise(stack, method), denoise(stack, method))


@pytest.mark.parametrize("method", CLASSICAL)
def test_each_image_is_processed_independently(method, stack):
    """A trial's output must not depend on which other trials were in the stack.

    If it did, the denoiser would be averaging an ensemble it would never have in practice,
    the trials would stop being independent, and the observers' variance estimates — and the
    ceiling test built on them — would be wrong.
    """
    together = denoise(stack, method)
    apart = np.stack([denoise(plane, method) for plane in stack])
    assert np.array_equal(together, apart)


def test_none_is_the_identity(stack):
    out = denoise(stack, "none")
    assert np.array_equal(out, stack)
    assert out is not stack


def test_gaussian_zero_sigma_is_the_identity(image):
    assert np.allclose(denoise(image, "gaussian", sigma=0.0), image)


def test_gaussian_reduces_noise(trials_high_dose):
    """Smoothing must actually smooth — otherwise every downstream comparison is vacuous."""
    absent = trials_high_dose.absent
    assert np.std(denoise(absent, "gaussian", sigma=1.5)) < 0.5 * np.std(absent)


def test_unknown_method_raises(image):
    with pytest.raises(ValueError, match="unknown denoiser"):
        denoise(image, "wavelet-magic")


def test_unknown_parameter_raises(image):
    """A typo in a parameter name must fail loudly.

    Silently ignored, it would be read as a property of the algorithm.
    """
    with pytest.raises(ValueError, match="unknown parameter"):
        denoise(image, "gaussian", sgima=1.0)


def test_non_finite_input_raises(image):
    bad = image.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        denoise(bad, "gaussian")


def test_wrong_dimensionality_raises():
    with pytest.raises(ValueError, match="2-D or a 3-D stack"):
        denoise(np.zeros((2, 2, 2, 2)), "gaussian")


def test_labels_are_stable_and_readable():
    assert denoiser_label("gaussian", {"sigma": 1.5}) == "gaussian(sigma=1.5)"
    assert denoiser_label("none") == "none"
    assert "weight=2" in denoiser_label("tv", {"weight": 2.0})


def test_every_default_is_documented():
    for method in available_methods():
        assert method in DENOISER_DEFAULTS


def test_cnn_needs_a_checkpoint(image):
    """The CNN entry exists only with the [dl] extra, and refuses to guess at weights."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="checkpoint"):
        denoise(image, "cnn")
