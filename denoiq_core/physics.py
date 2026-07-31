r"""A documented analytic model of what lowering kV and mAs does to an image.

What this module is
-------------------
Two scalars as a function of the acquisition setting: the contrast of the lesion, and the
standard deviation of the noise. Both **relative to a reference setting**, from three
textbook proportionalities:

.. math::

   N(kV, mAs) \;=\; mAs \left(\frac{kV}{kV_{ref}}\right)^{2}, \qquad
   \sigma \;=\; \sigma_{ref}\sqrt{\frac{N_{ref}}{N}}, \qquad
   c(kV) \;=\; c_{ref}\left(\frac{kV_{ref}}{kV}\right)^{p}

tube output rising roughly as the square of the tube voltage, quantum noise falling as one
over the square root of the photon count, and subject contrast falling as the beam hardens.
Dose is taken proportional to ``mAs`` at fixed kV.

What this module is **not**
---------------------------
It is not a calibration. There is no scanner here, no measured dose in mGy, no measured NPS
from any device, and no absolute kV/mAs threshold table. Everything is normalised to a
reference setting the caller chooses, and every coefficient is an explicit argument with a
documented default. The claim this package makes — that post-processing cannot exceed the
information already in the data — does not depend on the numerical values of these
coefficients; they only set where on the axis a given condition lands.

Reading an absolute threshold off this model for a particular scanner would be an error of
kind, not of degree: mapping a normalised model onto a specific device requires measured
dose-to-noise calibration, which is deliberately out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from taskiq_core import TrialSet, make_disk_signal, ske_bke_trials

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PHANTOM",
    "Acquisition",
    "AcquisitionModel",
    "PhantomSpec",
    "acquisition_params",
    "make_trials",
    "relative_dose",
    "relative_photons",
]


@dataclass(frozen=True)
class AcquisitionModel:
    r"""Coefficients of the relative acquisition model.

    These are **modelling assumptions, not a calibration**.

    Attributes
    ----------
    kv_ref, mas_ref:
        The reference setting. Everything is expressed relative to it; its numerical value
        carries no physical claim beyond "this is where the reference noise and contrast were
        defined".
    sigma_ref:
        Noise standard deviation at the reference setting, in the arbitrary intensity units
        the phantoms use (read them as HU if you like — nothing depends on it).
    c_ref:
        Lesion contrast at the reference kV, same units.
    contrast_exponent:
        The exponent ``p`` in :math:`c \\propto kV^{-p}`. The default ``1.5`` sits in the
        middle of the range implied by attenuation-coefficient differences over the
        diagnostic kV range for soft-tissue-like contrast; it is a documented default, not a
        measurement, and it is an argument precisely so that a reader can change it.

    """

    kv_ref: float = 120.0
    mas_ref: float = 100.0
    sigma_ref: float = 30.0
    c_ref: float = 20.0
    contrast_exponent: float = 1.5

    def __post_init__(self) -> None:
        """Reject a model that would make the relative formulae undefined."""
        for name in ("kv_ref", "mas_ref", "sigma_ref", "c_ref"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if not np.isfinite(self.contrast_exponent):
            raise ValueError("contrast_exponent must be finite")

    def to_dict(self) -> dict[str, float]:
        """The coefficients, for the record written into ``results/``."""
        return {
            "kv_ref": float(self.kv_ref),
            "mas_ref": float(self.mas_ref),
            "sigma_ref": float(self.sigma_ref),
            "c_ref": float(self.c_ref),
            "contrast_exponent": float(self.contrast_exponent),
        }


#: The reference model. Chosen so that the reference setting sits comfortably above the
#: Rose criterion and a four-fold dose reduction falls below it — i.e. so that the
#: interesting part of the kV-mAs plane is on screen.
DEFAULT_MODEL = AcquisitionModel()


@dataclass(frozen=True)
class Acquisition:
    """What one ``(kV, mAs)`` setting means for the image, under :class:`AcquisitionModel`."""

    kv: float
    mas: float
    contrast: float
    noise_sd: float
    relative_photons: float
    relative_dose: float

    def to_dict(self) -> dict[str, float]:
        """Machine-readable record of this condition."""
        return {
            "kv": float(self.kv),
            "mas": float(self.mas),
            "contrast": float(self.contrast),
            "noise_sd": float(self.noise_sd),
            "relative_photons": float(self.relative_photons),
            "relative_dose": float(self.relative_dose),
        }


def _check_setting(kv: float, mas: float) -> tuple[float, float]:
    kv = float(kv)
    mas = float(mas)
    if not np.isfinite(kv) or kv <= 0.0:
        raise ValueError(f"kV must be finite and > 0, got {kv!r}")
    if not np.isfinite(mas) or mas <= 0.0:
        raise ValueError(f"mAs must be finite and > 0, got {mas!r}")
    return kv, mas


def relative_photons(kv: float, mas: float, *, model: AcquisitionModel = DEFAULT_MODEL) -> float:
    r"""Detected photon count relative to ``mas_ref`` at ``kv_ref``: :math:`mAs\,(kV/kV_{ref})^2`.

    Dimensionless by construction — it is a ratio to the reference setting, so no quantum
    efficiency, filtration or patient attenuation enters.
    """
    kv, mas = _check_setting(kv, mas)
    return float(mas * (kv / model.kv_ref) ** 2)


def relative_dose(kv: float, mas: float, *, model: AcquisitionModel = DEFAULT_MODEL) -> float:
    """Dose relative to the reference setting, taken proportional to ``mAs``.

    Dose also rises with kV in reality; that dependence is left out because every dose
    comparison in this study is made **at fixed kV**, where ``dose ∝ mAs`` holds, and
    inventing a kV-dependence would give the model an authority it has not earned. Where the
    kV axis is swept (the red-lamp atlas), the axis reported is kV itself, not dose.
    """
    kv, mas = _check_setting(kv, mas)
    return float(mas / model.mas_ref)


def acquisition_params(
    kv: float, mas: float, *, model: AcquisitionModel = DEFAULT_MODEL
) -> Acquisition:
    r"""Contrast and noise standard deviation at ``(kV, mAs)``.

    .. math::  \sigma = \sigma_{ref}\sqrt{N_{ref}/N}, \qquad c = c_{ref}(kV_{ref}/kV)^{p}

    The returned ``contrast`` goes to :func:`~taskiq_core.make_disk_signal` and ``noise_sd``
    to :func:`~taskiq_core.ske_bke_trials`; that is the whole coupling between this model and
    the rest of the package.

    Returns
    -------
    Acquisition

    """
    kv, mas = _check_setting(kv, mas)
    n = relative_photons(kv, mas, model=model)
    n_ref = relative_photons(model.kv_ref, model.mas_ref, model=model)
    noise_sd = float(model.sigma_ref * np.sqrt(n_ref / n))
    contrast = float(model.c_ref * (model.kv_ref / kv) ** model.contrast_exponent)
    return Acquisition(
        kv=kv,
        mas=mas,
        contrast=contrast,
        noise_sd=noise_sd,
        relative_photons=n,
        relative_dose=relative_dose(kv, mas, model=model),
    )


@dataclass(frozen=True)
class PhantomSpec:
    """Geometry and noise texture of the detection experiment.

    Attributes
    ----------
    size, spacing:
        Image size in pixels and pitch in mm. The defaults describe a 32 x 32 mm field at
        0.5 mm — CT-like, and small enough that a sweep runs in seconds.
    radius_mm:
        Lesion radius. With the default spacing a 3 mm disk is 12 pixels across: resolved,
        but not so large that the task is trivial.
    background:
        Flat background level (the "BKE" in SKE/BKE).
    correlation_sigma_mm:
        Gaussian correlation length of the noise. ``0`` is white noise, for which the ideal
        observer has the closed form ``d' = ||s||/sigma`` that
        ``tests/test_closed_form.py`` checks against.
    white_floor_fraction:
        Width of the white noise floor as a fraction of ``noise_sd``. Correlated noise with
        *no* floor makes a prewhitening observer ill-posed (its power decays to nothing at
        high frequency, so ``1/NPS`` explodes); real detectors have electronic noise, and so
        does this model whenever the noise is correlated. Ignored when
        ``correlation_sigma_mm == 0``.
    edge_sigma_mm:
        Optional Gaussian blur of the disk edge, i.e. a crude system MTF.

    """

    size: int = 64
    spacing: float = 0.5
    radius_mm: float = 3.0
    background: float = 0.0
    correlation_sigma_mm: float = 0.0
    white_floor_fraction: float = 0.1
    edge_sigma_mm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the geometry."""
        return {
            "size": int(self.size),
            "spacing": float(self.spacing),
            "radius_mm": float(self.radius_mm),
            "background": float(self.background),
            "correlation_sigma_mm": float(self.correlation_sigma_mm),
            "white_floor_fraction": float(self.white_floor_fraction),
            "edge_sigma_mm": float(self.edge_sigma_mm),
        }


#: The reference geometry used by the examples, the sweeps and the tests.
DEFAULT_PHANTOM = PhantomSpec()


def make_trials(
    kv: float,
    mas: float,
    *,
    n_trials: int,
    seed: int,
    model: AcquisitionModel = DEFAULT_MODEL,
    phantom: PhantomSpec = DEFAULT_PHANTOM,
) -> TrialSet:
    """Build the SKE/BKE experiment for one ``(kV, mAs)`` setting.

    The single bridge from an acquisition setting to images: this is the only place where
    :func:`~taskiq_core.make_disk_signal` and :func:`~taskiq_core.ske_bke_trials` are called
    with model-derived numbers, so a change to the model propagates everywhere by
    construction rather than by hand.

    The returned :class:`~taskiq_core.TrialSet` carries the **analytic** NPS of its own
    noise, which is what lets the raw ideal observer be evaluated in closed form rather than
    estimated — see :mod:`denoiq_core.bound`.
    """
    acq = acquisition_params(kv, mas, model=model)
    signal = make_disk_signal(
        size=phantom.size,
        radius_mm=phantom.radius_mm,
        contrast=acq.contrast,
        spacing=phantom.spacing,
        edge_sigma_mm=phantom.edge_sigma_mm,
    )
    white_floor_sd = (
        acq.noise_sd * phantom.white_floor_fraction if phantom.correlation_sigma_mm > 0.0 else 0.0
    )
    return ske_bke_trials(
        signal.image,
        n_trials=n_trials,
        spacing=phantom.spacing,
        noise_sd=acq.noise_sd,
        seed=seed,
        background=phantom.background,
        correlation_sigma_mm=phantom.correlation_sigma_mm,
        white_floor_sd=white_floor_sd,
    )


def with_noise_sd(noise_sd: float, *, model: AcquisitionModel = DEFAULT_MODEL) -> AcquisitionModel:
    """A copy of ``model`` whose reference noise is ``noise_sd``.

    Convenience for dose sweeps expressed directly in noise units rather than in mAs.
    """
    noise_sd = float(noise_sd)
    if not np.isfinite(noise_sd) or noise_sd <= 0.0:
        raise ValueError(f"noise_sd must be finite and > 0, got {noise_sd!r}")
    return replace(model, sigma_ref=noise_sd)
