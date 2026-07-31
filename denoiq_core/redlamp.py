r"""The red lamp: a reliability gauge that says *why*, not just *how much*.

The ceiling (:mod:`denoiq_core.bound`) says what processing cannot do. This module says when
a particular acquisition has already fallen below what any processing needs: the
**information floor**.

The floor is operational, not informational
-------------------------------------------
:func:`information_floor` computes the analytic ideal-observer ``d'`` of the **unprocessed
input** across the kV-mAs plane and returns the contour ``d' = threshold``, with the Rose
criterion (``d' = 5``) as the default. That contour is a *prespecified task requirement*, not
a zero-information boundary: a condition below it still carries measurable task information
— ``d' = 3.47`` corresponds to an AUC of about 0.993 for an equal-variance Gaussian decision
variable — it simply no longer meets the chosen detectability criterion. Nothing in this
module licenses the statement that the task is impossible below the floor.

What *is* bounded below the floor is what processing can establish: no denoiser can lift
detectability above the ceiling of its own input (:mod:`denoiq_core.bound`), so once the
input's ideal ``d'`` is under the requirement, no post-processing can bring the requirement
back into reach.

Plausibility is not evidence of preserved information
----------------------------------------------------
A denoiser optimised for pixelwise fidelity will keep producing clean-looking images at any
dose. Its output can have smooth texture, high SSIM and lesion-like blobs while carrying less
task information than the noisy input it came from. Two measurements make that concrete and
auditable:

* :func:`false_structure_rate` — how often **signal-absent** images contain a lesion-shaped
  matched-filter response of at least a given fraction of a true lesion's amplitude. The
  underlying object is spatially uniform, but the acquired image is not: noise alone produces
  such responses, so the measure is only interpretable **relative to the same quantity on the
  unprocessed input**. What it detects is therefore an *excess* of lesion-like responses, not
  structure that processing alone invented.
* :func:`contrast_recovery` — how much of the true lesion contrast survives where the lesion
  *is*. The other failure: erasure.

:func:`assess` combines the floor, the task performance, the fidelity, and both of those
into green / amber / red **with a reason string**, because a gauge that cannot say why it is
red is not auditable.

Scope
-----
This is the *method* and a synthetic demonstration of it. Turning it into an absolute
kV/mAs threshold table for a particular scanner requires measured dose-to-noise calibration
of that scanner, which is deliberately not part of this repository — see
:mod:`denoiq_core.physics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from denoiq_core.bound import ideal_ceiling
from denoiq_core.denoisers import denoiser_label
from denoiq_core.evaluate import (
    DEFAULT_CONFIG,
    EvalConfig,
    denoise_trials,
    fidelity,
    prewhitening_estimate,
)
from denoiq_core.physics import (
    DEFAULT_MODEL,
    DEFAULT_PHANTOM,
    AcquisitionModel,
    PhantomSpec,
    acquisition_params,
    make_trials,
)

__all__ = [
    "GREEN",
    "AMBER",
    "RED",
    "Atlas",
    "RedLamp",
    "RedLampCriteria",
    "amplitude_map",
    "assess",
    "classify",
    "classify_grid",
    "contrast_recovery",
    "false_structure_rate",
    "information_floor",
]

GREEN = "green"
AMBER = "amber"
RED = "red"

#: Level codes for image-valued output (figures shade with these).
LEVEL_CODE = {GREEN: 0, AMBER: 1, RED: 2}


@dataclass(frozen=True)
class RedLampCriteria:
    """The thresholds the gauge fires on. Every one of them is an explicit, reportable number.

    Attributes
    ----------
    d_prime_threshold:
        The prespecified detectability requirement — the operational floor. ``5`` is the Rose
        criterion. It is a task requirement, not an information boundary.
    amber_factor:
        How far above the floor still counts as marginal: amber while
        ``threshold <= d' < amber_factor * threshold``.
    min_task_d_prime:
        Minimum *adequate* task detectability of the processed images. ``1.0`` corresponds to
        an AUC of about 0.76 for an equal-variance Gaussian decision variable, so it is a
        threshold for **inadequate** performance and emphatically not chance performance
        (which would be ``d' = 0``, AUC 0.5).
    plausible_ssim:
        SSIM at or above which the processed image looks close to the truth. Combined with
        inadequate task performance this is the fidelity-task discordance in its sharpest
        form.
    max_false_structure_rate:
        Fraction of signal-absent images allowed to show a lesion-like matched-filter response
        before the gauge reports an excess (it must also exceed the unprocessed rate).
    max_task_loss:
        Fractional drop in task ``d'`` (processed vs unprocessed) that counts as severe
        degradation.
    min_contrast_recovery:
        Fraction of the true lesion contrast that must survive processing before erasure is
        reported.

    """

    d_prime_threshold: float = 5.0
    amber_factor: float = 1.2
    min_task_d_prime: float = 1.0
    plausible_ssim: float = 0.7
    max_false_structure_rate: float = 0.2
    max_task_loss: float = 0.2
    min_contrast_recovery: float = 0.5

    def to_dict(self) -> dict[str, float]:
        """Machine-readable record of the criteria."""
        return {
            "d_prime_threshold": float(self.d_prime_threshold),
            "amber_factor": float(self.amber_factor),
            "min_task_d_prime": float(self.min_task_d_prime),
            "plausible_ssim": float(self.plausible_ssim),
            "max_false_structure_rate": float(self.max_false_structure_rate),
            "max_task_loss": float(self.max_task_loss),
            "min_contrast_recovery": float(self.min_contrast_recovery),
        }


DEFAULT_CRITERIA = RedLampCriteria()


# --------------------------------------------------------------------------------------
# the information floor
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Atlas:
    """Ideal-observer detectability across the kV-mAs plane, and the floor contour.

    Attributes
    ----------
    kv, mas:
        The grid axes.
    d_prime:
        Raw ideal-observer ``d'``, shape ``(len(mas), len(kv))`` — row-major in mAs so that
        ``imshow`` puts mAs on the vertical axis without a transpose.
    contrast, noise_sd:
        The model's contrast and noise at each grid point, same shape.
    floor_mas:
        For each kV, the mAs at which ``d' = threshold`` — the information floor. Computed in
        closed form from the model's exact ``d' ∝ sqrt(mAs)`` scaling at fixed kV, not by
        interpolating the grid, so it is accurate between grid points and its correctness is
        checkable (``tests/test_redlamp.py`` does).
    level:
        Green / amber / red code per grid point, from the floor alone.

    """

    kv: np.ndarray
    mas: np.ndarray
    d_prime: np.ndarray
    contrast: np.ndarray
    noise_sd: np.ndarray
    floor_mas: np.ndarray
    threshold: float
    level: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        return {
            "kv": self.kv.tolist(),
            "mas": self.mas.tolist(),
            "d_prime": self.d_prime.tolist(),
            "contrast": self.contrast.tolist(),
            "noise_sd": self.noise_sd.tolist(),
            "floor_mas": self.floor_mas.tolist(),
            "threshold": float(self.threshold),
            "level": self.level.tolist(),
            **self.meta,
        }


def _analytic_d_prime(
    kv: float,
    mas: float,
    *,
    model: AcquisitionModel,
    phantom: PhantomSpec,
) -> float:
    """Raw ideal-observer ``d'`` at one setting, in closed form.

    Builds the smallest legal trial set purely to obtain its **analytic** NPS and signal, and
    reads the ideal observer off those. No Monte-Carlo error enters the atlas.
    """
    trials = make_trials(kv, mas, n_trials=2, seed=0, model=model, phantom=phantom)
    return float(ideal_ceiling(trials).d_prime)


def information_floor(
    kv_grid: np.ndarray,
    mas_grid: np.ndarray,
    *,
    threshold: float = DEFAULT_CRITERIA.d_prime_threshold,
    model: AcquisitionModel = DEFAULT_MODEL,
    phantom: PhantomSpec = DEFAULT_PHANTOM,
    criteria: RedLampCriteria = DEFAULT_CRITERIA,
) -> Atlas:
    r"""Ideal-observer ``d'`` over the kV-mAs plane, and the contour where it crosses ``threshold``.

    The floor contour is exact rather than interpolated. At fixed kV the model gives
    :math:`\sigma \propto 1/\sqrt{mAs}` and a contrast that does not depend on mAs, so
    :math:`d' = \|s\|/\sigma \propto \sqrt{mAs}`; one evaluation per kV therefore determines
    the whole curve:

    .. math::  mAs_{floor}(kV) = mAs_0 \left(\frac{d'_{threshold}}{d'(kV, mAs_0)}\right)^{2}

    Parameters
    ----------
    kv_grid, mas_grid:
        Axes of the atlas, in kV and mAs.
    threshold:
        The floor, in ``d'``. Default is the Rose criterion.
    model, phantom:
        The acquisition model and the geometry the atlas is computed for.
    criteria:
        Supplies the amber band used to shade the plane; the floor itself is ``threshold``.

    Returns
    -------
    Atlas

    """
    kv = np.asarray(kv_grid, dtype=np.float64).ravel()
    mas = np.asarray(mas_grid, dtype=np.float64).ravel()
    if kv.size == 0 or mas.size == 0:
        raise ValueError("kv_grid and mas_grid must be non-empty")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be finite and > 0, got {threshold!r}")

    d_prime = np.empty((mas.size, kv.size), dtype=np.float64)
    contrast = np.empty_like(d_prime)
    noise_sd = np.empty_like(d_prime)
    for j, kv_value in enumerate(kv):
        for i, mas_value in enumerate(mas):
            acq = acquisition_params(kv_value, mas_value, model=model)
            contrast[i, j] = acq.contrast
            noise_sd[i, j] = acq.noise_sd
            d_prime[i, j] = _analytic_d_prime(kv_value, mas_value, model=model, phantom=phantom)

    # Exact contour from the sqrt(mAs) scaling, anchored at the first mAs on the grid.
    anchor = float(mas[0])
    d_anchor = d_prime[0, :]
    floor_mas = anchor * (threshold / d_anchor) ** 2

    level = classify_grid(d_prime, threshold=threshold, amber_factor=criteria.amber_factor)

    return Atlas(
        kv=kv,
        mas=mas,
        d_prime=d_prime,
        contrast=contrast,
        noise_sd=noise_sd,
        floor_mas=floor_mas,
        threshold=threshold,
        level=level,
        meta={
            "model": model.to_dict(),
            "phantom": phantom.to_dict(),
            "amber_factor": float(criteria.amber_factor),
            "observer": "ideal_linear on raw data (analytic NPS)",
        },
    )


def classify_grid(d_prime: np.ndarray, *, threshold: float, amber_factor: float) -> np.ndarray:
    """Green / amber / red codes (0 / 1 / 2) from raw detectability alone.

    Red below the floor, amber within ``amber_factor`` of it, green above. This is the atlas
    shading; the full gauge (:func:`assess`) adds what the *processing* did on top of it.
    """
    d = np.asarray(d_prime, dtype=np.float64)
    codes = np.full(d.shape, LEVEL_CODE[GREEN], dtype=np.int8)
    codes[d < amber_factor * threshold] = LEVEL_CODE[AMBER]
    codes[d < threshold] = LEVEL_CODE[RED]
    return codes


# --------------------------------------------------------------------------------------
# fabrication and erasure
# --------------------------------------------------------------------------------------


def amplitude_map(image: np.ndarray, signal: np.ndarray) -> np.ndarray:
    r"""Local lesion amplitude, in units of the true lesion contrast.

    The matched-filter estimate of "how much lesion is here" at every position:

    .. math::  a(x) = \frac{\langle I - \bar I,\; s_0(\cdot - x)\rangle}{\|s_0\|^2}

    with :math:`s_0` the mean-subtracted signal. It is scaled so that an image containing
    exactly the true lesion at position :math:`x` gives :math:`a(x) = 1`: the map reads
    directly as a fraction of a real lesion, which is what makes a threshold on it
    interpretable rather than arbitrary.
    """
    img = np.asarray(image, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if img.shape != sig.shape:
        raise ValueError(f"image shape {img.shape} does not match signal {sig.shape}")
    s0 = sig - sig.mean()
    norm = float(np.sum(s0**2))
    if norm <= 0.0:
        raise ValueError("the signal is constant: there is no template to match")
    return fftconvolve(img - img.mean(), s0[::-1, ::-1], mode="same") / norm


def false_structure_rate(
    absent_images: np.ndarray,
    signal: np.ndarray,
    *,
    amplitude_fraction: float = 0.5,
) -> dict[str, float]:
    """How often signal-absent images contain a lesion-shaped structure anyway.

    The truth in these images is a flat background: **every** lesion-like structure in them
    is false. The measure is therefore honest about noise as well as about denoisers — raw
    low-dose images have a high rate too — which is why :func:`assess` compares the processed
    rate against the raw one rather than reading it alone.

    Parameters
    ----------
    absent_images:
        Stack of signal-absent images, processed or raw.
    signal:
        The true lesion, used as the template and as the amplitude scale.
    amplitude_fraction:
        How large a structure has to be to count, as a fraction of the true lesion contrast.
        ``0.5`` is "half a lesion".

    Returns
    -------
    dict
        ``rate`` (fraction of images with such a structure), ``mean_max_amplitude`` and
        ``p95_max_amplitude`` (the amplitude distribution behind the rate), and the
        ``amplitude_fraction`` used.

    """
    stack = np.asarray(absent_images, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a stack (n, ny, nx), got shape {stack.shape}")
    frac = float(amplitude_fraction)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"amplitude_fraction must be finite and > 0, got {frac!r}")

    peaks = np.array([float(np.max(amplitude_map(img, signal))) for img in stack])
    return {
        "rate": float(np.mean(peaks >= frac)),
        "mean_max_amplitude": float(peaks.mean()),
        "p95_max_amplitude": float(np.percentile(peaks, 95)),
        "amplitude_fraction": frac,
    }


def contrast_recovery(
    present_images: np.ndarray, absent_images: np.ndarray, signal: np.ndarray
) -> float:
    r"""Fraction of the true lesion contrast that survives processing.

    .. math::  \rho = \frac{\langle \overline{I_1} - \overline{I_0},\; s_0\rangle}{\|s_0\|^2}

    ``1`` means the lesion comes through at full contrast, ``0`` that it has been erased.
    Values above 1 mean the processing has *amplified* the difference — which flatters the
    picture without adding information, and is why this is reported next to ``d'`` rather
    than instead of it.
    """
    present = np.asarray(present_images, dtype=np.float64)
    absent = np.asarray(absent_images, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if present.ndim != 3 or absent.ndim != 3:
        raise ValueError("present_images and absent_images must be stacks (n, ny, nx)")
    difference = present.mean(axis=0) - absent.mean(axis=0)
    s0 = sig - sig.mean()
    norm = float(np.sum(s0**2))
    if norm <= 0.0:
        raise ValueError("the signal is constant: there is no template to match")
    return float(np.sum((difference - difference.mean()) * s0) / norm)


# --------------------------------------------------------------------------------------
# the gauge
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RedLamp:
    """A green / amber / red verdict, the reasons for it, and every number behind them."""

    level: str
    reasons: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def code(self) -> int:
        """0 green, 1 amber, 2 red."""
        return LEVEL_CODE[self.level]

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        return {
            "level": self.level,
            "code": self.code,
            "reasons": list(self.reasons),
            **self.metrics,
        }

    def summary(self) -> str:
        """One line: the level and the first reason."""
        head = self.reasons[0] if self.reasons else "no criterion fired"
        return f"[{self.level.upper()}] {head}"


def classify(
    *,
    ceiling_d_prime: float,
    task_d_prime: float,
    raw_task_d_prime: float,
    ssim: float,
    false_rate: float,
    raw_false_rate: float,
    recovery: float,
    criteria: RedLampCriteria = DEFAULT_CRITERIA,
    context: str = "",
) -> RedLamp:
    """Apply the criteria to a set of already-measured numbers.

    Separated from :func:`assess` so the decision logic can be unit-tested without running an
    experiment, and so a caller with its own measurements can reuse the rules.

    The rules, in the order they are checked. ``C`` is the analytic ideal ``d'`` of the
    unprocessed input, ``T`` and ``T_raw`` the held-out prewhitening estimates on the processed
    and unprocessed images, ``S`` the SSIM, ``F`` and ``F_raw`` the lesion-like response rates
    on signal-absent images, and ``p`` the contrast recovery:

    1. **Red** — ``C < d_prime_threshold``: the input does not meet the prespecified
       detectability requirement, and no post-processing can restore it (:mod:`denoiq_core.bound`).
    2. **Red** — ``S >= plausible_ssim`` and ``T <= min_task_d_prime``: the image looks close to
       the truth while task performance is inadequate.
    3. **Red** — ``F > max_false_structure_rate`` **and** ``F > F_raw``: an excess of
       lesion-like responses over the unprocessed input.
    4. **Red** — ``p < min_contrast_recovery``: erasure.
    5. **Amber** — ``C < amber_factor * d_prime_threshold`` (marginal), or
       ``1 - T/T_raw > max_task_loss`` (severe task degradation relative to the input).
    6. **Green** — none of the above.

    Every rule that fires contributes a reason, in this order; the level is the most severe
    that fired, and :meth:`RedLamp.summary` reports the first.
    """
    reasons: list[str] = []
    prefix = f"{context}: " if context else ""

    if ceiling_d_prime < criteria.d_prime_threshold:
        reasons.append(
            f"{prefix}input ideal d'={ceiling_d_prime:.2f} < {criteria.d_prime_threshold:g} "
            "(prespecified floor, Rose): the input does not meet the detectability requirement "
            "and post-processing cannot restore it."
        )
    if ssim >= criteria.plausible_ssim and task_d_prime <= criteria.min_task_d_prime:
        reasons.append(
            f"{prefix}SSIM={ssim:.2f} is high but task d'={task_d_prime:.2f} is below the "
            f"adequacy threshold ({criteria.min_task_d_prime:g}): visually plausible, "
            "task-inadequate."
        )
    if false_rate > criteria.max_false_structure_rate and false_rate > raw_false_rate:
        reasons.append(
            f"{prefix}lesion-like responses in {false_rate:.0%} of signal-absent images "
            f"vs {raw_false_rate:.0%} unprocessed: excess false structure."
        )
    if recovery < criteria.min_contrast_recovery:
        reasons.append(
            f"{prefix}only {recovery:.0%} of the true lesion contrast survives: erasure."
        )
    if reasons:
        return RedLamp(RED, reasons)

    if ceiling_d_prime < criteria.amber_factor * criteria.d_prime_threshold:
        reasons.append(
            f"{prefix}input ideal d'={ceiling_d_prime:.2f} is within "
            f"{criteria.amber_factor:g}x of the floor ({criteria.d_prime_threshold:g}): "
            "marginal — small dose changes cross it."
        )
    loss = 1.0 - (task_d_prime / raw_task_d_prime) if raw_task_d_prime > 0 else 0.0
    if loss > criteria.max_task_loss:
        reasons.append(
            f"{prefix}task d' fell {loss:.0%} against the unprocessed input "
            f"({raw_task_d_prime:.2f} -> {task_d_prime:.2f}) while fidelity improved: "
            "fidelity-task discordance."
        )
    if reasons:
        return RedLamp(AMBER, reasons)

    return RedLamp(
        GREEN,
        [
            f"{prefix}input ideal d'={ceiling_d_prime:.2f} meets the floor and the processed "
            f"task d'={task_d_prime:.2f} follows it; no excess false structure or erasure."
        ],
    )


def assess(
    kv: float,
    mas: float,
    method: str = "none",
    params: dict[str, Any] | None = None,
    *,
    n_trials: int = 400,
    seed: int = 0,
    model: AcquisitionModel = DEFAULT_MODEL,
    phantom: PhantomSpec = DEFAULT_PHANTOM,
    config: EvalConfig = DEFAULT_CONFIG,
    criteria: RedLampCriteria = DEFAULT_CRITERIA,
    amplitude_fraction: float = 0.5,
    label: str | None = None,
) -> RedLamp:
    """Run the whole gauge for one acquisition setting and one denoiser.

    Generates the experiment, measures the input's analytic ceiling, the held-out prewhitening
    estimate on the processed and unprocessed images, the fidelity, the lesion-like response
    rate (processed and unprocessed) and the contrast recovery, then applies :func:`classify`.
    Everything the verdict rests on comes back in :attr:`RedLamp.metrics`, so the call is
    auditable end to end.

    Parameters
    ----------
    kv, mas:
        The acquisition setting to evaluate.
    method, params:
        The processing under test, as passed to :func:`denoiq_core.denoisers.denoise`.
        ``"none"`` gauges the unprocessed input itself.
    n_trials, seed:
        Size and seed of the experiment generated for this setting.
    model, phantom:
        The relative acquisition model and the phantom geometry.
    config:
        Estimator settings for the held-out observers.
    criteria:
        The thresholds the verdict is made against.
    amplitude_fraction:
        Fraction of a true lesion's amplitude a matched-filter response must reach to count in
        :func:`false_structure_rate`.
    label:
        Name of the processing in the reason strings. Defaults to
        :func:`~denoiq_core.denoisers.denoiser_label`, which spells out every parameter — pass
        something shorter when a parameter is a filesystem path, so that the reason (and the
        figure that prints it) stays machine-independent.

    """
    trials = make_trials(kv, mas, n_trials=n_trials, seed=seed, model=model, phantom=phantom)
    ceiling = ideal_ceiling(trials)

    present, absent = denoise_trials(trials, method, params)
    task = prewhitening_estimate(present, absent, trials.spacing, config=config)
    raw_task = prewhitening_estimate(trials.present, trials.absent, trials.spacing, config=config)

    clean_present = float(trials.background) + trials.signal
    fid = fidelity(present, clean_present, data_range=config.fidelity_data_range)

    false_processed = false_structure_rate(
        absent, trials.signal, amplitude_fraction=amplitude_fraction
    )
    false_raw = false_structure_rate(
        trials.absent, trials.signal, amplitude_fraction=amplitude_fraction
    )
    recovery = contrast_recovery(present, absent, trials.signal)

    lamp = classify(
        ceiling_d_prime=ceiling.d_prime,
        task_d_prime=task.d_prime,
        raw_task_d_prime=raw_task.d_prime,
        ssim=fid["ssim"],
        false_rate=false_processed["rate"],
        raw_false_rate=false_raw["rate"],
        recovery=recovery,
        criteria=criteria,
        context=f"kV={kv:g}, mAs={mas:g}, {label or denoiser_label(method, params)}",
    )
    return RedLamp(
        lamp.level,
        lamp.reasons,
        {
            "kv": float(kv),
            "mas": float(mas),
            "denoiser": label or denoiser_label(method, params),
            "ceiling_d_prime": float(ceiling.d_prime),
            "ceiling_auc": float(ceiling.auc),
            "task_d_prime": float(task.d_prime),
            "task_auc": float(task.auc),
            "raw_task_d_prime": float(raw_task.d_prime),
            "raw_task_auc": float(raw_task.auc),
            "ssim": float(fid["ssim"]),
            "psnr": float(fid["psnr"]),
            "rmse": float(fid["rmse"]),
            "false_structure_rate": float(false_processed["rate"]),
            "false_structure_mean_amplitude": float(false_processed["mean_max_amplitude"]),
            "raw_false_structure_rate": float(false_raw["rate"]),
            "contrast_recovery": float(recovery),
            "criteria": criteria.to_dict(),
            "n_trials": int(n_trials),
            "seed": int(seed),
        },
    )
