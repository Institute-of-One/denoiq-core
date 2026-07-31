r"""Measure a condition: fidelity to the truth, and detectability to an observer.

One function does the work — :func:`evaluate_condition` — and it returns a plain dictionary
of numbers, because every number this study reports is written to ``results/`` and read back
by ``tests/test_manuscript_consistency.py``. Nothing is computed twice, and nothing is typed
into the manuscript by hand.

Two kinds of number, and why both
---------------------------------
**Fidelity** (RMSE / PSNR / SSIM against the noise-free truth) is what denoising papers
usually report, and what a denoiser is usually trained on. It measures how much the picture
looks like the truth.

**Detectability** (``d'`` / AUC for a model observer) measures how much of the information
needed to answer the question — *is the lesion there?* — survived. It is the operational
reading of the image's information content, which is the quantity the data-processing
inequality actually bounds.

The interesting result of this package lives in the gap between them.

Estimating an observer on processed images
------------------------------------------
On **raw** images the ideal linear observer is available in closed form: the trials carry
the analytic NPS of their own noise, so the observer needs nothing estimated. On **denoised**
images neither the noise power spectrum nor the effective signal is known analytically — a
non-linear denoiser has no transfer function — so both must be estimated from the images.

That estimation is done on a **training split** and the resulting template is scored on a
**disjoint test split**. This is not a detail. A template fitted and scored on the same
images borrows the noise it was fitted to and reports a detectability that is biased high;
against a bound that says "processing cannot increase detectability", a biased-high
estimator would manufacture violations of the very inequality being tested. Held-out
scoring gives an unbiased estimate of the true performance of the template it actually
built, which can only ever be at or below the ideal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity
from taskiq_core import (
    TrialSet,
    auc_from_scores,
    burgess_eye_filter,
    cho,
    d_prime_from_scores,
    ideal_linear,
    laguerre_gauss_channels,
    nps_2d,
    npwe,
    score_images,
)

from denoiq_core.denoisers import denoise, denoiser_label

__all__ = [
    "EvalConfig",
    "ObserverEstimate",
    "analytic_ideal",
    "auc_standard_error",
    "crop_center",
    "cross_fitted_ideal",
    "cross_fitted_npwe",
    "denoise_trials",
    "empirical_ideal",
    "empirical_npwe",
    "estimate_cho",
    "evaluate_condition",
    "fidelity",
    "fold_indices",
    "npwe_estimate",
    "prewhitening_estimate",
    "split_stack",
]


@dataclass(frozen=True)
class EvalConfig:
    """Estimator settings. Fixed across a study so conditions are comparable.

    Attributes
    ----------
    estimator:
        ``"cross-fit"`` (default) estimates the template on ``n_folds - 1`` folds and scores
        the held-out one, rotating over all folds, so **every** trial contributes a score that
        no part of its own template saw. ``"split"`` uses a single ``train_fraction`` division
        and scores only the held-out part. Cross-fitting uses the whole sample twice over —
        once for estimation, once for scoring — which raises the efficiency of the estimate
        without letting a trial score a template that was fitted to it.
    n_folds:
        Number of cross-fitting folds. Fold membership is ``index % n_folds``: the trials come
        from one seeded i.i.d. stream, so a positional assignment is already a random one and
        needs no second random number to record.
    train_fraction:
        Fraction of trials used to *build* the observer templates under ``estimator="split"``;
        the rest are scored. The default splits evenly, so template and score have the same
        sample size and neither dominates the error. Ignored when cross-fitting.
    roi_size:
        Side of the central window, in pixels, that the estimated observers work in
        (``None`` uses the whole image). The task is signal-known-exactly, so the observer
        knows where to look, and a template estimated over a window that is mostly empty
        background is mostly noise: the estimate's efficiency falls like the number of
        pixels it has to estimate. Cropping is itself a processing step applied to the
        images, so — this is the point that matters for the bound — it can only *lower* the
        estimated detectability, never raise it above the raw ceiling.
    nps_ridge_fraction:
        Tikhonov ridge added to the measured NPS before it is inverted, as a fraction of the
        mean measured power. A prewhitening observer weights by ``1/NPS``, and a measured
        spectrum has bins that fluctuate to near zero — after smoothing, whole bands do. The
        ridge bounds that weight. Like the crop, it can only mis-match the template and
        lower the held-out score; it cannot inflate it.
    nps_floor_fraction:
        Final hard clamp on the regularised spectrum, as a fraction of its peak, so the
        dynamic range stays inside what a prewhitening observer can be trusted with.
    eye_peak_cycles_per_mm:
        Peak of the NPWE eye filter. ``0.25`` cycles/mm places it where a 3 mm disk has its
        spectral content, at the default geometry.
    n_channels, channel_width_mm:
        Laguerre-Gauss channel set for the CHO. The default width matches the default 3 mm
        lesion; :func:`denoiq_core.experiment` derives it from the phantom instead.
    cho_internal_noise:
        Internal-noise standard deviation added to the CHO decision variable. ``0`` keeps the
        CHO an efficiency measure rather than a human model.
    fidelity_data_range:
        Intensity range PSNR and SSIM are computed on — the display window, in the arbitrary
        units the phantoms use (read them as HU and this is a 400-wide soft-tissue window).
        It has to be stated, and stated once for the whole study: SSIM against a flat
        reference is essentially ``C2 / (sigma^2 + C2)`` with ``C2 = (0.03 * range)^2``, so a
        range chosen per condition would silently rescale the plausibility axis between
        conditions and make Figure 4 meaningless.

    """

    train_fraction: float = 0.5
    n_folds: int = 5
    estimator: str = "cross-fit"
    roi_size: int | None = 24
    nps_ridge_fraction: float = 0.01
    nps_floor_fraction: float = 1e-6
    eye_peak_cycles_per_mm: float = 0.25
    n_channels: int = 6
    channel_width_mm: float = 6.0
    cho_internal_noise: float = 0.0
    fidelity_data_range: float = 400.0

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the estimator settings."""
        return {
            "estimator": str(self.estimator),
            "n_folds": int(self.n_folds),
            "train_fraction": float(self.train_fraction),
            "roi_size": None if self.roi_size is None else int(self.roi_size),
            "nps_ridge_fraction": float(self.nps_ridge_fraction),
            "nps_floor_fraction": float(self.nps_floor_fraction),
            "eye_peak_cycles_per_mm": float(self.eye_peak_cycles_per_mm),
            "n_channels": int(self.n_channels),
            "channel_width_mm": float(self.channel_width_mm),
            "cho_internal_noise": float(self.cho_internal_noise),
            "fidelity_data_range": float(self.fidelity_data_range),
        }


DEFAULT_CONFIG = EvalConfig()


@dataclass(frozen=True)
class ObserverEstimate:
    """One observer's reading of one condition.

    Attributes
    ----------
    name:
        ``"ideal"``, ``"cho"`` or ``"npwe"``.
    d_prime, auc:
        Detectability and area under the ROC curve.
    auc_se:
        Standard error of ``auc`` (Hanley-McNeil), from the size of the *scored* sample. It
        is what the ceiling test's confidence margin is built from.
    estimator:
        How the number was obtained: ``"analytic"`` (closed form from a known NPS) or
        ``"held-out"`` (template estimated on a training split, scored on a disjoint one).

    """

    name: str
    d_prime: float
    auc: float
    auc_se: float
    n_present: int
    n_absent: int
    estimator: str
    meta: dict[str, Any] = field(default_factory=dict)
    #: The scored decision variables, kept for resampling (see
    #: :func:`denoiq_core.bound.bootstrap_auc_se`). Deliberately **not** in :meth:`to_dict`:
    #: ``results/`` holds the numbers a manuscript cites, not the raw score vectors.
    scores_present: np.ndarray | None = None
    scores_absent: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        return {
            "name": self.name,
            "d_prime": float(self.d_prime),
            "auc": float(self.auc),
            "auc_se": float(self.auc_se),
            "n_present": int(self.n_present),
            "n_absent": int(self.n_absent),
            "estimator": self.estimator,
            **{k: v for k, v in self.meta.items()},
        }


# --------------------------------------------------------------------------------------
# splitting, and the standard error of an AUC
# --------------------------------------------------------------------------------------


def split_stack(stack: np.ndarray, train_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Split an image stack into (train, test) along the trial axis.

    The split is by position, not by a shuffle: the trials are i.i.d. draws from one seeded
    stream, so the first ``k`` are already an unbiased sample, and taking them needs no
    second random number to record and reproduce.
    """
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError(f"expected a stack (n, ny, nx), got shape {arr.shape}")
    n = arr.shape[0]
    frac = float(train_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction!r}")
    k = int(round(n * frac))
    if k < 2 or n - k < 2:
        raise ValueError(
            f"a {frac:.0%}/{1 - frac:.0%} split of {n} trials leaves {k} to train and {n - k} "
            "to score; each side needs at least 2 for a variance"
        )
    return arr[:k], arr[k:]


def auc_standard_error(auc: float, n_present: int, n_absent: int) -> float:
    r"""Standard error of an AUC (Hanley & McNeil, 1982).

    .. math::  \mathrm{SE} = \sqrt{\frac{A(1-A) + (n_1-1)(Q_1 - A^2) + (n_0-1)(Q_2 - A^2)}
                                       {n_1 n_0}}

    with :math:`Q_1 = A/(2-A)` and :math:`Q_2 = 2A^2/(1+A)`. It assumes an exponential score
    distribution, which makes it mildly conservative for Gaussian scores — the right
    direction for a margin whose job is to avoid crying "violation" at sampling noise.
    :func:`denoiq_core.bound.bootstrap_auc_se` is the distribution-free alternative.
    """
    a = float(auc)
    n1 = int(n_present)
    n0 = int(n_absent)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"auc must be in [0, 1], got {auc!r}")
    if n1 < 1 or n0 < 1:
        raise ValueError("need at least one score in each class")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------------------
# observers
# --------------------------------------------------------------------------------------


def analytic_ideal(trials: TrialSet) -> ObserverEstimate:
    """Closed-form ideal linear observer on the raw trials — the ceiling itself.

    Uses the **analytic** NPS the :class:`~taskiq_core.TrialSet` carries and the true signal,
    so nothing is estimated: ``d'`` here is exact for this experiment, not a sample of it.
    In white noise it reduces to ``d' = ||s||/sigma``, which is what
    ``tests/test_closed_form.py`` verifies.
    """
    res = ideal_linear(trials.signal, trials.nps, trials.spacing, nps_layout="centered")
    return ObserverEstimate(
        name="ideal",
        d_prime=float(res.d_prime),
        auc=float(res.auc),
        auc_se=0.0,
        n_present=int(trials.n_trials),
        n_absent=int(trials.n_trials),
        estimator="analytic",
        meta={
            "observer": "analytic ideal linear (prewhitening) observer on the unprocessed input",
            "template_source": "true signal + analytic NPS",
        },
    )


def crop_center(stack: np.ndarray, size: int | None) -> np.ndarray:
    """Central ``size x size`` window of every image in a stack (``None`` or too large: no-op).

    The window the estimated observers work in. Cropping *is* processing — it throws data
    away — so it can only reduce the detectability that is measured through it, which is the
    safe direction for a study whose central claim is an upper bound.
    """
    arr = np.asarray(stack)
    if size is None:
        return arr
    size = int(size)
    ny, nx = arr.shape[-2:]
    if size <= 0:
        raise ValueError(f"roi_size must be positive or None, got {size!r}")
    if size >= min(ny, nx):
        return arr
    y0 = (ny - size) // 2
    x0 = (nx - size) // 2
    return arr[..., y0 : y0 + size, x0 : x0 + size]


def _prewhitening_nps(
    images: np.ndarray, spacing: float, *, ridge_fraction: float, floor_fraction: float
) -> np.ndarray:
    """Measured NPS of an image stack, made safe to invert. DC-centred.

    Three things happen to it, in order, and each has a reason:

    1. **The DC bin is filled** from the mean of its immediate neighbours. ``nps_2d``
       detrends, which zeroes DC by construction — it is not a measurement of zero power.
       Left at zero and then clamped to a small floor, ``1/NPS`` puts an enormous weight on
       the one frequency where a disk signal has most of its power, and the observer
       degenerates into "compare the image means".
    2. **A ridge is added**, ``ridge * mean(NPS)``, bounding the inverse where the measured
       power has fluctuated (or been smoothed) to nothing.
    3. **A floor is applied** at ``floor_fraction`` of the peak, keeping the dynamic range
       inside what a prewhitening observer can be trusted with.

    These steps stabilise the estimate but may reduce its efficiency relative to the unknown
    optimum; they are not claimed to be a one-sided guarantee.
    """
    ridge_fraction = float(ridge_fraction)
    floor_fraction = float(floor_fraction)
    if ridge_fraction < 0.0:
        raise ValueError(f"nps_ridge_fraction must be >= 0, got {ridge_fraction!r}")
    if not 0.0 < floor_fraction < 1.0:
        raise ValueError(f"nps_floor_fraction must be in (0, 1), got {floor_fraction!r}")

    measured = nps_2d(images, spacing, detrend="mean")
    grid = np.array(measured.nps, dtype=np.float64, copy=True)
    ny, nx = grid.shape
    cy, cx = ny // 2, nx // 2  # DC of a centred (fftshifted) spectrum

    neighbourhood = grid[max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2]
    others = [v for v in neighbourhood.ravel() if v > 0.0]
    grid[cy, cx] = float(np.mean(others)) if others else float(grid.mean())

    grid = grid + ridge_fraction * float(grid.mean())
    peak = float(grid.max())
    if peak <= 0.0:
        raise ValueError("the measured NPS is zero everywhere: the images carry no noise")
    return np.maximum(grid, floor_fraction * peak)


def _template_signal(present_train: np.ndarray, absent_train: np.ndarray) -> np.ndarray:
    """Effective signal seen by the observer: the class-mean difference on the training split.

    For a linear denoiser ``g`` this converges to ``g(s)``; for a non-linear one it is the
    right thing by definition — the mean shift the processing actually produces. It is
    estimated from images, never read off the ground truth, so the observer knows only what
    an observer could know.
    """
    return np.asarray(present_train.mean(axis=0) - absent_train.mean(axis=0), dtype=np.float64)


def _scored(
    name: str, template: np.ndarray, present: np.ndarray, absent: np.ndarray, meta: dict[str, Any]
) -> ObserverEstimate:
    """Apply a spatial template to held-out stacks and package the figures of merit."""
    sp = score_images(present, template)
    sa = score_images(absent, template)
    auc = auc_from_scores(sp, sa)
    return ObserverEstimate(
        name=name,
        d_prime=float(d_prime_from_scores(sp, sa)),
        auc=float(auc),
        auc_se=auc_standard_error(auc, sp.size, sa.size),
        n_present=int(sp.size),
        n_absent=int(sa.size),
        estimator="held-out",
        meta=meta,
        scores_present=sp,
        scores_absent=sa,
    )


def fold_indices(n: int, n_folds: int) -> list[np.ndarray]:
    """Fold membership by position: ``index % n_folds``.

    The trials are i.i.d. draws from one seeded stream, so a positional assignment is already
    a random partition — and unlike a shuffled one it needs no second random number to record,
    which keeps the whole estimate a function of the experiment's seed alone.
    """
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n < 2 * n_folds:
        raise ValueError(f"{n} trials cannot be split into {n_folds} usable folds")
    positions = np.arange(n)
    return [positions[positions % n_folds == k] for k in range(n_folds)]


def _prewhitening_template(
    present_train: np.ndarray,
    absent_train: np.ndarray,
    spacing: float,
    config: EvalConfig,
) -> tuple[np.ndarray, float]:
    """Template and its closed-form ``d'`` from a training set (already cropped)."""
    signal_hat = _template_signal(present_train, absent_train)
    nps_hat = _prewhitening_nps(
        absent_train,
        spacing,
        ridge_fraction=config.nps_ridge_fraction,
        floor_fraction=config.nps_floor_fraction,
    )
    res = ideal_linear(signal_hat, nps_hat, spacing, nps_layout="centered")
    return res.template, float(res.d_prime)


def cross_fitted_ideal(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """Prewhitening matched filter, cross-fitted: every trial scored out of fold.

    For each of ``config.n_folds`` folds the effective signal and the noise spectrum are
    estimated from the *other* folds and the resulting template is applied to this one. The
    out-of-fold scores are then pooled, so the reported ``d'`` and AUC use the whole sample
    while no score ever comes from a template that saw its own image.

    This is the estimator the study reports. Against a single 50/50 split it uses
    ``n_folds - 1`` times as many images per template and scores all of them rather than half,
    which raises the efficiency of the estimate substantially at the same leakage guarantee —
    and estimator efficiency matters here, because an estimate that falls far short of the
    analytic ceiling would satisfy the ceiling comparison for statistical rather than
    informational reasons.
    """
    roi_present = crop_center(present, config.roi_size)
    roi_absent = crop_center(absent, config.roi_size)
    n = int(roi_present.shape[0])
    if roi_absent.shape[0] != n:
        raise ValueError("the two stacks must have the same number of trials")

    folds = fold_indices(n, config.n_folds)
    scores_present: list[np.ndarray] = []
    scores_absent: list[np.ndarray] = []
    closed_form: list[float] = []
    for test in folds:
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        template, d_closed = _prewhitening_template(
            roi_present[mask], roi_absent[mask], spacing, config
        )
        scores_present.append(score_images(roi_present[test], template))
        scores_absent.append(score_images(roi_absent[test], template))
        closed_form.append(d_closed)

    sp = np.concatenate(scores_present)
    sa = np.concatenate(scores_absent)
    auc = auc_from_scores(sp, sa)
    return ObserverEstimate(
        name="ideal",
        d_prime=float(d_prime_from_scores(sp, sa)),
        auc=float(auc),
        auc_se=auc_standard_error(auc, sp.size, sa.size),
        n_present=int(sp.size),
        n_absent=int(sa.size),
        estimator=f"cross-fit ({config.n_folds}-fold)",
        meta={
            "observer": "held-out prewhitening linear observer",
            "template_source": "class-mean difference + measured NPS (out-of-fold)",
            "roi_size": None if config.roi_size is None else int(config.roi_size),
            "nps_ridge_fraction": float(config.nps_ridge_fraction),
            "mean_closed_form_d_prime_on_estimates": float(np.mean(closed_form)),
        },
        scores_present=sp,
        scores_absent=sa,
    )


def cross_fitted_npwe(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """Non-prewhitening observer with an eye filter, cross-fitted the same way."""
    roi_present = crop_center(present, config.roi_size)
    roi_absent = crop_center(absent, config.roi_size)
    n = int(roi_present.shape[0])
    eye = burgess_eye_filter(peak_cycles_per_mm=config.eye_peak_cycles_per_mm)

    scores_present: list[np.ndarray] = []
    scores_absent: list[np.ndarray] = []
    for test in fold_indices(n, config.n_folds):
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        signal_hat = _template_signal(roi_present[mask], roi_absent[mask])
        measured = nps_2d(roi_absent[mask], spacing, detrend="mean")
        res = npwe(signal_hat, measured, spacing, eye_filter=eye)
        scores_present.append(score_images(roi_present[test], res.template))
        scores_absent.append(score_images(roi_absent[test], res.template))

    sp = np.concatenate(scores_present)
    sa = np.concatenate(scores_absent)
    auc = auc_from_scores(sp, sa)
    return ObserverEstimate(
        name="npwe",
        d_prime=float(d_prime_from_scores(sp, sa)),
        auc=float(auc),
        auc_se=auc_standard_error(auc, sp.size, sa.size),
        n_present=int(sp.size),
        n_absent=int(sa.size),
        estimator=f"cross-fit ({config.n_folds}-fold)",
        meta={
            "observer": (
                "held-out non-prewhitening linear observer with an eye filter "
                "(a stylised surrogate for limited noise-prewhitening efficiency)"
            ),
            "template_source": "class-mean difference + eye filter (out-of-fold)",
            "eye_peak_cycles_per_mm": float(config.eye_peak_cycles_per_mm),
        },
        scores_present=sp,
        scores_absent=sa,
    )


def prewhitening_estimate(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """The configured held-out prewhitening estimate: cross-fitted, or a single split."""
    if config.estimator == "split":
        return empirical_ideal(present, absent, spacing, config=config)
    return cross_fitted_ideal(present, absent, spacing, config=config)


def npwe_estimate(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """The configured held-out non-prewhitening estimate."""
    if config.estimator == "split":
        return empirical_npwe(present, absent, spacing, config=config)
    return cross_fitted_npwe(present, absent, spacing, config=config)


def empirical_ideal(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """Prewhitening matched filter estimated from the images, scored on held-out trials.

    The observer this study applies to **processed** images. Both of its ingredients come
    from the training split — the effective signal from the class-mean difference, the noise
    covariance from the measured NPS of the signal-absent images — and the template is then
    scored on trials it has never seen.

    Because the score is held out, the number it returns is an unbiased estimate of the true
    detectability of *this* template, which is at best the ideal detectability of the
    processed images, which by the data-processing inequality is at best that of the raw
    images. That chain is what :func:`denoiq_core.bound.check_no_gain` tests.
    """
    roi_present = crop_center(present, config.roi_size)
    roi_absent = crop_center(absent, config.roi_size)
    p_train, p_test = split_stack(roi_present, config.train_fraction)
    a_train, a_test = split_stack(roi_absent, config.train_fraction)
    signal_hat = _template_signal(p_train, a_train)
    nps_hat = _prewhitening_nps(
        a_train,
        spacing,
        ridge_fraction=config.nps_ridge_fraction,
        floor_fraction=config.nps_floor_fraction,
    )
    res = ideal_linear(signal_hat, nps_hat, spacing, nps_layout="centered")
    return _scored(
        "ideal",
        res.template,
        p_test,
        a_test,
        {
            "observer": "held-out prewhitening linear observer",
            "template_source": "class-mean difference + measured NPS (train split)",
            "roi_size": None if config.roi_size is None else int(config.roi_size),
            "nps_ridge_fraction": float(config.nps_ridge_fraction),
        },
    )


def empirical_npwe(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
) -> ObserverEstimate:
    """Non-prewhitening observer with an eye filter, estimated and scored the same way.

    A stylised surrogate for limited noise-prewhitening efficiency — **not** a validated model
    of a human reader, and no human observer study is involved anywhere in this package. It does
    not invert the noise covariance, so correlated noise costs it performance that a prewhitening
    observer recovers. A denoiser can genuinely *help* this observer: that is a gain in observer
    efficiency, not in information, and Table 2 reports it.
    """
    p_train, p_test = split_stack(crop_center(present, config.roi_size), config.train_fraction)
    a_train, a_test = split_stack(crop_center(absent, config.roi_size), config.train_fraction)
    signal_hat = _template_signal(p_train, a_train)
    measured = nps_2d(a_train, spacing, detrend="mean")
    res = npwe(
        signal_hat,
        measured,
        spacing,
        eye_filter=burgess_eye_filter(peak_cycles_per_mm=config.eye_peak_cycles_per_mm),
    )
    return _scored(
        "npwe",
        res.template,
        p_test,
        a_test,
        {
            "observer": (
                "held-out non-prewhitening linear observer with an eye filter "
                "(a stylised surrogate for limited noise-prewhitening efficiency)"
            ),
            "template_source": "class-mean difference + eye filter (train split)",
            "eye_peak_cycles_per_mm": float(config.eye_peak_cycles_per_mm),
        },
    )


def estimate_cho(
    present: np.ndarray,
    absent: np.ndarray,
    spacing: float,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
    seed: int = 0,
) -> ObserverEstimate:
    """Channelized Hotelling observer on Laguerre-Gauss channels.

    Delegates to :func:`taskiq_core.cho` with ``method="split"``, which does its own
    train/test division — so this number is held out for the same reason and in the same way
    as the other two.
    """
    shape = (int(present.shape[1]), int(present.shape[2]))
    channels = laguerre_gauss_channels(
        shape,
        spacing,
        n_channels=config.n_channels,
        width_mm=config.channel_width_mm,
    )
    res = cho(
        present,
        absent,
        channels,
        method="split",
        internal_noise=config.cho_internal_noise,
        seed=seed,
    )
    return ObserverEstimate(
        name="cho",
        d_prime=float(res.d_prime),
        auc=float(res.auc),
        auc_se=auc_standard_error(res.auc, res.n_present, res.n_absent),
        n_present=int(res.n_present),
        n_absent=int(res.n_absent),
        estimator="held-out",
        meta={
            "observer": "channelized Hotelling observer (Laguerre-Gauss channels, split)",
            "n_channels": int(res.n_channels),
            "channel_width_mm": float(config.channel_width_mm),
            "condition_number": float(res.condition_number),
        },
        scores_present=np.asarray(res.scores_present, dtype=np.float64),
        scores_absent=np.asarray(res.scores_absent, dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------------------


def fidelity(
    images: np.ndarray, reference: np.ndarray, *, data_range: float | None = None
) -> dict[str, float]:
    """RMSE, PSNR and SSIM of a stack against one noise-free reference image.

    Parameters
    ----------
    images:
        Stack ``(n, ny, nx)`` (or a single image) to score.
    reference:
        The noise-free truth these images are estimates of.
    data_range:
        Intensity range for PSNR and SSIM. Defaults to the peak-to-peak of the reference —
        i.e. the lesion contrast — so the numbers are on the scale of the thing being
        detected rather than on an arbitrary window.

    Returns
    -------
    dict
        ``rmse``, ``psnr``, ``ssim`` (means over the stack) and the ``data_range`` used.

    """
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim == 2:
        stack = arr[None, ...]
    elif arr.ndim == 3:
        stack = arr
    else:
        raise ValueError(f"images must be 2-D or a 3-D stack, got shape {arr.shape}")
    ref = np.asarray(reference, dtype=np.float64)
    if stack.shape[1:] != ref.shape:
        raise ValueError(f"reference shape {ref.shape} does not match images {stack.shape[1:]}")

    rng = float(np.ptp(ref)) if data_range is None else float(data_range)
    if not np.isfinite(rng) or rng <= 0.0:
        raise ValueError(
            "data_range must be finite and > 0; a flat reference has none of its own, so pass "
            "the lesion contrast explicitly"
        )

    err = stack - ref[None, ...]
    mse = np.mean(err**2, axis=(1, 2))
    rmse = float(np.sqrt(mse).mean())
    psnr = float(np.mean(10.0 * np.log10(rng**2 / np.maximum(mse, 1e-300))))
    ssim = float(np.mean([structural_similarity(ref, img, data_range=rng) for img in stack]))
    return {"rmse": rmse, "psnr": psnr, "ssim": ssim, "data_range": rng}


# --------------------------------------------------------------------------------------
# the orchestration
# --------------------------------------------------------------------------------------


def denoise_trials(
    trials: TrialSet, method: str, params: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a denoiser to both stacks of a trial set, independently and identically.

    The denoiser never learns which stack it is processing — the same call, the same
    parameters, one image at a time.
    """
    kwargs = dict(params or {})
    return (
        denoise(trials.present, method, **kwargs),
        denoise(trials.absent, method, **kwargs),
    )


def evaluate_condition(
    trials: TrialSet,
    method: str = "none",
    params: dict[str, Any] | None = None,
    *,
    observers: tuple[str, ...] = ("ideal", "cho", "npwe"),
    config: EvalConfig = DEFAULT_CONFIG,
    condition: dict[str, Any] | None = None,
    seed: int = 0,
    images: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fidelity and detectability for one denoiser on one experimental condition.

    Parameters
    ----------
    trials:
        The raw experiment.
    method, params:
        The denoiser to apply. ``"none"`` evaluates the raw images, which is how the ceiling
        row of every table is produced.
    observers:
        Which observers to run. ``"ideal"`` is the one the bound is about; ``"cho"`` and
        ``"npwe"`` are the sub-optimal readings that a denoiser can genuinely improve.
    config:
        Estimator settings, held fixed across a study.
    condition:
        Caller-supplied description of the acquisition condition, copied into the record.
    seed:
        Seed for the CHO's internal split/noise. Everything else is deterministic.
    images:
        Pre-computed ``(present, absent)`` stacks, if the denoiser has already been run (the
        CNN path passes its output here rather than re-running the network).

    Returns
    -------
    dict
        A JSON-serialisable record: ``condition``, ``denoiser``, ``fidelity``, ``observers``
        and, for the raw condition, the analytic ideal observer under
        ``observers["ideal_analytic"]``.

    """
    params = dict(params or {})
    present, absent = images if images is not None else denoise_trials(trials, method, params)

    clean_present = float(trials.background) + trials.signal
    fid = fidelity(present, clean_present, data_range=config.fidelity_data_range)

    obs: dict[str, Any] = {}
    if "ideal" in observers:
        obs["ideal"] = prewhitening_estimate(
            present, absent, trials.spacing, config=config
        ).to_dict()
    if "cho" in observers:
        obs["cho"] = estimate_cho(
            present, absent, trials.spacing, config=config, seed=seed
        ).to_dict()
    if "npwe" in observers:
        obs["npwe"] = npwe_estimate(present, absent, trials.spacing, config=config).to_dict()
    if method == "none" and images is None:
        obs["ideal_analytic"] = analytic_ideal(trials).to_dict()

    return {
        "condition": dict(condition or {}),
        "denoiser": {
            "method": method,
            "params": params,
            "label": denoiser_label(method, params),
        },
        "trials": {
            "n_trials": int(trials.n_trials),
            "seed": int(trials.seed),
            "spacing": float(trials.spacing),
            "noise_sd": float(trials.noise_sd),
            "pixel_sd": float(trials.meta["pixel_sd"]),
            "correlation_sigma_mm": float(trials.correlation_sigma_mm),
            "white_floor_sd": float(trials.white_floor_sd),
            "signal_peak": float(np.max(np.abs(trials.signal))),
            "signal_energy": float(trials.meta["signal_energy"]),
        },
        "fidelity": fid,
        "observers": obs,
        "config": config.to_dict(),
    }
