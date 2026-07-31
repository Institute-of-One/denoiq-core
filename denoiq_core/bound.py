r"""The ceiling: post-processing cannot add information, and this checks that it does not.

The argument
------------
Let :math:`H \in \{0, 1\}` be the hypothesis (lesion absent / present), :math:`G` the raw
image, and :math:`\hat G = g(G)` any processed version of it — a Gaussian filter, total
variation, a convolutional network, anything at all, as long as it is a function of the
image and not of the truth. Then :math:`H \to G \to \hat G` is a Markov chain, and the
**data-processing inequality** gives

.. math::  I(H; \hat G) \;\le\; I(H; G).

Detection performance is a monotone read-out of that mutual information: by the
Neyman-Pearson lemma the likelihood-ratio test on :math:`G` maximises the ROC curve
*pointwise*, so no test on :math:`\hat G` can have a larger AUC than the ideal observer on
:math:`G`:

.. math::  \mathrm{AUC}_{\mathrm{ideal}}(\hat G) \;\le\; \mathrm{AUC}_{\mathrm{ideal}}(G).

Equality holds exactly when :math:`g` is invertible (a sufficient statistic is preserved) —
an invertible linear filter, for instance, changes the noise and the signal in the same way
and the prewhitening observer undoes both. Every non-invertible :math:`g` — every denoiser
anyone actually uses — can only lose.

What this module does with it
-----------------------------
:func:`ideal_ceiling` computes the right-hand side in closed form, from the analytic NPS the
trial set carries. :func:`check_no_gain` computes the left-hand side by the held-out
estimator of :mod:`denoiq_core.evaluate` and asserts the inequality within a confidence
margin. ``tests/test_bound.py`` runs it over every condition in the study.

**A violation is not a discovery; it is a bug.** Either the "denoiser" saw something it
should not have (the truth, the noise realisation, the other stack), or the observer
estimate is biased high (a template scored on the images that fitted it), or the confidence
margin is wrong. The test exists to catch exactly those three mistakes, and it is the
reason a result from this package can be believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from taskiq_core import TrialSet, auc_from_scores

from denoiq_core.denoisers import denoiser_label
from denoiq_core.evaluate import (
    DEFAULT_CONFIG,
    EvalConfig,
    ObserverEstimate,
    analytic_ideal,
    denoise_trials,
    prewhitening_estimate,
)

__all__ = [
    "BoundResult",
    "CeilingResult",
    "bootstrap_auc_se",
    "check_denoiser",
    "check_no_gain",
    "ideal_ceiling",
    "raw_estimator_check",
]

#: Two-sided 95 % normal quantile — the default width of the confidence margin.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class CeilingResult:
    """The information ceiling of a raw experiment.

    Attributes
    ----------
    d_prime, auc:
        Ideal-observer detectability and AUC on the **raw** images, in closed form.
    source:
        ``"analytic"`` — computed from the known signal and the analytic NPS, so it carries
        no sampling error of its own. That matters: the whole confidence margin can then be
        attributed to the processed-image estimate.

    """

    d_prime: float
    auc: float
    source: str = "analytic"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record."""
        return {
            "d_prime": float(self.d_prime),
            "auc": float(self.auc),
            "source": self.source,
            **self.meta,
        }


def ideal_ceiling(trials: TrialSet) -> CeilingResult:
    """Ideal-observer AUC of the raw data — the quantity no processing may exceed.

    Everything downstream of the detector is bounded by this number, including any denoiser,
    any network, and any human looking at any rendering of the image.
    """
    est = analytic_ideal(trials)
    return CeilingResult(
        d_prime=est.d_prime,
        auc=est.auc,
        source="analytic",
        meta={
            "noise_sd": float(trials.noise_sd),
            "pixel_sd": float(trials.meta["pixel_sd"]),
            "signal_energy": float(trials.meta["signal_energy"]),
        },
    )


def bootstrap_auc_se(
    scores_present: np.ndarray,
    scores_absent: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """Distribution-free standard error of an AUC, by resampling both score sets.

    The alternative to the Hanley-McNeil closed form in
    :func:`denoiq_core.evaluate.auc_standard_error` when the score distributions are not
    trusted to be well-behaved — after a non-linear denoiser, they may not be. Seeded, so
    the margin it produces is reproducible.
    """
    p = np.asarray(scores_present, dtype=np.float64).ravel()
    a = np.asarray(scores_absent, dtype=np.float64).ravel()
    if p.size < 2 or a.size < 2:
        raise ValueError("need at least 2 scores per class to bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        draws[i] = auc_from_scores(
            p[rng.integers(0, p.size, p.size)], a[rng.integers(0, a.size, a.size)]
        )
    return float(draws.std(ddof=1))


@dataclass(frozen=True)
class BoundResult:
    """The verdict of the ceiling test for one processed condition.

    Attributes
    ----------
    ceiling_auc:
        Ideal-observer AUC on the raw images (closed form).
    processed_auc, processed_auc_se:
        Held-out ideal-observer AUC on the processed images, and its standard error.
    margin:
        How far above the ceiling the processed estimate is allowed to sit before the result
        counts as a violation. It is a *sampling* allowance, not a fudge factor: the ceiling
        is exact, so the only thing that can push the comparison the wrong way is noise in
        the processed estimate.
    excess:
        ``processed_auc - ceiling_auc``. Negative for every non-invertible processing;
        approximately zero for an invertible one.
    ok:
        ``excess <= margin``. **False means something is wrong with the experiment**, not
        that the inequality failed.
    efficiency:
        ``(d'_processed / d'_ceiling)**2`` — the fraction of the raw information the
        processed images still deliver to an ideal observer. This is the number that says
        how much a denoiser threw away.

    """

    ceiling_auc: float
    ceiling_d_prime: float
    processed_auc: float
    processed_auc_se: float
    processed_d_prime: float
    margin: float
    excess: float
    ok: bool
    efficiency: float
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        return {
            "label": self.label,
            "ceiling_auc": float(self.ceiling_auc),
            "ceiling_d_prime": float(self.ceiling_d_prime),
            "processed_auc": float(self.processed_auc),
            "processed_auc_se": float(self.processed_auc_se),
            "processed_d_prime": float(self.processed_d_prime),
            "margin": float(self.margin),
            "excess": float(self.excess),
            "ok": bool(self.ok),
            "efficiency": float(self.efficiency),
            **self.meta,
        }

    def reason(self) -> str:
        """One line saying what happened, for logs and for a failing assertion."""
        verdict = "within the ceiling" if self.ok else "ABOVE THE CEILING"
        return (
            f"{self.label or 'processed'}: ideal AUC {self.processed_auc:.4f} "
            f"vs raw ceiling {self.ceiling_auc:.4f} "
            f"(excess {self.excess:+.4f}, margin {self.margin:.4f}) — {verdict}"
        )


def check_no_gain(
    raw_trials: TrialSet,
    present: np.ndarray,
    absent: np.ndarray,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
    z: float = Z95,
    ci_margin: float | None = None,
    bootstrap: bool = False,
    seed: int = 0,
    label: str = "",
    estimate: ObserverEstimate | None = None,
) -> BoundResult:
    """Test the data-processing ceiling for one pair of processed stacks.

    Parameters
    ----------
    raw_trials:
        The unprocessed experiment, which supplies the ceiling.
    present, absent:
        The processed image stacks. They must be the processed versions of *these* trials —
        comparing against a ceiling from a different experiment tests nothing.
    config:
        Estimator settings for the held-out ideal observer.
    z, ci_margin:
        The margin is ``z * SE`` unless ``ci_margin`` is given explicitly.
    bootstrap:
        Use a resampled standard error instead of the Hanley-McNeil closed form.
    seed:
        Seed for the bootstrap resampling; ignored otherwise.
    label:
        Name of the processing under test, carried into the record and the reason string.
    estimate:
        A previously computed held-out ideal estimate for these stacks, to avoid scoring
        them twice when :func:`denoiq_core.evaluate.evaluate_condition` has already run.

    Returns
    -------
    BoundResult

    """
    ceiling = ideal_ceiling(raw_trials)
    est = estimate or prewhitening_estimate(present, absent, raw_trials.spacing, config=config)

    se = est.auc_se
    se_source = "hanley-mcneil"
    if bootstrap:
        if est.scores_present is None or est.scores_absent is None:  # pragma: no cover
            raise ValueError("bootstrap needs the scored decision variables")
        se = bootstrap_auc_se(est.scores_present, est.scores_absent, seed=seed)
        se_source = "bootstrap"

    # The Mann-Whitney AUC is quantised: with n1 present and n0 absent scores it can only
    # move in steps of 1/(n1*n0), and it saturates at exactly 1. A ceiling AUC of 0.9999995
    # and an estimate of 1.0000000 differ by less than one such step — that is the estimator
    # running out of resolution, not processing creating information — so one quantum is
    # added to the margin. It is ~2.5e-5 at 200 x 200 scores and never rescues a real
    # violation.
    resolution = 1.0 / float(est.n_present * est.n_absent)
    margin = float(ci_margin) if ci_margin is not None else float(z) * se + resolution
    excess = float(est.auc - ceiling.auc)
    efficiency = (
        float((est.d_prime / ceiling.d_prime) ** 2) if ceiling.d_prime > 0 else float("nan")
    )
    saturated = bool(ceiling.auc > 1.0 - 10.0 * resolution)

    return BoundResult(
        ceiling_auc=ceiling.auc,
        ceiling_d_prime=ceiling.d_prime,
        processed_auc=est.auc,
        processed_auc_se=se,
        processed_d_prime=est.d_prime,
        margin=margin,
        excess=excess,
        ok=bool(excess <= margin),
        efficiency=efficiency,
        label=label,
        meta={
            "se_source": se_source,
            "z": float(z),
            "auc_resolution": resolution,
            # Both AUCs sit against the top of the scale: the inequality still holds, but it
            # is no longer informative there, and the d' comparison is the one to read.
            "saturated": saturated,
        },
    )


def check_denoiser(
    trials: TrialSet,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    config: EvalConfig = DEFAULT_CONFIG,
    z: float = Z95,
    bootstrap: bool = False,
    seed: int = 0,
) -> BoundResult:
    """Run a denoiser on a trial set and test the ceiling on its output.

    The convenience entry point used by the sweeps and by ``tests/test_bound.py``.
    """
    present, absent = denoise_trials(trials, method, params)
    return check_no_gain(
        trials,
        present,
        absent,
        config=config,
        z=z,
        bootstrap=bootstrap,
        seed=seed,
        label=denoiser_label(method, params),
    )


def raw_estimator_check(trials: TrialSet, *, config: EvalConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Sanity check: how close the held-out estimator gets to the closed-form ceiling on raw data.

    The estimator builds its template from a finite sample (a class-mean difference and a
    measured, clamped NPS), so it must fall *short* of the analytic ideal — by a little, not
    by a lot. This ratio is what makes the ceiling test meaningful in both directions: an
    estimator that landed far below the ceiling would satisfy the inequality trivially and
    prove nothing.

    Returns
    -------
    dict
        ``ceiling_auc``, ``estimated_auc``, ``ratio`` (estimated / ceiling), and the same for
        ``d'``.

    """
    ceiling = ideal_ceiling(trials)
    est = prewhitening_estimate(trials.present, trials.absent, trials.spacing, config=config)
    return {
        "ceiling_auc": ceiling.auc,
        "ceiling_d_prime": ceiling.d_prime,
        "estimated_auc": est.auc,
        "estimated_auc_se": est.auc_se,
        "estimated_d_prime": est.d_prime,
        "auc_ratio": float(est.auc / ceiling.auc),
        "d_prime_ratio": float(est.d_prime / ceiling.d_prime),
    }
