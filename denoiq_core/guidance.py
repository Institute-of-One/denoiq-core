"""Turn the measurements into a protocol decision.

Why this module exists
----------------------
The rest of this package measures. It says what the ceiling is, what a real observer reaches
against it, and how far fidelity and task performance part company. None of that is a decision.
The person who has to make one is choosing an exposure, and what they need from this work is an
answer to "how far can this protocol be cut, and does reconstruction or denoising change that".
This module is that answer, in the units the question is asked in.

The three steps
---------------
1. :func:`exposure_for_requirement` -- the exposure at which detectability falls to a
   prespecified requirement, from one measurement at the exposure in use. This is the number the
   decision is.
2. :func:`achievable_floor` -- the correction that stops that number being optimistic. A floor
   computed from an ideal observer is not reachable; a real observer reaches a fraction of the
   ideal, and the exposure scales as the inverse *square* of that fraction. An efficiency of
   0.75 -- roughly what this study measures on real liver data -- means the reachable floor
   needs 78 % more exposure than the ideal-observer floor. Protocols set from an ideal-observer
   calculation are under-exposed by that much.
3. :func:`apparent_exposure` -- what the processed picture claims about its own exposure. This
   is the quantity the decision must *not* be made from, and it is computed here so that the
   size of the discrepancy can be stated rather than warned about in the abstract.

:func:`decide` runs all three and returns the verdict with every assumption attached.

What is transferable and what is not
------------------------------------
The absolute exposures this study reports are specific to its task, its lesion and its idealised
background, and are not a clinical recommendation. What transfers is the procedure and three
findings that enter it:

* processing did not extend the reachable floor; on the data measured here all but one
  arm shortened it, so nothing in this work supports spending a fidelity gain as exposure;
* the discrepancy between claimed and delivered exposure *grows* as the exposure falls, so it is
  largest where the task has already been lost;
* observer efficiency is not constant along the dose axis, so a single measured efficiency
  extrapolates only near where it was measured. :func:`decide` says so when it is asked to
  extrapolate far.

The noise model, stated once
----------------------------
Detectability of a fixed signal in quantum noise goes as :math:`d' \\propto \\sqrt{D}`, because
the noise power goes as :math:`1/D`. Every conversion here uses that and nothing else, so a
caller whose noise is not quantum-limited -- an image that is electronic-noise limited at very
low exposure, for instance -- is outside what these functions describe.

This is deliberately *not* the convention used on the dose axis of this study's real-data arm.
There the background is the routine-dose reconstruction treated as known, so only the noise the
reduction *inserted* is stochastic and its power goes as :math:`1/D - 1`. That convention
measures what dose reduction takes away and is the right one for a data-processing question; it
is the wrong one for a site measuring its own images, which carry their total noise. The two are
kept apart on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Below this ratio between the exposure asked about and the exposure the efficiency was
#: measured at, :func:`decide` stops trusting the efficiency. The dose axis of the real-data
#: arm shows observer efficiency varying by 16-47 % over an eight-fold range, so an efficiency
#: measured at one exposure is a local quantity.
EXTRAPOLATION_LIMIT = 2.0


def exposure_for_requirement(
    current_exposure: float, measured_d_prime: float, requirement: float
) -> float:
    r"""The exposure at which detectability would fall to ``requirement``.

    From :math:`d' \propto \sqrt{D}`:

    .. math::  D^{*} = D \left(\frac{d'_{\text{req}}}{d'}\right)^{2}

    Parameters
    ----------
    current_exposure:
        The exposure the measurement was made at, in whatever units the answer is wanted in
        (mAs, CTDIvol, or a fraction of the routine protocol).
    measured_d_prime:
        Detectability measured at that exposure, by an observer the reader is willing to stand
        behind. Using an ideal observer's value here is the mistake :func:`achievable_floor`
        exists to prevent.
    requirement:
        The prespecified task requirement.

    Returns
    -------
    float
        The exposure, in the units of ``current_exposure``. Larger than it when the measurement
        is below the requirement.

    """
    for name, value in (("current_exposure", current_exposure), ("requirement", requirement)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if not np.isfinite(measured_d_prime) or measured_d_prime <= 0.0:
        raise ValueError(f"measured_d_prime must be finite and > 0, got {measured_d_prime!r}")
    return float(current_exposure) * (float(requirement) / float(measured_d_prime)) ** 2


def achievable_floor(ideal_floor: float | np.ndarray, efficiency: float) -> float | np.ndarray:
    r"""Correct an ideal-observer floor into one a real observer reaches.

    An observer reaching a fraction :math:`\eta` of the ideal detectability needs the exposure
    at which the *ideal* detectability is :math:`d'_{\text{req}}/\eta`, and exposure enters
    detectability as a square root, so

    .. math::  D_{\text{achievable}} = D_{\text{ideal}} \, / \, \eta^{2}

    The square is the point. An efficiency of 0.75 sounds like a 25 % shortfall and costs 78 %
    more exposure; an efficiency of 0.5 costs four times as much.

    Parameters
    ----------
    ideal_floor:
        The floor from an ideal-observer calculation -- for example
        :attr:`denoiq_core.redlamp.Atlas.floor_mas`, in mAs, for each kV.
    efficiency:
        Measured ratio of achieved to ideal detectability, in ``(0, 1]``. It has no default:
        an efficiency belongs to a task, an observer and an exposure, and inheriting someone
        else's silently is the error this signature is shaped to prevent.

    Returns
    -------
    float or ndarray
        The floor a real observer reaches, in the units of ``ideal_floor``.

    """
    eta = float(efficiency)
    if not np.isfinite(eta) or not 0.0 < eta <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency!r}")
    floor = np.asarray(ideal_floor, dtype=np.float64)
    if np.any(floor <= 0.0) or not np.all(np.isfinite(floor)):
        raise ValueError("ideal_floor must be finite and > 0")
    out = floor / eta**2
    return float(out) if out.ndim == 0 else out


def apparent_exposure(current_exposure: float, fidelity_gain_db: float) -> float:
    r"""The exposure the processed image's fidelity claims for it.

    The mean-squared error against the truth is the noise power, which goes as :math:`1/D`, so
    a gain of :math:`\Delta` dB in PSNR is the fidelity of

    .. math::  D^{*} = D \cdot 10^{\Delta/10}

    A reader who judges exposure by appearance is reading this number. It is reported so that
    the gap between it and :func:`exposure_for_requirement` can be stated as a factor.
    """
    if not np.isfinite(current_exposure) or current_exposure <= 0.0:
        raise ValueError(f"current_exposure must be finite and > 0, got {current_exposure!r}")
    if not np.isfinite(fidelity_gain_db):
        raise ValueError(f"fidelity_gain_db must be finite, got {fidelity_gain_db!r}")
    return float(current_exposure) * 10.0 ** (float(fidelity_gain_db) / 10.0)


@dataclass(frozen=True)
class DoseDecision:
    """An exposure decision, the numbers behind it, and what it assumes.

    Attributes
    ----------
    verdict:
        ``"concede"`` -- the task has headroom and exposure can be reduced to
        ``minimum_exposure``; ``"hold"`` -- the exposure is at the requirement within the stated
        margin; ``"raise"`` -- the requirement is not met at the exposure in use.
    minimum_exposure:
        The exposure at which detectability reaches the requirement, in the caller's units.
    headroom:
        ``current_exposure / minimum_exposure``. Above one there is room to concede.
    claimed_exposure:
        What the processed image's fidelity claims, or ``None`` if no fidelity gain was given.
    overstatement:
        ``claimed_exposure / current_exposure``: how many times the exposure the picture looks
        like it was given. ``None`` when no fidelity gain was given.
    warnings:
        Every reason this decision might not hold. An empty list is meaningful; a non-empty one
        is not a formality.

    """

    verdict: str
    requirement: float
    current_exposure: float
    current_d_prime: float
    minimum_exposure: float
    headroom: float
    efficiency: float | None = None
    claimed_exposure: float | None = None
    overstatement: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        return {
            "verdict": self.verdict,
            "requirement": float(self.requirement),
            "current_exposure": float(self.current_exposure),
            "current_d_prime": float(self.current_d_prime),
            "minimum_exposure": float(self.minimum_exposure),
            "headroom": float(self.headroom),
            "efficiency": None if self.efficiency is None else float(self.efficiency),
            "claimed_exposure": (
                None if self.claimed_exposure is None else float(self.claimed_exposure)
            ),
            "overstatement": (
                None if self.overstatement is None else float(self.overstatement)
            ),
            "warnings": list(self.warnings),
        }

    def explain(self) -> str:
        """The decision in sentences, for a report or a console."""
        lines = [
            f"{self.verdict.upper()}: the task needs {self.minimum_exposure:.3g} and the "
            f"protocol is at {self.current_exposure:.3g} "
            f"({self.headroom:.2f}x the requirement).",
            f"Measured d' = {self.current_d_prime:.3g} against a requirement of "
            f"{self.requirement:.3g}.",
        ]
        if self.overstatement is not None and self.claimed_exposure is not None:
            lines.append(
                f"The processed image has the fidelity of {self.claimed_exposure:.3g} "
                f"({self.overstatement:.2f}x the exposure it was given). That is appearance, "
                f"not detectability, and it is not part of this decision."
            )
        lines.extend(f"Caution: {w}" for w in self.warnings)
        return "\n".join(lines)


def decide(
    current_exposure: float,
    measured_d_prime: float,
    requirement: float,
    *,
    efficiency: float | None = None,
    efficiency_measured_at: float | None = None,
    fidelity_gain_db: float | None = None,
    hold_margin: float = 0.05,
) -> DoseDecision:
    """Decide what to do with the exposure, and say what the decision assumes.

    Parameters
    ----------
    current_exposure:
        The exposure in use, in any units; the answer comes back in the same ones.
    measured_d_prime:
        Detectability measured at that exposure, by an observer the caller stands behind.
    requirement:
        The prespecified task requirement -- a property of the task, fixed before the
        measurement, not read off it.
    efficiency:
        Ratio of achieved to ideal detectability, if known. Recorded and reported; it does not
        change the arithmetic, because ``measured_d_prime`` is already an achieved value. It is
        used to warn when an efficiency is being carried further than it was measured.
    efficiency_measured_at:
        The exposure ``efficiency`` was measured at. Given both, an extrapolation further than
        :data:`EXTRAPOLATION_LIMIT` raises a warning, because efficiency is not constant along
        the dose axis.
    fidelity_gain_db:
        PSNR gain of the processed image over the unprocessed one at this exposure, if the
        decision is being made about a processed image. Supplying it adds the claimed exposure
        and the overstatement to the result.
    hold_margin:
        Fractional band around the requirement counted as ``"hold"`` rather than
        ``"concede"`` or ``"raise"``.

    Returns
    -------
    DoseDecision

    """
    minimum = exposure_for_requirement(current_exposure, measured_d_prime, requirement)
    headroom = float(current_exposure) / minimum
    if headroom > 1.0 + float(hold_margin):
        verdict = "concede"
    elif headroom < 1.0 - float(hold_margin):
        verdict = "raise"
    else:
        verdict = "hold"

    warnings: list[str] = []
    claimed = overstatement = None
    if fidelity_gain_db is not None:
        claimed = apparent_exposure(current_exposure, fidelity_gain_db)
        overstatement = claimed / float(current_exposure)
        if overstatement > 1.2:
            warnings.append(
                f"the image looks like {overstatement:.1f}x the exposure it was given; on the "
                "data behind this module that overstatement grows as exposure falls and is "
                "largest below the requirement, where the task has already been lost"
            )
        if verdict == "raise":
            warnings.append(
                "the requirement is not met, and processing cannot restore it: a denoiser is a "
                "function of the image it is given and cannot add task information to it"
            )
    if efficiency is not None:
        if not 0.0 < float(efficiency) <= 1.0:
            raise ValueError(f"efficiency must be in (0, 1], got {efficiency!r}")
        if efficiency_measured_at is not None:
            ratio = max(
                float(current_exposure) / float(efficiency_measured_at),
                float(efficiency_measured_at) / float(current_exposure),
            )
            if ratio > EXTRAPOLATION_LIMIT:
                warnings.append(
                    f"the efficiency was measured {ratio:.1f}x away from this exposure, and "
                    "efficiency is not constant along the dose axis"
                )
        if minimum / float(current_exposure) < 1.0 / EXTRAPOLATION_LIMIT:
            warnings.append(
                f"the answer is {float(current_exposure) / minimum:.1f}x below the exposure it "
                "was measured at; confirm it by measuring there rather than concede it on this "
                "extrapolation"
            )
    return DoseDecision(
        verdict=verdict,
        requirement=float(requirement),
        current_exposure=float(current_exposure),
        current_d_prime=float(measured_d_prime),
        minimum_exposure=minimum,
        headroom=headroom,
        efficiency=None if efficiency is None else float(efficiency),
        claimed_exposure=claimed,
        overstatement=overstatement,
        warnings=warnings,
    )
