r"""denoiq-core: the information limits of image denoising, on synthetic phantoms.

Post-processing cannot create information. For a detection task that has two measurable
consequences, and this package computes both.

**The ceiling** (:mod:`denoiq_core.bound`). The chain hypothesis → raw image → processed
image is Markov, so by the data-processing inequality and the Neyman-Pearson lemma the ideal
observer's AUC on the raw data bounds the performance of any observer on any processed
version of it. The ceiling is computed in closed form; processed performance is estimated
with a held-out template so the comparison cannot be flattered by its own estimator.

**The floor** (:mod:`denoiq_core.redlamp`). Below a certain exposure the input's ideal
detectability falls under a *prespecified* requirement — the Rose criterion by default. This is
an operational threshold, not a zero-information boundary: measurable task information remains
below it, but no post-processing can bring the requirement back into reach, because processing
cannot exceed its input's ceiling. The red lamp reports that, with the reason attached, next to
what the processing did to fidelity, to the task estimate, to lesion-like false responses and
to lesion contrast.

Everything here is synthetic, analytic and seeded. Model observers, SKE/BKE trials and the
ROC machinery come from ``taskiq-core`` and are not reimplemented. The deep-learning path
(:mod:`denoiq_core.cnn`, :mod:`denoiq_core.train`) is an optional extra; nothing else needs
it.

Modules
-------
``denoisers``   classical denoisers behind one interface, plus registration for the optional CNN
``physics``     a documented *relative* (kV, mAs) → contrast/noise model — not a calibration
``evaluate``    fidelity and model-observer detectability for one condition
``bound``       the data-processing ceiling and the test of it
``redlamp``     the information floor, fabrication and erasure, the green/amber/red gauge
``experiment``  condition sweeps; the only place numbers come from
``figures``     Figures 1-7, drawn from ``results/``
``cnn``/``train``  the optional residual network and its deterministic training
"""

from __future__ import annotations

from denoiq_core.bound import (
    BoundResult,
    CeilingResult,
    check_denoiser,
    check_no_gain,
    ideal_ceiling,
    raw_estimator_check,
)
from denoiq_core.denoisers import (
    available_methods,
    denoise,
    denoiser_label,
    register_denoiser,
)
from denoiq_core.evaluate import (
    EvalConfig,
    ObserverEstimate,
    analytic_ideal,
    empirical_ideal,
    empirical_npwe,
    estimate_cho,
    evaluate_condition,
    fidelity,
)
from denoiq_core.physics import (
    DEFAULT_MODEL,
    DEFAULT_PHANTOM,
    Acquisition,
    AcquisitionModel,
    PhantomSpec,
    acquisition_params,
    make_trials,
    relative_dose,
    relative_photons,
)
from denoiq_core.redlamp import (
    Atlas,
    RedLamp,
    RedLampCriteria,
    assess,
    classify,
    contrast_recovery,
    false_structure_rate,
    information_floor,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # denoisers
    "denoise",
    "available_methods",
    "denoiser_label",
    "register_denoiser",
    # physics
    "Acquisition",
    "AcquisitionModel",
    "PhantomSpec",
    "DEFAULT_MODEL",
    "DEFAULT_PHANTOM",
    "acquisition_params",
    "relative_photons",
    "relative_dose",
    "make_trials",
    # evaluate
    "EvalConfig",
    "ObserverEstimate",
    "evaluate_condition",
    "fidelity",
    "analytic_ideal",
    "empirical_ideal",
    "empirical_npwe",
    "estimate_cho",
    # bound
    "CeilingResult",
    "BoundResult",
    "ideal_ceiling",
    "check_no_gain",
    "check_denoiser",
    "raw_estimator_check",
    # redlamp
    "Atlas",
    "RedLamp",
    "RedLampCriteria",
    "information_floor",
    "false_structure_rate",
    "contrast_recovery",
    "classify",
    "assess",
]
