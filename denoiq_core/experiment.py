r"""Condition sweeps, and the only place numbers are allowed to come from.

Every figure and every number in the manuscript is produced here and written to
``results/``. Nothing is computed in a figure script, nothing is typed into a table by hand,
and ``tests/test_manuscript_consistency.py`` re-derives the manuscript's numbers from these
files. A number that is not in ``results/`` is not a result.

The axes
--------
``sweep_dose``
    mAs at fixed kV — the dose axis, from comfortably above the information floor to well
    below it. This is the sweep behind Figures 2-5.
``sweep_texture``
    Noise correlation length, with a white floor. Correlated noise is where a prewhitening
    observer and a non-prewhitening one come apart, and where a denoiser can genuinely help
    the latter without touching the ceiling.
``sweep_signal``
    Lesion contrast and radius — the same dose seen by an easier and a harder task.
``sweep_kv_mas``
    The kV-mAs plane: the red-lamp atlas and its information floor.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import taskiq_core

import denoiq_core
from denoiq_core.bound import check_no_gain, ideal_ceiling, raw_estimator_check
from denoiq_core.denoisers import available_methods
from denoiq_core.evaluate import (
    DEFAULT_CONFIG,
    EvalConfig,
    ObserverEstimate,
    denoise_trials,
    evaluate_condition,
)
from denoiq_core.physics import (
    DEFAULT_MODEL,
    DEFAULT_PHANTOM,
    AcquisitionModel,
    PhantomSpec,
    acquisition_params,
    make_trials,
)
from denoiq_core.redlamp import (
    DEFAULT_CRITERIA,
    Atlas,
    RedLampCriteria,
    classify,
    contrast_recovery,
    false_structure_rate,
    information_floor,
)

__all__ = [
    "DEFAULT_DENOISERS",
    "PRIMARY_SEEDS",
    "DenoiserSpec",
    "SweepConfig",
    "closed_form_validation",
    "results_dir",
    "ceiling_summary",
    "design_matrix",
    "estimator_sensitivity",
    "run_all",
    "run_condition",
    "run_multiseed",
    "run_primary",
    "run_seed",
    "sweep_dose",
    "sweep_kv_mas",
    "sweep_signal",
    "sweep_texture",
    "summarize",
    "task_gain_table",
    "write_csv",
    "write_json",
]


# --------------------------------------------------------------------------------------
# what gets swept
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DenoiserSpec:
    """A denoiser and how its parameters are set as the noise changes.

    Attributes
    ----------
    method, params:
        As passed to :func:`denoiq_core.denoisers.denoise`.
    noise_scaled:
        Names of parameters expressed **in units of the noise standard deviation**. A
        denoiser whose strength is fixed in absolute terms is barely filtering at high dose
        and destroying the image at low dose, which would confound the dose axis with a
        strength axis; real denoisers are given the noise level, and so are these.
    label:
        Legend label, stable across conditions (the resolved parameters are not).

    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    noise_scaled: tuple[str, ...] = ()
    label: str | None = None

    def resolve(self, noise_sd: float) -> dict[str, Any]:
        """Concrete parameters for a given noise level."""
        out = dict(self.params)
        for name in self.noise_scaled:
            if name not in out:
                raise ValueError(f"{name!r} is marked noise-scaled but is not a parameter")
            out[name] = float(out[name]) * float(noise_sd)
        return out

    @property
    def name(self) -> str:
        """Stable label for figures and tables."""
        if self.label is not None:
            return self.label
        scaled = "".join(f", {k}={self.params[k]:g}s" for k in self.noise_scaled)
        fixed = ", ".join(
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(self.params.items())
            if k not in self.noise_scaled
        )
        inner = ", ".join(x for x in (fixed, scaled.lstrip(", ")) if x)
        return f"{self.method}({inner})" if inner else self.method

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record."""
        return {
            "method": self.method,
            "params": dict(self.params),
            "noise_scaled": list(self.noise_scaled),
            "label": self.name,
        }


#: The denoiser panel. Deliberately conventional, and deliberately not tuned: the ceiling
#: does not care how good a denoiser is, and a tuned panel would invite the reading that the
#: result is about these particular settings.
DEFAULT_DENOISERS: tuple[DenoiserSpec, ...] = (
    DenoiserSpec("none", {}, label="raw"),
    DenoiserSpec("gaussian", {"sigma": 1.5}, label="gaussian(1.5 px)"),
    DenoiserSpec(
        "tv", {"weight": 0.4, "max_num_iter": 200}, noise_scaled=("weight",), label="TV(0.4 sd)"
    ),
    DenoiserSpec(
        "nlm",
        {"h": 0.6, "patch_size": 5, "patch_distance": 6},
        noise_scaled=("h",),
        label="NLM(0.6 sd)",
    ),
)


@dataclass(frozen=True)
class SweepConfig:
    """Everything a sweep needs that is not the axis being swept."""

    n_trials: int = 1200
    seed: int = 20260101
    model: AcquisitionModel = DEFAULT_MODEL
    phantom: PhantomSpec = DEFAULT_PHANTOM
    eval_config: EvalConfig = DEFAULT_CONFIG
    criteria: RedLampCriteria = DEFAULT_CRITERIA
    denoisers: tuple[DenoiserSpec, ...] = DEFAULT_DENOISERS
    amplitude_fraction: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the whole configuration."""
        return {
            "n_trials": int(self.n_trials),
            "seed": int(self.seed),
            "model": self.model.to_dict(),
            "phantom": self.phantom.to_dict(),
            "eval_config": self.eval_config.to_dict(),
            "criteria": self.criteria.to_dict(),
            "denoisers": [d.to_dict() for d in self.denoisers],
            "amplitude_fraction": float(self.amplitude_fraction),
        }


DEFAULT_SWEEP = SweepConfig()

#: The ten independent realisations the primary analysis is run over. The first is the seed
#: the single-realisation study used, kept so that the earlier result stays inspectable; the
#: rest are generated deterministically from it by a fixed stride, so the list is a property of
#: the code rather than of anyone's session. The stride is a prime well above any per-condition
#: offset used inside a sweep, so no two realisations share a trial stream.
SEED_STRIDE = 1_000_003
PRIMARY_SEEDS: tuple[int, ...] = tuple(
    DEFAULT_SWEEP.seed + SEED_STRIDE * index for index in range(10)
)


# --------------------------------------------------------------------------------------
# writing results
# --------------------------------------------------------------------------------------


def results_dir() -> Path:
    """The ``results/`` directory of this checkout, created if missing."""
    path = Path(__file__).resolve().parent.parent / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Version stamp written into every results file.

    The interpreter version and the platform are deliberately **not** recorded: the files are
    compared across the CI matrix, and a field that differs between Python 3.10 and 3.12
    would make a determinism test fail for a reason that has nothing to do with the science.
    """
    return {
        "denoiq_core": denoiq_core.__version__,
        "taskiq_core": taskiq_core.__version__,
        **(extra or {}),
    }


def write_json(payload: dict[str, Any], name: str, *, directory: Path | None = None) -> Path:
    """Write a results file, with sorted keys and a trailing newline.

    Sorted keys and a fixed indent make the file diffable: a rerun that changes a number
    shows up as a one-line change rather than a reordering.
    """
    path = (directory or results_dir()) / name
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_jsonable)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__} into results/")


#: Columns that lead a CSV, in reading order; anything else follows alphabetically.
_CSV_LEADING = (
    "sweep",
    "kv",
    "mas",
    "relative_dose",
    "noise_sd",
    "correlation_sigma_mm",
    "contrast",
    "radius_mm",
    "denoiser",
    "method",
    "ceiling_d_prime",
    "ceiling_auc",
    "d_prime_ideal",
    "auc_ideal",
    "d_prime_cho",
    "auc_cho",
    "d_prime_npwe",
    "auc_npwe",
    "ssim",
    "psnr",
    "rmse",
    "excess",
    "margin",
    "bound_ok",
    "efficiency",
    "false_structure_rate",
    "contrast_recovery",
    "level",
)


def write_csv(rows: list[dict[str, Any]], name: str, *, directory: Path | None = None) -> Path:
    """Write flat rows as CSV, with a stable column order."""
    path = (directory or results_dir()) / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = set().union(*(set(r) for r in rows))
    ordered = [k for k in _CSV_LEADING if k in keys]
    ordered += sorted(k for k in keys if k not in _CSV_LEADING)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# --------------------------------------------------------------------------------------
# one condition
# --------------------------------------------------------------------------------------


def run_condition(
    trials: Any,
    spec: DenoiserSpec,
    *,
    config: SweepConfig = DEFAULT_SWEEP,
    condition: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Everything measurable about one denoiser on one experiment, as a flat row.

    Fidelity, three observers, the ceiling test, fabrication and erasure, and the red-lamp
    verdict — computed once, from one set of images, so that the entries of a row cannot
    disagree with each other.
    """
    params = spec.resolve(trials.noise_sd)
    present, absent = denoise_trials(trials, spec.method, params)

    record = evaluate_condition(
        trials,
        spec.method,
        params,
        config=config.eval_config,
        condition=condition,
        seed=seed,
        images=(present, absent),
    )
    ideal = record["observers"]["ideal"]
    bound = check_no_gain(
        trials,
        present,
        absent,
        config=config.eval_config,
        label=spec.name,
        estimate=ObserverEstimate(
            name="ideal",
            d_prime=ideal["d_prime"],
            auc=ideal["auc"],
            auc_se=ideal["auc_se"],
            n_present=ideal["n_present"],
            n_absent=ideal["n_absent"],
            estimator=ideal["estimator"],
        ),
    )

    false_processed = false_structure_rate(
        absent, trials.signal, amplitude_fraction=config.amplitude_fraction
    )
    recovery = contrast_recovery(present, absent, trials.signal)

    ceiling = ideal_ceiling(trials)
    raw_ideal = raw_estimator_check(trials, config=config.eval_config)
    false_raw = false_structure_rate(
        trials.absent, trials.signal, amplitude_fraction=config.amplitude_fraction
    )
    lamp = classify(
        ceiling_d_prime=ceiling.d_prime,
        task_d_prime=ideal["d_prime"],
        raw_task_d_prime=raw_ideal["estimated_d_prime"],
        ssim=record["fidelity"]["ssim"],
        false_rate=false_processed["rate"],
        raw_false_rate=false_raw["rate"],
        recovery=recovery,
        criteria=config.criteria,
        context=f"{spec.name}",
    )

    row: dict[str, Any] = dict(condition or {})
    row.update(
        {
            "denoiser": spec.name,
            "method": spec.method,
            "params": json.dumps(params, sort_keys=True),
            "noise_sd": float(trials.noise_sd),
            "pixel_sd": float(trials.meta["pixel_sd"]),
            "n_trials": int(trials.n_trials),
            "seed": int(trials.seed),
            "ssim": record["fidelity"]["ssim"],
            "psnr": record["fidelity"]["psnr"],
            "rmse": record["fidelity"]["rmse"],
            "ceiling_d_prime": bound.ceiling_d_prime,
            "ceiling_auc": bound.ceiling_auc,
            "raw_estimated_d_prime": raw_ideal["estimated_d_prime"],
            "raw_estimated_auc": raw_ideal["estimated_auc"],
            "estimator_d_prime_ratio": raw_ideal["d_prime_ratio"],
            "excess": bound.excess,
            "margin": bound.margin,
            "processed_auc_se": bound.processed_auc_se,
            "bound_ok": bound.ok,
            "efficiency": bound.efficiency,
            "saturated": bound.meta["saturated"],
            "false_structure_rate": false_processed["rate"],
            "false_structure_mean_amplitude": false_processed["mean_max_amplitude"],
            "raw_false_structure_rate": false_raw["rate"],
            "contrast_recovery": recovery,
            "level": lamp.level,
            "reason": lamp.reasons[0] if lamp.reasons else "",
        }
    )
    for observer in ("ideal", "cho", "npwe"):
        if observer in record["observers"]:
            row[f"d_prime_{observer}"] = record["observers"][observer]["d_prime"]
            row[f"auc_{observer}"] = record["observers"][observer]["auc"]
            row[f"auc_se_{observer}"] = record["observers"][observer]["auc_se"]
    return row


# --------------------------------------------------------------------------------------
# the sweeps
# --------------------------------------------------------------------------------------


def _sweep(
    conditions: list[tuple[dict[str, Any], Any]],
    config: SweepConfig,
    sweep_name: str,
) -> dict[str, Any]:
    """Run every denoiser on every prepared condition and package the rows."""
    rows: list[dict[str, Any]] = []
    for index, (condition, trials) in enumerate(conditions):
        for spec in config.denoisers:
            row = run_condition(
                trials,
                spec,
                config=config,
                condition={"sweep": sweep_name, **condition},
                seed=config.seed + index,
            )
            rows.append(row)
    return {
        "sweep": sweep_name,
        "provenance": provenance(),
        "config": config.to_dict(),
        "rows": rows,
    }


def sweep_dose(
    mas_values: np.ndarray | list[float] | None = None,
    *,
    kv: float = 120.0,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """Dose sweep at fixed kV — the backbone of the study.

    The default range spans the information floor: the top of it sits above the Rose
    criterion, the bottom well below, and the interesting behaviour is in between.
    """
    values = (
        np.asarray(mas_values, dtype=np.float64)
        if mas_values is not None
        else np.geomspace(200.0, 3.125, 7)
    )
    conditions = []
    for i, mas in enumerate(values):
        acq = acquisition_params(kv, mas, model=config.model)
        trials = make_trials(
            kv,
            mas,
            n_trials=config.n_trials,
            seed=config.seed + 1000 * i,
            model=config.model,
            phantom=config.phantom,
        )
        conditions.append(
            (
                {
                    "kv": float(kv),
                    "mas": float(mas),
                    "relative_dose": acq.relative_dose,
                    "contrast": acq.contrast,
                    "correlation_sigma_mm": float(config.phantom.correlation_sigma_mm),
                },
                trials,
            )
        )
    return _sweep(conditions, config, "dose")


def sweep_texture(
    correlation_values: np.ndarray | list[float] | None = None,
    *,
    kv: float = 120.0,
    mas: float = 25.0,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """Noise correlation-length sweep at fixed dose.

    Correlated noise separates the observers: the ideal one prewhitens it away, the
    non-prewhitening surrogate cannot. It is the condition under which a denoiser can raise
    the *human-surrogate* reading without raising — because it cannot raise — the ceiling.
    """
    values = (
        np.asarray(correlation_values, dtype=np.float64)
        if correlation_values is not None
        else np.array([0.0, 0.25, 0.5, 1.0])
    )
    conditions = []
    for i, corr in enumerate(values):
        phantom = replace(config.phantom, correlation_sigma_mm=float(corr))
        trials = make_trials(
            kv,
            mas,
            n_trials=config.n_trials,
            seed=config.seed + 2000 * (i + 1),
            model=config.model,
            phantom=phantom,
        )
        conditions.append(
            (
                {
                    "kv": float(kv),
                    "mas": float(mas),
                    "correlation_sigma_mm": float(corr),
                    "white_floor_sd": float(trials.white_floor_sd),
                },
                trials,
            )
        )
    return _sweep(conditions, config, "texture")


def sweep_signal(
    radii_mm: list[float] | None = None,
    contrast_scales: list[float] | None = None,
    *,
    kv: float = 120.0,
    mas: float = 25.0,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """Lesion size and contrast sweep — the same dose, an easier and a harder task."""
    radii = list(radii_mm) if radii_mm is not None else [1.5, 3.0, 5.0]
    scales = list(contrast_scales) if contrast_scales is not None else [0.5, 1.0]
    conditions = []
    for i, radius in enumerate(radii):
        for j, scale in enumerate(scales):
            model = replace(config.model, c_ref=config.model.c_ref * float(scale))
            phantom = replace(config.phantom, radius_mm=float(radius))
            trials = make_trials(
                kv,
                mas,
                n_trials=config.n_trials,
                seed=config.seed + 3000 * (i + 1) + 100 * j,
                model=model,
                phantom=phantom,
            )
            acq = acquisition_params(kv, mas, model=model)
            conditions.append(
                (
                    {
                        "kv": float(kv),
                        "mas": float(mas),
                        "radius_mm": float(radius),
                        "contrast": acq.contrast,
                        "contrast_scale": float(scale),
                    },
                    trials,
                )
            )
    return _sweep(conditions, config, "signal")


def sweep_kv_mas(
    kv_grid: np.ndarray | None = None,
    mas_grid: np.ndarray | None = None,
    *,
    config: SweepConfig = DEFAULT_SWEEP,
    demo_denoiser: DenoiserSpec | None = None,
    demo_points: list[tuple[float, float]] | None = None,
    n_trials_demo: int = 400,
) -> tuple[Atlas, dict[str, Any]]:
    """The red-lamp atlas, plus a Monte-Carlo demonstration at a few points on it.

    The atlas itself is analytic — the ideal ``d'`` of the unprocessed input at every
    ``(kV, mAs)``, and the exact contour where it crosses the prespecified criterion. The
    demonstration points then *run* the experiment at a handful of settings: above the floor a
    denoiser tracks the input's detectability, below it the picture improves while the task
    estimate falls further behind.
    """
    kv = np.asarray(kv_grid, dtype=np.float64) if kv_grid is not None else np.linspace(70, 140, 15)
    mas = (
        np.asarray(mas_grid, dtype=np.float64)
        if mas_grid is not None
        else np.geomspace(2.0, 300.0, 15)
    )
    atlas = information_floor(
        kv,
        mas,
        threshold=config.criteria.d_prime_threshold,
        model=config.model,
        phantom=config.phantom,
        criteria=config.criteria,
    )

    spec = demo_denoiser or DEFAULT_DENOISERS[1]
    points = (
        demo_points
        if demo_points is not None
        else [(120.0, 200.0), (120.0, 50.0), (120.0, 12.0), (80.0, 12.0)]
    )
    demo_config = replace(config, n_trials=n_trials_demo, denoisers=(DEFAULT_DENOISERS[0], spec))
    rows: list[dict[str, Any]] = []
    for i, (kv_value, mas_value) in enumerate(points):
        trials = make_trials(
            kv_value,
            mas_value,
            n_trials=n_trials_demo,
            seed=config.seed + 4000 * (i + 1),
            model=config.model,
            phantom=config.phantom,
        )
        acq = acquisition_params(kv_value, mas_value, model=config.model)
        for demo_spec in demo_config.denoisers:
            rows.append(
                run_condition(
                    trials,
                    demo_spec,
                    config=demo_config,
                    condition={
                        "sweep": "atlas_demo",
                        "kv": float(kv_value),
                        "mas": float(mas_value),
                        "relative_dose": acq.relative_dose,
                        "contrast": acq.contrast,
                    },
                    seed=config.seed + i,
                )
            )
    demo = {
        "sweep": "atlas_demo",
        "provenance": provenance(),
        "config": demo_config.to_dict(),
        "rows": rows,
    }
    return atlas, demo


# --------------------------------------------------------------------------------------
# the whole study
# --------------------------------------------------------------------------------------


def estimator_sensitivity(
    mas_values: Sequence[float] | None = None,
    *,
    kv: float = 120.0,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """Supplementary Table S2: cross-fitting against a single 50/50 split, on unprocessed images.

    The estimator's efficiency is what makes the ceiling comparison informative, so the choice
    between the two schemes is reported rather than asserted. Both are applied to the same
    images of the same conditions, and recovery is measured against the analytic ceiling of that
    input — which is exact, and identical for both.
    """
    from dataclasses import replace as _replace

    values = (
        list(mas_values)
        if mas_values is not None
        else [200.0, 100.0, 50.0, 25.0, 12.5, 6.25, 3.125]
    )
    split_config = _replace(config.eval_config, estimator="split")
    rows: list[dict[str, Any]] = []
    for index, mas in enumerate(values):
        trials = make_trials(
            kv,
            mas,
            n_trials=config.n_trials,
            seed=config.seed + 1000 * index,
            model=config.model,
            phantom=config.phantom,
        )
        ceiling = ideal_ceiling(trials).d_prime
        cross = raw_estimator_check(trials, config=config.eval_config)
        split = raw_estimator_check(trials, config=split_config)
        rows.append(
            {
                "kv": float(kv),
                "mas": float(mas),
                "ceiling_d_prime": ceiling,
                "cross_fit_d_prime": cross["estimated_d_prime"],
                "cross_fit_recovery": cross["d_prime_ratio"],
                "split_d_prime": split["estimated_d_prime"],
                "split_recovery": split["d_prime_ratio"],
            }
        )
    return {
        "table": "estimator_sensitivity",
        "provenance": provenance(),
        "seed": int(config.seed),
        "n_folds": int(config.eval_config.n_folds),
        "train_fraction": float(config.eval_config.train_fraction),
        "rows": rows,
    }


def run_primary(
    *,
    seeds: Sequence[int] = PRIMARY_SEEDS,
    config: SweepConfig = DEFAULT_SWEEP,
    directory: Path | None = None,
    workers: int | None = None,
) -> dict[str, Path]:
    """The primary analysis: the experimental matrix over independent realisations.

    Writes three files — the per-arm rows of every realisation, the derived endpoints, and the
    statistical summary — and nothing in the manuscript comes from anywhere else.
    """
    from denoiq_core.statistics import compute_endpoints

    multiseed = run_multiseed(seeds, config=config, workers=workers)
    written = {
        "multiseed": write_json(multiseed, "multiseed.json", directory=directory),
        "multiseed_csv": write_csv(multiseed["rows"], "multiseed.csv", directory=directory),
    }

    endpoints = compute_endpoints(
        multiseed["rows"],
        threshold=config.criteria.d_prime_threshold,
        amber_factor=config.criteria.amber_factor,
    )
    written["endpoints"] = write_json(
        {
            "provenance": provenance(),
            "criteria": config.criteria.to_dict(),
            "definitions": {
                "delta_ssim": "SSIM(processed) - SSIM(unprocessed input), same condition and seed",
                "delta_d_pw": "held-out prewhitening d'(processed) - d'(input)",
                "benefit": "delta_d_npwe - delta_d_pw",
                "divergent": "delta_ssim > 0 and delta_d_pw < 0",
                "stratum": "floor stratum of the input condition's analytic ceiling",
            },
            "rows": endpoints,
        },
        "endpoints.json",
        directory=directory,
    )
    written["endpoints_csv"] = write_csv(endpoints, "endpoints.csv", directory=directory)

    sensitivity = estimator_sensitivity(config=config)
    written["estimator_sensitivity"] = write_json(
        sensitivity, "estimator_sensitivity.json", directory=directory
    )

    statistics = summarise_rows(multiseed["rows"], seeds=list(seeds), config=config)
    written["statistics"] = write_json(statistics, "statistics.json", directory=directory)
    return written


def summarise_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int],
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    """The statistical summary of a set of per-arm rows.

    Every quantity in ``results/statistics.json`` is a function of the rows already stored in
    ``results/multiseed.json``, so the summary can be rebuilt — after a change to an endpoint
    definition or to the ceiling accounting — without re-running the simulations that produced
    them. :func:`run_primary` and ``--reaggregate`` share this function so the two can never
    diverge.
    """
    from denoiq_core.statistics import aggregate_endpoints, compute_endpoints

    endpoints = compute_endpoints(
        rows,
        threshold=config.criteria.d_prime_threshold,
        amber_factor=config.criteria.amber_factor,
    )
    statistics = aggregate_endpoints(endpoints)
    statistics["provenance"] = provenance()
    statistics["design"] = design_matrix(rows, seeds=list(seeds))
    statistics["ceiling"] = ceiling_summary(rows)
    return statistics


def design_matrix(rows: Sequence[dict[str, Any]], *, seeds: Sequence[int]) -> dict[str, Any]:
    """The experimental matrix, counted from the rows rather than described in prose."""
    per_seed = [row for row in rows if row["realisation_seed"] == seeds[0]]
    sweeps: dict[str, Any] = {}
    for name in sorted({row["sweep"] for row in per_seed}):
        subset = [row for row in per_seed if row["sweep"] == name]
        processed = [row for row in subset if row["method"] != "none"]
        unprocessed = [row for row in subset if row["method"] == "none"]
        sweeps[name] = {
            "input_conditions": len(unprocessed),
            "denoisers": sorted({row["denoiser"] for row in processed}),
            "n_denoisers": len({row["denoiser"] for row in processed}),
            "unprocessed_arms": len(unprocessed),
            "processed_arms": len(processed),
            "arms": len(subset),
            # Not uniform across the matrix: the atlas demonstration settings use fewer trials
            # than the three sweeps, so the trial total has to be summed rather than assumed.
            "trials_per_class_per_arm": sorted({int(row["n_trials"]) for row in subset}),
        }
    processed = [row for row in per_seed if row["method"] != "none"]
    return {
        "sweeps": sweeps,
        "input_conditions": len(per_seed) - len(processed),
        "unique_arms": len(per_seed),
        "unprocessed_arms": len(per_seed) - len(processed),
        "processed_arms": len(processed),
        "n_seeds": len(seeds),
        "arm_seed_evaluations": len(rows),
        # The matrix is not uniform in trial count — the atlas demonstration settings use fewer
        # — so both are reported: the range across arms, and the exact total.
        "trials_per_class_per_arm": sorted({int(row["n_trials"]) for row in per_seed}),
        "trials_per_class_main_sweeps": max(int(row["n_trials"]) for row in per_seed),
        # Each arm scores both classes of its own trial set.
        "scored_image_trials": int(2 * sum(int(row["n_trials"]) for row in rows)),
    }


def ceiling_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Ceiling comparison over all arm-seed evaluations, with the saturated subset separated.

    The armwise margin is a per-comparison bound, so over many evaluations a small number of
    exceedances is expected from sampling alone. Two things are therefore recorded beside the
    count: a Bonferroni-widened margin for a family-wise statement, and — for every exceedance —
    whether it occurred on a *processed* arm at all, since an exceedance on the unprocessed arm
    cannot represent processing creating information.
    """
    from scipy.stats import norm

    excesses = [row["excess"] for row in rows]
    violations = [row for row in rows if not row["bound_ok"]]
    unsaturated = [row for row in rows if not row["saturated"]]

    # Family-wise: widen z from 1.96 to the Bonferroni quantile for this many comparisons.
    z_family = float(norm.ppf(1.0 - 0.05 / (2 * max(len(rows), 1))))

    def _standard_error(row: dict[str, Any]) -> float:
        """The AUC standard error behind this row's margin.

        Recorded directly by newer runs; for rows written before that field existed it is
        recovered exactly from the margin, which is ``z*SE`` plus one quantisation step.
        """
        if "processed_auc_se" in row:
            return float(row["processed_auc_se"])
        quantum = 1.0 / (int(row["n_trials"]) ** 2)
        return max(0.0, (float(row["margin"]) - quantum) / 1.959963984540054)

    family_exceedances = [
        row
        for row in rows
        if row["excess"] > z_family * _standard_error(row) + 1.0 / (int(row["n_trials"]) ** 2)
    ]
    return {
        "violation_detail": [
            {
                "sweep": row["sweep"],
                "denoiser": row["denoiser"],
                "method": row["method"],
                "processed": row["method"] != "none",
                "seed": row.get("realisation_seed"),
                "excess": row["excess"],
                "margin": row["margin"],
                "saturated": row["saturated"],
                "ceiling_auc": row["ceiling_auc"],
                "auc": row["auc_ideal"],
            }
            for row in violations
        ],
        "n_violations_processed_arms": sum(1 for row in violations if row["method"] != "none"),
        "family_wise": {
            "z": z_family,
            "n_comparisons": len(rows),
            "n_processed_comparisons": sum(1 for row in rows if row["method"] != "none"),
            "n_unprocessed_comparisons": sum(1 for row in rows if row["method"] == "none"),
            "n_exceedances": len(family_exceedances),
            # The claim the paper makes is about processing, so the count that carries it is
            # the one restricted to processed arms. An unprocessed arm compared against its own
            # analytic ceiling is a self-comparison: it measures the estimator, not processing.
            "n_exceedances_processed_arms": sum(
                1 for row in family_exceedances if row["method"] != "none"
            ),
            "n_exceedances_unprocessed_arms": sum(
                1 for row in family_exceedances if row["method"] == "none"
            ),
            "detail": [
                {
                    "sweep": row["sweep"],
                    "denoiser": row["denoiser"],
                    "method": row["method"],
                    "processed": row["method"] != "none",
                    "seed": row.get("realisation_seed"),
                    "excess": row["excess"],
                    "margin": row["margin"],
                    "saturated": row["saturated"],
                    "ceiling_auc": row["ceiling_auc"],
                    "auc": row["auc_ideal"],
                    "n_trials": row["n_trials"],
                }
                for row in family_exceedances
            ],
            "rule": "Bonferroni-widened armwise margin at family-wise alpha 0.05",
        },
        "n_evaluations": len(rows),
        "n_violations": len(violations),
        "max_excess": float(max(excesses)),
        "mean_excess": float(np.mean(excesses)),
        "margin_at_max_excess": float(max(rows, key=lambda r: r["excess"])["margin"]),
        "n_saturated": len(rows) - len(unsaturated),
        "saturation_rule": "analytic ceiling AUC > 1 - 10/(n_present*n_absent)",
        "unsaturated": {
            "n_evaluations": len(unsaturated),
            "max_excess": float(max(row["excess"] for row in unsaturated)),
            "mean_excess": float(np.mean([row["excess"] for row in unsaturated])),
            "n_violations": sum(1 for row in unsaturated if not row["bound_ok"]),
        },
        "estimator_recovery": {
            "min": float(min(row["estimator_d_prime_ratio"] for row in rows)),
            "median": float(np.median([row["estimator_d_prime_ratio"] for row in rows])),
            "q1": float(np.percentile([row["estimator_d_prime_ratio"] for row in rows], 25)),
            "q3": float(np.percentile([row["estimator_d_prime_ratio"] for row in rows], 75)),
            "above_floor_min": float(
                min(
                    (
                        row["estimator_d_prime_ratio"]
                        for row in rows
                        if row["ceiling_d_prime"] >= 5.0
                    ),
                    default=float("nan"),
                )
            ),
        },
    }


def run_all(
    *,
    config: SweepConfig = DEFAULT_SWEEP,
    directory: Path | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    """Run every sweep and write ``results/``.

    Parameters
    ----------
    config:
        The sweep configuration.
    directory:
        Where to write; defaults to ``results/`` beside the package.
    quick:
        A small, fast version of every sweep — for smoke tests and for CI, not for the
        manuscript. Files written in quick mode say so in their provenance, so a figure
        built from them cannot be mistaken for the real thing.

    Returns
    -------
    dict
        Name to path for every file written.

    """
    if quick:
        config = replace(config, n_trials=200)

    written: dict[str, Path] = {}
    tag = {"quick": bool(quick), "available_denoisers": list(available_methods())}

    dose = sweep_dose(
        np.geomspace(200.0, 6.25, 4) if quick else None,
        config=config,
    )
    dose["provenance"].update(tag)
    written["dose_sweep_json"] = write_json(dose, "dose_sweep.json", directory=directory)
    written["dose_sweep_csv"] = write_csv(dose["rows"], "dose_sweep.csv", directory=directory)

    texture = sweep_texture(
        [0.0, 0.5] if quick else None,
        config=config,
    )
    texture["provenance"].update(tag)
    written["texture_sweep_json"] = write_json(texture, "texture_sweep.json", directory=directory)
    written["texture_sweep_csv"] = write_csv(
        texture["rows"], "texture_sweep.csv", directory=directory
    )

    signal = sweep_signal(
        [1.5, 3.0] if quick else None,
        [1.0] if quick else None,
        config=config,
    )
    signal["provenance"].update(tag)
    written["signal_sweep_json"] = write_json(signal, "signal_sweep.json", directory=directory)
    written["signal_sweep_csv"] = write_csv(signal["rows"], "signal_sweep.csv", directory=directory)

    atlas, demo = sweep_kv_mas(
        np.linspace(80, 140, 4) if quick else None,
        np.geomspace(4.0, 300.0, 5) if quick else None,
        config=config,
        n_trials_demo=200 if quick else 400,
        demo_points=[(120.0, 200.0), (120.0, 12.0)] if quick else None,
    )
    atlas_payload = {"provenance": provenance(tag), "atlas": atlas.to_dict()}
    demo["provenance"].update(tag)
    written["atlas_json"] = write_json(atlas_payload, "redlamp_atlas.json", directory=directory)
    written["atlas_demo_json"] = write_json(demo, "redlamp_demo.json", directory=directory)
    written["atlas_demo_csv"] = write_csv(demo["rows"], "redlamp_demo.csv", directory=directory)

    closed_form = closed_form_validation(
        [30.0, 120.0] if quick else None,
        [3.0] if quick else None,
        config=config,
    )
    closed_form["provenance"].update(tag)
    written["closed_form_json"] = write_json(closed_form, "closed_form.json", directory=directory)
    written["closed_form_csv"] = write_csv(
        closed_form["rows"], "closed_form.csv", directory=directory
    )

    gains = task_gain_table(dose)
    gains["provenance"].update(tag)
    written["task_gains_json"] = write_json(gains, "task_gains.json", directory=directory)
    written["task_gains_csv"] = write_csv(gains["rows"], "task_gains.csv", directory=directory)

    summary = summarize([dose, texture, signal, demo])
    summary["provenance"] = provenance(tag)
    summary["max_closed_form_relative_error"] = closed_form["max_relative_error"]
    written["summary_json"] = write_json(summary, "summary.json", directory=directory)
    return written


def summarize(sweeps: list[dict[str, Any]]) -> dict[str, Any]:
    """The headline numbers, in one small file the manuscript can quote directly.

    Chiefly: did the ceiling hold everywhere, by how much was it approached, and how large
    does the fidelity-task divergence get.
    """
    rows = [row for sweep in sweeps for row in sweep["rows"]]
    excesses = [row["excess"] for row in rows]
    violations = [row for row in rows if not row["bound_ok"]]
    processed = [row for row in rows if row["method"] != "none"]

    npwe_gains = []
    for row in processed:
        raw = _matching_raw(rows, row)
        if raw is not None:
            npwe_gains.append(row["d_prime_npwe"] - raw["d_prime_npwe"])
    best_gain_npwe = max(npwe_gains, default=float("nan"))
    processed_excesses = [row["excess"] for row in processed]
    worst = max(rows, key=lambda row: row["excess"]) if rows else None
    return {
        # `n_conditions` counts *evaluated arms*: every (condition, denoiser) pair, including
        # the unprocessed arm of each condition. The breakdown is spelled out so the manuscript
        # can state it without a reader having to re-derive it.
        "n_conditions": len(rows),
        "n_processed_conditions": len(processed),
        "n_unprocessed_conditions": len(rows) - len(processed),
        "arms_per_sweep": {
            sweep["sweep"]: len(sweep["rows"]) for sweep in sweeps if sweep.get("rows")
        },
        "denoisers_in_primary_analysis": sorted({row["denoiser"] for row in processed}),
        # The optional CNN is evaluated separately (examples/run_denoise_taskbench.py --cnn and
        # paper/redlamp_console.py) and is deliberately not part of this count, so that the
        # primary analysis reproduces without torch.
        "cnn_in_primary_analysis": any(row["method"] == "cnn" for row in rows),
        "n_bound_violations": len(violations),
        "bound_holds_everywhere": not violations,
        "max_excess_over_ceiling": float(max(excesses)) if excesses else float("nan"),
        "max_excess_over_ceiling_processed_only": float(max(processed_excesses))
        if processed_excesses
        else float("nan"),
        "mean_excess_over_ceiling": float(np.mean(excesses)) if excesses else float("nan"),
        "mean_excess_over_ceiling_processed_only": float(np.mean(processed_excesses))
        if processed_excesses
        else float("nan"),
        # The largest excess and the margin it was judged against, so "within the sampling
        # margin" can be checked rather than taken on trust.
        "margin_at_max_excess": float(worst["margin"]) if worst else float("nan"),
        "max_excess_is_auc_saturated": bool(worst["saturated"]) if worst else False,
        "n_auc_saturated_arms": sum(1 for row in rows if row["saturated"]),
        "max_npwe_gain_from_denoising": float(best_gain_npwe),
        "min_estimator_d_prime_ratio": float(min(row["estimator_d_prime_ratio"] for row in rows))
        if rows
        else float("nan"),
        "levels": {
            level: sum(1 for row in rows if row["level"] == level)
            for level in ("green", "amber", "red")
        },
        "violations": [row["reason"] for row in violations],
    }


def run_seed(seed: int, *, config: SweepConfig = DEFAULT_SWEEP) -> list[dict[str, Any]]:
    """Every arm of the primary experimental matrix, for one independent realisation.

    The four sweeps share their design; only the seed of the trial stream changes, so the arms
    of two realisations correspond one-to-one and can be paired.
    """
    seeded = replace(config, seed=int(seed))
    rows: list[dict[str, Any]] = []
    for sweep in (
        sweep_dose(config=seeded),
        sweep_texture(config=seeded),
        sweep_signal(config=seeded),
    ):
        rows.extend(sweep["rows"])
    _, demo = sweep_kv_mas(config=seeded)
    rows.extend(demo["rows"])
    for row in rows:
        row["realisation_seed"] = int(seed)
    return rows


def _run_seed_worker(seed: int) -> list[dict[str, Any]]:
    """Module-level entry point for the process pool (must be importable to be picklable)."""
    return run_seed(seed)


def run_multiseed(
    seeds: Sequence[int] = PRIMARY_SEEDS,
    *,
    config: SweepConfig = DEFAULT_SWEEP,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the primary matrix over independent realisations, in parallel across seeds.

    Parallelism is over realisations rather than over arms: each worker owns one whole seed,
    which keeps every trial stream inside a single process and makes the result independent of
    how many workers happen to be available.
    """
    seeds = [int(s) for s in seeds]
    rows: list[dict[str, Any]] = []
    if workers == 1 or len(seeds) == 1:
        for seed in seeds:
            rows.extend(run_seed(seed, config=config))
    else:
        import concurrent.futures as futures

        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(_run_seed_worker, seeds):
                rows.extend(chunk)
    rows.sort(key=lambda r: (r["realisation_seed"], r["sweep"], str(r.get("mas")), r["denoiser"]))
    return {
        "sweep": "multiseed",
        "provenance": provenance(),
        "seeds": seeds,
        "seed_stride": SEED_STRIDE,
        "config": config.to_dict(),
        "rows": rows,
    }


def closed_form_validation(
    noise_sds: list[float] | None = None,
    radii_mm: list[float] | None = None,
    *,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Any]:
    r"""Table 1: the ideal observer against its closed form, in white noise.

    For a signal-known-exactly task in white Gaussian noise the prewhitening observer has an
    exact answer, :math:`d' = \|s\|_2/\sigma`, with no free parameters and nothing estimated.
    Checking the machinery against it is the difference between a pipeline that has been
    validated and one that has merely been run.

    The comparison uses **analytic** quantities on both sides, so the residual is numerical,
    not statistical: it should be at the level of floating-point summation, far below the 1 %
    the acceptance criterion asks for.
    """
    sigmas = list(noise_sds) if noise_sds is not None else [15.0, 30.0, 60.0, 120.0]
    radii = list(radii_mm) if radii_mm is not None else [1.5, 3.0, 5.0]
    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        for radius in radii:
            model = replace(config.model, sigma_ref=float(sigma))
            phantom = replace(config.phantom, radius_mm=float(radius), correlation_sigma_mm=0.0)
            trials = make_trials(
                config.model.kv_ref,
                config.model.mas_ref,
                n_trials=2,
                seed=config.seed,
                model=model,
                phantom=phantom,
            )
            analytic = float(np.sqrt(np.sum(trials.signal**2)) / trials.noise_sd)
            observed = float(ideal_ceiling(trials).d_prime)
            rows.append(
                {
                    "noise_sd": float(sigma),
                    "radius_mm": float(radius),
                    "contrast": float(np.max(np.abs(trials.signal))),
                    "d_prime_closed_form": analytic,
                    "d_prime_ideal_linear": observed,
                    "relative_error": float(abs(observed - analytic) / analytic),
                }
            )
    return {
        "table": "closed_form",
        "provenance": provenance(),
        "identity": "d' = ||s||_2 / sigma  (SKE, white Gaussian noise)",
        "max_relative_error": float(max(row["relative_error"] for row in rows)),
        "rows": rows,
    }


def task_gain_table(dose: dict[str, Any]) -> dict[str, Any]:
    """Table 2: what each denoiser did to each observer, signed.

    The table the argument turns on. A denoiser can raise the reading of a *sub-optimal*
    observer — the non-prewhitening one especially — because that observer was
    not using all the information to begin with. It cannot raise the ideal observer's
    reading, because that one already was. Positive numbers in the NPWE column next to
    non-positive numbers in the ideal column are the whole story in one table.
    """
    rows = dose["rows"]
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == "none":
            continue
        raw = _matching_raw(rows, row)
        if raw is None:  # pragma: no cover - defensive
            continue
        entry: dict[str, Any] = {
            "sweep": row.get("sweep", ""),
            "kv": row.get("kv"),
            "mas": row.get("mas"),
            "relative_dose": row.get("relative_dose"),
            "denoiser": row["denoiser"],
            "ceiling_d_prime": row["ceiling_d_prime"],
            "delta_ssim": row["ssim"] - raw["ssim"],
        }
        for observer in ("ideal", "cho", "npwe"):
            key = f"d_prime_{observer}"
            if key in row and key in raw:
                entry[f"delta_d_prime_{observer}"] = row[key] - raw[key]
        out.append(entry)

    by_denoiser: dict[str, dict[str, float]] = {}
    for denoiser in sorted({row["denoiser"] for row in out}):
        subset = [row for row in out if row["denoiser"] == denoiser]
        by_denoiser[denoiser] = {
            f"mean_delta_d_prime_{observer}": float(
                np.mean([row[f"delta_d_prime_{observer}"] for row in subset])
            )
            for observer in ("ideal", "cho", "npwe")
            if all(f"delta_d_prime_{observer}" in row for row in subset)
        }
        by_denoiser[denoiser]["mean_delta_ssim"] = float(
            np.mean([row["delta_ssim"] for row in subset])
        )
    return {
        "table": "task_gains",
        "provenance": provenance(),
        "rows": out,
        "by_denoiser": by_denoiser,
    }


def _matching_raw(rows: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    """The raw row of the same condition as ``row``, if it is in the same sweep."""
    keys = ("sweep", "kv", "mas", "correlation_sigma_mm", "radius_mm", "contrast_scale")
    for candidate in rows:
        if candidate["method"] != "none":
            continue
        if all(candidate.get(k) == row.get(k) for k in keys):
            return candidate
    return None
