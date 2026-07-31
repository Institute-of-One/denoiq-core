r"""Primary endpoints, effect sizes and their uncertainty.

The study's questions are comparative, so the numbers that answer them are differences, and a
difference without an interval is an anecdote. This module turns the per-arm measurements into
the three prespecified endpoints and prices each one's uncertainty.

The endpoints
-------------
For every **processed arm** — one denoiser at one input condition in one realisation — with
the unprocessed arm of that same condition and realisation as its reference:

``ΔSSIM``
    ``SSIM(processed) − SSIM(input)``.
``Δd'`` per observer
    the same difference for the held-out prewhitening (PW), channelised Hotelling (CHO) and
    non-prewhitening (NPWE) estimates.
**divergent arm**
    ``ΔSSIM > 0`` **and** ``Δd'_PW < 0``: fidelity improved while the task estimate fell.
**observer-dependent benefit**
    ``B = Δd'_NPWE − Δd'_PW``, the extent to which processing helped the inefficient observer
    more than the efficient one. Positive ``B`` with ``Δd'_PW ≤ 0`` is redistribution rather
    than creation of information.

Why the resampling is clustered
-------------------------------
Arms are **not** independent observations. All arms of one realisation share its noise
realisation, and the arms of one condition share its images. The bootstrap therefore resamples
**whole realisations** (the outer cluster), carrying every arm of a drawn realisation with it.
Pooling arm-level values as if they were independent would give intervals that are too narrow
by roughly the square root of the number of arms per realisation, which is the kind of error
that turns a null result into a finding.

Two-sided bootstrap p-values are reported next to the intervals, Holm-adjusted within each
family of comparisons, but the intervals and effect sizes are what the conclusions rest on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

#: A summary computed over a set of endpoint records — the thing the cluster bootstrap
#: resamples. Every statistic in this module has this shape, including the closures built by
#: :func:`_mean` and :func:`_rate`.
Statistic: TypeAlias = Callable[[Sequence[dict[str, Any]]], float]

__all__ = [
    "STRATA",
    "aggregate_endpoints",
    "cluster_bootstrap",
    "compute_endpoints",
    "holm",
    "spearman",
    "wilson_interval",
]

#: Floor strata, by the analytic ceiling ``d'`` of the *input* condition. The boundaries are
#: the gauge's own prespecified criteria; nothing here is chosen after seeing the results.
STRATA = ("above floor", "marginal", "below floor")

#: Bootstrap replicates. Enough that the reported interval endpoints are stable to the third
#: decimal, and seeded, so the interval is reproducible.
N_BOOT = 4000
BOOT_SEED = 20260731


# --------------------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------------------


def _condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of an input condition within a realisation, independent of the denoiser."""
    return (
        row["realisation_seed"],
        row.get("sweep"),
        row.get("kv"),
        row.get("mas"),
        row.get("correlation_sigma_mm"),
        row.get("radius_mm"),
        row.get("contrast_scale"),
        row.get("contrast"),
    )


def stratum_of(ceiling_d_prime: float, threshold: float, amber_factor: float) -> str:
    """Which floor stratum an input condition falls in."""
    if ceiling_d_prime < threshold:
        return "below floor"
    if ceiling_d_prime < amber_factor * threshold:
        return "marginal"
    return "above floor"


def compute_endpoints(
    rows: Sequence[dict[str, Any]], *, threshold: float, amber_factor: float
) -> list[dict[str, Any]]:
    """One record per processed arm, paired with the unprocessed arm of its own condition.

    Raises
    ------
    ValueError
        If a processed arm has no unprocessed reference in the same realisation and condition.
        Silently dropping such an arm would bias every endpoint towards whichever conditions
        happened to be complete.

    """
    inputs = {_condition_key(row): row for row in rows if row["method"] == "none"}
    endpoints: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == "none":
            continue
        key = _condition_key(row)
        reference = inputs.get(key)
        if reference is None:
            raise ValueError(f"no unprocessed reference for {row['denoiser']} at {key}")
        delta = {
            observer: row[f"d_prime_{observer}"] - reference[f"d_prime_{observer}"]
            for observer in ("ideal", "cho", "npwe")
        }
        delta_ssim = row["ssim"] - reference["ssim"]
        endpoints.append(
            {
                "seed": row["realisation_seed"],
                "sweep": row["sweep"],
                "denoiser": row["denoiser"],
                "kv": row.get("kv"),
                "mas": row.get("mas"),
                "correlation_sigma_mm": row.get("correlation_sigma_mm"),
                "radius_mm": row.get("radius_mm"),
                "ceiling_d_prime": row["ceiling_d_prime"],
                "stratum": stratum_of(row["ceiling_d_prime"], threshold, amber_factor),
                "delta_ssim": delta_ssim,
                "delta_d_pw": delta["ideal"],
                "delta_d_cho": delta["cho"],
                "delta_d_npwe": delta["npwe"],
                "benefit": delta["npwe"] - delta["ideal"],
                "divergent": bool(delta_ssim > 0.0 and delta["ideal"] < 0.0),
                "input_d_pw": reference["d_prime_ideal"],
                "processed_d_pw": row["d_prime_ideal"],
                "task_degradation": (
                    1.0 - row["d_prime_ideal"] / reference["d_prime_ideal"]
                    if reference["d_prime_ideal"] > 0
                    else float("nan")
                ),
                "contrast_recovery": row["contrast_recovery"],
                "erasure": bool(row["contrast_recovery"] < 0.5),
                "excess_response": bool(
                    row["false_structure_rate"] > 0.2
                    and row["false_structure_rate"] > row["raw_false_structure_rate"]
                ),
                "level": row["level"],
                "excess": row["excess"],
                "margin": row["margin"],
                "bound_ok": row["bound_ok"],
                "saturated": row["saturated"],
                "estimator_d_prime_ratio": row["estimator_d_prime_ratio"],
            }
        )
    return endpoints


# --------------------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval and a two-sided bootstrap p-value."""

    value: float
    low: float
    high: float
    p_value: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record."""
        return {
            "value": float(self.value),
            "ci_low": float(self.low),
            "ci_high": float(self.high),
            "p_value": float(self.p_value),
            "n": int(self.n),
        }


def cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    statistic: Statistic,
    *,
    cluster: str = "seed",
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    null: float = 0.0,
) -> Estimate:
    """Bootstrap a statistic by resampling whole clusters (realisations) with replacement.

    ``statistic`` maps a list of records to a float. Clusters are drawn with replacement and
    all of their records are carried along, which preserves the dependence between arms that
    share a realisation.

    The p-value is the standard two-sided bootstrap value: twice the smaller tail mass on the
    far side of ``null``, floored at ``1/(n_boot + 1)`` because a bootstrap cannot resolve a
    smaller one.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record[cluster], []).append(record)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(f"need at least two {cluster} clusters to bootstrap, got {len(keys)}")

    point = float(statistic(list(records)))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        picked = rng.integers(0, len(keys), len(keys))
        sample = [record for index in picked for record in groups[keys[index]]]
        draws[i] = statistic(sample)

    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5])
    below = float(np.mean(finite <= null))
    above = float(np.mean(finite >= null))
    p = min(1.0, 2.0 * min(below, above))
    return Estimate(point, float(low), float(high), max(p, 1.0 / (len(finite) + 1)), len(records))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation, with ties given average ranks."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
    return float(np.sum(ra * rb) / denom) if denom > 0 else float("nan")


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a proportion — reported alongside the clustered one.

    It ignores the clustering, so it is the *narrower* of the two and is given only as the
    conventional reference; the clustered bootstrap interval is the one to read.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment within a family of comparisons."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * p))
        adjusted[key] = running
    return adjusted


# --------------------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------------------


def _mean(field: str) -> Statistic:
    def statistic(records: Sequence[dict[str, Any]]) -> float:
        values = [record[field] for record in records]
        return float(np.mean(values)) if values else float("nan")

    return statistic


def _rate(field: str) -> Statistic:
    def statistic(records: Sequence[dict[str, Any]]) -> float:
        values = [bool(record[field]) for record in records]
        return float(np.mean(values)) if values else float("nan")

    return statistic


def _spearman_stat(records: Sequence[dict[str, Any]]) -> float:
    return spearman(
        [record["delta_ssim"] for record in records],
        [record["delta_d_pw"] for record in records],
    )


def aggregate_endpoints(
    endpoints: Sequence[dict[str, Any]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Every prespecified endpoint, with clustered intervals and Holm-adjusted p-values."""
    denoisers = sorted({record["denoiser"] for record in endpoints})
    seeds = sorted({record["seed"] for record in endpoints})

    def boot(
        records: Sequence[dict[str, Any]], statistic: Statistic, null: float = 0.0
    ) -> dict[str, Any]:
        return cluster_bootstrap(records, statistic, n_boot=n_boot, seed=seed, null=null).to_dict()

    # --- H1: fidelity-task divergence ------------------------------------------------
    divergence: dict[str, Any] = {
        "spearman_delta_ssim_delta_d_pw": boot(endpoints, _spearman_stat),
        "divergence_rate": boot(endpoints, _rate("divergent")),
        "mean_delta_ssim": boot(endpoints, _mean("delta_ssim")),
        "mean_delta_d_pw": boot(endpoints, _mean("delta_d_pw")),
        "wilson_divergence_rate": wilson_interval(
            sum(1 for record in endpoints if record["divergent"]), len(endpoints)
        ),
        "by_denoiser": {},
        "by_stratum": {},
    }
    p_divergence: dict[str, float] = {}
    for name in denoisers:
        subset = [record for record in endpoints if record["denoiser"] == name]
        entry = {
            "spearman": boot(subset, _spearman_stat),
            "divergence_rate": boot(subset, _rate("divergent")),
            "mean_delta_ssim": boot(subset, _mean("delta_ssim")),
            "mean_delta_d_pw": boot(subset, _mean("delta_d_pw")),
        }
        divergence["by_denoiser"][name] = entry
        p_divergence[name] = entry["spearman"]["p_value"]
    divergence["spearman_p_holm"] = holm(p_divergence)

    for stratum in STRATA:
        subset = [record for record in endpoints if record["stratum"] == stratum]
        if len(subset) < 2:
            continue
        divergence["by_stratum"][stratum] = {
            "n": len(subset),
            "divergence_rate": boot(subset, _rate("divergent")),
            "mean_delta_ssim": boot(subset, _mean("delta_ssim")),
            "mean_delta_d_pw": boot(subset, _mean("delta_d_pw")),
        }

    # --- H2: observer-dependent benefit ----------------------------------------------
    observers = {
        "benefit": _mean("benefit"),
        "delta_d_pw": _mean("delta_d_pw"),
        "delta_d_cho": _mean("delta_d_cho"),
        "delta_d_npwe": _mean("delta_d_npwe"),
    }
    observer_block: dict[str, Any] = {
        "overall": {name: boot(endpoints, statistic) for name, statistic in observers.items()},
        "npwe_improved_pw_did_not": boot(
            endpoints,
            lambda records: float(
                np.mean(
                    [record["delta_d_npwe"] > 0.0 >= record["delta_d_pw"] for record in records]
                )
            ),
        ),
        "by_denoiser": {},
        "by_correlation": {},
    }
    p_benefit: dict[str, float] = {}
    for name in denoisers:
        subset = [record for record in endpoints if record["denoiser"] == name]
        entry = {key: boot(subset, statistic) for key, statistic in observers.items()}
        observer_block["by_denoiser"][name] = entry
        p_benefit[name] = entry["benefit"]["p_value"]
    observer_block["benefit_p_holm"] = holm(p_benefit)

    for correlation in sorted(
        {
            record["correlation_sigma_mm"]
            for record in endpoints
            if record["correlation_sigma_mm"] is not None
        }
    ):
        subset = [record for record in endpoints if record["correlation_sigma_mm"] == correlation]
        if len(subset) < 2:
            continue
        observer_block["by_correlation"][f"{correlation:g}"] = {
            "n": len(subset),
            "benefit": boot(subset, _mean("benefit")),
            "delta_d_pw": boot(subset, _mean("delta_d_pw")),
            "delta_d_npwe": boot(subset, _mean("delta_d_npwe")),
        }

    # --- H3: floor-stratified failure patterns ----------------------------------------
    strata: dict[str, Any] = {}
    for stratum in STRATA:
        subset = [record for record in endpoints if record["stratum"] == stratum]
        if not subset:
            continue
        strata[stratum] = {
            "n_arm_seed": len(subset),
            "n_arms": len(
                {
                    (
                        r["sweep"],
                        r["kv"],
                        r["mas"],
                        r["correlation_sigma_mm"],
                        r["radius_mm"],
                        r["denoiser"],
                    )
                    for r in subset
                }
            ),
            "divergence_rate": boot(subset, _rate("divergent")),
            "erasure_rate": boot(subset, _rate("erasure")),
            "excess_response_rate": boot(subset, _rate("excess_response")),
            "benefit": boot(subset, _mean("benefit")),
            "task_degradation": boot(subset, _mean("task_degradation")),
            "mean_delta_ssim": boot(subset, _mean("delta_ssim")),
            "verdicts": {
                level: sum(1 for record in subset if record["level"] == level)
                for level in ("green", "amber", "red")
            },
        }
    # Contrasts between strata, Holm-adjusted within the family.
    contrasts: dict[str, Any] = {}
    p_contrast: dict[str, float] = {}
    # Reported name -> the statistic it is computed from. Spelled out rather than derived from
    # the name: the endpoint field for the divergence rate is `divergent`, not `divergence`.
    contrast_fields = {
        "divergence_rate": _rate("divergent"),
        "erasure_rate": _rate("erasure"),
        "excess_response_rate": _rate("excess_response"),
        "task_degradation": _mean("task_degradation"),
    }
    for field, statistic in contrast_fields.items():
        for a, b in (("below floor", "above floor"), ("marginal", "above floor")):
            left = [r for r in endpoints if r["stratum"] == a]
            right = [r for r in endpoints if r["stratum"] == b]
            if not left or not right:
                continue
            key = f"{field}: {a} - {b}"

            def difference(
                records: Sequence[dict[str, Any]],
                _a: str = a,
                _b: str = b,
                _stat: Statistic = statistic,
            ) -> float:
                first = [r for r in records if r["stratum"] == _a]
                second = [r for r in records if r["stratum"] == _b]
                if not first or not second:
                    return float("nan")
                return _stat(first) - _stat(second)

            contrasts[key] = boot(endpoints, difference)
            p_contrast[key] = contrasts[key]["p_value"]
    if p_contrast:
        contrasts["p_holm"] = holm(p_contrast)

    return {
        "n_arm_seed_evaluations": len(endpoints),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_unique_processed_arms": len(endpoints) // max(len(seeds), 1),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "cluster": "realisation seed",
        "divergence": divergence,
        "observer_dependence": observer_block,
        "floor_strata": strata,
        "stratum_contrasts": contrasts,
    }
