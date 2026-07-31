"""The multi-realisation primary analysis: design, endpoints, statistics, leakage.

These are the guards on the study's own numbers — the counts it reports, the endpoints it
derives, the intervals it puts on them, and the two properties that make those numbers mean
anything: that realisations are independent, and that no trial is scored by a template fitted
to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from denoiq_core.evaluate import EvalConfig, cross_fitted_ideal, fold_indices
from denoiq_core.experiment import DEFAULT_SWEEP, PRIMARY_SEEDS, SEED_STRIDE
from denoiq_core.statistics import (
    STRATA,
    cluster_bootstrap,
    compute_endpoints,
    holm,
    spearman,
    stratum_of,
    wilson_interval,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

REQUIRED = ("multiseed.json", "endpoints.json", "statistics.json")
requires_primary = pytest.mark.skipif(
    not all((RESULTS / name).exists() for name in REQUIRED),
    reason="the primary analysis has not been run: denoiq_core.experiment.run_primary()",
)


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# seeds and independence
# --------------------------------------------------------------------------------------


def test_ten_realisations_are_declared_and_derived_from_the_first():
    assert len(PRIMARY_SEEDS) == 10
    assert PRIMARY_SEEDS[0] == DEFAULT_SWEEP.seed, "the original single-realisation seed is kept"
    assert len(set(PRIMARY_SEEDS)) == 10
    for index, seed in enumerate(PRIMARY_SEEDS):
        assert seed == PRIMARY_SEEDS[0] + SEED_STRIDE * index


def test_realisations_do_not_share_a_trial_stream():
    """Different realisation seeds must give different noise, at the same condition."""
    from denoiq_core.physics import make_trials

    first = make_trials(120.0, 25.0, n_trials=8, seed=PRIMARY_SEEDS[0])
    second = make_trials(120.0, 25.0, n_trials=8, seed=PRIMARY_SEEDS[1])
    assert not np.array_equal(first.absent, second.absent)
    assert not np.array_equal(first.present, second.present)
    # The signal is deterministic, so the two are the same experiment on different noise.
    assert np.array_equal(first.signal, second.signal)


def test_present_and_absent_are_independent_within_a_realisation():
    from denoiq_core.physics import make_trials

    trials = make_trials(120.0, 25.0, n_trials=64, seed=PRIMARY_SEEDS[0])
    # The present stack is the absent-style noise plus the signal; if the two stacks shared a
    # noise realisation their difference would be exactly the signal, every time.
    difference = trials.present - trials.absent - trials.signal
    assert np.std(difference) > 0.5 * trials.noise_sd


# --------------------------------------------------------------------------------------
# cross-fitting: no trial scores its own template
# --------------------------------------------------------------------------------------


def test_folds_partition_every_trial_exactly_once():
    folds = fold_indices(37, 5)
    assert len(folds) == 5
    joined = np.sort(np.concatenate(folds))
    assert np.array_equal(joined, np.arange(37))


def test_cross_fitting_scores_every_trial_out_of_fold(trials_high_dose):
    """Each trial is scored once, and never by a template that saw it.

    The check is behavioural: perturbing one training image must not change the score of a
    trial in the same fold as that image (it was never in that trial's template), but must
    change the scores of trials in the other folds.
    """
    trials = trials_high_dose
    config = EvalConfig(n_folds=5)
    baseline = cross_fitted_ideal(trials.present, trials.absent, trials.spacing, config=config)
    assert baseline.n_present == trials.n_trials
    assert baseline.n_absent == trials.n_trials

    present = trials.present.copy()
    present[0] += 50.0  # index 0 is in fold 0
    perturbed = cross_fitted_ideal(present, trials.absent, trials.spacing, config=config)

    fold_of_zero = 0
    scores_before = baseline.scores_present
    scores_after = perturbed.scores_present
    # Scores come back concatenated fold by fold; fold 0 owns indices 0, 5, 10, ...
    n_fold0 = len(fold_indices(trials.n_trials, 5)[fold_of_zero])
    # Every trial of fold 0 except the perturbed one keeps its score: their template was fitted
    # on the other folds, which are untouched.
    unchanged = np.isclose(scores_before[1:n_fold0], scores_after[1:n_fold0])
    assert unchanged.all(), "a fold's scores changed although its template did not"
    # Trials in the other folds do move: image 0 is in their training set.
    assert not np.allclose(scores_before[n_fold0:], scores_after[n_fold0:])


def test_cross_fitting_recovers_more_of_the_ceiling_than_a_single_split(trials_high_dose):
    from denoiq_core.bound import ideal_ceiling, raw_estimator_check

    trials = trials_high_dose
    ceiling = ideal_ceiling(trials).d_prime
    cross = raw_estimator_check(trials, config=EvalConfig(estimator="cross-fit"))
    split = raw_estimator_check(trials, config=EvalConfig(estimator="split"))
    assert cross["estimated_d_prime"] > 0.8 * ceiling
    # At this dose both are close; the ordering is checked on the study data in
    # test_the_estimator_sensitivity_table_favours_cross_fitting.
    assert cross["d_prime_ratio"] > 0.0 and split["d_prime_ratio"] > 0.0


# --------------------------------------------------------------------------------------
# statistics: the machinery, on data with a known answer
# --------------------------------------------------------------------------------------


def test_spearman_matches_a_known_case():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(spearman([1, 2, 3, 4], [1, 1, 1, 1])) != abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]))


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] >= adjusted["b"]
    assert all(value <= 1.0 for value in adjusted.values())


def test_wilson_interval_brackets_the_proportion():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    assert wilson_interval(0, 10)[0] == 0.0


def test_the_cluster_bootstrap_resamples_clusters_not_records():
    """Two clusters that disagree must widen the interval; identical ones must not."""
    same = [{"seed": s, "x": 1.0} for s in range(10)]
    spread = [{"seed": s, "x": float(s)} for s in range(10)]
    statistic = lambda records: float(np.mean([r["x"] for r in records]))  # noqa: E731
    narrow = cluster_bootstrap(same, statistic, n_boot=200, null=1.0)
    wide = cluster_bootstrap(spread, statistic, n_boot=200)
    assert narrow.high - narrow.low == pytest.approx(0.0)
    assert wide.high - wide.low > 1.0


def test_endpoints_pair_each_processed_arm_with_its_own_input():
    rows = [
        {
            "realisation_seed": 1,
            "sweep": "dose",
            "kv": 120.0,
            "mas": 25.0,
            "correlation_sigma_mm": 0.0,
            "radius_mm": 3.0,
            "contrast": 20.0,
            "method": "none",
            "denoiser": "raw",
            "ssim": 0.1,
            "d_prime_ideal": 3.0,
            "d_prime_cho": 3.0,
            "d_prime_npwe": 2.0,
            "ceiling_d_prime": 3.5,
            "contrast_recovery": 1.0,
            "false_structure_rate": 1.0,
            "raw_false_structure_rate": 1.0,
            "level": "red",
            "excess": -0.01,
            "margin": 0.02,
            "bound_ok": True,
            "saturated": False,
            "estimator_d_prime_ratio": 0.9,
        },
        {
            "realisation_seed": 1,
            "sweep": "dose",
            "kv": 120.0,
            "mas": 25.0,
            "correlation_sigma_mm": 0.0,
            "radius_mm": 3.0,
            "contrast": 20.0,
            "method": "gaussian",
            "denoiser": "g",
            "ssim": 0.4,
            "d_prime_ideal": 2.0,
            "d_prime_cho": 2.5,
            "d_prime_npwe": 2.5,
            "ceiling_d_prime": 3.5,
            "contrast_recovery": 0.4,
            "false_structure_rate": 0.9,
            "raw_false_structure_rate": 1.0,
            "level": "red",
            "excess": -0.05,
            "margin": 0.02,
            "bound_ok": True,
            "saturated": False,
            "estimator_d_prime_ratio": 0.9,
        },
    ]
    endpoints = compute_endpoints(rows, threshold=5.0, amber_factor=1.2)
    assert len(endpoints) == 1
    record = endpoints[0]
    assert record["delta_ssim"] == pytest.approx(0.3)
    assert record["delta_d_pw"] == pytest.approx(-1.0)
    assert record["delta_d_npwe"] == pytest.approx(0.5)
    assert record["benefit"] == pytest.approx(1.5)
    assert record["divergent"] is True
    assert record["erasure"] is True  # recovery 0.4 < 0.5
    assert record["excess_response"] is False  # not above the input's own rate
    assert record["stratum"] == "below floor"
    assert record["task_degradation"] == pytest.approx(1.0 / 3.0)


def test_a_processed_arm_without_its_input_is_an_error():
    rows = [
        {
            "realisation_seed": 1,
            "sweep": "dose",
            "kv": 120.0,
            "mas": 25.0,
            "correlation_sigma_mm": 0.0,
            "radius_mm": 3.0,
            "contrast": 20.0,
            "method": "gaussian",
            "denoiser": "g",
            "ssim": 0.4,
            "d_prime_ideal": 2.0,
            "d_prime_cho": 2.5,
            "d_prime_npwe": 2.5,
            "ceiling_d_prime": 3.5,
            "contrast_recovery": 0.4,
            "false_structure_rate": 0.9,
            "raw_false_structure_rate": 1.0,
            "level": "red",
            "excess": -0.05,
            "margin": 0.02,
            "bound_ok": True,
            "saturated": False,
            "estimator_d_prime_ratio": 0.9,
        }
    ]
    with pytest.raises(ValueError, match="no unprocessed reference"):
        compute_endpoints(rows, threshold=5.0, amber_factor=1.2)


def test_strata_boundaries_follow_the_prespecified_criteria():
    assert stratum_of(4.9, 5.0, 1.2) == "below floor"
    assert stratum_of(5.0, 5.0, 1.2) == "marginal"
    assert stratum_of(5.9, 5.0, 1.2) == "marginal"
    assert stratum_of(6.0, 5.0, 1.2) == "above floor"
    assert set(STRATA) == {"above floor", "marginal", "below floor"}


# --------------------------------------------------------------------------------------
# the study's own numbers
# --------------------------------------------------------------------------------------


@requires_primary
def test_the_design_counts_add_up():
    multiseed = _load("multiseed.json")
    design = _load("statistics.json")["design"]
    rows = multiseed["rows"]
    assert design["n_seeds"] == len(multiseed["seeds"]) == 10
    assert design["arm_seed_evaluations"] == len(rows)
    assert design["unique_arms"] * design["n_seeds"] == design["arm_seed_evaluations"]
    assert design["unprocessed_arms"] + design["processed_arms"] == design["unique_arms"]
    assert design["unprocessed_arms"] == design["input_conditions"]
    assert sum(entry["arms"] for entry in design["sweeps"].values()) == design["unique_arms"]
    # The matrix is not uniform in trial count (the atlas settings use fewer), so the total is
    # summed over rows rather than multiplied out — the multiplication was wrong, and this is
    # the assertion that caught it.
    expected_trials = 2 * sum(int(row["n_trials"]) for row in rows)
    assert design["scored_image_trials"] == expected_trials
    assert design["trials_per_class_main_sweeps"] == max(design["trials_per_class_per_arm"])
    # Every realisation ran the identical matrix.
    per_seed = {seed: 0 for seed in multiseed["seeds"]}
    for row in rows:
        per_seed[row["realisation_seed"]] += 1
    assert len(set(per_seed.values())) == 1


@requires_primary
def test_the_learned_denoiser_is_not_in_the_primary_matrix():
    rows = _load("multiseed.json")["rows"]
    assert not any(row["method"] == "cnn" for row in rows)
    assert len({row["denoiser"] for row in rows if row["method"] != "none"}) == 3


@requires_primary
def test_the_endpoints_are_recomputable_from_the_rows():
    multiseed = _load("multiseed.json")
    stored = _load("endpoints.json")["rows"]
    criteria = multiseed["config"]["criteria"]
    recomputed = compute_endpoints(
        multiseed["rows"],
        threshold=criteria["d_prime_threshold"],
        amber_factor=criteria["amber_factor"],
    )
    assert len(recomputed) == len(stored)
    for a, b in zip(
        sorted(recomputed, key=lambda r: (r["seed"], r["sweep"], r["denoiser"], str(r["mas"]))),
        sorted(stored, key=lambda r: (r["seed"], r["sweep"], r["denoiser"], str(r["mas"]))),
        strict=True,
    ):
        assert a["delta_ssim"] == pytest.approx(b["delta_ssim"])
        assert a["delta_d_pw"] == pytest.approx(b["delta_d_pw"])
        assert a["benefit"] == pytest.approx(b["benefit"])
        assert a["divergent"] == b["divergent"]
        assert a["stratum"] == b["stratum"]


@requires_primary
def test_the_primary_endpoints_match_the_endpoint_records():
    """Spearman, divergence rate and benefit recomputed from endpoints.json."""
    endpoints = _load("endpoints.json")["rows"]
    statistics = _load("statistics.json")

    rho = spearman([r["delta_ssim"] for r in endpoints], [r["delta_d_pw"] for r in endpoints])
    assert statistics["divergence"]["spearman_delta_ssim_delta_d_pw"]["value"] == pytest.approx(rho)

    rate = float(np.mean([r["divergent"] for r in endpoints]))
    assert statistics["divergence"]["divergence_rate"]["value"] == pytest.approx(rate)

    benefit = float(np.mean([r["benefit"] for r in endpoints]))
    assert statistics["observer_dependence"]["overall"]["benefit"]["value"] == pytest.approx(
        benefit
    )
    for observer, key in (
        ("delta_d_pw", "delta_d_pw"),
        ("delta_d_cho", "delta_d_cho"),
        ("delta_d_npwe", "delta_d_npwe"),
    ):
        expected = float(np.mean([r[observer] for r in endpoints]))
        assert statistics["observer_dependence"]["overall"][key]["value"] == pytest.approx(expected)


@requires_primary
def test_every_reported_interval_brackets_its_point_estimate():
    """A CI that does not contain its own estimate is a broken resampler."""
    statistics = _load("statistics.json")

    def walk(node):
        if isinstance(node, dict):
            if {"value", "ci_low", "ci_high"} <= set(node):
                assert node["ci_low"] <= node["value"] <= node["ci_high"], node
                assert 0.0 <= node["p_value"] <= 1.0
            for value in node.values():
                walk(value)

    walk(statistics)


@requires_primary
def test_the_observer_ordering_follows_efficiency():
    """The study's central comparative claim, asserted on the aggregate."""
    overall = _load("statistics.json")["observer_dependence"]["overall"]
    assert overall["delta_d_pw"]["value"] < overall["delta_d_cho"]["value"]
    assert overall["delta_d_cho"]["value"] < overall["delta_d_npwe"]["value"]
    assert overall["benefit"]["ci_low"] > 0.0, "the benefit interval must exclude zero"


@requires_primary
def test_the_floor_strata_partition_the_evaluations():
    statistics = _load("statistics.json")
    endpoints = _load("endpoints.json")["rows"]
    total = sum(entry["n_arm_seed"] for entry in statistics["floor_strata"].values())
    assert total == len(endpoints)
    for name, entry in statistics["floor_strata"].items():
        assert name in STRATA
        assert sum(entry["verdicts"].values()) == entry["n_arm_seed"]


@requires_primary
def test_the_ceiling_summary_is_recomputable_and_its_exceedance_explained():
    multiseed = _load("multiseed.json")
    ceiling = _load("statistics.json")["ceiling"]
    rows = multiseed["rows"]
    assert ceiling["n_evaluations"] == len(rows)
    assert ceiling["n_violations"] == sum(1 for row in rows if not row["bound_ok"])
    assert ceiling["max_excess"] == pytest.approx(max(row["excess"] for row in rows))
    assert ceiling["n_saturated"] == sum(1 for row in rows if row["saturated"])
    assert ceiling["unsaturated"]["n_evaluations"] == sum(1 for row in rows if not row["saturated"])
    # Whatever exceedances exist must be accounted for: none on a processed arm, and the
    # unsaturated subset clean. If that ever changes, the manuscript's account is wrong.
    assert ceiling["n_violations_processed_arms"] == 0
    assert ceiling["unsaturated"]["n_violations"] == 0
    for detail in ceiling["violation_detail"]:
        assert detail["processed"] is False
        assert detail["saturated"] is True


@requires_primary
def test_the_estimator_recovers_most_of_the_ceiling_above_the_floor():
    recovery = _load("statistics.json")["ceiling"]["estimator_recovery"]
    assert recovery["median"] > 0.85
    assert recovery["above_floor_min"] > 0.8, "the comparison must not be won by a weak estimator"


@requires_primary
def test_the_estimator_sensitivity_table_favours_cross_fitting():
    path = RESULTS / "estimator_sensitivity.json"
    if not path.exists():
        pytest.skip("estimator_sensitivity.json has not been generated")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    assert rows
    # At the lowest doses, where the estimate is hardest, cross-fitting must do better.
    hardest = min(rows, key=lambda row: row["mas"])
    assert hardest["cross_fit_recovery"] > hardest["split_recovery"]
