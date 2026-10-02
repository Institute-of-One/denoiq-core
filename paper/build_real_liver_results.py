"""Consolidate the real low-dose CT results into one file the manuscript can cite.

The measurements themselves are made by ``ldct-io``'s examples, which read
LDCT-and-Projection-data from disk and write one file per run. This script does no
science: it merges those runs into ``results/real_liver.json`` in the shape the
manuscript's ``[[results:...]]`` markers address, and computes the few derived
quantities the text quotes (rank correlation, parameter ratio, exceedance counts) so
that they are resolved from data rather than typed in by hand.

    python paper/build_real_liver_results.py

Inputs, produced by ldct-io and committed under ``paper/results/``:

    liver_cnn_small.json   held-out split, five classical denoisers + the 21 k network
    liver_cnn_large.json   the same split and denoisers + the 1.85 M network
    liver_ceiling.json     all twelve cases, no learned denoiser
    dose_axis.json         liver and chest, two operating points
    dose_decision.json     the measured dose axis: where the task is lost, and what
                           processing does and does not return
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr

PAPER_DIR = Path(__file__).resolve().parent
SRC = PAPER_DIR / "results"
OUT = PAPER_DIR.parent / "results" / "real_liver.json"

# Parameter counts are a property of the architecture, not of a training run, and are
# reproduced by denoiq_core.cnn.build_cnn on the presets ldct-io uses.
PARAMETERS = {"small": 21385, "large": 1849633}
TRAINING = {
    "small": {"patches_per_case": 800, "epochs": 16, "batch": 32},
    "large": {"patches_per_case": 3000, "epochs": 60, "batch": 64},
}


#: Configuration of the real-data arm, transcribed from ldct-io examples/liver_cnn.py
#: and denoiq_core/denoisers.py. Recorded so the Methods can resolve them as markers
#: rather than typing numbers into the prose, where they could drift from the code.
SPEC = {
    "lesion": {
        "diameter_mm": 8.0,
        "contrast_hu": -25.0,
        "edge_sigma_mm": 0.5,
        "supersample": 4,
        "roi_px": 48,
    },
    "sites": {
        "hu_range_low": 0.0,
        "hu_range_high": 160.0,
        "max_sd_hu": 60.0,
        "slice_halfwidth": 25,
        "max_per_case": 250,
        "min_per_case": 16,
    },
    "cnn": {
        "small": {"depth": 6, "width": 24, "kernel": 3},
        "large": {"depth": 10, "width": 96, "kernel": 5},
        "patch_px": 64,
        "validation_fraction": 0.15,
        "optimiser": "Adam",
        "learning_rate": 0.001,
        "formulation": "the network predicts the noise, which is subtracted from its input",
    },
    "arms": {
        "gaussian_small_mm": 0.75,
        "gaussian_large_mm": 1.00,
        "tv_weight_x_noise": 1.0,
        "nlm_h_x_noise": 0.8,
    },
    # How the observer is realised on anatomical background, and how d' is computed from its
    # scores. Transcribed from ldct-io examples/liver_cnn.py (cross_fitted_paired, invertible)
    # and ldct_io/lesion.py (paired_d_prime). Medical Physics returned MS 26-1820 partly
    # because the manuscript described neither.
    "observer": {
        "name": "cross-fitted prewhitening matched filter on paired trials",
        "folds": 5,
        "fold_assignment": "trial index modulo the number of folds",
        "signal_estimate": "difference of the training fold's present and absent means",
        "noise_estimate": "two-dimensional noise power spectrum of the paired differences, "
        "with their mean removed",
        "nps_dc_fill": "the zeroed DC bin is replaced by the mean of its four neighbours",
        "nps_ridge_x_mean": 0.01,
        "nps_floor_x_max": 0.001,
        "detectability": "paired d' = sqrt(2) * mean(delta) / sd(delta), delta being the "
        "per-trial difference of the two scores",
        "sqrt_two_reason": "the pair shares a background, so the difference carries two "
        "independent noise draws and has twice the variance of one image",
        # What dropping the sqrt(2) would cost, so the manuscript can quote it without a
        # number being typed into the prose. 1 - 1/sqrt(2), as a percentage.
        "understatement_without_sqrt_two_percent": round(100.0 * (1.0 - 2.0**-0.5)),
    },
    "implementation": {
        "gaussian": "scipy.ndimage.gaussian_filter, mode='nearest'",
        "tv": "skimage.restoration.denoise_tv_chambolle, channel_axis=None, applied to the "
        "mean-removed plane and the mean restored, so the filter is shift-invariant",
        "nlm": "skimage.restoration.denoise_nl_means, fast_mode=True, patch_size=5, "
        "patch_distance=6, sigma=h",
    },
}


SLUGS = {
    "none": "unprocessed",
    "gaussian 0.75 mm": "gauss075",
    "gaussian 1.00 mm": "gauss100",
    "tv 1x noise": "tv",
    "nlm 0.8x noise": "nlm",
    "CNN small (21k)": "cnn_small",
    "CNN large (1850k)": "cnn_large",
    "GAN small (21k)": "gan_small",
    "GAN large (1850k)": "gan_large",
}

#: The sensitivity run labels its arms its own way; map them to the same slugs the table uses.
SLUGS_SENS = {
    "unprocessed": "unprocessed",
    "gaussian 0.75 mm": "gauss075",
    "gaussian 1.00 mm": "gauss100",
    "tv 1x noise": "tv",
    "nlm 0.8x noise": "nlm",
    "CNN 21k": "cnn_small",
    "CNN 1.85M": "cnn_large",
    "GAN 21k": "gan_small",
    "GAN 1.85M": "gan_large",
}

#: The four learned arms, as ``preset -> the label its run writes``. Each run writes the five
#: classical arms as well, identically, which :func:`_merged_rows` checks rather than trusts.
LEARNED = {
    "small": "CNN small (21k)",
    "large": "CNN large (1850k)",
    "small_gan": "GAN small (21k)",
    "large_gan": "GAN large (1850k)",
}


def _load(name: str) -> dict:
    return json.loads((SRC / f"{name}.json").read_text(encoding="utf-8"))


def _guidance(run: dict) -> dict:
    """The dose axis, reduced to the numbers a protocol decision is made from.

    Everything here is derived from ``dose_decision.json``; nothing is typed. The quantities
    are the ones the Results section states, in the order it states them: where the task is
    lost, whether processing moves that point, how much exposure the processed picture claims,
    and how far the ideal-observer scaling law can be trusted.
    """
    rows = run["rows"]
    crossing = run["dose_at_requirement_measured"]
    nominal = run["alpha_measured_at"]
    raw_label = "unprocessed"
    raw_crossing = crossing[raw_label]
    processed = {k: v for k, v in crossing.items() if k != raw_label}

    at_nominal = {r["label"]: r for r in rows if r["dose"] == nominal}
    below = [r for r in rows if r["dose"] < raw_crossing and r["label"] != raw_label]
    # The overstatement is only alarming where the task has already failed, so it is
    # summarised separately above and below the crossing rather than pooled.
    worst_below = max(below, key=lambda r: r["apparent_dose"] / r["dose"])

    efficiency = {
        label: row["d_prime"] / row["ceiling"] for label, row in at_nominal.items()
    }
    eta = efficiency[raw_label]

    spreads = {k: v["spread_fraction"] for k, v in run["scaling_law_check"].items()}
    worst_observer = max(spreads, key=lambda k: spreads[k])

    # The ideal-observer floor and the reachable one, on THIS axis.
    #
    # denoiq_core.guidance converts between them with 1/eta^2, which follows from
    # d' proportional to sqrt(D) -- the total noise a site measures in its own images. This
    # study's dose axis is not that one: only the inserted noise is stochastic, its power goes
    # as 1/beta - 1, and so d' goes as sqrt(beta / (1 - beta)). Quoting 1/eta^2 here would mix
    # the two. The ceiling obeys this axis exactly, so its crossing is available in closed form,
    # and the reachable floor follows from holding the efficiency constant -- an assumption,
    # which is why its agreement with the measured crossing is reported beside it.
    ceiling_at_nominal = at_nominal[raw_label]["ceiling"]
    ideal_excess = (1.0 / nominal - 1.0) * (ceiling_at_nominal / run["requirement"]) ** 2
    ideal_floor = 1.0 / (1.0 + ideal_excess)
    reachable_floor = 1.0 / (1.0 + ideal_excess * eta**2)
    floor_ratio = raw_crossing / ideal_floor

    return {
        "requirement": run["requirement"],
        "requirement_source": run["requirement_source"],
        "nominal_dose": nominal,
        "n_pairs": run["n_pairs"],
        "n_cases": len(run["cases"]),
        "doses": run["doses"],
        "crossing": {
            "unprocessed": raw_crossing,
            "best_processed": min(processed.values()),
            "best_processed_label": min(processed, key=lambda k: processed[k]),
            "worst_processed": max(processed.values()),
            "worst_processed_label": max(processed, key=lambda k: processed[k]),
            "by_method": crossing,
        },
        "processing_penalty": {
            "n_processed": len(processed),
            "n_needing_more_dose": sum(1 for v in processed.values() if v > raw_crossing * 1.02),
            "worst_percent": 100.0 * (max(processed.values()) / raw_crossing - 1.0),
            "best_percent": 100.0 * (min(processed.values()) / raw_crossing - 1.0),
        },
        "overstatement": {
            "at_nominal_max": max(
                r["apparent_dose"] / r["dose"] for k, r in at_nominal.items() if k != raw_label
            ),
            "at_nominal_max_label": max(
                (k for k in at_nominal if k != raw_label),
                key=lambda k: at_nominal[k]["apparent_dose"] / at_nominal[k]["dose"],
            ),
            "worst_below_crossing": worst_below["apparent_dose"] / worst_below["dose"],
            "worst_below_crossing_label": worst_below["label"],
            "worst_below_crossing_dose": worst_below["dose"],
            "worst_below_crossing_apparent": worst_below["apparent_dose"],
            "worst_below_crossing_d_prime": worst_below["d_prime"],
        },
        "efficiency": {
            "at_nominal": efficiency,
            "unprocessed_at_nominal": eta,
            # The exposure penalty an ideal-observer floor hides, from guidance.achievable_floor.
            #
            # Two numbers, because they answer different questions and are easy to confuse.
            # `achievable_floor_percent` is how much MORE exposure the reachable floor needs
            # than the ideal-observer one, with the ideal floor as the denominator.
            # `ideal_floor_shortfall_percent` is how much of the REQUIRED exposure an
            # ideal-observer calculation fails to supply, with the requirement as the
            # denominator. At eta = 0.745 they are 80 % and 44 %, and saying the ideal
            # calculation "underestimates the requirement by 80 %" is the second quantity
            # named with the first one's value.
            # The sqrt(D) conversion, kept because denoiq_core.guidance offers it to a site
            # working in total noise, and labelled so it is not mistaken for this axis.
            "sqrt_law_floor_factor": 1.0 / eta**2,
            "sqrt_law_floor_percent": 100.0 * (1.0 / eta**2 - 1.0),
            # This axis, where the conversion belongs.
            "ideal_observer_floor": ideal_floor,
            "reachable_floor_constant_efficiency": reachable_floor,
            "reachable_floor_measured": raw_crossing,
            "constant_efficiency_agreement_percent": 100.0
            * abs(reachable_floor - raw_crossing)
            / raw_crossing,
            "floor_ratio": floor_ratio,
            "floor_ratio_percent": 100.0 * (floor_ratio - 1.0),
            "ideal_floor_fraction_of_required": ideal_floor / raw_crossing,
            "ideal_floor_shortfall_percent": 100.0 * (1.0 - ideal_floor / raw_crossing),
        },
        "scaling_law": {
            "ceiling_spread": run["ceiling_scaling_spread"],
            "worst_observer_spread": spreads[worst_observer],
            "worst_observer_label": worst_observer,
            "dose_range_fold": max(run["doses"]) / min(run["doses"]),
            "by_method": spreads,
        },
    }


def _merged_rows(runs: dict[str, dict]) -> list[dict]:
    """The five classical arms, identical in every run, plus each run's learned arm.

    The equality check is the point. All four runs evaluate the same held-out pairs with the
    same seed, so their classical rows have to agree bit for bit; if they do not, something has
    drifted between the runs and merging them would hide it behind a plausible table.
    """
    reference = None
    learned: list[dict] = []
    for preset, label in LEARNED.items():
        run = runs[preset]
        classical = [r for r in run["rows"] if r["label"] not in LEARNED.values()]
        if reference is None:
            reference = classical
        elif classical != reference:
            raise SystemExit(
                f"the classical arms in the {preset} run differ from the others; they share a "
                "split and a seed and must be identical, so something has drifted"
            )
        learned.append(next(r for r in run["rows"] if r["label"] == label))
    rows = [*(reference or []), *learned]

    unprocessed = next(r for r in rows if r["label"] == "none")
    for r in rows:
        r["delta_psnr"] = r["psnr"] - unprocessed["psnr"]
        r["delta_d_prime"] = r["d_prime"] - unprocessed["d_prime"]
    return rows


def main() -> int:
    runs = {preset: _load(f"liver_cnn_{preset}") for preset in LEARNED}
    small, large = runs["small"], runs["large"]
    ceiling_run, dose = _load("liver_ceiling"), _load("dose_axis")

    for preset, run in runs.items():
        if run["ceiling"] != small["ceiling"]:
            raise SystemExit(f"the {preset} run disagrees about the ceiling")
        if run["test"] != small["test"]:
            raise SystemExit(f"the {preset} run used different held-out cases")

    rows = _merged_rows(runs)
    rho, p_value = spearmanr([r["psnr"] for r in rows], [r["d_prime"] for r in rows])

    by_psnr = sorted(rows, key=lambda r: -r["psnr"])
    by_task = sorted(rows, key=lambda r: -r["d_prime"])
    ceiling = small["ceiling"]

    payload = {
        "held_out": {
            # Keyed by slug as well as listed, because the manuscript's marker syntax
            # cannot address a list element by a label containing spaces and brackets.
            "by_method": {SLUGS[r["label"]]: r for r in rows if r["label"] in SLUGS},
            "ceiling": ceiling,
            "train_cases": small["train"],
            "test_cases": sorted(small["test"]),
            "n_train_cases": len(small["train"]),
            "n_test_cases": len(small["test"]),
            "n_pairs": sum(small["test"].values()),
            "rows": rows,
            "n_methods": len(rows),
            "n_exceeding_ceiling": sum(1 for r in rows if r["d_prime"] > ceiling),
            "spearman_psnr_vs_d_prime": {
                "rho": float(rho),
                "p_value": float(p_value),
                "n": len(rows),
            },
            "rank_by_psnr": [r["label"] for r in by_psnr],
            "rank_by_task": [r["label"] for r in by_task],
            "best_psnr": by_psnr[0],
            "best_task": by_task[0],
            "psnr_winner_task_rank": by_task.index(by_psnr[0]) + 1,
        },
        "capacity": {
            "small": {
                "parameters": PARAMETERS["small"],
                **TRAINING["small"],
                # The checkpoint saved is the last epoch, not the lowest-validation one, so the
                # best loss a run reached is reported beside it rather than implied by it.
                "best_val_loss": runs["small"]["best_val_loss"],
                **next(r for r in rows if r["label"] == LEARNED["small"]),
            },
            "large": {
                "parameters": PARAMETERS["large"],
                **TRAINING["large"],
                "best_val_loss": runs["large"]["best_val_loss"],
                **next(r for r in rows if r["label"] == LEARNED["large"]),
            },
            "parameter_ratio": PARAMETERS["large"] / PARAMETERS["small"],
            "checkpoint_sha256": {preset: run["sha256"] for preset, run in runs.items()},
        },
        # The controlled comparison the adversarial arms exist for. At each capacity the two
        # networks share an architecture, a training split, a patch set, an epoch count, a
        # batch size, a learning rate and a seed; only the objective differs. Anything that
        # separates them is therefore attributable to optimising appearance rather than
        # fidelity, and not to capacity, data or training length.
        "objective": {
            capacity: {
                "mse": next(r for r in rows if r["label"] == LEARNED[capacity]),
                "adversarial": next(
                    r for r in rows if r["label"] == LEARNED[f"{capacity}_gan"]
                ),
                "adversarial_config": runs[f"{capacity}_gan"]["adversarial"]["objective"],
                "epochs": runs[f"{capacity}_gan"]["epochs"],
                "d_prime_ratio": (
                    next(r for r in rows if r["label"] == LEARNED[f"{capacity}_gan"])["d_prime"]
                    / next(r for r in rows if r["label"] == LEARNED[capacity])["d_prime"]
                ),
                "psnr_difference": (
                    next(r for r in rows if r["label"] == LEARNED[f"{capacity}_gan"])["psnr"]
                    - next(r for r in rows if r["label"] == LEARNED[capacity])["psnr"]
                ),
            }
            for capacity in ("small", "large")
        },
        # What the adversarial arm was built to provoke, and did not. `added` counts
        # lesion-shaped structure the processing put into a lesion-free image, with the
        # anatomy cancelled; the unprocessed row is the noise's own rate, and is the only
        # reference under which the number means anything.
        "fabrication": {
            "noise_itself": next(r for r in rows if r["label"] == "none")[
                "added_structure_rate"
            ],
            "by_method": {
                SLUGS[r["label"]]: r["added_structure_rate"]
                for r in rows
                if r["label"] in SLUGS
            },
            "max_processed": max(
                r["added_structure_rate"] for r in rows if r["label"] != "none"
            ),
            "n_above_the_noise": sum(
                1
                for r in rows
                if r["label"] != "none"
                and r["added_structure_rate"]
                > next(x for x in rows if x["label"] == "none")["added_structure_rate"]
            ),
            "n_processed": len(rows) - 1,
            "lowest_contrast_recovery": min(
                (r for r in rows if r["label"] != "none"), key=lambda r: r["contrast_recovery"]
            ),
        },
        "spec": SPEC,
        "all_cases": {
            "ceiling": ceiling_run["ceiling"],
            "n_cases": len(ceiling_run["cases"]),
            "n_slices": sum(ceiling_run["cases"].values()),
            "rows": ceiling_run["rows"],
            "n_exceeding_ceiling": sum(
                1 for r in ceiling_run["rows"] if r["d_prime"] > ceiling_run["ceiling"]
            ),
        },
        "operating_points": {
            region: {
                "region": block["region"],
                "dose": block["dose"],
                "noise_sd": block["noise_sd"],
                "ceiling": block["ceiling"],
                "rows": block["rows"],
                "n_exceeding_ceiling": sum(
                    1 for r in block["rows"] if r["d_prime"] > block["ceiling"]
                ),
            }
            for region, block in (("liver", dose["liver"]), ("chest", dose["chest"]))
        },
    }
    payload["operating_points"]["noise_ratio_chest_over_liver"] = (
        dose["chest"]["noise_sd"] / dose["liver"]["noise_sd"]
    )
    payload["guidance"] = _guidance(_load("dose_decision"))

    # How much the held-out ranking depends on where the lesions were put. Drawing the admitted
    # sites showed the admission rule passes bowel, mesentery and vessel structure as well as
    # parenchyma, so the evaluation was repeated with the one criterion that addresses it added
    # and everything else held fixed, including the pairs per case.
    sensitivity = _load("site_sensitivity")
    published = sensitivity["rules"]["as published"]
    strict = sensitivity["rules"]["low structure"]
    swaps = sum(
        1
        for a, b in zip(
            sensitivity["comparison"]["ranking_as_published"],
            sensitivity["comparison"]["ranking_low_structure"],
        )
        if a != b
    )
    payload["site_sensitivity"] = {
        "n_pairs": published["n_pairs"],
        "structure_sd_median": published["structure_sd_median"],
        "structure_sd_median_strict": strict["structure_sd_median"],
        "structure_over_lesion_fraction": published["structure_over_lesion_fraction"],
        "lesion_depth_hu": published["lesion_depth_hu"],
        "structure_threshold_hu": strict["rule"]["structure_sd"],
        "ceiling": published["ceiling"],
        "ceiling_strict": strict["ceiling"],
        "spearman_rho": sensitivity["comparison"]["spearman_rho"],
        "spearman_p": sensitivity["comparison"]["spearman_p"],
        "n_methods": len(published["rows"]),
        "n_positions_changed": swaps,
        "by_method": {
            SLUGS_SENS[r["label"]]: {
                "published": r["d_prime"],
                "strict": next(
                    x["d_prime"] for x in strict["rows"] if x["label"] == r["label"]
                ),
            }
            for r in published["rows"]
            if r["label"] in SLUGS_SENS
        },
    }
    payload["all_exceedances"] = (
        payload["held_out"]["n_exceeding_ceiling"]
        + payload["all_cases"]["n_exceeding_ceiling"]
        + payload["operating_points"]["liver"]["n_exceeding_ceiling"]
        + payload["operating_points"]["chest"]["n_exceeding_ceiling"]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    h = payload["held_out"]
    print(f"held-out split: {h['n_pairs']} pairs, ceiling d' = {ceiling:.3f}")
    for r in sorted(h["rows"], key=lambda r: -r["d_prime"]):
        print(
            f"  {r['label']:20s} d' {r['d_prime']:6.3f}  {r['ratio']:.2f}x  "
            f"PSNR {r['psnr']:6.2f} dB"
        )
    s = h["spearman_psnr_vs_d_prime"]
    print(f"\nSpearman(PSNR, d') = {s['rho']:+.3f} (p = {s['p_value']:.2f}, n = {s['n']})")
    print(
        f"best PSNR is {h['best_psnr']['label']}, ranked {h['psnr_winner_task_rank']} of "
        f"{h['n_methods']} on the task"
    )
    print(f"capacity ratio {payload['capacity']['parameter_ratio']:.1f}x")
    print(f"exceedances of the ceiling, all arms: {payload['all_exceedances']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
