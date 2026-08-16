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


SLUGS = {
    "none": "unprocessed",
    "gaussian 0.75 mm": "gauss075",
    "gaussian 1.00 mm": "gauss100",
    "tv 1x noise": "tv",
    "nlm 0.8x noise": "nlm",
    "CNN small (21k)": "cnn_small",
    "CNN large (1850k)": "cnn_large",
}


def _load(name: str) -> dict:
    return json.loads((SRC / f"{name}.json").read_text(encoding="utf-8"))


def _merged_rows(small: dict, large: dict) -> list[dict]:
    """The five classical arms (identical in both runs) plus both networks."""
    classical = [r for r in small["rows"] if not r["label"].startswith("CNN")]
    other = [r for r in large["rows"] if not r["label"].startswith("CNN")]
    if classical != other:
        raise SystemExit(
            "the classical arms differ between the two capacity runs; they share a "
            "split and a seed and must be identical, so something has drifted"
        )
    cnn_small = next(r for r in small["rows"] if r["label"].startswith("CNN"))
    cnn_large = next(r for r in large["rows"] if r["label"].startswith("CNN"))
    rows = [*classical, cnn_small, cnn_large]

    unprocessed = next(r for r in rows if r["label"] == "none")
    for r in rows:
        r["delta_psnr"] = r["psnr"] - unprocessed["psnr"]
        r["delta_d_prime"] = r["d_prime"] - unprocessed["d_prime"]
    return rows


def main() -> int:
    small, large = _load("liver_cnn_small"), _load("liver_cnn_large")
    ceiling_run, dose = _load("liver_ceiling"), _load("dose_axis")

    if small["ceiling"] != large["ceiling"]:
        raise SystemExit("the two capacity runs disagree about the ceiling")
    if small["test"] != large["test"]:
        raise SystemExit("the two capacity runs used different held-out cases")

    rows = _merged_rows(small, large)
    rho, p_value = spearmanr([r["psnr"] for r in rows], [r["d_prime"] for r in rows])

    by_psnr = sorted(rows, key=lambda r: -r["psnr"])
    by_task = sorted(rows, key=lambda r: -r["d_prime"])
    ceiling = small["ceiling"]

    payload = {
        "held_out": {
            # Keyed by slug as well as listed, because the manuscript's marker syntax
            # cannot address a list element by a label containing spaces and brackets.
            "by_method": {
                SLUGS[r["label"]]: r for r in rows if r["label"] in SLUGS
            },
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
                **next(r for r in rows if "21k" in r["label"]),
            },
            "large": {
                "parameters": PARAMETERS["large"],
                **TRAINING["large"],
                **next(r for r in rows if "1850k" in r["label"]),
            },
            "parameter_ratio": PARAMETERS["large"] / PARAMETERS["small"],
            "checkpoint_sha256": {
                "small": small["sha256"],
                "large": large["sha256"],
            },
        },
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
    print(f"best PSNR is {h['best_psnr']['label']}, ranked {h['psnr_winner_task_rank']} of "
          f"{h['n_methods']} on the task")
    print(f"capacity ratio {payload['capacity']['parameter_ratio']:.1f}x")
    print(f"exceedances of the ceiling, all arms: {payload['all_exceedances']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
