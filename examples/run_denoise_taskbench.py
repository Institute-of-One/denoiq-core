"""Raw vs classical vs (optionally) CNN at one acquisition setting.

    python examples/run_denoise_taskbench.py
    python examples/run_denoise_taskbench.py --mas 12 --cnn checkpoints/cnn.pt

Prints a table of fidelity, task detectability and the data-processing ceiling, and writes
the numbers to ``results/taskbench.json`` (plus ``results/taskbench_cnn.json`` when a
checkpoint is given, so that the deep-learning numbers stay in their own file and the rest
of the study remains reproducible without torch).

The line to read is the last column: no processing beats the ceiling, while SSIM happily
goes up.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from denoiq_core.experiment import (
    DEFAULT_DENOISERS,
    DEFAULT_SWEEP,
    DenoiserSpec,
    provenance,
    run_condition,
    write_csv,
    write_json,
)
from denoiq_core.physics import acquisition_params, make_trials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kv", type=float, default=120.0)
    parser.add_argument("--mas", type=float, default=25.0)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_SWEEP.n_trials)
    parser.add_argument("--seed", type=int, default=DEFAULT_SWEEP.seed)
    parser.add_argument(
        "--cnn",
        type=Path,
        default=None,
        help="checkpoint from `python -m denoiq_core.train`; adds the CNN row",
    )
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args(argv)

    config = replace(DEFAULT_SWEEP, n_trials=args.n_trials, seed=args.seed)
    trials = make_trials(
        args.kv,
        args.mas,
        n_trials=args.n_trials,
        seed=args.seed,
        model=config.model,
        phantom=config.phantom,
    )
    acq = acquisition_params(args.kv, args.mas, model=config.model)
    condition = {
        "sweep": "taskbench",
        "kv": args.kv,
        "mas": args.mas,
        "relative_dose": acq.relative_dose,
        "contrast": acq.contrast,
    }

    specs = list(DEFAULT_DENOISERS)
    if args.cnn is not None:
        specs.append(DenoiserSpec("cnn", {"checkpoint": str(args.cnn)}, label="CNN"))

    rows = [run_condition(trials, spec, config=config, condition=condition) for spec in specs]

    header = (
        f"{'denoiser':<18}{'SSIM':>7}{'PSNR':>8}{'d ideal':>9}{'d CHO':>8}{'d NPWE':>8}"
        f"{'AUC ideal':>11}{'ceiling':>9}{'excess':>9}  verdict"
    )
    print(
        f"\nkV={args.kv:g}  mAs={args.mas:g}  noise sd={trials.noise_sd:.1f}  "
        f"contrast={acq.contrast:.1f}  n_trials={args.n_trials}"
    )
    print(
        f"raw ideal-observer ceiling: d'={rows[0]['ceiling_d_prime']:.3f}  "
        f"AUC={rows[0]['ceiling_auc']:.5f}\n"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['denoiser']:<18}{row['ssim']:>7.3f}{row['psnr']:>8.2f}"
            f"{row['d_prime_ideal']:>9.3f}{row['d_prime_cho']:>8.3f}{row['d_prime_npwe']:>8.3f}"
            f"{row['auc_ideal']:>11.4f}{row['ceiling_auc']:>9.4f}{row['excess']:>9.4f}"
            f"  {'ok' if row['bound_ok'] else 'ABOVE CEILING'}"
        )
    print(
        "\nThe held-out estimator recovers "
        f"{100 * rows[0]['estimator_d_prime_ratio']:.0f}% of the analytic ceiling d' on the raw "
        "images, so the comparison above is not trivially satisfied."
    )

    payload = {
        "sweep": "taskbench",
        "provenance": provenance({"cnn": str(args.cnn) if args.cnn else None}),
        "config": config.to_dict(),
        "rows": rows,
    }
    name = "taskbench_cnn" if args.cnn is not None else "taskbench"
    path = write_json(payload, f"{name}.json", directory=args.results)
    write_csv(rows, f"{name}.csv", directory=args.results)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
