"""The red-lamp atlas over the kV-mAs plane, and Figure 6.

    python examples/run_redlamp_atlas.py
    python examples/run_redlamp_atlas.py --threshold 4 --quick

Computes the raw ideal-observer detectability across the plane, the information floor
(``d' = 5``, the Rose criterion) as an exact contour, and a green / amber / red verdict with
a written reason at a few demonstration points. Writes ``results/redlamp_atlas.json`` and
``results/redlamp_demo.json`` and draws ``paper/figures/fig6_redlamp_atlas.png``.

What this is and is not: the atlas is a *method* demonstrated on a normalised analytic
acquisition model. It is not a calibration of any scanner, and the mAs numbers on its axis
are relative to a reference setting, not absolute exposure values.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from denoiq_core.experiment import (
    DEFAULT_SWEEP,
    provenance,
    sweep_kv_mas,
    write_csv,
    write_json,
)
from denoiq_core.figures import figure6_redlamp_atlas
from denoiq_core.redlamp import assess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=5.0, help="the floor, in d'")
    parser.add_argument("--n-kv", type=int, default=15)
    parser.add_argument("--n-mas", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="coarse grid, fewer trials")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args(argv)

    criteria = replace(DEFAULT_SWEEP.criteria, d_prime_threshold=args.threshold)
    config = replace(DEFAULT_SWEEP, criteria=criteria)
    n_kv = 6 if args.quick else args.n_kv
    n_mas = 6 if args.quick else args.n_mas

    atlas, demo = sweep_kv_mas(
        np.linspace(70.0, 140.0, n_kv),
        np.geomspace(2.0, 300.0, n_mas),
        config=config,
        n_trials_demo=200 if args.quick else 400,
    )

    print(f"\ninformation floor at d' = {args.threshold:g} (Rose criterion)")
    print(f"{'kV':>6}{'floor mAs':>12}")
    for kv, mas in zip(atlas.kv, atlas.floor_mas, strict=True):
        print(f"{kv:>6.0f}{mas:>12.1f}")

    print("\nred lamp at the demonstration points:")
    for kv, mas in ((120.0, 200.0), (120.0, 50.0), (120.0, 12.0), (80.0, 12.0)):
        lamp = assess(
            kv,
            mas,
            "gaussian",
            {"sigma": 1.5},
            n_trials=200 if args.quick else 400,
            seed=config.seed,
            model=config.model,
            phantom=config.phantom,
            config=config.eval_config,
            criteria=criteria,
        )
        print("  " + lamp.summary())

    # The provenance records --quick, because this script overwrites the study's atlas files:
    # a coarse grid written over the published one would otherwise be invisible.
    stamp = provenance({"quick": bool(args.quick)})
    atlas_path = write_json(
        {"provenance": stamp, "atlas": atlas.to_dict()},
        "redlamp_atlas.json",
        directory=args.results,
    )
    demo["provenance"] = stamp
    demo_path = write_json(demo, "redlamp_demo.json", directory=args.results)
    write_csv(demo["rows"], "redlamp_demo.csv", directory=args.results)

    figures = args.figures or (Path(__file__).resolve().parent.parent / "paper" / "figures")
    figures.mkdir(parents=True, exist_ok=True)
    fig = figure6_redlamp_atlas(
        figures / "fig6_redlamp_atlas.png",
        {"provenance": stamp, "atlas": atlas.to_dict()},
    )
    print(f"\nwrote {atlas_path}\n      {demo_path}\n      {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
