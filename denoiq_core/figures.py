r"""Figures 1-8, built from ``results/`` and nothing else.

Every number plotted here was computed by :mod:`denoiq_core.experiment` and written to
``results/``. This module reads those files and draws them; it does not measure anything.
The exceptions are the figures that show **example images**: Figures 1 and 7 regenerate theirs
from the same seeded configuration the sweeps used, and Figure 8 reads the tiles that
``paper/redlamp_console.py`` wrote. An image is not a number, and storing image stacks in
``results/`` would serve nothing — but every number annotating those images still comes from
``results/``.

Determinism
-----------
The backend is fixed to Agg, the style is set explicitly rather than inherited from a user's
matplotlibrc, and no figure draws a random number. Re-running produces byte-identical PNGs,
which ``tests/test_determinism.py`` checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from denoiq_core.denoisers import denoise  # noqa: E402
from denoiq_core.experiment import DEFAULT_SWEEP, SweepConfig, results_dir  # noqa: E402
from denoiq_core.physics import acquisition_params, make_trials  # noqa: E402

__all__ = [
    "figure1_pipeline",
    "figure2_fidelity_vs_dose",
    "figure3_task_vs_dose",
    "figure4_divergence",
    "figure5_observer_dependence",
    "figure6_dpi_ceiling",
    "figure7_redlamp_atlas",
    "figure8_floor_strata",
    "figureS1_console_panel",
    "load_results",
    "make_all_figures",
]

#: Green / amber / red, in that order — used for the atlas shading and the lamp markers.
LEVEL_COLOURS = ("#2e7d32", "#f9a825", "#c62828")

_STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
}


def load_results(name: str, directory: Path | None = None) -> dict[str, Any]:
    """Read one results file. Raises if it is missing — a figure without its data is a lie."""
    path = (directory or results_dir()) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist: run denoiq_core.experiment.run_all() (or "
            "examples/run_denoise_taskbench.py) before drawing figures"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a PNG with no software/timestamp metadata, so the bytes are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _denoiser_order(rows: list[dict[str, Any]]) -> list[str]:
    """Denoiser labels with the raw condition first, then the rest in first-seen order."""
    seen: list[str] = []
    for row in rows:
        if row["denoiser"] not in seen:
            seen.append(row["denoiser"])
    raw = [d for d in seen if d == "raw"]
    return raw + [d for d in seen if d != "raw"]


def _series(
    rows: list[dict[str, Any]], denoiser: str, x: str, y: str
) -> tuple[np.ndarray, np.ndarray]:
    subset = sorted((row for row in rows if row["denoiser"] == denoiser), key=lambda r: float(r[x]))
    return (
        np.array([float(row[x]) for row in subset]),
        np.array([float(row[y]) for row in subset]),
    )


# --------------------------------------------------------------------------------------
# Figure 1 — the pipeline, and what the images look like
# --------------------------------------------------------------------------------------


def figure1_pipeline(
    path: Path,
    *,
    config: SweepConfig = DEFAULT_SWEEP,
    kv: float = 120.0,
    mas: float = 25.0,
    seed: int = 20260101,
) -> Path:
    """Representative signal-present / signal-absent images through the pipeline.

    One row per stage — the noise-free object, the unprocessed input, then each classical
    denoiser — with signal-present and signal-absent side by side and **one common display
    window** for every panel, stated in the title. The processed rows look better than the
    input in every case, which is precisely why looking at them is not evidence about
    detectability. The optional CNN is not shown here: it is evaluated separately (Section 3.5),
    and this figure is built from the classical panel that the 76-arm analysis uses.
    """
    trials = make_trials(kv, mas, n_trials=8, seed=seed, model=config.model, phantom=config.phantom)
    stages: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "object\n(noise-free)",
            (config.phantom.background + trials.signal)[None, ...],
            np.full((1, *trials.shape), config.phantom.background),
        ),
        ("unprocessed\ninput", trials.present, trials.absent),
    ]
    for spec in config.denoisers:
        if spec.method == "none":
            continue
        params = spec.resolve(trials.noise_sd)
        stages.append(
            (
                spec.name,
                denoise(trials.present, spec.method, **params),
                denoise(trials.absent, spec.method, **params),
            )
        )

    with plt.rc_context(_STYLE):
        # One column per stage, not one row. Stacked vertically five stages came to
        # 11.9 in, taller than the page they print on, so the figure was cut in half and
        # its caption stranded overleaf.
        fig, axes = plt.subplots(
            2, len(stages), figsize=(1.5 * len(stages), 3.7), constrained_layout=True
        )
        # One window for every row, set by the *lesion contrast* rather than by the noise:
        # a window wide enough to contain low-dose noise renders the lesion invisible
        # everywhere, including in the truth panel. The raw row therefore saturates, which is
        # the honest depiction — that noise really is several times the contrast.
        window = 2.5 * float(np.ptp(trials.signal))
        centre = float(config.phantom.background)
        for col, (name, present, absent) in enumerate(stages):
            for row, stack in enumerate((present, absent)):
                ax = axes[row, col]
                ax.imshow(
                    stack[0],
                    cmap="gray",
                    vmin=centre - window,
                    vmax=centre + window,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(name.replace("\n", " "), fontsize=8)
                if col == 0:
                    ax.set_ylabel(
                        "signal\npresent" if row == 0 else "signal\nabsent", fontsize=8
                    )
        fig.suptitle(
            f"{kv:g} kV, {mas:g} mAs  ·  noise SD {trials.noise_sd:.0f}, "
            f"lesion contrast {np.max(np.abs(trials.signal)):.0f}\n"
            f"common display window ±{window:.0f} (the unprocessed row therefore saturates)",
            fontsize=9,
        )
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 2 — fidelity against dose
# --------------------------------------------------------------------------------------


def figure2_fidelity_vs_dose(path: Path, dose: dict[str, Any]) -> Path:
    """SSIM and PSNR against relative dose, per denoiser — the picture-quality view.

    Read on its own, this figure says denoising works and works best where the dose is
    lowest. Figure 3 is the same conditions measured by the task.
    """
    rows = dose["rows"]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for name in _denoiser_order(rows):
            for ax, metric, label in (
                (axes[0], "ssim", "SSIM"),
                (axes[1], "psnr", "PSNR (dB)"),
            ):
                x, y = _series(rows, name, "relative_dose", metric)
                ax.plot(x, y, marker="o", label=name)
                ax.set_xscale("log")
                ax.set_xlabel("relative dose (mAs / mAs$_{ref}$)")
                ax.set_ylabel(label)
        axes[0].legend(fontsize=7)
        fig.suptitle("Fidelity to the noise-free truth", fontsize=9)
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 3 — task performance against dose, with the ceiling
# --------------------------------------------------------------------------------------


def figure3_task_vs_dose(path: Path, dose: dict[str, Any]) -> Path:
    """Estimated detectability against dose per observer, with the input's analytic ceiling.

    The dashed line is the analytic ideal linear (prewhitening) observer on the **unprocessed
    input**: the ceiling. The three panels are the estimated observers, all held out. Nothing
    crosses the ceiling in any panel at any dose — Figure 6 states the same thing as a scatter.
    """
    rows = dose["rows"]
    order = _denoiser_order(rows)
    with plt.rc_context(_STYLE):
        # 9.6 in reduces to 0.67 in a text column and takes the tick labels with it.
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), constrained_layout=True, sharey=True)
        for ax, observer, title in zip(
            axes,
            ("ideal", "cho", "npwe"),
            (
                "held-out prewhitening linear",
                "CHO (Laguerre-Gauss channels)",
                "non-prewhitening + eye filter",
            ),
            strict=False,
        ):
            for name in order:
                x, y = _series(rows, name, "relative_dose", f"d_prime_{observer}")
                ax.plot(x, y, marker="o", label=name)
            x, ceiling = _series(rows, order[0], "relative_dose", "ceiling_d_prime")
            ax.plot(x, ceiling, "k--", lw=1.2, label="analytic ceiling (unprocessed input)")
            ax.axhline(5.0, color="0.4", lw=0.9, ls=":", label="prespecified floor ($d'=5$, Rose)")
            ax.set_xscale("log")
            ax.set_xlabel("relative dose (mAs / mAs$_{ref}$)")
            ax.set_title(title, fontsize=9)
        axes[0].set_ylabel("estimated task $d'$")
        # Below the panels, not inside the left one: in the left panel the entries sat on
        # the rising curves and on the ceiling line they name.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5)
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 6 — the ceiling itself
# --------------------------------------------------------------------------------------


def figure6_dpi_ceiling(path: Path, sweeps: list[dict[str, Any]]) -> Path:
    """Held-out prewhitening AUC after processing against the input's analytic ceiling.

    Every point lies on or below the diagonal, as it must. Note what the diagonal does and does
    not mean: a *true* ideal observer would preserve performance under invertible processing and
    would therefore sit on it, but the observer plotted here is an estimated linear one, which
    need not attain that equality after a non-linear transformation. The error bars are the
    Hanley-McNeil standard error of the plotted AUC; the ceiling is analytic and carries no
    sampling error of its own.
    """
    rows = [row for sweep in sweeps for row in sweep["rows"]]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
        lo = min(min(r["ceiling_auc"] for r in rows), min(r["auc_ideal"] for r in rows))
        lo = max(0.45, lo - 0.02)
        ax.plot(
            [lo, 1.0],
            [lo, 1.0],
            "k--",
            lw=1.0,
            label="identity (equality requires a true ideal observer)",
        )
        for name in sorted({row["denoiser"] for row in rows}):
            subset = [row for row in rows if row["denoiser"] == name]
            ax.errorbar(
                [row["ceiling_auc"] for row in subset],
                [row["auc_ideal"] for row in subset],
                yerr=[row["auc_se_ideal"] for row in subset],
                fmt="o",
                ms=4,
                lw=0.8,
                alpha=0.85,
                label=name,
            )
        ax.set_xlim(lo, 1.005)
        ax.set_ylim(lo, 1.005)

        # Inset over the crowded top-right corner, where the ceiling is close to 1 and the rank
        # statistic runs out of resolution. The saturation boundary is drawn so that the reader
        # can see which points the saturated-subset analysis of the manuscript excludes.
        inset = ax.inset_axes((0.50, 0.155, 0.46, 0.385))
        inset.set_facecolor("white")
        inset.set_zorder(5)  # the parent's identity line must not show through the panel
        for name in sorted({row["denoiser"] for row in rows}):
            subset = [row for row in rows if row["denoiser"] == name and row["ceiling_auc"] >= 0.95]
            if not subset:
                continue
            inset.errorbar(
                [row["ceiling_auc"] for row in subset],
                [row["auc_ideal"] for row in subset],
                yerr=[row["auc_se_ideal"] for row in subset],
                fmt="o",
                ms=3,
                lw=0.7,
                alpha=0.85,
            )
        inset.plot([0.95, 1.0], [0.95, 1.0], "k--", lw=0.9)
        saturation = 1.0 - 10.0 / (rows[0]["n_trials"] ** 2)
        inset.axvline(saturation, color="0.45", lw=0.9, ls=":")
        inset.text(
            0.985,
            0.04,
            "saturated →",
            transform=inset.transAxes,
            fontsize=6,
            color="0.35",
            va="bottom",
            ha="right",
        )
        inset.set_xlim(0.95, 1.002)
        inset.set_ylim(0.95, 1.002)
        inset.tick_params(labelsize=6)
        inset.set_title("unsaturated range", fontsize=7).set_bbox(
            {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
        )
        inset.grid(alpha=0.2)
        # The "raw" series is below the diagonal too, and it should be: it is the same
        # held-out estimator applied to unprocessed images, so its distance from the line is
        # the price of estimating an observer from a finite sample — not information lost.
        ax.text(
            0.98,
            0.03,
            "the unprocessed series measures the estimator's\nown cost, not a loss of information",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="0.35",
        )
        ax.set_xlabel("analytic ideal-observer AUC of the unprocessed input (ceiling)")
        ax.set_ylabel("held-out prewhitening AUC\nafter processing")
        ax.set_title("No processing exceeds its input's ceiling", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 7 — the red-lamp atlas
# --------------------------------------------------------------------------------------


def figure7_redlamp_atlas(path: Path, atlas_payload: dict[str, Any]) -> Path:
    """The kV-mAs plane: detectability contours, the operational floor, and the lamp bands.

    The floor is the contour where the **unprocessed input's** analytic ideal ``d'`` crosses the
    prespecified criterion (Rose, ``d' = 5`` by default). It is a task requirement, not a
    zero-information boundary: conditions below it retain measurable task information but no
    longer meet the requirement, and — this is the part processing cannot argue with — no
    post-processing can bring them back above it.

    The three bands are named in words as well as coloured, and their two boundaries differ in
    both colour and line style, so nothing here depends on colour vision or on colour printing.
    """
    atlas = atlas_payload["atlas"]
    kv = np.array(atlas["kv"], dtype=np.float64)
    mas = np.array(atlas["mas"], dtype=np.float64)
    d_prime = np.array(atlas["d_prime"], dtype=np.float64)
    floor_mas = np.array(atlas["floor_mas"], dtype=np.float64)
    threshold = float(atlas["threshold"])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
        amber_factor = float(atlas_payload["atlas"].get("amber_factor", 1.2))
        # The bands are filled between the *exact* contours rather than shaded per grid cell:
        # cell shading drew a staircase that could be mistaken for the floor itself, when the
        # floor is a smooth closed-form curve.
        marginal_top = floor_mas * amber_factor**2
        ax.fill_between(kv, marginal_top, mas.max(), color=LEVEL_COLOURS[0], alpha=0.16)
        ax.fill_between(kv, floor_mas, marginal_top, color=LEVEL_COLOURS[1], alpha=0.20)
        ax.fill_between(kv, mas.min(), floor_mas, color=LEVEL_COLOURS[2], alpha=0.14)
        contours = ax.contour(
            kv, mas, d_prime, levels=[1, 2, 3, 5, 8, 12], colors="0.25", linewidths=0.8
        )
        ax.clabel(contours, fmt="$d'$=%g", fontsize=7)
        ax.plot(
            kv,
            floor_mas,
            color="#c62828",
            lw=2.2,
            label=f"operational floor: input $d'={threshold:g}$ (Rose)",
        )
        ax.plot(
            kv,
            marginal_top,
            color="#f9a825",
            lw=1.4,
            ls="--",
            label=f"marginal band: input $d'={amber_factor * threshold:g}$",
        )
        # Redundant encoding without hatching: the two boundaries differ in colour *and* line
        # style, and each band is named in words. Area hatching was tried and buried the d'
        # contour labels, which are the quantitative content of the figure.
        mid = len(kv) // 4
        floor_here = float(floor_mas[mid])
        bands = (
            ("meets requirement", np.sqrt(floor_here * amber_factor**2 * mas.max()), "#1b5e20"),
            ("marginal", floor_here * amber_factor, "#8d6e00"),
            ("below requirement", np.sqrt(floor_here * mas.min()), "#8e1c1c"),
        )
        for text, y, colour in bands:
            ax.text(
                kv[mid],
                y,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )
        ax.set_yscale("log")
        ax.set_xlabel("tube voltage (kV)")
        ax.set_ylabel("tube current-time product, mAs (dose $\\propto$ mAs at fixed kV)")
        # `shading="nearest"` pads half a cell beyond the grid; clip back to the swept range
        # so the shading cannot suggest the atlas covers settings it never evaluated.
        ax.set_xlim(kv.min(), kv.max())
        ax.set_ylim(mas.min(), mas.max())
        ax.set_title(
            "Detectability atlas of the unprocessed input, and the operational floor\n"
            "(normalised relative model — not a scanner calibration)",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85, frameon=True)
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 4 (primary) — fidelity-task divergence over every evaluation
# --------------------------------------------------------------------------------------


def figure4_divergence(path: Path, endpoints: dict[str, Any], statistics: dict[str, Any]) -> Path:
    """ΔSSIM against Δd'(PW) for every arm-realisation evaluation: the first endpoint.

    One point per processed arm per realisation, coloured by denoiser, with the divergent
    quadrant shaded and the clustered correlation and divergence rate printed on the axes.
    Realisation structure is shown two ways: the marginal spread of the per-realisation
    divergence rates as a rug on the right, and the cluster-bootstrap interval in the annotation
    — a single scatter of pooled points would otherwise suggest more independent evidence than
    ten realisations provide.
    """
    rows = endpoints["rows"]
    divergence = statistics["divergence"]
    denoisers = sorted({row["denoiser"] for row in rows})

    with plt.rc_context(_STYLE):
        fig, (ax, rug) = plt.subplots(
            1, 2, figsize=(6.9, 4.6), width_ratios=[6, 1], constrained_layout=True
        )
        xs = [row["delta_ssim"] for row in rows]
        ys = [row["delta_d_pw"] for row in rows]
        span_x = (min(xs), max(xs))
        span_y = (min(ys), max(ys))
        ax.fill_between(
            [0.0, span_x[1] * 1.05],
            0.0,
            span_y[0] * 1.05,
            color=LEVEL_COLOURS[2],
            alpha=0.07,
            zorder=0,
        )
        for name in denoisers:
            subset = [row for row in rows if row["denoiser"] == name]
            ax.scatter(
                [row["delta_ssim"] for row in subset],
                [row["delta_d_pw"] for row in subset],
                s=9,
                alpha=0.55,
                linewidths=0,
                label=name,
            )
        ax.axhline(0.0, color="0.3", lw=1.0)
        ax.axvline(0.0, color="0.3", lw=1.0)
        rho = divergence["spearman_delta_ssim_delta_d_pw"]
        rate = divergence["divergence_rate"]
        ax.text(
            0.98,
            0.03,
            f"divergent: {rate['value']:.1%}\n[{rate['ci_low']:.1%}, {rate['ci_high']:.1%}]\n"
            f"Spearman ρ = {rho['value']:.2f}\n[{rho['ci_low']:.2f}, {rho['ci_high']:.2f}]",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#7a1f1f",
        )
        ax.set_xlabel(r"$\Delta$SSIM (processed $-$ unprocessed input)")
        ax.set_ylabel("$\\Delta d'$, cross-fitted prewhitening observer\n(processed $-$ input)")
        ax.set_title("Fidelity-task divergence", fontsize=9.5)
        ax.legend(fontsize=7, loc="lower left", markerscale=1.6)

        seeds = sorted({row["seed"] for row in rows})
        per_seed = [
            float(np.mean([row["divergent"] for row in rows if row["seed"] == seed]))
            for seed in seeds
        ]
        rug.scatter([0.0] * len(per_seed), per_seed, marker="_", s=260, color="#7a1f1f")
        rug.set_xlim(-0.5, 0.5)
        rug.set_xticks([])
        rug.set_ylim(0.0, 1.0)
        rug.set_ylabel("divergence rate per realisation", fontsize=8)
        rug.set_title(f"{len(seeds)} realisations", fontsize=8)
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 5 (primary) — observer-dependent benefit
# --------------------------------------------------------------------------------------


def figure5_observer_dependence(path: Path, statistics: dict[str, Any]) -> Path:
    """Δd' per observer per denoiser, and the observer-dependent benefit, with intervals.

    The second endpoint, drawn as a forest plot: the same processing on the same images, read
    by observers of different efficiency. The point is the ordering — the efficient observer
    gains least and the inefficient one most — and the intervals are cluster-bootstrap over
    realisations, so they answer "would another ten realisations say the same".
    """
    observers = statistics["observer_dependence"]
    denoisers = sorted(observers["by_denoiser"])
    series = (
        ("delta_d_pw", "prewhitening (PW)", "#1f77b4"),
        ("delta_d_cho", "channelised Hotelling (CHO)", "#8c564b"),
        ("delta_d_npwe", "non-prewhitening (NPWE)", "#ff7f0e"),
        ("benefit", "benefit $B$ = NPWE $-$ PW", "#2e7d32"),
    )

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6.9, 4.1), constrained_layout=True)
        offsets = np.linspace(0.3, -0.3, len(series))
        for (key, label, colour), offset in zip(series, offsets, strict=True):
            positions, values, lows, highs = [], [], [], []
            for index, name in enumerate(denoisers):
                entry = observers["by_denoiser"][name][key]
                positions.append(index + offset)
                values.append(entry["value"])
                lows.append(entry["value"] - entry["ci_low"])
                highs.append(entry["ci_high"] - entry["value"])
            ax.errorbar(
                values,
                positions,
                xerr=[lows, highs],
                fmt="o",
                ms=5,
                lw=1.4,
                capsize=2.5,
                color=colour,
                label=label,
            )
        ax.axvline(0.0, color="0.3", lw=1.0)
        ax.set_yticks(range(len(denoisers)))
        ax.set_yticklabels(denoisers)
        ax.set_ylim(-0.6, len(denoisers) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel("$\\Delta d'$ (processed $-$ unprocessed input), mean with 95 % CI")
        ax.set_title(
            "Denoising benefit depends on observer efficiency\n"
            "(cluster bootstrap over realisations)",
            fontsize=9.5,
        )
        # Below the axes: any in-axes corner sits on top of one of the four series.
        ax.legend(
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=4,
            frameon=False,
            columnspacing=1.4,
        )
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Figure 8 (primary) — failure patterns across the floor
# --------------------------------------------------------------------------------------


def figure8_floor_strata(path: Path, statistics: dict[str, Any]) -> Path:
    """Failure patterns by floor stratum: the third endpoint, quantitatively.

    Four measures — divergence, erasure, excess lesion-like responses and task degradation —
    against the detectability available in the input. The floor stratifies; it is not claimed to
    cause. Intervals are cluster-bootstrap over realisations.
    """
    strata = statistics["floor_strata"]
    order = [name for name in ("above floor", "marginal", "below floor") if name in strata]
    measures = (
        ("divergence_rate", "divergent", True),
        ("erasure_rate", "contrast erasure", True),
        ("excess_response_rate", "excess lesion-like\nresponses", True),
        ("task_degradation", "mean task\ndegradation", True),
    )
    colours = {
        "above floor": LEVEL_COLOURS[0],
        "marginal": LEVEL_COLOURS[1],
        "below floor": LEVEL_COLOURS[2],
    }

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(measures), figsize=(7.4, 3.4), constrained_layout=True)
        for ax, (key, title, as_percent) in zip(axes, measures, strict=True):
            for index, name in enumerate(order):
                entry = strata[name][key]
                ax.bar(
                    index,
                    entry["value"],
                    color=colours[name],
                    alpha=0.75,
                    yerr=[[entry["value"] - entry["ci_low"]], [entry["ci_high"] - entry["value"]]],
                    capsize=3,
                    error_kw={"lw": 1.0},
                )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([name.replace(" ", "\n") for name in order], fontsize=7.5)
            ax.set_title(title, fontsize=8.5)
            if as_percent:
                ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
            ax.margins(y=0.18)
        axes[0].set_ylabel("fraction of evaluations")
        fig.suptitle(
            "Failure patterns by the detectability available in the input\n"
            "(strata of the operational floor; 95 % CI, cluster bootstrap over realisations)",
            fontsize=9.5,
        )
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# Supplementary Figure S1 — the console, as a figure rather than a screenshot
# --------------------------------------------------------------------------------------


def figureS1_console_panel(path: Path, console: dict[str, Any], tiles: Path) -> Path:
    """The red-lamp console at journal text width, drawn from ``results/redlamp_console.json``.

    ``paper/redlamp_console.py`` also writes a full HTML console, which is the interactive
    artefact and reads well on screen; at a single text column it does not. This figure carries
    the same three scenes and the same numbers, laid out to be read on paper — and, unlike a
    screenshot of the HTML, it is regenerated from the results file every time, so it cannot
    fall out of step with the measurements or with the wording of the verdicts.

    Each column is one acquisition setting: the unprocessed input, the processed image, the two
    verdicts, and the reason the processing earned.
    """
    import textwrap

    import matplotlib.image as mpimg

    scenes = console["scenes"]
    rose = float(console["rose_threshold"])
    processed_name = str(console["denoiser"]).upper()

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(6.9, 6.4), constrained_layout=True)
        grid = fig.add_gridspec(3, len(scenes), height_ratios=[1.0, 1.0, 1.25])

        for col, scene in enumerate(scenes):
            for row, (kind, key) in enumerate(
                (("unprocessed input", "raw"), (processed_name, "proc"))
            ):
                ax = fig.add_subplot(grid[row, col])
                tile = tiles / f"{scene['name']}_{key}.png"
                if not tile.exists():
                    raise FileNotFoundError(
                        f"{tile} is missing: run `python paper/redlamp_console.py` to write the "
                        "console tiles before drawing Figure 8"
                    )
                ax.imshow(mpimg.imread(str(tile)))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(
                        f"{scene['kv']:.0f} kV · {scene['mas']:.0f} mAs\n"
                        f"(relative dose ×{scene['relative_dose']:.2f})",
                        fontsize=9,
                    )
                ax.set_xlabel(kind, fontsize=8.5)

            level = scene["dose_level"]
            proc_level = scene["proc_level"]
            reason = scene["proc_reason"].split(": ", 1)[-1]
            ax = fig.add_subplot(grid[2, col])
            ax.axis("off")
            ax.text(
                0.0,
                1.0,
                f"input: {level.upper()}",
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color=LEVEL_COLOURS[["green", "amber", "red"].index(level)],
            )
            ax.text(
                0.0,
                0.88,
                f"processed: {proc_level.upper()}",
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color=LEVEL_COLOURS[["green", "amber", "red"].index(proc_level)],
            )
            body = (
                f"input $d'$ {scene['ceiling_d_prime']:.2f} (floor {rose:.0f})\n"
                f"SSIM {scene['raw_ssim']:.2f} → {scene['proc_ssim']:.2f}\n"
                f"task $d'$ {scene['raw_task_d_prime']:.2f} → {scene['proc_task_d_prime']:.2f}\n"
                f"contrast recovered {scene['proc_contrast_recovery']:.0%}\n"
                f"lesion-like responses {scene['proc_false_structure_rate']:.1%}"
            )
            ax.text(0.0, 0.74, body, transform=ax.transAxes, va="top", fontsize=8.5)
            ax.text(
                0.0,
                0.24,
                textwrap.fill(reason, 42),
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                color="0.25",
            )

        fig.suptitle(
            "Red-lamp console: the input's detectability against the operational floor,\n"
            "and what the processing did to it (every value from results/redlamp_console.json)",
            fontsize=9.5,
        )
        return _save(fig, path)


# --------------------------------------------------------------------------------------
# all of them
# --------------------------------------------------------------------------------------


def make_all_figures(
    *,
    results: Path | None = None,
    figures: Path | None = None,
    config: SweepConfig = DEFAULT_SWEEP,
) -> dict[str, Path]:
    """Draw the manuscript's Figures 1-8 and the supplementary figures into ``paper/figures/``.

    Figures 4, 5 and 8 are the primary-endpoint figures and need the multi-realisation analysis
    (``results/endpoints.json`` and ``results/statistics.json``); Figure S1 needs the console
    record and its tiles. Anything whose inputs are absent is skipped, and its key is missing
    from the returned mapping, so the results-only figures are never blocked by an optional
    path.

    Returns
    -------
    dict
        Figure name to path.

    """
    results = results or results_dir()
    figures = figures or (Path(__file__).resolve().parent.parent / "paper" / "figures")
    figures.mkdir(parents=True, exist_ok=True)

    dose = load_results("dose_sweep.json", results)
    texture = load_results("texture_sweep.json", results)
    signal = load_results("signal_sweep.json", results)
    atlas = load_results("redlamp_atlas.json", results)
    demo = load_results("redlamp_demo.json", results)
    all_sweeps = [dose, texture, signal, demo]

    written = {
        "fig1": figure1_pipeline(figures / "fig1_pipeline.png", config=config),
        "fig2": figure2_fidelity_vs_dose(figures / "fig2_fidelity_vs_dose.png", dose),
        "fig3": figure3_task_vs_dose(figures / "fig3_task_vs_dose.png", dose),
    }

    endpoints_path = results / "endpoints.json"
    statistics_path = results / "statistics.json"
    if endpoints_path.exists() and statistics_path.exists():
        endpoints = json.loads(endpoints_path.read_text(encoding="utf-8"))
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        written["fig4"] = figure4_divergence(figures / "fig4_divergence.png", endpoints, statistics)
        written["fig5"] = figure5_observer_dependence(
            figures / "fig5_observer_dependence.png", statistics
        )
        written["fig8"] = figure8_floor_strata(figures / "fig8_floor_strata.png", statistics)

    written["fig6"] = figure6_dpi_ceiling(figures / "fig6_dpi_ceiling.png", all_sweeps)
    written["fig7"] = figure7_redlamp_atlas(figures / "fig7_redlamp_atlas.png", atlas)

    console_path = results / "redlamp_console.json"
    tiles = figures / "console"
    if console_path.exists() and tiles.exists():
        written["figS1"] = figureS1_console_panel(
            figures / "figS1_redlamp_console.png",
            json.loads(console_path.read_text(encoding="utf-8")),
            tiles,
        )
    return written


def _example_acquisition_note(kv: float, mas: float) -> str:
    """One-line caption fragment describing a setting in relative terms."""
    acq = acquisition_params(kv, mas)
    return (
        f"kV={kv:g}, mAs={mas:g}: relative dose {acq.relative_dose:.3g}, "
        f"noise sd {acq.noise_sd:.1f}, contrast {acq.contrast:.1f}"
    )
