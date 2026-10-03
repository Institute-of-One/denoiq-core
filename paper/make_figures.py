"""Draw Figures 1-8 from ``results/`` into ``paper/figures/``.

    python paper/make_figures.py                # refresh the console record, then Figures 1-8
    python paper/make_figures.py --no-console   # Figures 1-7 (and 8 from the existing record)

Figures 1-7 measure nothing: run the sweeps first (``python -c "from denoiq_core.experiment
import run_all; run_all()"`` or the example scripts). Figure 8 is drawn by
:func:`denoiq_core.figures.figure8_console_panel` from ``results/redlamp_console.json`` and the
tiles in ``figures/console/``; ``paper/redlamp_console.py`` writes both, together with the
interactive HTML console. That script needs the ``[dl]`` extra and a trained checkpoint, so it
is attempted first and skipped with a note when either is missing — in which case Figure 8 is
still drawn if a previous record is present, and omitted if it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from denoiq_core.figures import make_all_figures

PAPER_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PAPER_DIR))
import figure_sources  # noqa: E402  (path is set just above)

#: What each figure is drawn from, so a change to any of it marks the figure stale. The results
#: are named here rather than returned by :func:`make_all_figures`, so :func:`record_sources`
#: checks that this list and that function still agree about which figures exist.
FIGURE_RESULTS = {
    "fig1": (),  # drawn from the sweep configuration, not from a results file
    "fig2": ("dose_sweep.json",),
    "fig3": ("dose_sweep.json",),
    "fig4": ("endpoints.json", "statistics.json"),
    "fig5": ("statistics.json",),
    "fig6": ("dose_sweep.json", "texture_sweep.json", "signal_sweep.json", "redlamp_demo.json"),
    "fig7": ("redlamp_atlas.json",),
    "fig8": ("statistics.json",),
    "figS1": ("redlamp_console.json",),
}

#: The code that decides what the figure says. Recording it is what makes an edit to a legend or
#: an axis label -- a spelling change, say -- show up as a stale figure rather than as a figure
#: that disagrees with the manuscript in print.
DRAWING_CODE = PAPER_DIR.parent / "denoiq_core" / "figures.py"


def record_sources(written: dict[str, Path], results: Path) -> None:
    """Record each drawn figure's results and drawing code."""
    unknown = sorted(set(written) - set(FIGURE_RESULTS) - {"provenance"})
    if unknown:
        raise SystemExit(
            f"make_all_figures now writes {unknown}, which FIGURE_RESULTS does not describe; "
            "add them there so the staleness check covers them"
        )
    for name, path in written.items():
        if name == "provenance":
            continue
        sources = [results / each for each in FIGURE_RESULTS[name]]
        figure_sources.record(path, results=sources, code=[DRAWING_CODE])


def refresh_console(results: Path, figures: Path) -> bool:
    """Re-measure the console record if the optional deep-learning path is available."""
    checkpoint = PAPER_DIR.parent / "checkpoints" / "cnn.pt"
    try:
        import denoiq_core.cnn as cnn

        ready = cnn.torch_available() and checkpoint.exists()
    except Exception:  # noqa: BLE001 - any import failure just means "no console here"
        ready = False

    if not ready:
        print(
            "console record: not refreshed — needs the [dl] extra and checkpoints/cnn.pt "
            "(Figure 8 will use the existing results/redlamp_console.json, if any)"
        )
        return False

    from redlamp_console import main as console_main

    console_main(["--results", str(results), "--figures", str(figures)])
    return True


def render_stack() -> dict[str, str]:
    """The versions that decide the bytes of a PNG.

    matplotlib and FreeType lay out and rasterise the text; a different version of either
    produces a visually identical figure with different bytes. Recording them is what lets a
    byte comparison against a committed figure mean "this figure is stale" rather than "this
    machine is not the machine that drew it".
    """
    import matplotlib
    from matplotlib import ft2font

    return {
        "matplotlib": matplotlib.__version__,
        "freetype": ft2font.__freetype_version__,
    }


def write_provenance(figures: Path) -> Path:
    """Write ``PROVENANCE.json`` next to the committed figures."""
    path = figures / "PROVENANCE.json"
    path.write_text(json.dumps(render_stack(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=PAPER_DIR.parent / "results")
    parser.add_argument("--figures", type=Path, default=PAPER_DIR / "figures")
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="do not re-measure the console record before drawing (Figure 8 uses the existing one)",
    )
    args = parser.parse_args(argv)

    if not args.no_console:
        refresh_console(args.results, args.figures)

    written = make_all_figures(results=args.results, figures=args.figures)
    record_sources(written, args.results)
    written["provenance"] = write_provenance(args.figures)
    for name, path in written.items():
        print(f"{name}: {path}")
    if "fig8" not in written:
        print("fig8: skipped — no results/redlamp_console.json or figures/console/ tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
