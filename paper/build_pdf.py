"""Render the built manuscript to a review/submission PDF.

    python paper/build_pdf.py            # -> paper/build/manuscript.pdf
    python paper/build_pdf.py --check    # verify build/manuscript.md is current; write nothing

The PDF is a **derivative**, never a source. It is assembled from three things that are
themselves derived from ``results/``:

1. ``paper/build/manuscript.md`` — the built manuscript. Before anything is rendered, this
   script re-renders ``paper/manuscript.md`` and refuses to continue if the built file is out
   of date, so a stale number cannot be baked into a PDF that then gets circulated.
2. **Tables 1 and 2**, generated here from ``results/closed_form.json`` and
   ``results/task_gains.json``. Not typed, for the same reason nothing else in this
   repository is typed.
3. **Figures 1-8** from ``paper/figures/``, with the captions read out of
   ``paper/README.md`` so the two cannot drift apart.

Because it regenerates from text every time, proof-reading can never reach a state that
cannot be rolled back: corrections go into ``paper/manuscript.md``, and the PDF is rebuilt.

Why this renderer
-----------------
Pure Python (reportlab), with the DejaVu Serif/Sans faces that ship inside matplotlib — a
runtime dependency of this package, so the fonts are present wherever the study runs and the
typographic result does not depend on what happens to be installed on the machine. That
matters here specifically because the manuscript contains ``‖ ₂ ² ′ σ ∝ √ Δ``, and a font
without those glyphs renders them as empty boxes.

Pandoc with a LaTeX engine is the conventional route and produces a comparable document from
the same file — ``pandoc paper/build/manuscript.md -o manuscript.pdf`` — but it needs a TeX
installation, and WeasyPrint needs pango/cairo. Neither is a dependency this repository is
willing to require for a derivative artefact, so the PDF build stays out of CI and is
documented as a local step (see ``paper/README.md``).

Determinism
-----------
``rl_config.invariant`` is set, no date is written into the document or its metadata, and the
tables and figures are assembled in a fixed order. The same inputs produce a byte-identical
PDF.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent

DEFAULT_SOURCE = PAPER_DIR / "manuscript.md"
DEFAULT_BUILT = PAPER_DIR / "build" / "manuscript.md"
DEFAULT_RESULTS = REPO_DIR / "results"
DEFAULT_FIGURES = PAPER_DIR / "figures"
DEFAULT_OUTPUT = PAPER_DIR / "build" / "manuscript.pdf"


def document_paths(kind: str) -> tuple[Path, Path, Path]:
    """Return ``(source, built, output)`` for a document kind.

    The two documents share every stage of the build, so the only thing that distinguishes
    them on the command line is ``--document``. Deriving the paths from it here keeps
    ``--document supplementary`` from rendering the manuscript's source, or from writing over
    the manuscript's PDF.
    """
    stem = "manuscript" if kind == "manuscript" else "supplementary"
    return (
        PAPER_DIR / f"{stem}.md",
        PAPER_DIR / "build" / f"{stem}.md",
        PAPER_DIR / "build" / f"{stem}.pdf",
    )


#: The serif used for the JMI layout: metric-compatible with Times, and the only bundled face
#: that covers the Greek letters, primes and superscripts this text needs.
JMI_SERIF = "STIXGeneral"

#: Printed under the author block of the *preprint* layout only. The JMI layout is the
#: submission artefact and carries no reviewer-facing furniture.
REVIEW_NOTE = (
    "Review draft — content of record, built from results/; "
    "format to the SPIE JMI template at submission."
)

#: Figure files per document, in order. Captions come from paper/README.md (see
#: :func:`figure_captions`), so the manuscript and the repository documentation cannot describe
#: a figure differently.
FIGURE_FILES = (
    "fig1_pipeline.png",
    "fig2_fidelity_vs_dose.png",
    "fig3_task_vs_dose.png",
    "fig4_divergence.png",
    "fig5_observer_dependence.png",
    "fig6_dpi_ceiling.png",
    "fig7_redlamp_atlas.png",
    "fig8_floor_strata.png",
)
SUPPLEMENTARY_FIGURE_FILES = ("figS1_redlamp_console.png",)

#: Order Table 2's rows are reported in: weakest smoothing first, so the NPWE column reads as
#: a trend rather than as an unordered list.
DENOISER_ORDER = ("gaussian(1.5 px)", "TV(0.4 sd)", "NLM(0.6 sd)")

# Page geometry (§5 of the proposal): Letter, single column, generous but not wasteful.
PAGE_SIZE = letter
MARGIN_TOP = 20 * mm
MARGIN_SIDE = 19 * mm
MARGIN_BOTTOM = 18 * mm
#: Largest figure height that still leaves room for a caption on one page.
FIGURE_MAX_HEIGHT = 205 * mm


class BuildError(RuntimeError):
    """Something that must be fixed before a PDF may be produced."""


# --------------------------------------------------------------------------------------
# fonts
# --------------------------------------------------------------------------------------


def register_fonts() -> tuple[str, str]:
    """Register the DejaVu families bundled with matplotlib. Returns ``(serif, sans)`` names.

    matplotlib ships DejaVu Serif and Sans in all four styles and is a runtime dependency of
    this package, so this needs no system font lookup and gives the same output everywhere.
    """
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    faces = {
        "DejaVuSerif": (
            "DejaVuSerif.ttf",
            "DejaVuSerif-Bold.ttf",
            "DejaVuSerif-Italic.ttf",
            "DejaVuSerif-BoldItalic.ttf",
        ),
        "DejaVuSans": (
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique.ttf",
        ),
        # STIX is metric-compatible with Times and covers the Greek, arrows and superscripts
        # this text uses; it is what makes a Times-set JMI page possible without a system font.
        "STIXGeneral": (
            "STIXGeneral.ttf",
            "STIXGeneralBol.ttf",
            "STIXGeneralItalic.ttf",
            "STIXGeneralBolIta.ttf",
        ),
    }
    for family, (regular, bold, italic, bold_italic) in faces.items():
        for suffix, filename in (
            ("", regular),
            ("-Bold", bold),
            ("-Italic", italic),
            ("-BoldItalic", bold_italic),
        ):
            path = ttf / filename
            if not path.exists():  # pragma: no cover - would mean a broken matplotlib
                raise BuildError(f"{path} is missing: matplotlib's bundled fonts are required")
            pdfmetrics.registerFont(TTFont(f"{family}{suffix}", str(path)))
        pdfmetrics.registerFontFamily(
            family,
            normal=family,
            bold=f"{family}-Bold",
            italic=f"{family}-Italic",
            boldItalic=f"{family}-BoldItalic",
        )
    return "DejaVuSerif", "DejaVuSans"


# --------------------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------------------


def check_freshness(source: Path, built: Path, results: Path, *, submission: bool = False) -> str:
    """Return the built manuscript, having verified it matches a fresh render.

    Raises
    ------
    BuildError
        If the built file is missing or out of date. A PDF is circulated; a stale number in
        one is far harder to retract than a stale number in a working file.

    """
    sys.path.insert(0, str(PAPER_DIR))
    import build_manuscript

    if not built.exists():
        raise BuildError(f"{built} does not exist — run: python paper/build_manuscript.py")
    try:
        fresh = build_manuscript.render(
            source.read_text(encoding="utf-8"), results, submission=submission
        )
    except build_manuscript.ResolutionError as exc:
        raise BuildError(str(exc)) from exc
    current = built.read_text(encoding="utf-8")
    if current != fresh:
        raise BuildError(
            f"{built} is out of date with respect to {source} and {results}\n"
            "run: python paper/build_manuscript.py"
        )
    return current


# --------------------------------------------------------------------------------------
# markdown -> blocks
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """The parsed manuscript: a title, an author block, and a list of typed blocks."""

    title: str
    authors: list[str]
    blocks: list[tuple[str, Any]]


_COMMENT = re.compile(r"<!--.*?-->", re.S)


def parse_manuscript(text: str) -> Document:
    """Parse the controlled Markdown subset the manuscript uses.

    Recognised: ``#`` title, an author block of plain lines before the first ``##``, ``##``
    and ``###`` headings, paragraphs, numbered list items (including their indented
    continuation lines), and pipe tables. That is the whole vocabulary of
    ``paper/manuscript.md``, and anything else appearing there should extend this function
    rather than be silently flattened.
    """
    text = _COMMENT.sub("", text)
    lines = text.splitlines()

    title = ""
    authors: list[str] = []
    blocks: list[tuple[str, Any]] = []
    para: list[str] = []
    item: list[str] = []
    section = ""

    def flush_para() -> None:
        if para:
            blocks.append(("p", " ".join(para).strip()))
            para.clear()

    def flush_item() -> None:
        if item:
            kind = "ref" if section.lower().startswith("references") else "li"
            blocks.append((kind, " ".join(item).strip()))
            item.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            flush_item()
            flush_para()
            section = stripped[3:].strip()
            blocks.append(("h1", section))
            continue
        if stripped.startswith("### "):
            flush_item()
            flush_para()
            blocks.append(("h2", stripped[4:].strip()))
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_item()
            flush_para()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            # The |---|---| separator row carries no content.
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                if blocks and blocks[-1][0] == "table":
                    blocks[-1][1].append(cells)
                else:
                    blocks.append(("table", [cells]))
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush_item()
            flush_para()
            item.append(stripped)
            continue
        if not stripped:
            flush_item()
            flush_para()
            continue
        if item and raw.startswith("   "):  # continuation of a list item
            item.append(stripped)
            continue
        flush_item()
        if title and not blocks and not para:
            # Between the title and the first heading: the author block.
            authors.append(stripped)
            continue
        para.append(stripped)

    flush_item()
    flush_para()
    return Document(title=title, authors=authors, blocks=blocks)


#: A bracketed citation — ``[10]``, ``[10,11]``, ``[10-12]``. Confidence intervals are also
#: written in brackets, but always carry a decimal point or a percent sign, so requiring bare
#: integers keeps the two apart without a lookup table. The optional leading space is part of
#: the match: a superscript citation sits against the word it follows, not a space away from it.
CITATION = re.compile(r" ?\[(\d{1,2}(?:\s*[,–-]\s*\d{1,2})*)\]")
#: ``Figure 4`` becomes ``Fig. 4`` in SPIE style everywhere except at the start of a sentence,
#: where the word is spelled out — hence the two lookbehinds.
FIGURE_REF = re.compile(r"(?<!^)(?<![.!?] )Figure (\d)")
#: The plural, which occurs in headings and cross-references: "Figures 7-8" -> "Figs. 7-8".
FIGURES_REF = re.compile(r"(?<!^)(?<![.!?] )Figures (\d)")

#: Unicode super- and subscripts, and their ASCII equivalents. The manuscript writes exponents
#: as characters (``4.76×10⁻⁷``) because that is what a plain-text reader should see; a Times
#: face does not necessarily carry those code points, so the JMI layout raises the ASCII digits
#: with real ``<super>`` runs instead of relying on the font.
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉₊₋"
_SUPER_ASCII = str.maketrans(_SUPERSCRIPTS, "0123456789+-")
_SUB_ASCII = str.maketrans(_SUBSCRIPTS, "0123456789+-")
_SUPER_RUN = re.compile(f"[{_SUPERSCRIPTS}]+")
_SUB_RUN = re.compile(f"[{_SUBSCRIPTS}]+")


def typeset_scripts(text: str) -> str:
    """Replace runs of Unicode super/subscript characters with reportlab markup."""
    text = _SUPER_RUN.sub(lambda m: f"<super>{m.group(0).translate(_SUPER_ASCII)}</super>", text)
    return _SUB_RUN.sub(lambda m: f"<sub>{m.group(0).translate(_SUB_ASCII)}</sub>", text)


def inline(text: str, *, style: str = "preprint") -> str:
    """Markdown inline markup to reportlab markup, for the subset the manuscript uses.

    Code spans become **ordinary body text** rather than monospace: in this manuscript they
    hold mathematical notation (``d'``, ``‖s‖₂/σ``, ``1.96``), which reads as mathematics in
    the serif face and as source code in a typewriter one. Their spaces become non-breaking so
    an expression is never split across lines.

    In ``jmi`` style two further journal conventions are applied: numbered citations are set as
    superscripts, and figure references are abbreviated except at the start of a sentence.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<u>\1</u>", text)
    text = re.sub(r"`([^`]+)`", lambda m: m.group(1).replace(" ", " "), text)
    # `x^(n)` becomes a real superscript: an exponent is the one place where the source's ASCII
    # notation would read as a typing error on the page.
    text = re.sub(r"\^\(([^)]+)\)", r"<super>\1</super>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    if style == "jmi":
        text = CITATION.sub(lambda m: f"<super>{m.group(1)}</super>", text)
        text = FIGURES_REF.sub(r"Figs. \1", text)
        text = FIGURE_REF.sub(r"Fig. \1", text)
        text = typeset_scripts(text)
    return text


# --------------------------------------------------------------------------------------
# tables, from results/
# --------------------------------------------------------------------------------------


def scientific(value: float, digits: int) -> str:
    """Journal-style scientific notation, shared with the manuscript builder.

    Imported lazily from ``build_manuscript`` so the two renderings of a number — the one in
    the prose and the one in a table caption — cannot drift apart.
    """
    sys.path.insert(0, str(PAPER_DIR))
    import build_manuscript

    return build_manuscript.scientific(value, digits)


def _load(results: Path, name: str) -> dict[str, Any]:
    path = results / name
    if not path.exists():
        raise BuildError(
            f'{path} is missing: run `python -c "from denoiq_core.experiment import run_all; '
            'run_all()"` before building the PDF'
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(entry: dict, spec: str = "+.2f") -> str:
    """``value [low, high]`` from a bootstrap record."""
    return f"{entry['value']:{spec}} [{entry['ci_low']:{spec}}, {entry['ci_high']:{spec}}]"


def _pct(entry: dict) -> str:
    return f"{entry['value']:.1%} [{entry['ci_low']:.1%}, {entry['ci_high']:.1%}]"


def table1_data(results: Path) -> tuple[list[list[str]], str]:
    """Main Table 1 — the experimental design, counted from the results rather than described."""
    statistics = _load(results, "statistics.json")
    design = statistics["design"]
    labels = {
        "dose": ("Dose sweep", "mAs at fixed kV"),
        "texture": ("Noise-correlation sweep", "noise correlation length"),
        "signal": ("Lesion sweep", "lesion radius and contrast"),
        "atlas_demo": ("Atlas / floor conditions", "kV and mAs on the atlas"),
    }
    header = [
        "domain",
        "varied factor",
        "input conditions",
        "denoisers",
        "unprocessed arms",
        "processed arms",
    ]
    body = []
    for key in ("dose", "texture", "signal", "atlas_demo"):
        if key not in design["sweeps"]:
            continue
        entry = design["sweeps"][key]
        name, factor = labels.get(key, (key, "—"))
        body.append(
            [
                name,
                factor,
                f"{entry['input_conditions']}",
                f"{entry['n_denoisers']}",
                f"{entry['unprocessed_arms']}",
                f"{entry['processed_arms']}",
            ]
        )
    body.append(
        [
            "Total",
            "—",
            f"{design['input_conditions']}",
            "—",
            f"{design['unprocessed_arms']}",
            f"{design['processed_arms']}",
        ]
    )
    caption = (
        "**Table 1.** The experimental matrix. An *input condition* is one acquisition and "
        "phantom setting; an *arm* is one condition with one processing, including the "
        f"unprocessed one. The {design['unique_arms']} unique arms were each evaluated in "
        f"{design['n_seeds']} independent realisations, giving "
        f"{design['arm_seed_evaluations']:,} arm–realisation evaluations and "
        f"{design['scored_image_trials']:,} scored image trials. The three sweeps use "
        f"{design['trials_per_class_main_sweeps']:,} trials per class per arm and the atlas "
        f"settings fewer, so the trial total is summed rather than assumed. Conditions are not "
        "shared between domains, so no deduplication is needed; the atlas domain applies a "
        "single denoiser rather than all three, which is why the processed-arm total is not "
        "three times the number of input conditions. The learned denoiser is evaluated "
        "separately and is not part of this matrix."
    )
    return [header, *body], caption


def table2_data(results: Path) -> tuple[list[list[str]], str]:
    """Main Table 2 — observer-dependent effects, with clustered intervals."""
    statistics = _load(results, "statistics.json")
    observers = statistics["observer_dependence"]
    by_denoiser = observers["by_denoiser"]
    missing = [name for name in DENOISER_ORDER if name not in by_denoiser]
    if missing:
        raise BuildError(f"statistics.json has no observer entry for {missing}")

    header = [
        "denoiser",
        "ΔSSIM",
        "Δd′ PW",
        "Δd′ CHO",
        "Δd′ NPWE",
        "benefit B",
        "p (Holm)",
    ]
    body = []
    for name in DENOISER_ORDER:
        entry = by_denoiser[name]
        divergence = statistics["divergence"]["by_denoiser"][name]
        body.append(
            [
                name,
                _ci(divergence["mean_delta_ssim"], "+.3f"),
                _ci(entry["delta_d_pw"]),
                _ci(entry["delta_d_cho"]),
                _ci(entry["delta_d_npwe"]),
                _ci(entry["benefit"]),
                f"{observers['benefit_p_holm'][name]:.3f}",
            ]
        )
    overall = observers["overall"]
    body.append(
        [
            "All",
            _ci(statistics["divergence"]["mean_delta_ssim"], "+.3f"),
            _ci(overall["delta_d_pw"]),
            _ci(overall["delta_d_cho"]),
            _ci(overall["delta_d_npwe"]),
            _ci(overall["benefit"]),
            "—",
        ]
    )
    caption = (
        "**Table 2.** Observer-dependent effects: processed minus unprocessed at the same "
        "condition and realisation, as mean [95 % CI] from a bootstrap resampling whole "
        "realisations. PW is the cross-fitted prewhitening observer, CHO the channelised "
        "Hotelling observer, NPWE the non-prewhitening observer with an eye filter — a "
        "stylised surrogate for limited prewhitening efficiency, not a human-reader model. "
        "The benefit is B = Δd′(NPWE) − Δd′(PW); p-values are two-sided bootstrap values for "
        "B, Holm-adjusted across the three denoisers. None of these columns is the analytic "
        "ceiling, which is a property of the unprocessed input and is unchanged by processing."
    )
    return [header, *body], caption


def table_s1_data(results: Path) -> tuple[list[list[str]], str]:
    """Supplementary Table S1 — closed-form validation of the analytic observer."""
    payload = _load(results, "closed_form.json")
    rows = sorted(payload["rows"], key=lambda r: (r["noise_sd"], r["radius_mm"]))
    header = ["noise σ", "radius (mm)", "contrast", "d′ closed form", "d′ computed", "rel. error"]
    body = [
        [
            f"{row['noise_sd']:g}",
            f"{row['radius_mm']:g}",
            f"{row['contrast']:g}",
            f"{row['d_prime_closed_form']:.4f}",
            f"{row['d_prime_ideal_linear']:.4f}",
            f"{row['relative_error']:.1e}",
        ]
        for row in rows
    ]
    caption = (
        "**Table S1.** The analytic ideal linear (prewhitening) observer against its closed "
        "form on the unprocessed input. Both columns are analytic and neither is estimated "
        "from image samples, so the residual is numerical rather than statistical. Largest "
        f"relative error: {scientific(payload['max_relative_error'], 1)}, against a criterion "
        "of 1 % fixed in advance."
    )
    return [header, *body], caption


def table_s2_data(results: Path) -> tuple[list[list[str]], str]:
    """Supplementary Table S2 — cross-fitting against a single 50/50 split."""
    payload = _load(results, "estimator_sensitivity.json")
    header = ["mAs", "ceiling d′", "cross-fit d′", "recovery", "split d′", "recovery"]
    body = [
        [
            f"{row['mas']:g}",
            f"{row['ceiling_d_prime']:.3f}",
            f"{row['cross_fit_d_prime']:.3f}",
            f"{row['cross_fit_recovery']:.0%}",
            f"{row['split_d_prime']:.3f}",
            f"{row['split_recovery']:.0%}",
        ]
        for row in payload["rows"]
    ]
    caption = (
        "**Table S2.** Estimator sensitivity on unprocessed images of the dose sweep in the "
        f"representative realisation (seed {payload['seed']}): the "
        f"{payload['n_folds']}-fold cross-fitted estimator used throughout against the single "
        "50/50 split used in the earlier single-realisation analysis. Recovery is the estimated "
        "d′ as a fraction of the analytic ceiling of the same input. Recovery slightly above "
        "100 % at the highest doses is sampling noise in the d′ estimate, not a breached "
        "bound: the ceiling test is armwise, in AUC, and carries its own margin (Section 3.4)."
    )
    return [header, *body], caption


def table_s3_data(results: Path) -> tuple[list[list[str]], str]:
    """Supplementary Table S3 — the full floor-stratified comparison."""
    statistics = _load(results, "statistics.json")
    strata = statistics["floor_strata"]
    header = [
        "stratum",
        "n",
        "divergent",
        "erasure",
        "excess resp.",
        "task degr.",
        "benefit B",
        "green / amber / red",
    ]
    body = []
    for name in ("above floor", "marginal", "below floor"):
        if name not in strata:
            continue
        entry = strata[name]
        verdicts = entry["verdicts"]
        body.append(
            [
                name,
                f"{entry['n_arm_seed']}",
                _pct(entry["divergence_rate"]),
                _pct(entry["erasure_rate"]),
                _pct(entry["excess_response_rate"]),
                _pct(entry["task_degradation"]),
                _ci(entry["benefit"]),
                f"{verdicts['green']} / {verdicts['amber']} / {verdicts['red']}",
            ]
        )
    caption = (
        "**Table S3.** Failure patterns by floor stratum, as rate or mean [95 % CI] from the "
        "clustered bootstrap. The columns are the divergence rate, the contrast-erasure rate, "
        "the excess lesion-like response rate (excess resp.), the mean task degradation "
        "(task degr.), the observer-dependent benefit and the gauge verdict counts. "
        "Strata are defined by the analytic ceiling of the *input* "
        "condition against the prespecified requirement; the floor stratifies conditions and "
        "is not claimed to cause the failures. Verdict counts are the gauge's, and below the "
        "floor they are red by construction, since an input that fails the requirement is "
        "itself a red rule."
    )
    return [header, *body], caption


def table_s4_data(results: Path) -> tuple[list[list[str]], str]:
    """Supplementary Table S4 — per-denoiser divergence endpoints."""
    statistics = _load(results, "statistics.json")
    divergence = statistics["divergence"]
    header = ["denoiser", "Spearman ρ", "p (Holm)", "divergent", "mean ΔSSIM", "mean Δd′ PW"]
    body = []
    for name in DENOISER_ORDER:
        entry = divergence["by_denoiser"][name]
        body.append(
            [
                name,
                _ci(entry["spearman"], "+.2f"),
                f"{divergence['spearman_p_holm'][name]:.3f}",
                _pct(entry["divergence_rate"]),
                _ci(entry["mean_delta_ssim"], "+.3f"),
                _ci(entry["mean_delta_d_pw"], "+.2f"),
            ]
        )
    body.append(
        [
            "All",
            _ci(divergence["spearman_delta_ssim_delta_d_pw"], "+.2f"),
            "—",
            _pct(divergence["divergence_rate"]),
            _ci(divergence["mean_delta_ssim"], "+.3f"),
            _ci(divergence["mean_delta_d_pw"], "+.2f"),
        ]
    )
    caption = (
        "**Table S4.** Fidelity–task divergence per denoiser: the Spearman correlation between "
        "ΔSSIM and Δd′(PW), the fraction of divergent evaluations (ΔSSIM > 0 with Δd′(PW) < 0), "
        "and the mean changes, each as value [95 % CI] from the clustered bootstrap. p-values "
        "are two-sided bootstrap values for the correlation, Holm-adjusted across denoisers."
    )
    return [header, *body], caption


#: Which generated tables belong to which document, and the heading text that anchors them.
DOCUMENT_TABLES = {
    "manuscript": {"table 1": table1_data, "table 2": table2_data},
    "supplementary": {
        "table s1": table_s1_data,
        "table s2": table_s2_data,
        "table s3": table_s3_data,
        "table s4": table_s4_data,
    },
}


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def figure_captions(readme: Path, *, supplementary: bool = False) -> list[str]:
    """Figure captions, read from the table in ``paper/README.md``.

    Reading them rather than restating them is the same discipline the numbers follow: there
    is one place a caption is written, so the PDF and the repository documentation cannot
    describe a figure differently.
    """
    text = readme.read_text(encoding="utf-8")
    pattern = (
        r"^\|\s*Fig\s*S(\d+)\s*\|\s*(.+?)\s*\|\s*$"
        if supplementary
        else (r"^\|\s*Fig\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$")
    )
    captions = re.findall(pattern, text, re.M)
    ordered = [caption for _, caption in sorted(captions, key=lambda pair: int(pair[0]))]
    expected = len(SUPPLEMENTARY_FIGURE_FILES if supplementary else FIGURE_FILES)
    if len(ordered) != expected:
        raise BuildError(
            f"{readme} lists {len(ordered)} figure captions, expected {expected}: the PDF takes "
            "its captions from that table"
        )
    return ordered


def png_size(path: Path) -> tuple[int, int]:
    """Pixel size of a PNG, from its IHDR chunk (no image library needed)."""
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise BuildError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def scaled_image(path: Path, max_width: float) -> Image:
    """A figure scaled to the text width, or to :data:`FIGURE_MAX_HEIGHT` if that is binding.

    Figure 1 is a tall column of image panels; without the height cap it would run off the
    page, and reportlab would scale it silently rather than complain.
    """
    pixel_width, pixel_height = png_size(path)
    width = max_width
    height = width * pixel_height / pixel_width
    if height > FIGURE_MAX_HEIGHT:
        height = FIGURE_MAX_HEIGHT
        width = height * pixel_width / pixel_height
    return Image(str(path), width=width, height=height)


# --------------------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Styles:
    """The paragraph styles of §5, built once so the story code stays readable."""

    title: ParagraphStyle
    author: ParagraphStyle
    note: ParagraphStyle
    h1: ParagraphStyle
    h2: ParagraphStyle
    body: ParagraphStyle
    abstract: ParagraphStyle
    listitem: ParagraphStyle
    reference: ParagraphStyle
    caption: ParagraphStyle
    figure_caption: ParagraphStyle


def build_styles(serif: str, sans: str, *, style: str = "preprint") -> Styles:
    """Typography for one of the two layouts.

    ``preprint`` is the working single-column layout: serif body, sans headings. ``jmi`` is the
    submission layout for SPIE's Journal of Medical Imaging — Times-metric serif throughout,
    12 pt on a wide leading, headings set in the same face, and no reviewer-facing furniture.
    The content is identical; only the setting differs, so a number cannot change with a style.
    """
    if style == "jmi":
        return build_jmi_styles(JMI_SERIF)
    body = ParagraphStyle(
        "body",
        fontName=serif,
        fontSize=10.5,
        leading=15.75,  # 1.5 lines
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )
    return Styles(
        title=ParagraphStyle(
            "title",
            fontName=f"{sans}-Bold",
            fontSize=17,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        author=ParagraphStyle(
            "author", parent=body, fontSize=10, leading=13.5, alignment=TA_CENTER, spaceAfter=1
        ),
        note=ParagraphStyle(
            "note",
            parent=body,
            fontName=f"{sans}",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceBefore=8,
            spaceAfter=12,
        ),
        h1=ParagraphStyle(
            "h1",
            fontName=f"{sans}-Bold",
            fontSize=12.5,
            leading=16,
            spaceBefore=14,
            spaceAfter=4,
        ),
        h2=ParagraphStyle(
            "h2",
            fontName=f"{sans}-Bold",
            fontSize=10.5,
            leading=14,
            spaceBefore=10,
            spaceAfter=3,
            textColor=colors.HexColor("#222222"),
        ),
        body=body,
        abstract=ParagraphStyle(
            "abstract", parent=body, fontSize=10, leading=14.5, leftIndent=6, rightIndent=6
        ),
        listitem=ParagraphStyle(
            "listitem", parent=body, leftIndent=16, firstLineIndent=-16, spaceAfter=4
        ),
        reference=ParagraphStyle(
            "reference",
            parent=body,
            fontSize=9,
            leading=12.5,
            alignment=TA_LEFT,
            leftIndent=16,
            firstLineIndent=-16,
            spaceAfter=3,
        ),
        caption=ParagraphStyle(
            "caption",
            parent=body,
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=12,
        ),
        figure_caption=ParagraphStyle(
            "figure_caption",
            parent=body,
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=14,
        ),
    )


def build_jmi_styles(serif: str) -> Styles:
    """SPIE Journal of Medical Imaging submission typography.

    12 pt Times-metric text on 1.9 leading (a wide-spaced review manuscript), section headings
    in bold and subsection headings in bold italic as in SPIE's own class file, and captions
    at 11 pt. Reference entries are hanging-indented and numbered in citation order.
    """
    body = ParagraphStyle(
        "body",
        fontName=serif,
        fontSize=12,
        leading=22.8,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    return Styles(
        title=ParagraphStyle(
            "title",
            fontName=f"{serif}-Bold",
            fontSize=15,
            leading=19,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        author=ParagraphStyle(
            "author", parent=body, fontSize=11.5, leading=15, alignment=TA_CENTER, spaceAfter=2
        ),
        note=ParagraphStyle(
            "note",
            parent=body,
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=14,
        ),
        h1=ParagraphStyle(
            "h1",
            fontName=f"{serif}-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=16,
            spaceAfter=6,
        ),
        h2=ParagraphStyle(
            "h2",
            fontName=f"{serif}-BoldItalic",
            fontSize=12,
            leading=16,
            spaceBefore=12,
            spaceAfter=4,
        ),
        body=body,
        abstract=ParagraphStyle("abstract", parent=body, fontSize=11, leading=16.5, spaceAfter=8),
        listitem=ParagraphStyle(
            "listitem", parent=body, leftIndent=18, firstLineIndent=-18, spaceAfter=4
        ),
        reference=ParagraphStyle(
            "reference",
            parent=body,
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
            leftIndent=18,
            firstLineIndent=-18,
            spaceAfter=4,
        ),
        caption=ParagraphStyle(
            "caption",
            parent=body,
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=10,
        ),
        figure_caption=ParagraphStyle(
            "figure_caption",
            parent=body,
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=16,
        ),
    )


def _fits(rows: list[list[str]], serif: str, width: float) -> bool:
    """Whether a table at its natural column widths stays inside the text block."""
    probe = Table(rows, hAlign="LEFT")
    probe.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), serif),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    natural, _height = probe.wrap(0, 0)
    return bool(natural <= width)


def table_flowable(rows: list[list[str]], serif: str, *, wrap_width: float | None = None) -> Table:
    """A table: header shaded, labels left, numbers right, kept on one page.

    Column widths are left to reportlab for the numeric tables, which are narrower than the
    text block — stretching those to the margins would only put air between the digits. Pass
    ``wrap_width`` for a prose table (the decision rules), where cells must wrap instead of
    running off the page; each cell then becomes a paragraph and the columns share the width.
    """
    # Wrap only when the natural layout would run past the margin: numeric tables read better
    # right-aligned at their natural widths, and a table that fits should keep them.
    if wrap_width is not None and not _fits(rows, serif, wrap_width):
        style = ParagraphStyle("cell", fontName=serif, fontSize=8.5, leading=11, alignment=TA_LEFT)
        header_style = ParagraphStyle("cellhead", parent=style, fontName=f"{serif}-Bold")
        n_columns = max(len(row) for row in rows)
        # First column narrower: it holds short labels, the rest hold sentences.
        first = 0.21 * wrap_width
        rest = (wrap_width - first) / max(n_columns - 1, 1)
        rows = [
            [Paragraph(cell, header_style if index == 0 else style) for cell in row]
            for index, row in enumerate(rows)
        ]
        table = Table(
            rows,
            colWidths=[first] + [rest] * (n_columns - 1),
            hAlign="LEFT",
            repeatRows=1,
        )
    else:
        table = Table(rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), serif),
                ("FONTNAME", (0, 0), (-1, 0), f"{serif}-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#666666")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    if wrap_width is not None:
        # Prose cells read left-aligned and top-aligned; only numeric columns want the right.
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
    return table


# --------------------------------------------------------------------------------------
# assembling the story
# --------------------------------------------------------------------------------------


def build_story(
    document: Document,
    *,
    styles: Styles,
    serif: str,
    results: Path,
    figures: Path,
    readme: Path,
    text_width: float,
    document_kind: str = "manuscript",
    style: str = "preprint",
) -> list[Any]:
    """Turn the parsed manuscript into a reportlab story, inserting the tables and figures.

    Tables go where the text refers to them — the headings that name them are the anchors —
    rather than into an appendix, because a reviewer reading section 3.1 should not have to
    go looking for Table 1.
    """
    story: list[Any] = [Paragraph(document.title, styles.title)]
    for index, line in enumerate(document.authors):
        story.append(Paragraph(inline(line, style=style), styles.author))
        if index == 0:
            story.append(Spacer(1, 1))
    if style == "jmi":
        story.append(Paragraph(correspondence_line(document.authors), styles.note))
    else:
        story.append(Paragraph(REVIEW_NOTE, styles.note))

    generators = DOCUMENT_TABLES[document_kind]
    tables = {name: builder(results) for name, builder in generators.items()}
    figure_files = SUPPLEMENTARY_FIGURE_FILES if document_kind == "supplementary" else FIGURE_FILES
    figure_prefix = "Figure S" if document_kind == "supplementary" else "Figure "
    pending: str | None = None
    in_abstract = False

    for kind, payload in document.blocks:
        if kind == "h1":
            in_abstract = payload.strip().lower() == "abstract"
            story.append(Paragraph(section_heading(payload, style), styles.h1))
            continue
        if kind == "h2":
            heading = payload
            story.append(Paragraph(section_heading(heading, style), styles.h2))
            lowered = heading.lower()
            # The heading may name several artefacts — "(Figure 5, Table 2)" — so the
            # anchor is the table's name anywhere in it, not a parenthesis of its own.
            pending = next((key for key in tables if key in lowered), None)
            continue
        if kind == "p":
            story.append(
                Paragraph(
                    inline(payload, style=style), styles.abstract if in_abstract else styles.body
                )
            )
            if pending:
                rows, caption = tables[pending]
                if style == "jmi":
                    # SPIE labels a table "Table 1", with no period after the number.
                    caption = re.sub(r"^\*\*(Table S?\d+)\.\*\*", r"**\1**", caption)
                table = table_flowable(rows, serif, wrap_width=text_width)
                label = Paragraph(inline(caption, style=style), styles.caption)
                # SPIE sets table captions above the table; the preprint layout keeps them below.
                parts = [Spacer(1, 4), label, table, Spacer(1, 10)]
                if style != "jmi":
                    parts = [Spacer(1, 4), table, label]
                story.append(KeepTogether(parts))
                pending = None
            continue
        if kind == "table":
            # A table written in the manuscript itself (the decision rules of Section 2.6);
            # Tables 1 and 2 are generated from results/ and inserted above.
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 4),
                        table_flowable(
                            [[inline(cell, style=style) for cell in row] for row in payload],
                            serif,
                            wrap_width=text_width,
                        ),
                        Spacer(1, 8),
                    ]
                )
            )
            continue
        if kind == "li":
            story.append(Paragraph(inline(payload, style=style), styles.listitem))
            continue
        if kind == "ref":
            story.append(Paragraph(inline(payload, style=style), styles.reference))
            continue
        raise BuildError(f"unhandled block type {kind!r}")  # pragma: no cover - defensive

    story.append(PageBreak())
    story.append(
        Paragraph(
            "Supplementary figures" if document_kind == "supplementary" else "Figures", styles.h1
        )
    )
    captions = figure_captions(readme, supplementary=document_kind == "supplementary")
    for filename, caption in zip(figure_files, captions, strict=True):
        path = figures / filename
        if not path.exists():
            raise BuildError(
                f"{path} is missing: run `python paper/make_figures.py` before building the PDF"
            )
        number = filename.split("_")[0].removeprefix("fig").removeprefix("S")
        if style == "jmi":
            # SPIE style: "Fig. 4 Caption." — abbreviated, no bold run-in, no period after
            # the number.
            prefix = "Fig. S" if document_kind == "supplementary" else "Fig. "
            text = f"**{prefix}{number}** {caption[0].upper()}{caption[1:]}"
        else:
            text = f"**{figure_prefix}{number}.** {caption[0].upper()}{caption[1:]}"
        if not text.rstrip().endswith("."):
            text = f"{text}."
        story.append(
            KeepTogether(
                [
                    scaled_image(path, text_width),
                    Paragraph(inline(text, style=style), styles.figure_caption),
                ]
            )
        )
    return story


def section_heading(text: str, style: str) -> str:
    """A heading, in the numbering convention of the chosen layout.

    SPIE numbers sections without a trailing period — "1 Introduction", "3.1 Study design" —
    while the manuscript source writes "1." for readability as markdown.
    """
    if style != "jmi":
        return text
    heading = re.sub(r"^(\d+)\.(\s)", r"\1\2", text.strip())
    # A heading is not run through inline(), so the figure convention is applied here as well.
    heading = FIGURES_REF.sub(r"Figs. \1", heading)
    return FIGURE_REF.sub(r"Fig. \1", heading)


def correspondence_line(authors: list[str]) -> str:
    """The SPIE corresponding-author line, built from the manuscript's author block.

    SPIE asks for "Address all correspondence to <name>, <e-mail>". Both are already in the
    author block, so they are read from it rather than configured a second time.
    """
    joined = " ".join(authors)
    name = re.sub(r"[*]", "", authors[0]).strip() if authors else ""
    email = next(iter(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", joined)), "")
    if not name or not email:  # pragma: no cover - the author block always carries both
        return ""
    return f"Address all correspondence to {name}, {email}"


def build_pdf(
    *,
    source: Path = DEFAULT_SOURCE,
    built: Path = DEFAULT_BUILT,
    results: Path = DEFAULT_RESULTS,
    figures: Path = DEFAULT_FIGURES,
    output: Path = DEFAULT_OUTPUT,
    readme: Path | None = None,
    document_kind: str = "manuscript",
    style: str = "jmi",
    submission: bool = False,
) -> Path:
    """Verify freshness, assemble, and write the PDF. Returns the output path."""
    text = check_freshness(source, built, results, submission=submission)
    # Deterministic output: fixed document id, no creation timestamp.
    rl_config.invariant = 1

    serif, sans = register_fonts()
    if style == "jmi":
        serif = JMI_SERIF
    styles = build_styles(serif, sans, style=style)
    document = parse_manuscript(text)
    if not document.title:
        raise BuildError(f"{built} has no '# ' title line")

    text_width = PAGE_SIZE[0] - 2 * MARGIN_SIDE
    story = build_story(
        document,
        styles=styles,
        serif=serif,
        results=results,
        figures=figures,
        readme=readme or (PAPER_DIR / "README.md"),
        text_width=text_width,
        document_kind=document_kind,
        style=style,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_SIDE,
        rightMargin=MARGIN_SIDE,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=document.title,
        author=document.authors[0].replace("**", "") if document.authors else "",
        subject=(
            "Manuscript submitted to the Journal of Medical Imaging, built from results/"
            if style == "jmi"
            else "denoiq-core review draft, built from results/"
        ),
        invariant=1,
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output


def _page_number(canvas: Any, doc: Any) -> None:
    """Centred page number in the bottom margin — required for a reviewed manuscript."""
    canvas.saveState()
    canvas.setFont(JMI_SERIF, 10)
    canvas.drawCentredString(PAGE_SIZE[0] / 2.0, MARGIN_BOTTOM / 2.0, str(doc.page))
    canvas.restoreState()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # These four default to the document selected by --document; an explicit path still wins.
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--built", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--document",
        choices=sorted(DOCUMENT_TABLES),
        default="manuscript",
        help="which document to render: the manuscript or the supplementary material",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the built manuscript is current; write no PDF",
    )
    parser.add_argument(
        "--style",
        choices=("jmi", "preprint"),
        default="jmi",
        help="page layout: the SPIE JMI submission setting (default) or the working preprint",
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="build for submission: fail unless the archived release DOI has been minted",
    )
    args = parser.parse_args(argv)
    source, built, output = document_paths(args.document)
    args.source = args.source or source
    args.built = args.built or built
    args.output = args.output or output

    try:
        if args.check:
            check_freshness(args.source, args.built, args.results, submission=args.submission)
            print(f"{args.built} is up to date")
            return 0
        path = build_pdf(
            source=args.source,
            built=args.built,
            results=args.results,
            figures=args.figures,
            output=args.output,
            document_kind=args.document,
            style=args.style,
            submission=args.submission,
        )
    except BuildError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
