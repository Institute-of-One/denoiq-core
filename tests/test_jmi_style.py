"""The SPIE JMI submission setting: conventions, and glyphs that actually exist.

The journal setting is not cosmetic. Three of its rules can silently produce a wrong page —
a citation that reads as an interval, an exponent that vanishes because the Times-metric face
has no such code point, and a section number that keeps the markdown's period — and none of
them would fail a build. They are checked here on the text that reaches the page.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
RESULTS = REPO / "results"

# The PDF builder needs the [pdf] extra, which CI deliberately does not install: a PDF is a
# derivative, not a dependency. Skip the module rather than fail collection without it.
sys.path.insert(0, str(PAPER))
pytest.importorskip("reportlab")
build_pdf = pytest.importorskip("build_pdf")

#: Code points that STIXGeneral maps to a glyph that is not the character. The capital forms
#: "Ŝ" (U+015C) and "Ŵ" (U+0174) draw correctly and are what the manuscript uses.
MISDRAWN_BY_THE_JMI_SERIF = "ŝ"

requires_build = pytest.mark.skipif(
    not (PAPER / "build" / "manuscript.md").exists(), reason="manuscript not built"
)


def test_citations_are_superscripts_and_intervals_are_not():
    """``[10]`` is a citation; ``[86.7%, 89.6%]`` and ``[+0.63, +0.65]`` are intervals."""
    assert build_pdf.inline("denoising [10] degrades", style="jmi") == (
        "denoising<super>10</super> degrades"
    )
    assert build_pdf.inline("as reported [11,12]", style="jmi") == (
        "as reported<super>11,12</super>"
    )
    for interval in ("[86.7%, 89.6%]", "[+0.63, +0.65]", "[1.5, 6.2]"):
        assert "<super>" not in build_pdf.inline(f"rate {interval}", style="jmi"), interval
    # The preprint setting leaves both alone.
    assert build_pdf.inline("denoising [10] degrades", style="preprint") == (
        "denoising [10] degrades"
    )


def test_figure_references_are_abbreviated_except_at_the_start_of_a_sentence():
    text = "Figure 4 shows it. See Figure 5, and (Figure 6) too."
    assert build_pdf.inline(text, style="jmi") == (
        "Figure 4 shows it. See Fig. 5, and (Fig. 6) too."
    )
    assert build_pdf.inline(text, style="preprint") == text


def test_section_numbers_lose_the_markdown_period():
    assert build_pdf.section_heading("1. Introduction", "jmi") == "1 Introduction"
    assert build_pdf.section_heading("3.4 Performance", "jmi") == "3.4 Performance"
    # A heading never passes through inline(), so it abbreviates its own figure references.
    assert (
        build_pdf.section_heading("3.5 Failures (Figures 7-8)", "jmi") == "3.5 Failures (Figs. 7-8)"
    )
    assert build_pdf.section_heading("3.4 Ceiling (Figure 6)", "jmi") == "3.4 Ceiling (Fig. 6)"
    assert build_pdf.section_heading("Abstract", "jmi") == "Abstract"
    assert build_pdf.section_heading("1. Introduction", "preprint") == "1. Introduction"


def test_unicode_exponents_become_real_superscript_runs():
    """``4.76×10⁻⁷`` must not depend on the font carrying U+207B and U+2077."""
    rendered = build_pdf.inline("a value of 4.76×10⁻⁷ here", style="jmi")
    assert rendered == "a value of 4.76×10<super>-7</super> here"
    assert build_pdf.inline("‖s‖₂/σ", style="jmi") == "‖s‖<sub>2</sub>/σ"


@requires_build
def test_every_character_that_reaches_the_page_exists_in_the_font():
    """The JMI serif is Times-metric, and Times-metric faces are not fully covered.

    Anything the font cannot draw comes out as a blank box or as nothing at all, which is how
    an exponent disappears from a result. The check runs over both built documents, after the
    markup conversions the JMI setting applies.
    """
    from reportlab.pdfbase.ttfonts import TTFont

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "STIXGeneral.ttf"
    cmap = TTFont(build_pdf.JMI_SERIF, str(ttf)).face.charToGlyph

    text = "".join(
        build_pdf.inline((PAPER / "build" / f"{name}.md").read_text(encoding="utf-8"), style="jmi")
        for name in ("manuscript", "supplementary")
    )
    # Table cells are drawn directly, without inline(), so they are checked as they are.
    for generators in build_pdf.DOCUMENT_TABLES.values():
        for builder in generators.values():
            rows, _ = builder(RESULTS)
            text += "".join(cell for row in rows for cell in row)

    missing = sorted({c for c in set(text) if ord(c) not in cmap and c not in "\n\t"})
    assert not missing, f"the JMI serif cannot draw: {missing}"

    # A code point can be in the cmap and still come out as the wrong glyph. U+015D is drawn as
    # a section sign by this face — invisible to a coverage check, very visible on the page — so
    # the characters known to misdraw are named rather than merely counted.
    misdrawn = sorted(set(text) & set(MISDRAWN_BY_THE_JMI_SERIF))
    assert not misdrawn, f"the JMI serif draws the wrong glyph for: {misdrawn}"


@requires_build
def test_the_submission_setting_carries_no_reviewer_furniture():
    """No "review draft" note, and the corresponding-author line the journal asks for."""
    built = (PAPER / "build" / "manuscript.md").read_text(encoding="utf-8")
    document = build_pdf.parse_manuscript(built)
    line = build_pdf.correspondence_line(document.authors)
    assert line.startswith("Address all correspondence to ")
    assert "yamamoto@lisit.jp" in line
    assert "Institute of One, LISIT Co., Ltd., Tokyo, Japan" in " ".join(document.authors)
    assert "0000-0001-9211-1071" in " ".join(document.authors)


@requires_build
def test_the_two_layouts_report_the_same_numbers():
    """A style may change the setting of a page; it may not change what the page says."""
    numbers = {}
    for style in ("jmi", "preprint"):
        story = build_pdf.build_story(
            build_pdf.parse_manuscript(
                (PAPER / "build" / "manuscript.md").read_text(encoding="utf-8")
            ),
            styles=build_pdf.build_styles(*build_pdf.register_fonts(), style=style),
            serif=build_pdf.JMI_SERIF if style == "jmi" else "DejaVuSerif",
            results=RESULTS,
            figures=PAPER / "figures",
            readme=PAPER / "README.md",
            text_width=400.0,
            style=style,
        )
        text = " ".join(getattr(item, "text", "") for item in story)
        numbers[style] = _numbers(text)
    assert numbers["jmi"] == numbers["preprint"]


def _numbers(text: str) -> list[str]:
    """Every numeric token in a rendered story, independent of how it was typeset.

    The two layouts write the same exponent differently — ``10⁻⁷`` against
    ``10<super>-7</super>`` — so the markup is stripped and the Unicode scripts are folded to
    ASCII before the digits are compared.
    """
    plain = re.sub(r"<[^>]+>", "", text).translate(_TO_ASCII)
    # A trailing period is punctuation, not a digit: "1." in a heading and "1" in the
    # same heading set differently are the same number.
    return sorted(token.rstrip(".") for token in re.findall(r"[-+]?\d[\d.]*", plain))


#: Unicode super/subscript digits and signs, folded onto their ASCII equivalents.
_TO_ASCII = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉₊₋",
    "0123456789+-0123456789+-",
)


def test_the_supplementary_document_has_its_own_files():
    for kind in ("manuscript", "supplementary"):
        source, built, output = build_pdf.document_paths(kind)
        assert source.stem == kind and built.stem == kind and output.stem == kind


def test_the_release_file_is_json_and_declares_the_archive():
    release = json.loads((PAPER / "release.json").read_text(encoding="utf-8"))
    assert release["archive"] == "Zenodo"
    assert release["license"] == "MIT"


@requires_build
def test_no_citation_sits_against_a_numeral():
    """A superscript citation after a number reads as an exponent.

    "d' = 5 [7]" sets as "5" with a raised 7 beside it, which a reader parses as 5 to the
    seventh. The fix is editorial — put the citation somewhere else in the sentence — so the
    check is on the source, where the fix has to be made.
    """
    for name in ("manuscript", "supplementary"):
        text = (PAPER / "build" / f"{name}.md").read_text(encoding="utf-8")
        text = text.split("## References")[0]
        offenders = re.findall(r"\d\s*\[\d{1,2}[,–\d\s-]*\]", text)
        assert not offenders, f"{name}: citation against a numeral: {offenders}"
