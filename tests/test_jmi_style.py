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
from reportlab.platypus import KeepTogether, Paragraph

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


# --------------------------------------------------------------------------------------
# line numbers
# --------------------------------------------------------------------------------------


class _RecordingCanvas:
    """Just enough canvas to record what the numbering would draw."""

    def __init__(self):
        self.calls = []
        self.font = None
        self.colour = None

    def saveState(self):  # noqa: N802 - reportlab's spelling
        pass

    def restoreState(self):  # noqa: N802
        pass

    def setFont(self, name, size):  # noqa: N802
        self.font = (name, size)

    def setFillColor(self, colour):  # noqa: N802
        self.colour = colour

    def drawRightString(self, x, y, text):  # noqa: N802
        self.calls.append((x, y, text))


def _style(leading=22.8, size=12.0):
    from reportlab.lib.styles import ParagraphStyle

    return ParagraphStyle("probe", fontName=build_pdf.JMI_SERIF, fontSize=size, leading=leading)


def test_line_numbers_run_continuously_and_sit_in_the_left_margin():
    build_pdf.register_fonts()
    numbering = build_pdf.LineNumbering(font=build_pdf.JMI_SERIF)
    canvas = _RecordingCanvas()
    style = _style()

    numbering.draw(canvas, height=3 * style.leading, style=style, n_lines=3)
    numbering.draw(canvas, height=2 * style.leading, style=style, n_lines=2)

    assert [text for _, _, text in canvas.calls] == ["1", "2", "3", "4", "5"]
    # Left of the text block, so the numbers land in the margin rather than on the words.
    assert all(x < 0 for x, _, _ in canvas.calls)
    # One leading apart, descending down the paragraph.
    first = [y for _, y, _ in canvas.calls[:3]]
    assert first[0] > first[1] > first[2]
    assert abs((first[0] - first[1]) - style.leading) < 1e-6
    assert canvas.font == (build_pdf.JMI_SERIF, numbering.size)


def test_a_build_starts_its_numbering_at_one():
    numbering = build_pdf.LineNumbering(font=build_pdf.JMI_SERIF, counter=417)
    numbering.reset()
    assert numbering.counter == 0


def test_a_split_paragraph_keeps_numbering_its_lines():
    """A paragraph broken over a page break must not stop counting."""
    build_pdf.register_fonts()
    numbering = build_pdf.LineNumbering(font=build_pdf.JMI_SERIF)
    style = _style()
    paragraph = build_pdf.NumberedParagraph(" ".join(["word"] * 200), style, numbering)
    paragraph.wrap(300, 60)
    parts = paragraph.split(300, 60)
    assert parts, "the probe paragraph should be long enough to split"
    assert all(part.numbering is numbering for part in parts)


@requires_build
def test_the_story_numbers_running_text_and_not_captions():
    numbering = build_pdf.LineNumbering(font=build_pdf.JMI_SERIF)
    story = build_pdf.build_story(
        build_pdf.parse_manuscript((PAPER / "build" / "manuscript.md").read_text(encoding="utf-8")),
        styles=build_pdf.build_styles(*build_pdf.register_fonts(), style="jmi"),
        serif=build_pdf.JMI_SERIF,
        results=RESULTS,
        figures=PAPER / "figures",
        readme=PAPER / "README.md",
        text_width=400.0,
        style="jmi",
        numbering=numbering,
    )
    numbered = [item for item in story if isinstance(item, build_pdf.NumberedParagraph)]
    plain = [
        item
        for item in story
        if type(item).__name__ == "Paragraph" and not isinstance(item, build_pdf.NumberedParagraph)
    ]
    assert len(numbered) > 100, "the body of the manuscript should be numbered"
    # The title block is unnumbered.
    assert any("Address all correspondence" in item.text for item in plain)
    # So are the figure captions, which travel with their image inside a KeepTogether.
    captions = [
        item
        for group in story
        if isinstance(group, KeepTogether)
        for item in group._content
        if isinstance(item, Paragraph)
    ]
    assert captions, "the figures should carry captions"
    assert all(not isinstance(item, build_pdf.NumberedParagraph) for item in captions)
    assert any(item.text.startswith("<b>Fig. ") for item in captions)


@requires_build
def test_the_submission_pdf_carries_one_continuous_sequence(tmp_path):
    """Every text page numbered, no gaps, no restarts — a reviewer cites "line 214"."""
    pypdf = pytest.importorskip("pypdfium2")
    path = build_pdf.build_pdf(output=tmp_path / "numbered.pdf", style="jmi", line_numbers=True)
    document = pypdf.PdfDocument(str(path))

    expected = 1
    pages_with_numbers = 0
    for index in range(len(document)):
        page = document[index]
        # Read the left margin only: the footer page number is drawn in the same face and would
        # otherwise be indistinguishable from a line number that happens to share its value.
        found = [
            int(token)
            for token in re.findall(
                r"\d{1,4}",
                page.get_textpage().get_text_bounded(
                    left=0,
                    bottom=build_pdf.MARGIN_BOTTOM + 2,
                    right=build_pdf.MARGIN_SIDE,
                    top=float(page.get_height()),
                ),
            )
        ]
        if not found:
            continue  # a figure page: captions are deliberately not numbered
        pages_with_numbers += 1
        assert found == list(range(expected, expected + len(found))), (
            f"page {index + 1} numbers {found[:3]}... do not continue from {expected}"
        )
        expected += len(found)
    assert pages_with_numbers >= 15
    assert expected > 400, "the manuscript should number several hundred lines"

    # And the plain build has none of them: page three carries its page number and nothing else
    # that stands alone on a line.
    plain = build_pdf.build_pdf(output=tmp_path / "plain.pdf", style="jmi", line_numbers=False)
    plain_document = pypdf.PdfDocument(str(plain))
    page = plain_document[2]
    margin = page.get_textpage().get_text_bounded(
        left=0,
        bottom=build_pdf.MARGIN_BOTTOM + 2,
        right=build_pdf.MARGIN_SIDE,
        top=float(page.get_height()),
    )
    assert not re.findall(r"\d", margin), margin
