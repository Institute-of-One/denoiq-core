"""The PDF build: the parser, the results-derived tables, and byte-identical output.

The PDF is the artefact that gets circulated, so the things worth pinning down are that its
tables come from ``results/`` rather than from a keyboard, that its captions come from the one
place captions are written, and that the same inputs give the same file. Needs the ``[pdf]``
extra; skipped without it, exactly like the deep-learning tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
RESULTS = REPO / "results"
BUILT = PAPER / "build" / "manuscript.md"

sys.path.insert(0, str(PAPER))
pytest.importorskip("reportlab")
build_pdf = pytest.importorskip("build_pdf")

requires_inputs = pytest.mark.skipif(
    not (BUILT.exists() and (RESULTS / "closed_form.json").exists()),
    reason="paper/build/manuscript.md and results/ are needed: run build_manuscript.py first",
)


def test_the_markdown_subset_parses_into_the_expected_shape():
    text = """<!-- a comment -->

# A title

**Someone**
Somewhere

## Abstract

**Purpose.** One *sentence* with `code`.

## 1. Introduction

A paragraph
across two lines.

### Contributions

1. First item
   continued here.
2. Second item.

## References

1. Author A. A title. Journal, 2026.
"""
    doc = build_pdf.parse_manuscript(text)
    assert doc.title == "A title"
    assert doc.authors == ["**Someone**", "Somewhere"]
    kinds = [kind for kind, _ in doc.blocks]
    assert kinds == ["h1", "p", "h1", "p", "h2", "li", "li", "h1", "ref"]
    # The comment is gone, and a wrapped paragraph is rejoined into one.
    assert "a comment" not in json.dumps(doc.blocks)
    assert ("p", "A paragraph across two lines.") in doc.blocks
    # An indented continuation line stays with its list item.
    assert ("li", "1. First item continued here.") in doc.blocks


def test_inline_markup_becomes_reportlab_markup():
    assert build_pdf.inline("**bold**") == "<b>bold</b>"
    assert build_pdf.inline("*italic*") == "<i>italic</i>"
    assert build_pdf.inline("a < b & c") == "a &lt; b &amp; c"


def test_code_spans_become_body_text_that_cannot_break():
    """Code spans hold mathematics here, so they are set in the body face, unbroken."""
    out = build_pdf.inline("the margin is `1.96 standard errors`")
    assert "1.96\xa0standard\xa0errors" in out
    assert "`" not in out


@requires_inputs
def test_table_1_is_the_design_matrix_counted_from_the_results():
    rows, caption = build_pdf.table1_data(RESULTS)
    design = json.loads((RESULTS / "statistics.json").read_text(encoding="utf-8"))["design"]
    assert rows[0][0] == "domain"
    body = rows[1:]
    assert body[-1][0] == "Total"
    # The per-domain arms add up to the totals the manuscript quotes.
    assert sum(int(row[4]) for row in body[:-1]) == design["unprocessed_arms"]
    assert sum(int(row[5]) for row in body[:-1]) == design["processed_arms"]
    assert int(body[-1][5]) == design["processed_arms"]
    assert str(design["arm_seed_evaluations"]) in caption
    assert str(design["n_seeds"]) in caption


@requires_inputs
def test_table_s1_still_carries_the_closed_form_validation():
    rows, caption = build_pdf.table_s1_data(RESULTS)
    payload = json.loads((RESULTS / "closed_form.json").read_text(encoding="utf-8"))
    assert len(rows) == len(payload["rows"]) + 1  # + header
    printed = {(row[3], row[4]) for row in rows[1:]}
    expected = {
        (f"{r['d_prime_closed_form']:.4f}", f"{r['d_prime_ideal_linear']:.4f}")
        for r in payload["rows"]
    }
    assert printed == expected
    # The caption uses the same journal-style notation as the prose (4.76×10⁻⁷, not 4.76e-07).
    assert build_pdf.scientific(payload["max_relative_error"], 1) in caption


@requires_inputs
def test_table_2_reports_observer_effects_with_intervals():
    rows, caption = build_pdf.table2_data(RESULTS)
    statistics = json.loads((RESULTS / "statistics.json").read_text(encoding="utf-8"))
    by_denoiser = statistics["observer_dependence"]["by_denoiser"]
    assert [row[0] for row in rows[1:-1]] == list(build_pdf.DENOISER_ORDER)
    assert rows[-1][0] == "All"
    for row in rows[1:-1]:
        entry = by_denoiser[row[0]]
        # Every performance cell carries its interval, as the journal requires.
        for cell in row[1:-1]:
            assert "[" in cell and "]" in cell, cell
        assert f"{entry['delta_d_pw']['value']:+.2f}" in row[2]
        assert f"{entry['delta_d_npwe']['value']:+.2f}" in row[4]
        # The point of the table: the inefficient observer gains, the efficient one does not.
        assert entry["delta_d_npwe"]["value"] > 0.0
        assert entry["delta_d_pw"]["value"] <= 0.0
        assert entry["benefit"]["ci_low"] > 0.0
    assert "Holm" in caption


def test_a_missing_results_file_is_an_error_not_an_empty_table(tmp_path):
    with pytest.raises(build_pdf.BuildError, match="missing"):
        build_pdf.table1_data(tmp_path)


def test_figure_captions_are_read_from_the_readme():
    captions = build_pdf.figure_captions(PAPER / "README.md")
    assert len(captions) == len(build_pdf.FIGURE_FILES)
    assert "divergence" in captions[3], "Figure 4 is the divergence figure"
    assert "observer-dependent benefit" in captions[4], "Figure 5 is the observer figure"
    supplementary = build_pdf.figure_captions(PAPER / "README.md", supplementary=True)
    assert len(supplementary) == len(build_pdf.SUPPLEMENTARY_FIGURE_FILES)
    assert "console" in supplementary[0]


def test_a_readme_without_the_figure_table_is_an_error(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(build_pdf.BuildError, match="figure captions"):
        build_pdf.figure_captions(readme)


@requires_inputs
def test_a_stale_built_manuscript_blocks_the_pdf(tmp_path):
    """The check that stops a circulated PDF carrying yesterday's numbers."""
    stale = tmp_path / "manuscript.md"
    stale.write_text("# not the built manuscript\n", encoding="utf-8")
    with pytest.raises(build_pdf.BuildError, match="out of date"):
        build_pdf.check_freshness(PAPER / "manuscript.md", stale, RESULTS)


@requires_inputs
def test_the_pdf_builds_and_is_byte_identical_on_a_rebuild(tmp_path):
    first = build_pdf.build_pdf(output=tmp_path / "a.pdf")
    second = build_pdf.build_pdf(output=tmp_path / "b.pdf")
    assert first.exists() and second.exists()
    assert first.read_bytes() == second.read_bytes(), "the PDF build is not deterministic"
    head = first.read_bytes()[:8]
    assert head.startswith(b"%PDF-")


@requires_inputs
def test_the_pdf_contains_the_manuscript_the_tables_and_the_figures(tmp_path):
    """Read the text back out of the PDF and check the sections and tables are in it."""
    pypdf = pytest.importorskip("pypdfium2")
    path = build_pdf.build_pdf(output=tmp_path / "manuscript.pdf")
    document = pypdf.PdfDocument(str(path))
    text = "\n".join(document[i].get_textpage().get_text_range() for i in range(len(document)))
    for expected in (
        "Abstract",
        # SPIE numbers sections without a trailing period.
        "1 Introduction",
        "3 Results",
        "Discussion",
        "Conclusion",
        "Disclosures",
        "Code and Data Availability",
        "References",
        "Table 1",
        "Table 2",
        "Figures",
        "Fig. 8",
        "Denoising under a data-processing ceiling",
        "Address all correspondence to Shuji Yamamoto",
    ):
        assert expected in text, f"{expected!r} is not in the PDF"
    # The submission setting carries no reviewer-facing furniture.
    assert "Review draft" not in text
    # Numbers, from results/, reached the page.
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert str(summary["n_conditions"]) in text
    for forbidden in ("National Cancer Center", "Tohoku University"):
        assert forbidden not in text
    assert "Institute of One, LISIT Co., Ltd., Tokyo, Japan" in text


def test_each_document_builds_to_its_own_files():
    """``--document supplementary`` must not read, or overwrite, the manuscript.

    The two documents share the whole build, so a shared default output path would have the
    supplementary material silently replace the manuscript PDF.
    """
    source, built, output = build_pdf.document_paths("manuscript")
    assert source.name == "manuscript.md"
    assert output.name == "manuscript.pdf"
    s_source, s_built, s_output = build_pdf.document_paths("supplementary")
    assert s_source.name == "supplementary.md"
    assert s_built.name == "supplementary.md"
    assert s_output.name == "supplementary.pdf"
    assert {source, built, output}.isdisjoint({s_source, s_built, s_output})
