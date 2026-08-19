"""The two Word defaults the docx build has to override, checked without Word.

Both were found by reading a converted PDF, one round trip each. A table row broke across
a page, so three numbers appeared under a repeated header with no row label; then the
table itself broke between two rows even though the whole thing would have fitted
overleaf, so four rows of a six-row Table 2 arrived on the next page.

Neither setting is reachable from pandoc, so the build patches the written file. What is
checked here is that patch, on a minimal document: the transform is pure XML, so it needs
neither pandoc nor Word.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parent.parent / "paper"
sys.path.insert(0, str(PAPER))
build_docx = pytest.importorskip("build_docx")

DOCUMENT = (
    "<w:document><w:body>"
    "<w:p><w:r><w:t>A paragraph before the caption.</w:t></w:r></w:p>\n    "
    "<w:p><w:r><w:t>Table 1. The caption.</w:t></w:r></w:p>\n    "
    "<w:tbl>"
    "<w:tr><w:tc><w:p><w:r><w:t>header</w:t></w:r></w:p></w:tc></w:tr>"
    "<w:tr><w:tc><w:p><w:pPr /><w:r><w:t>middle</w:t></w:r></w:p></w:tc></w:tr>"
    "<w:tr><w:tc><w:p><w:r><w:t>last</w:t></w:r></w:p></w:tc></w:tr>"
    "</w:tbl>"
    "<w:p><w:r><w:t>A paragraph after the table.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


@pytest.fixture
def written(tmp_path: Path) -> Path:
    path = tmp_path / "minimal.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", DOCUMENT)
    return path


def _document(path: Path) -> str:
    return zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")


def test_every_row_is_told_not_to_split_across_a_page(written: Path):
    build_docx.keep_table_rows_whole(written)
    rows = re.findall(r"<w:tr\b.*?</w:tr>", _document(written), re.S)
    assert len(rows) == 3
    for row in rows:
        assert "<w:cantSplit/>" in row


def test_the_caption_and_all_but_the_last_row_keep_with_what_follows(written: Path):
    """The caption travels with its table, and the table does not break between rows.

    The last row is deliberately left free: binding it would drag the paragraph after the
    table onto the same page, which is a different defect rather than a fix.
    """
    build_docx.keep_table_rows_whole(written)
    document = _document(written)
    rows = re.findall(r"<w:tr\b.*?</w:tr>", document, re.S)
    assert all("<w:keepNext/>" in row for row in rows[:-1])
    assert "<w:keepNext/>" not in rows[-1]

    caption, after = document.split("<w:tbl>")[0], document.split("</w:tbl>")[1]
    assert "The caption." in caption and caption.count("<w:keepNext/>") == 1, (
        "the caption keeps with the table, and the paragraph before it does not"
    )
    assert "<w:keepNext/>" not in after


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        ("<w:p><w:r/></w:p>", "<w:p><w:pPr><w:keepNext/></w:pPr><w:r/></w:p>"),
        ("<w:p><w:pPr /><w:r/></w:p>", "<w:p><w:pPr><w:keepNext/></w:pPr><w:r/></w:p>"),
        (
            "<w:p><w:pPr><w:x/></w:pPr></w:p>",
            "<w:p><w:pPr><w:keepNext/><w:x/></w:pPr></w:p>",
        ),
        ("<w:p/>", "<w:p/>"),  # nothing inside to set a property on
    ],
)
def test_properties_are_set_whatever_form_the_element_is_in(paragraph: str, expected: str):
    """Absent, present-but-empty, present-with-children, and a closed empty paragraph.

    Missing one of these does not fail loudly. It appends a *second* properties element,
    which the schema forbids and Word then ignores, so the setting reads as applied in
    the file and does nothing on the page.
    """
    out, _count = build_docx._keep_with_next(paragraph)
    assert out == expected
    assert out.count("<w:pPr>") <= 1


def test_a_paragraph_without_properties_still_gets_them(written: Path):
    """``<w:pPr>`` has to be the first child of its paragraph, present or not."""
    build_docx.keep_table_rows_whole(written)
    document = _document(written)
    # The header row's paragraph had no properties and the middle row's had an empty
    # element; both end up with keepNext first inside a well-formed <w:pPr>.
    for paragraph in re.findall(r"<w:p>.*?</w:p>", document, re.S):
        if "<w:keepNext/>" in paragraph:
            assert paragraph.startswith("<w:p><w:pPr>")
            assert paragraph.count("<w:pPr>") == 1, "no duplicated properties element"
