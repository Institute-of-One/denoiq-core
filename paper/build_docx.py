"""Render the built manuscript to a review ``.docx``, figures and tables in place.

    python paper/build_docx.py                       # -> paper/build/manuscript_v2.docx
    python paper/build_docx.py --document supplementary
    python paper/build_docx.py --output some/where.docx

Like :mod:`build_pdf`, this is a **derivative**: it re-renders ``paper/manuscript.md`` and
refuses to continue if the committed ``build/*.md`` is out of date, so a stale number cannot be
baked into a circulated document. The prose comes from ``build/<document>.md``; Tables 1-2 (or
S1-S4) are generated from ``results/`` by the same functions the PDF uses, and Figures 1-8 (or
Figure S1) are placed **immediately after the paragraph that first cites them** — the same
near-first-mention layout as the JMI PDF, expressed as Markdown so a reviewer gets an editable
document.

The one external dependency is pandoc, reached through ``pypandoc``. It is not required to run
the study or the tests; it is required only to produce this one editable artefact, exactly as
``build_pdf`` requires reportlab only to produce the PDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
DEFAULT_RESULTS = REPO_DIR / "results"
DEFAULT_FIGURES = PAPER_DIR / "figures"

sys.path.insert(0, str(PAPER_DIR))
import build_pdf  # noqa: E402  (path is set just above)


def _table_markdown(rows: list[list[str]], caption: str) -> str:
    """A generated table as a GitHub-flavoured Markdown table, caption above.

    The caption keeps its ``**Table N.**`` run-in; the header row is separated from the body by
    the usual ``---`` rule so pandoc renders a real table rather than a run of pipes.
    """
    header, *body = rows
    lines = [f"| {' | '.join(header)} |", f"|{'|'.join(['---'] * len(header))}|"]
    lines += [f"| {' | '.join(cell for cell in row)} |" for row in body]
    return f"{caption}\n\n" + "\n".join(lines)


def _figure_markdown(token: str, filename: str, caption: str, figures: Path) -> str:
    """One figure as an implicit-figure image, labelled as the manuscript labels it."""
    path = figures / filename
    if not path.exists():
        raise build_pdf.BuildError(
            f"{path} is missing: run `python paper/make_figures.py` before building the docx"
        )
    supplementary = token.startswith("S")
    number = token[1:] if supplementary else token
    prefix = "Figure S" if supplementary else "Figure "
    label = f"**{prefix}{number}.** {caption[0].upper()}{caption[1:]}"
    if not label.rstrip().endswith("."):
        label = f"{label}."
    # An image alone in its paragraph becomes a captioned figure under pandoc's
    # implicit_figures; the alt text is the caption. The path is absolute so pandoc finds it
    # regardless of the working directory it is invoked from.
    return f"![{label}]({path.as_posix()})"


def render_markdown(
    built_text: str,
    *,
    results: Path,
    figures: Path,
    readme: Path,
    document_kind: str = "manuscript",
) -> str:
    """Augment the built prose with its tables and figures, in Markdown.

    The placement rules match :func:`build_pdf.build_story` exactly: a generated table follows
    the first paragraph of the section whose heading names it, and each figure follows the
    paragraph that first cites it, figures named together emitted in ascending order.
    """
    document = build_pdf.parse_manuscript(built_text)
    generators = build_pdf.DOCUMENT_TABLES[document_kind]
    tables = {name: builder(results) for name, builder in generators.items()}

    figure_files = (
        build_pdf.SUPPLEMENTARY_FIGURE_FILES
        if document_kind == "supplementary"
        else build_pdf.FIGURE_FILES
    )
    captions = build_pdf.figure_captions(readme, supplementary=document_kind == "supplementary")
    figure_entries: dict[str, tuple[str, str]] = {}
    for filename, caption in zip(figure_files, captions, strict=True):
        entry_token = filename.split("_")[0].removeprefix("fig").upper()
        figure_entries[entry_token] = (filename, caption)
    placed: set[str] = set()

    out: list[str] = [f"# {document.title}", "", *document.authors, ""]
    pending: str | None = None

    def emit_figures(payload: str) -> None:
        wanted = [
            token
            for token in dict.fromkeys(build_pdf.referenced_figures(payload))
            if token in figure_entries and token not in placed
        ]
        for token in sorted(wanted, key=lambda name: int(name.lstrip("S"))):
            filename, caption = figure_entries[token]
            out.extend([_figure_markdown(token, filename, caption, figures), ""])
            placed.add(token)

    for kind, payload in document.blocks:
        if kind == "h1":
            out.extend([f"## {payload}", ""])
            continue
        if kind == "h2":
            out.extend([f"### {payload}", ""])
            pending = next((key for key in tables if key in payload.lower()), None)
            continue
        if kind == "p":
            out.extend([payload, ""])
            if pending:
                rows, caption = tables[pending]
                out.extend([_table_markdown(rows, caption), ""])
                pending = None
            emit_figures(payload)
            continue
        if kind == "table":
            header, *body = payload
            out.append(f"| {' | '.join(header)} |")
            out.append(f"|{'|'.join(['---'] * len(header))}|")
            out.extend(f"| {' | '.join(row)} |" for row in body)
            out.append("")
            continue
        if kind in ("li", "ref"):
            out.extend([payload, ""])
            continue
        raise build_pdf.BuildError(f"unhandled block type {kind!r}")  # pragma: no cover

    remaining = [token for token in figure_entries if token not in placed]
    if remaining:
        heading = "Supplementary figures" if document_kind == "supplementary" else "Figures"
        out.extend([f"## {heading}", ""])
        for token in sorted(remaining, key=lambda name: int(name.lstrip("S"))):
            filename, caption = figure_entries[token]
            out.extend([_figure_markdown(token, filename, caption, figures), ""])

    return "\n".join(out).rstrip() + "\n"


#: Medical Physics has been double-anonymised since 1 July 2026. The manuscript source
#: keeps its author block, because build_pdf.py still needs it for the SPIE layout this
#: paper was first written for; anonymity is a property of the artefact that has to
#: carry it, not of the source. So it is applied here, on the way into the .docx.
#:
#: Set False only for a venue that reviews single-anonymised.
ANONYMOUS = True

#: The author block: three lines that follow the title. Matched rather than located by
#: line number so that reordering the front matter cannot silently defeat this.
_AUTHOR_BLOCK = re.compile(
    r"^\*\*Shuji Yamamoto\*\*\n"
    r"Institute of One[^\n]*\n"
    r"[^\n]*ORCID[^\n]*\n",
    re.M,
)

#: The supplement carries a one-line byline instead. It needs the same treatment: the
#: journal's checklist says no names or affiliations in the manuscript, **supplementary
#: information**, or figures. Anonymising only the main document leaves the author's
#: name on the first page of the file that goes to the same reviewers.
_SUPPLEMENT_BYLINE = re.compile(r"^Shuji Yamamoto\s*·[^\n]*\n", re.M)

_AVAILABILITY = re.compile(
    r"(## Code and Data Availability\n)(.*?)(?=\n## )", re.S
)
_REPOSITORY_URL = re.compile(r"is public at\s+https://github\.com/\S+")
_ARCHIVE_PHRASE = re.compile(r"archived at Zenodo as version [^.]*\.")

_WITHHELD_URL = "is public"
_WITHHELD_ARCHIVE = (
    "archived under a version DOI. The repository URL and that DOI both name the "
    "author, so they are withheld from this anonymised copy and given on the title "
    "page; they are restored at acceptance."
)


def anonymise(markdown: str) -> str:
    """Remove what identifies the author, and only that.

    Citing one's own published work is explicitly allowed -- what the journal forbids
    is referring to it in the first person, which this manuscript does not do. So the
    Zenodo DOI in reference 18 stays, and the substitutions below are scoped to the
    availability section rather than applied to the whole document. Removing the
    reference would be over-anonymising, and would cost a citation for nothing.
    """
    if _SUPPLEMENT_BYLINE.search(markdown):
        return _SUPPLEMENT_BYLINE.sub("", markdown, count=1)

    if not _AUTHOR_BLOCK.search(markdown):
        raise SystemExit(
            "the author block is not where anonymisation expects it; refusing to "
            "build a .docx that may still carry it"
        )
    markdown = _AUTHOR_BLOCK.sub("", markdown, count=1)

    section = _AVAILABILITY.search(markdown)
    if section is None:
        raise SystemExit("no Code and Data Availability section to anonymise")
    body = section.group(2)
    for pattern, replacement in (
        (_REPOSITORY_URL, _WITHHELD_URL),
        (_ARCHIVE_PHRASE, _WITHHELD_ARCHIVE),
    ):
        if not pattern.search(body):
            raise SystemExit(f"availability section has no {pattern.pattern!r}")
        body = pattern.sub(replacement, body, count=1)
    return markdown[: section.start(2)] + body + markdown[section.end(2) :]


def build_docx(
    *,
    results: Path = DEFAULT_RESULTS,
    figures: Path = DEFAULT_FIGURES,
    document_kind: str = "manuscript",
    output: Path | None = None,
    anonymous: bool = ANONYMOUS,
) -> Path:
    """Verify freshness, assemble the augmented Markdown, and convert it to ``.docx``."""
    import pypandoc

    source, built, _pdf = build_pdf.document_paths(document_kind)
    text = build_pdf.check_freshness(source, built, results)
    markdown = render_markdown(
        text,
        results=results,
        figures=figures,
        readme=PAPER_DIR / "README.md",
        document_kind=document_kind,
    )
    if anonymous:
        markdown = anonymise(markdown)
    if output is None:
        output = PAPER_DIR / "build" / f"{document_kind}_v2.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    # implicit_figures turns a lone image into a captioned figure; the resource path lets a
    # relative image reference resolve even though the paths written here are absolute.
    pypandoc.convert_text(
        markdown,
        to="docx",
        format="markdown+implicit_figures",
        outputfile=str(output),
        extra_args=["--resource-path", str(PAPER_DIR)],
    )
    keep_table_rows_whole(output)
    number_pages_and_lines(output)
    return output


#: The footer part, holding a centred PAGE field. Word evaluates the field on open; the
#: literal "1" is what a reader sees before it does.
_FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    "<w:r><w:t>1</w:t></w:r>"
    '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    "</w:p></w:ftr>"
)

_FOOTER_PART = "word/footer1.xml"
_FOOTER_RELATIONSHIP = "rIdPageFooter"
_FOOTER_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"

#: Continuous line numbering down the left margin, every line, restarting never.
#: ``distance`` is twentieths of a point: 360 is a quarter inch clear of the text.
_LINE_NUMBERING = '<w:lnNumType w:countBy="1" w:restart="continuous" w:distance="360"/>'


def number_pages_and_lines(path: Path) -> Path:
    """Add page numbers and continuous line numbering to a written ``.docx``.

    Medical Physics returned MS 26-1820 before peer review for the want of both. Pandoc
    emits neither and no reference document supplies them: line numbering is a section
    property, and a page number needs a footer part, a content-type override and a
    relationship as well as the reference to it. All of that is done here so the next
    build carries them rather than the next submission discovering it again.

    Order inside ``sectPr`` is not free. The schema wants ``footerReference`` before
    ``footnotePr`` and ``lnNumType`` after it; Word rejects the part outright if they
    are the other way round.
    """
    import re  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        contents = {name: archive.read(name) for name in names}

    document = contents["word/document.xml"].decode("utf-8")
    if "lnNumType" in document:
        return path

    reference = f'<w:footerReference w:type="default" r:id="{_FOOTER_RELATIONSHIP}"/>'
    document = document.replace("<w:sectPr>", f"<w:sectPr>{reference}", 1)
    document = document.replace("</w:footnotePr>", f"</w:footnotePr>{_LINE_NUMBERING}", 1)
    contents["word/document.xml"] = document.encode("utf-8")

    rels = contents["word/_rels/document.xml.rels"].decode("utf-8")
    relationship = (
        f'<Relationship Id="{_FOOTER_RELATIONSHIP}" Type="http://schemas.openxmlformats'
        f'.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
    )
    rels = rels.replace("</Relationships>", f"{relationship}</Relationships>", 1)
    contents["word/_rels/document.xml.rels"] = rels.encode("utf-8")

    types = contents["[Content_Types].xml"].decode("utf-8")
    override = f'<Override PartName="/{_FOOTER_PART}" ContentType="{_FOOTER_TYPE}"/>'
    types = types.replace("</Types>", f"{override}</Types>", 1)
    contents["[Content_Types].xml"] = types.encode("utf-8")

    contents[_FOOTER_PART] = _FOOTER_XML.encode("utf-8")
    names = [*names, _FOOTER_PART]

    temporary = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, contents[name])
    shutil.move(str(temporary), str(path))
    lines = len(re.findall(r"<w:p[ >]", document))
    print(f"  page numbers and continuous line numbering added ({lines} paragraphs)")
    return path


#: Word's default is to let a table row break across a page. In the converted PDF that put
#: half of a Table 2 row on one page and half on the next, with the repeated header in
#: between, so a reader met three numbers with no idea which denoiser they belonged to.
#: Rows here are two or three lines; moving a whole row to the next page costs nothing.
def _set_property(fragment: str, container: str, element: str, prop: str) -> tuple[str, int]:
    """Add ``prop`` to every ``container``'s properties element, creating one if needed.

    The properties element comes in three forms and all three occur: open with children,
    self-closing and empty, and absent. Missing the self-closing form does not fail
    loudly -- it appends a *second* properties element, which the schema forbids and
    which Word then silently ignores, so the setting looks applied and is not.
    """
    import re

    opened = f"<{element}>"
    fragment, with_children = re.subn(re.escape(opened), f"{opened}{prop}", fragment)
    fragment, empty = re.subn(
        rf"<{re.escape(element)}\s*/>", f"{opened}{prop}</{element}>", fragment
    )
    # The lookbehind skips a self-closing container: it has no inside to put a property
    # in, and appending one would land it after the element rather than within it.
    fragment, absent = re.subn(
        rf"(<{re.escape(container)}\b[^>]*>)(?<!/>)(?!<{re.escape(element)})",
        rf"\1{opened}{prop}</{element}>",
        fragment,
    )
    return fragment, with_children + empty + absent


def _keep_with_next(fragment: str) -> tuple[str, int]:
    """Mark every paragraph in ``fragment`` as staying with the paragraph after it."""
    return _set_property(fragment, "w:p", "w:pPr", "<w:keepNext/>")


def keep_table_rows_whole(path: Path) -> Path:
    """Keep table rows unbroken, and keep each table with its caption on one page.

    Two separate Word defaults, both wrong for a table of results. A row may break across
    a page: ``cantSplit`` stops that. And a table may break between any two rows even when
    the whole thing would fit overleaf, which put four rows of a six-row Table 2 on the
    next page. ``keepNext`` on every paragraph of the caption and of all but the last row
    moves the table whole instead.

    Neither is reachable from pandoc or from a reference document -- one is a row
    property and the other has to be set paragraph by paragraph -- so they are set here,
    on the written file, which is a zip of XML.
    """
    import re
    import shutil
    import zipfile

    entry = "word/document.xml"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        contents = {name: archive.read(name) for name in names}

    document = contents[entry].decode("utf-8")
    # The properties element must be the first child of the row, whether it was already
    # there, there but empty, or absent.
    document, split_stopped = _set_property(document, "w:tr", "w:trPr", "<w:cantSplit/>")

    kept = 0

    def bind_table(match: re.Match) -> str:
        """Bind a table's caption and rows to what follows them."""
        nonlocal kept
        caption, table = match.group(1), match.group(2)
        rows = re.findall(r"<w:tr\b.*?</w:tr>", table, re.S)
        caption, n = _keep_with_next(caption)
        kept += n
        # The last row must not keep with what follows it, or the paragraph after the
        # table gets dragged onto the same page too.
        for row in rows[:-1]:
            bound, n = _keep_with_next(row)
            kept += n
            table = table.replace(row, bound, 1)
        return caption + table

    # Pandoc writes the caption paragraph, a newline and some indentation, then the table.
    document = re.sub(
        r"(<w:p\b(?:(?!<w:p\b).)*?</w:p>\s*)(<w:tbl>.*?</w:tbl>)",
        bind_table,
        document,
        flags=re.S,
    )
    contents[entry] = document.encode("utf-8")

    temporary = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:  # preserve the original entry order
            archive.writestr(name, contents[name])
    shutil.move(str(temporary), str(path))
    print(f"  rows kept whole: {split_stopped}, paragraphs bound to the next: {kept}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--document",
        choices=sorted(build_pdf.DOCUMENT_TABLES),
        default="manuscript",
        help="which document to render: the manuscript or the supplementary material",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        path = build_docx(
            results=args.results,
            figures=args.figures,
            document_kind=args.document,
            output=args.output,
        )
    except build_pdf.BuildError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
