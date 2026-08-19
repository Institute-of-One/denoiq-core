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


def build_docx(
    *,
    results: Path = DEFAULT_RESULTS,
    figures: Path = DEFAULT_FIGURES,
    document_kind: str = "manuscript",
    output: Path | None = None,
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
    return output


#: Word's default is to let a table row break across a page. In the converted PDF that put
#: half of a Table 2 row on one page and half on the next, with the repeated header in
#: between, so a reader met three numbers with no idea which denoiser they belonged to.
#: Rows here are two or three lines; moving a whole row to the next page costs nothing.
_ROW_PROPERTIES = "<w:trPr><w:cantSplit/></w:trPr>"


def keep_table_rows_whole(path: Path) -> Path:
    """Set ``cantSplit`` on every table row of a written ``.docx``.

    Pandoc has no option for this and no reference document can supply it, because it is a
    row property rather than a style. The file is a zip of XML, so it is set afterwards.
    """
    import re
    import shutil
    import zipfile

    entry = "word/document.xml"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        contents = {name: archive.read(name) for name in names}

    document = contents[entry].decode("utf-8")
    # A row that already carries properties gets cantSplit added to them; one that carries
    # none gets a properties element, which must be the first child of the row.
    document, with_props = re.subn(r"(<w:trPr>)", r"\1<w:cantSplit/>", document)
    document, without_props = re.subn(
        r"(<w:tr\b[^>]*>)(?!<w:trPr>)", rf"\1{_ROW_PROPERTIES}", document
    )
    contents[entry] = document.encode("utf-8")

    temporary = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:  # preserve the original entry order
            archive.writestr(name, contents[name])
    shutil.move(str(temporary), str(path))
    print(f"  rows kept whole: {with_props + without_props}")
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
