"""Assemble the files a submission actually uploads, named the way the system expects.

What a journal receives is not what a repository holds. Physica Medica wants the manuscript as an
editable ``.docx`` in a single column, each figure as its own file under a logical name, the
highlights as a separate file with "highlights" in its name, and the competing-interest
declaration as a Word document of its own for the declarations tool. This script produces that
set in one directory, and checks what it can check before it does.

    python paper/make_submission_files.py            # -> paper/build/submission/

What it verifies, because an editorial office verifying it instead costs a round trip:

* every figure cited in the manuscript exists and is exported;
* each figure meets the journal's raster minimum -- at least 300 dpi, and at least 1063 pixels
  wide for a single-column figure;
* the manuscript's own gates have passed, by requiring the built Markdown to be current;
* the ``.docx`` carries page numbers and, for this journal, no line numbers, because the
  submission system adds its own and two columns of them reached a PMB proof once.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))
import build_docx  # noqa: E402
import build_pdf  # noqa: E402

#: Elsevier's artwork minimum for a halftone: 300 dpi, and 1063 pixels across a single column.
MINIMUM_DPI = 300
MINIMUM_WIDTH = 1063


def png_geometry(path: Path) -> tuple[int, int, int | None]:
    """Width, height and dpi, read from the PNG header rather than from an image library."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    dpi = None
    marker = data.find(b"pHYs")
    if marker > 0:
        per_metre_x, _per_metre_y, unit = struct.unpack(">IIB", data[marker + 4 : marker + 13])
        if unit == 1:
            dpi = round(per_metre_x * 0.0254)
    return width, height, dpi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PAPER / "build" / "submission")
    parser.add_argument("--journal", default=build_docx.DEFAULT_JOURNAL,
                        choices=sorted(build_docx.LINE_NUMBERS))
    args = parser.parse_args(argv)

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    problems: list[str] = []

    # The manuscript and the supplement, editable.
    for kind, name in (("manuscript", "Manuscript.docx"), ("supplementary", "Supplementary.docx")):
        built = build_docx.build_docx(document_kind=kind, journal=args.journal)
        shutil.copy2(built, out / name)
        print(f"  {name}")

    # Figures, one file each, under the names the system asks for.
    manuscript = (PAPER / "build" / "manuscript.md").read_text(encoding="utf-8")
    cited = sorted({int(n) for n in re.findall(r"\bFigure (\d+)\b", manuscript)})
    for number, source_name in enumerate(build_pdf.FIGURE_FILES, start=1):
        source = PAPER / "figures" / source_name
        if not source.exists():
            problems.append(f"Figure {number}: {source} is missing")
            continue
        width, _height, dpi = png_geometry(source)
        if dpi is None or dpi < MINIMUM_DPI:
            problems.append(f"Figure {number} ({source_name}): {dpi} dpi, {MINIMUM_DPI} required")
        if width < MINIMUM_WIDTH:
            problems.append(
                f"Figure {number} ({source_name}): {width} px wide, {MINIMUM_WIDTH} required"
            )
        target = out / f"Figure_{number}.png"
        shutil.copy2(source, target)
        print(f"  {target.name}  <- {source_name}  ({width} px, {dpi} dpi)")
    for number in cited:
        if number > len(build_pdf.FIGURE_FILES):
            problems.append(f"the manuscript cites Figure {number} and there are only "
                            f"{len(build_pdf.FIGURE_FILES)} figure files")

    for number, source_name in enumerate(build_pdf.SUPPLEMENTARY_FIGURE_FILES, start=1):
        source = PAPER / "figures" / source_name
        if source.exists():
            shutil.copy2(source, out / f"Figure_S{number}.png")
            print(f"  Figure_S{number}.png  <- {source_name}")

    # Highlights: a separate editable file with "highlights" in its name.
    highlights = PAPER / "highlights.md"
    if highlights.exists():
        text = re.sub(r"<!--.*?-->", "", highlights.read_text(encoding="utf-8"), flags=re.S)
        bullets = [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]
        (out / "Highlights.txt").write_text(
            "\n".join(f"• {b}" for b in bullets) + "\n", encoding="utf-8"
        )
        print(f"  Highlights.txt  ({len(bullets)} bullets)")
    else:
        problems.append("paper/highlights.md is missing")

    # The competing-interest declaration, as its own Word document for the declarations tool.
    declaration = PAPER / "declaration_of_interest.md"
    if declaration.exists():
        import pypandoc  # noqa: PLC0415

        body = re.sub(r"<!--.*?-->", "", declaration.read_text(encoding="utf-8"), flags=re.S)
        pypandoc.convert_text(
            body, "docx", format="markdown", outputfile=str(out / "Declaration_of_Interest.docx")
        )
        print("  Declaration_of_Interest.docx")
        # It has to agree with the manuscript, or the pair of them is worse than either.
        disclosures = manuscript.split("## Disclosures", 1)[1].split("## Code and Data", 1)[0]
        if "no financial or commercial conflicts of interest" not in disclosures:
            problems.append(
                "the manuscript's Disclosures section no longer states the same competing "
                "interest as the declaration file"
            )
    else:
        problems.append("paper/declaration_of_interest.md is missing")

    print()
    if problems:
        print("not ready to submit:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"wrote {out}")
    print("Still to add by hand: the cover letter, and the declarations tool's own form if the")
    print("portal asks for one in addition to this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
