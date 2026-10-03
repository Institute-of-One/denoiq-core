"""Figure captions, read from the one place they are written.

Why this is its own module
--------------------------
These two functions are text: they read a table out of ``paper/README.md`` and render it. They
lived in ``build_pdf``, which imports reportlab at the top, so building the *Markdown* manuscript
imported a PDF engine. Continuous integration does not install reportlab for the plain test job,
so every test that rendered the manuscript failed with ``ModuleNotFoundError`` — and because the
lint step failed first, nobody saw that the repository had been red for a month.

A caption does not need a PDF library. Both builders import it from here instead.
"""

from __future__ import annotations

import re
from pathlib import Path


class CaptionError(RuntimeError):
    """The caption table does not say what the document needs it to say."""


def figure_captions(
    readme: Path, *, supplementary: bool = False, expected: int | None = None
) -> list[str]:
    """Figure captions, read from the table in ``paper/README.md``.

    Reading them rather than restating them is the same discipline the numbers follow: there
    is one place a caption is written, so the PDF and the repository documentation cannot
    describe a figure differently.

    ``expected`` is the number of figure files the caller is going to place. Passing it turns a
    silent mismatch — a figure added without a caption, or the reverse — into a refusal.
    """
    text = readme.read_text(encoding="utf-8")
    pattern = (
        r"^\|\s*Fig\s*S(\d+)\s*\|\s*(.+?)\s*\|\s*$"
        if supplementary
        else (r"^\|\s*Fig\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$")
    )
    captions = re.findall(pattern, text, re.M)
    ordered = [caption for _, caption in sorted(captions, key=lambda pair: int(pair[0]))]
    if expected is not None and len(ordered) != expected:
        raise CaptionError(
            f"{readme} lists {len(ordered)} figure captions, expected {expected}: the PDF takes "
            "its captions from that table"
        )
    return ordered


def caption_list_block(readme: Path, *, supplementary: bool = False) -> str:
    """The captions again, as a list after the references.

    Some journals ask for captions beneath each figure *and* listed at the end. They are still
    written once, in ``paper/README.md``; this renders that same list a second time rather than
    letting anyone maintain two copies.
    """
    captions = figure_captions(readme, supplementary=supplementary)
    prefix = "Figure S" if supplementary else "Figure "
    lines = ["## Figure captions", ""]
    # Capitalised the same way the caption beneath the figure is, so the two renderings of one
    # caption do not differ by a letter.
    lines += [f"**{prefix}{i}.** {c[0].upper()}{c[1:]}" for i, c in enumerate(captions, 1)]
    return "\n\n".join(lines) + "\n"
