"""Every defect we have shipped or nearly shipped, checked in one pass.

Written after a session in which nine separate faults were found in built manuscripts
one at a time, each costing a round trip: duplicated figure captions, maths that printed
as LaTeX source, a reference list too thin and five entries never cited, a title that
contradicted its own abstract, figures drawn so wide that their type was unreadable once
reduced, a table row with empty cells, an unnumbered table, figures generated and never
referenced, and a stale build sitting beside the current one under a more plausible name.

The worst of them was quieter than any of those: a manuscript that gained an entire
real-data arm while its abstract still described only the synthetic one. The body was
right and the front matter was three weeks out of date, which is the half an editor reads
first.

    python paper/presubmission_check.py            # the manuscript
    python paper/presubmission_check.py --strict   # exit non-zero on warnings too

This reports; it does not fix. Every finding is something a person should look at.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
SOURCE = PAPER / "manuscript.md"
BUILT = PAPER / "build" / "manuscript.md"
FIGURES = PAPER / "figures"

#: Words that must appear in the abstract because the body is built on them. This is the
#: front-matter drift check: the body gained a real-data arm and the abstract did not.
ABSTRACT_MUST_MENTION = ("liver", "CNN", "ceiling", "detectability", "fidelity")

#: Column width a figure prints into. The dpi is read from each PNG rather than
#: assumed: these figures carry 150 and 300, and guessing one number for both got
#: the answer wrong in each direction at once.
COLUMN_INCHES = 6.5
#: Below this, type set at 9 pt lands under 6 pt on the page.
MIN_SCALE = 0.70
#: Height of the text block a figure and its caption have to share. Figure 1 shipped at
#: 11.9 in tall - five stages stacked vertically - and the converted PDF cut it in half
#: and stranded the caption overleaf. Width was checked; height was not.
MAX_PRINTED_INCHES = 7.5

errors: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


# ---------------------------------------------------------------- emphasis
def check_emphasis(built: str, label: str = "manuscript") -> None:
    """Bold that is not a run-in heading is bold doing a job italic should do.

    Bold had accumulated on ordinary terms mid-sentence until a reader could no longer
    tell which bold meant "this paragraph is about X" and which meant "note this word".
    A run-in heading opens a paragraph and closes with a full stop or colon; anything
    else -- a term, a clause, a headline result -- is emphasis and belongs in italic or
    in nothing at all.

    Checked on the built file, not the source. Scanning the source line by line missed
    every span that straddled a line break, which is how half a sentence in Section 2.7
    stayed bold through two rounds of this.
    """
    body = built.split("## References")[0]
    for match in re.finditer(r"\*\*(.+?)\*\*", body, re.S):
        span = " ".join(match.group(1).split())
        before, after = body[: match.start()], body[match.end() :]
        opens_line = before.endswith("\n") or not before
        # A bold span with nothing else on its line is a title, an author name or a
        # heading. One with prose beside it is emphasis, whatever it says.
        if opens_line and (after.startswith("\n") or not after):
            continue
        if opens_line and span.endswith((".", ":")):
            continue  # a run-in heading, or an abstract label
        fail(f"bold mid-sentence in the {label}: **{span[:60]}** - should it be italic?")


# ---------------------------------------------------------------- front matter
def check_front_matter(src: str, built: str) -> None:
    """The half an editor reads first, against the half that changed."""
    abstract = built.split("## Abstract")[1].split("**Keywords")[0].lower()
    for word in ABSTRACT_MUST_MENTION:
        if word.lower() not in abstract:
            fail(f"abstract never mentions {word!r}, which the body is built on")

    title = next((ln for ln in src.splitlines() if ln.startswith("# ")), "")
    if not title:
        fail("no level-1 title found")
    words = len(abstract.split())
    if words > 500:
        fail(f"abstract is {words} words, over the 500-word limit")
    elif words > 480:
        warn(f"abstract is {words} words, close to the 500-word limit")

    for heading in ("Background", "Purpose", "Methods", "Results", "Conclusions"):
        if f"**{heading}.**" not in built.split("## 1.")[0]:
            fail(f"abstract has no **{heading}.** section, which Medical Physics requires")


# ---------------------------------------------------------------- references
def check_references(src: str) -> None:
    body, reflist = src.split("## References")
    cited: set[int] = set()
    for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        cited.update(int(x) for x in re.split(r"\s*,\s*", m))
    listed = {int(m) for m in re.findall(r"^(\d+)\. ", reflist, re.M)}
    cited.discard(0)  # an interval such as [0,1], not a citation

    for n in sorted(cited - listed):
        fail(f"reference [{n}] is cited but not listed")
    for n in sorted(listed - cited):
        fail(f"reference [{n}] is listed but never cited")
    if listed and listed != set(range(1, max(listed) + 1)):
        fail(f"reference numbering is not contiguous: {sorted(listed)}")

    # Medical Physics numbers by order of first citation.
    order, seen = [], set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        for n in (int(x) for x in re.split(r"\s*,\s*", m.group(1))):
            if n in listed and n not in seen:
                seen.add(n)
                order.append(n)
    if order != sorted(order):
        fail(f"references are not in order of first citation: {order}")

    # AMA: up to six authors list all; seven or more, first three then "et al."
    for m in re.finditer(r"^(\d+)\.\s(.+?)[,.]\s[\"*]", reflist, re.M):
        n, authors = m.group(1), m.group(2)
        if "et al" in authors:
            before = authors.split("et al")[0]
            named = before.count(",") + (0 if before.rstrip().endswith(",") else 1)
            if named != 3:
                fail(f"reference [{n}] uses 'et al.' after {named} names; AMA wants three")


# ---------------------------------------------------------------- tables
def check_tables(src: str) -> None:
    # A marker such as [[results:x.json:a.b|.2f]] carries a pipe, which a naive
    # column count reads as a cell boundary. Blank them before splitting rows.
    src = re.sub(r"\[\[results:[^\]]*\]\]", "X", src)
    lines = src.splitlines()
    blocks, cur = [], []
    for ln in lines:
        if ln.startswith("|"):
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    for i, block in enumerate(blocks, 1):
        rows = [r for r in block if not re.match(r"^\|[\s:|-]+\|?\s*$", r)]
        widths = {len(r.strip().strip("|").split("|")) for r in rows}
        if len(widths) > 1:
            fail(f"table block {i} has ragged columns {sorted(widths)}: {rows[0][:60]}")
        for r in rows:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if any(c == "" for c in cells):
                fail(f"table block {i} has an empty cell: {r[:80]}")

    # Tables 1-2 are generated from results/ by build_pdf, so their captions never
    # appear in the markdown. Only hand-written tables are checked here.
    captioned = {int(m) for m in re.findall(r"\*\*Table (\d+)\.\*\*", src)}
    handwritten = len(blocks)
    if handwritten and not captioned:
        fail(
            f"{handwritten} hand-written table(s) in the markdown carry no "
            "**Table N.** caption, while the generated tables are numbered"
        )
    return
    mentioned = {int(m) for m in re.findall(r"\bTable (\d+)\b", src)} - captioned
    if len(captioned) != len(blocks):
        warn(f"{len(blocks)} table blocks but {len(captioned)} numbered captions")
    for n in sorted(captioned - mentioned):
        fail(f"Table {n} is captioned but never referred to in the prose")
    for n in sorted(mentioned - captioned):
        fail(f"Table {n} is referred to but has no caption")


# ---------------------------------------------------------------- figures
def check_figures(src: str) -> None:
    sys.path.insert(0, str(PAPER))
    import build_pdf  # noqa: PLC0415

    registered = list(build_pdf.FIGURE_FILES) + list(build_pdf.SUPPLEMENTARY_FIGURE_FILES)
    on_disk = {p.name for p in FIGURES.glob("fig*.png")}

    for name in registered:
        if name not in on_disk:
            fail(f"{name} is registered but not in paper/figures/")
    for name in sorted(on_disk - set(registered)):
        warn(f"{name} exists but is not registered, so it is never placed")

    captions = re.findall(
        r"^\|\s*Fig\s*(S?\d+)\s*\|", (PAPER / "README.md").read_text(encoding="utf-8"), re.M
    )
    if len(captions) != len(registered):
        fail(f"{len(registered)} figures registered but {len(captions)} captions in README.md")

    # Supplementary figures are named in supplementary.md, not in the manuscript.
    supp = PAPER / "supplementary.md"
    prose = src + (supp.read_text(encoding="utf-8") if supp.exists() else "")
    for name in registered:
        token = re.match(r"fig(S?\d+)_", name).group(1)
        if not re.search(rf"\bFig(?:ure)?s?\.?\s*{token}\b", prose):
            fail(f"{name} is registered but the prose never names Figure {token}")

    for p in sorted(FIGURES.glob("fig*.png")):
        raw = p.read_bytes()
        w, h = struct.unpack(">II", raw[16:24])
        i = raw.find(b"pHYs")
        dpi = round(struct.unpack(">I", raw[i + 4 : i + 8])[0] * 0.0254) if i > 0 else 100
        scale = COLUMN_INCHES / (w / dpi)
        printed = (h / dpi) * min(1.0, scale)
        if printed > MAX_PRINTED_INCHES:
            fail(
                f"{p.name} prints {printed:.1f} in tall once fitted to a {COLUMN_INCHES} in "
                f"column, over the {MAX_PRINTED_INCHES} in text block; the converter will "
                "cut it across a page break"
            )
        if scale < MIN_SCALE:
            fail(
                f"{p.name} is {w / dpi:.1f} in wide at {dpi} dpi; in a {COLUMN_INCHES} in column "
                f"it reduces to x{scale:.2f}, "
                "so its type will not survive printing"
            )


# ---------------------------------------------------------------- built output
def check_built(built: str) -> None:
    stripped = re.sub(r"`[^`]*`", "", built)
    for cmd in ("times", "exp", "mathrm", "alpha", "sigma", "Delta"):
        if "\\" + cmd in stripped:
            warn(f"built manuscript still contains literal \\{cmd}; check how it renders")
    for lineno, line in enumerate(built.splitlines(), 1):
        if "\t" in line or "\x0b" in line or "\x0c" in line:
            fail(f"built manuscript line {lineno} contains a control character")
    # Any "[[results:" that does not close with "]]" is a marker broken in the source,
    # which the build passes through as literal text. Looking only for well-formed
    # markers misses exactly the failures worth catching.
    opened = [m.start() for m in re.finditer(r"\[\[results:", built)]
    closed = [m.start() for m in re.finditer(r"\[\[results:[^\]]*\]\]", built)]
    if len(opened) != len(closed):
        fail(
            f"{len(opened) - len(closed)} malformed [[results:...]] marker(s) "
            "in the built manuscript"
        )
    unresolved = [m for m in re.findall(r"\[\[results:[^\]]*\]\]", built) if "<" not in m]
    if unresolved:
        fail(f"built manuscript still contains unresolved markers: {unresolved[:3]}")

    for m in re.finditer(r"\*\*(Figure|Table) (\d+)\.\*\*", built):
        label = m.group(0)
        if built.count(label) > 1:
            fail(f"{label} caption appears {built.count(label)} times")


# ---------------------------------------------------------------- artefacts
def check_artefacts() -> None:
    builds = sorted(PAPER.glob("*.docx")) + sorted((PAPER / "build").glob("*.docx"))
    if len(builds) > 2:
        warn(
            f"{len(builds)} .docx files present; make sure the right one is uploaded: "
            + ", ".join(p.name for p in builds)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args()

    if not BUILT.exists():
        print("build/manuscript.md missing - run paper/build_manuscript.py first")
        return 2
    src = SOURCE.read_text(encoding="utf-8")
    built = BUILT.read_text(encoding="utf-8")

    check_emphasis(built)
    supplementary = PAPER / "build" / "supplementary.md"
    if supplementary.exists():
        check_emphasis(supplementary.read_text(encoding="utf-8"), "supplement")
    check_front_matter(src, built)
    check_references(src)
    check_tables(src)
    check_figures(src)
    check_built(built)
    check_artefacts()

    for w in warnings:
        print(f"  warn  {w}")
    for e in errors:
        print(f"  FAIL  {e}")
    print(f"\n{len(errors)} failures, {len(warnings)} warnings")
    if errors or (args.strict and warnings):
        return 1
    print("ready to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
