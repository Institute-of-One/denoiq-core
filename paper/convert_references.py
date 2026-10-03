"""Rewrite the reference list in the journal's numbered Vancouver style, from Crossref.

Why from Crossref and not by hand
---------------------------------
Reformatting twenty-four references by hand is twenty-four chances to drop an author, mistype a
page range or keep a volume that belongs to a different paper. This programme has shipped a DOI
pointing at the wrong article before. So the entries are rebuilt from the publisher's own record:
the DOI is taken from the existing entry, resolved, and the Vancouver line generated from what
comes back. What the author supplies is the DOI; what the publisher supplies is everything else.

The two books have no DOI and are carried through by hand, marked as such in ``BOOKS``.

Physica Medica's requirements, from its Guide for Authors:

* numbered in the order they appear in the text, cited as ``[1]``;
* surname then initials, no punctuation inside the initials;
* more than six authors: the first six, then ``et al.``;
* journal titles abbreviated per the List of Title Word Abbreviations;
* shortened last page number, so ``2295-305`` rather than ``2295-2305``.

Run it, read the output, and paste it in::

    python paper/convert_references.py            # print the converted list
    python paper/convert_references.py --write    # rewrite the manuscript's References section
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "manuscript.md"
MAILTO = "yamamoto@lisit.jp"

#: Entries with no DOI, written out because a publisher record cannot supply them. Keyed by the
#: number they carry in the current list so the order is preserved exactly.
BOOKS = {
    1: "Cover TM, Thomas JA. Elements of information theory. 2nd ed. Hoboken (NJ): Wiley; 2006.",
    16: "Barrett HH, Myers KJ. Foundations of image science. Hoboken (NJ): Wiley; 2004.",
}

#: Zenodo mints DataCite DOIs, which Crossref does not hold, so a software citation has to be
#: written here rather than resolved. The DOI is still checked: it is the one the archive
#: returned for that release.
SOFTWARE = {
    18: (
        "Yamamoto S. taskiq-core: task-based image quality on synthetic phantoms. Zenodo; "
        "2026. https://doi.org/10.5281/zenodo.21422924."
    ),
}

#: LTWA abbreviations Crossref does not supply in the form the journal wants. Crossref returns
#: the full container title and sometimes a short-container-title that is not LTWA; this maps the
#: ones this paper cites. Anything missing is reported rather than guessed.
ABBREVIATIONS = {
    (
        "Philosophical Transactions of the Royal Society of London. Series A, "
        "Containing Papers of a Mathematical or Physical Character"
    ): "Philos Trans R Soc Lond A",
    "IEEE Transactions on Medical Imaging": "IEEE Trans Med Imaging",
    "IEEE Transactions on Image Processing": "IEEE Trans Image Process",
    "Medical Physics": "Med Phys",
    "Journal of the Optical Society of America A": "J Opt Soc Am A",
    "Journal of the Optical Society of America": "J Opt Soc Am",
    "Radiology": "Radiology",
    "American Journal of Roentgenology": "AJR Am J Roentgenol",
    "European Radiology": "Eur Radiol",
    "Physics in Medicine & Biology": "Phys Med Biol",
    "Physics in Medicine and Biology": "Phys Med Biol",
    "Medical Image Analysis": "Med Image Anal",
    "Scientific Reports": "Sci Rep",
    "Journal of Medical Imaging": "J Med Imaging",
}


def crossref(doi: str) -> dict:
    """The publisher's record for one DOI."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    request = urllib.request.Request(url, headers={"User-Agent": f"ldct-paper (mailto:{MAILTO})"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["message"]


def authors(record: dict) -> str:
    """Surname then initials, first six then ``et al.``."""
    people = record.get("author") or []
    names = []
    for person in people:
        family = person.get("family")
        if not family:
            names.append(person.get("name", "?"))
            continue
        given = person.get("given", "")
        initials = "".join(part[0] for part in re.split(r"[\s.\-]+", given) if part)
        names.append(f"{family} {initials}".strip())
    if len(names) > 6:
        return ", ".join(names[:6]) + ", et al."
    return ", ".join(names)


def shorten_pages(pages: str) -> str:
    """``2295-2305`` becomes ``2295-305``: the journal asks for the shortened last page."""
    match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", pages.strip())
    if not match:
        return pages.strip()
    first, last = match.groups()
    if len(first) == len(last):
        for index in range(len(first)):
            if first[index] != last[index]:
                return f"{first}-{last[index:]}"
        return f"{first}-{last[-1]}"
    return f"{first}-{last}"


def vancouver(record: dict, doi: str, problems: list[str]) -> str:
    """One Vancouver entry from one Crossref record."""
    # Publishers put footnote marks in titles; they are not part of the title.
    title = re.sub(r"\s+", " ", (record.get("title") or ["?"])[0]).strip().rstrip(".*†‡§ ")
    container = (record.get("container-title") or [""])[0]
    abbreviation = ABBREVIATIONS.get(container)
    if abbreviation is None:
        short = (record.get("short-container-title") or [""])[0]
        abbreviation = short or container
        # Proceedings are cited as "In: <full name>", so a missing abbreviation is not a defect
        # there; only a journal article needs one.
        if container and record.get("type") not in {"proceedings-article", "book-chapter"}:
            problems.append(f"{doi}: no LTWA abbreviation for {container!r}, used {abbreviation!r}")
    year = record.get("issued", {}).get("date-parts", [[None]])[0][0]
    volume = record.get("volume", "")
    pages = shorten_pages(record.get("page", ""))

    # authors() already ends in a period when it ends in "et al.", so do not add a second one.
    names = authors(record)
    parts = [names if names.endswith(".") else names + ".", f"{title}."]
    if record.get("type") in {"proceedings-article", "book-chapter"}:
        parts.append(f"In: {container}; {year}." if container else f"{year}.")
        if pages:
            parts[-1] = parts[-1][:-1] + f", p. {pages}."
    else:
        tail = f"{abbreviation} {year}"
        if volume:
            tail += f";{volume}"
        if pages:
            tail += f":{pages}"
        parts.append(tail + ".")
    parts.append(f"https://doi.org/{doi}.")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the References section")
    args = parser.parse_args(argv)

    # The console here is cp932; the references are not.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    text = SOURCE.read_text(encoding="utf-8")
    head, _, tail = text.partition("## References")
    entries = re.findall(r"^(\d+)\. (.+)$", tail, flags=re.M)
    if not entries:
        print("no reference list found")
        return 1

    problems: list[str] = []
    converted: list[tuple[int, str]] = []
    for number, entry in entries:
        index = int(number)
        if index in BOOKS or index in SOFTWARE:
            converted.append((index, BOOKS.get(index) or SOFTWARE[index]))
            continue
        found = re.search(r"doi:([^\]\s]+)", entry)
        if not found:
            problems.append(f"[{index}] has no DOI and is not a known book: {entry[:70]}")
            converted.append((index, entry))
            continue
        doi = found.group(1).rstrip(".")
        try:
            record = crossref(doi)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            problems.append(f"[{index}] {doi} did not resolve: {exc}")
            converted.append((index, entry))
            continue
        converted.append((index, vancouver(record, doi, problems)))
        time.sleep(0.3)

    # The list stays a markdown ordered list, which both PDF layouts already render and number;
    # writing "[1] " as literal text instead gave two layouts that disagreed about what the page
    # said, which the style suite catches. What has to be Vancouver is the content of an entry,
    # not the bullet in front of it, and the journal typesets the bullet itself.
    lines = [f"{index}. {entry}" for index, entry in converted]
    print("\n".join(lines))
    if problems:
        print("\nproblems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)

    if args.write:
        if problems:
            print("\nrefusing to write while anything is unresolved", file=sys.stderr)
            return 1
        comment = (
            "<!-- Generated by paper/convert_references.py from Crossref records. Numbered\n"
            "     Vancouver style as Physica Medica requires: surname and initials, first six\n"
            "     authors then et al., LTWA journal abbreviations, shortened last page. Do not\n"
            "     hand-edit; rerun the script. -->\n\n"
        )
        SOURCE.write_text(
            head + "## References\n\n" + comment + "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {SOURCE}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
