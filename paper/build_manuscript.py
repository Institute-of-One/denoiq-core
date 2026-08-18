"""Resolve the manuscript's number markers from ``results/``.

    python paper/build_manuscript.py            # -> paper/build/{manuscript,supplementary}.md
    python paper/build_manuscript.py --check    # fail if either built file is out of date

Every number in ``paper/manuscript.md`` is written as a marker naming the results file and
the path it comes from::

    [[results:summary.json:max_excess_over_ceiling|.2e]]
    [[results:dose_sweep.json:rows[mas=25.0,denoiser=raw].d_prime_ideal|.2f]]

The build substitutes the value found in ``results/``. A number that cannot be resolved is an
error, not a warning — which means a stale or hand-edited figure in the text cannot survive a
rebuild, and ``tests/test_manuscript_consistency.py`` runs the rebuild on every CI run.

Path syntax
-----------
``a.b``            nested keys
``a["k.with.dots"]``  a key that a dotted path could not express (denoiser labels, mostly)
``a[2]``           list index (negative counts from the end)
``a[k=v,k2=v2]``   the single element of a list whose fields match (ambiguity is an error)
``|fmt``           optional trailing Python format spec, e.g. ``.3f``
``|sciN``          scientific notation with N decimals, typeset as ``4.76×10⁻⁷`` rather than
                   ``4.76e-07`` — journal prose wants the former, and a number that has to be
                   retyped to look right is a number that can drift
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
DEFAULT_RESULTS = REPO_DIR / "results"
DEFAULT_SOURCE = PAPER_DIR / "manuscript.md"
DEFAULT_OUTPUT = PAPER_DIR / "build" / "manuscript.md"

#: Every document rendered by a bare ``python paper/build_manuscript.py``. The supplementary
#: material cites ``results/`` through the same markers, so it goes stale in the same way; the
#: PDF build tells the reader to run this script, and that instruction has to be sufficient.
DOCUMENTS = ("manuscript", "supplementary")

#: Where the archived-release metadata lives. The version DOI is not a measurement, so it does
#: not belong in ``results/``; it is minted by Zenodo when the release is cut.
DEFAULT_RELEASE = PAPER_DIR / "release.json"

#: ``[[results:<file>:<path>]]`` or ``[[results:<file>:<path>|<format>]]``. The format is
#: separated by a pipe rather than a colon because paths contain brackets and dots but never
#: a pipe, and a colon would be ambiguous with a slice-looking path.
#: The trailing lookahead is what lets a path end in a bracket: ``atlas.kv[0]]]`` must close
#: the marker after ``[0]``, not after ``[0``.
MARKER = re.compile(r"\[\[results:([^:|\]]+):(.+?)(?:\|([^|\]]+))?\]\](?!\])")
#: ``[[release:<field>]]`` — the archived release's metadata, from ``paper/release.json``.
#: ``archive_statement`` is derived rather than stored: it is the sentence the manuscript needs,
#: and it differs between a draft (no DOI yet) and a submission (DOI required).
RELEASE_MARKER = re.compile(r"\[\[release:([a-z_]+)\]\]")
_STEP = re.compile(r"([^.\[\]]+)|\[([^\]]*)\]")
#: Kept as a capturing group so ``re.split`` returns the comments too, in place.
_COMMENT = re.compile(r"(<!--.*?-->)", re.S)


class ResolutionError(LookupError):
    """A marker that does not name anything in ``results/``."""


#: Unicode superscripts, for the exponent of ``|sciN``.
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def scientific(value: float, digits: int) -> str:
    """``4.756e-07`` → ``4.76×10⁻⁷``: the form journal prose uses.

    Exact powers of ten keep the mantissa (``1×10⁻⁶``) rather than dropping it, so the reader
    never has to wonder whether a mantissa was rounded away.
    """
    text = f"{float(value):.{int(digits)}e}"
    mantissa, exponent = text.split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def _match_value(candidate: Any, wanted: str) -> bool:
    """Compare a field to a marker's literal, numerically when both look like numbers.

    The numeric comparison is tolerant, because a sweep axis built with ``geomspace`` stores
    ``12.500000000000002`` and a manuscript will always write ``12.5``. The tolerance is
    relative and tiny — it identifies grid points, it does not merge them.
    """
    if str(candidate) == wanted:
        return True
    try:
        left, right = float(candidate), float(wanted)
    except (TypeError, ValueError):
        return False
    return abs(left - right) <= 1e-9 * max(1.0, abs(right))


def resolve(payload: Any, path: str) -> Any:
    """Follow a marker path into a loaded results file."""
    current = payload
    for name, bracket in _STEP.findall(path):
        if name:
            if not isinstance(current, dict) or name not in current:
                raise ResolutionError(f"{path!r}: no key {name!r} at this level")
            current = current[name]
            continue
        if len(bracket) >= 2 and bracket[0] == bracket[-1] and bracket[0] in "\"'":
            key = bracket[1:-1]
            if not isinstance(current, dict) or key not in current:
                raise ResolutionError(f"{path!r}: no key {key!r} at this level")
            current = current[key]
            continue
        if not isinstance(current, list):
            raise ResolutionError(f"{path!r}: {bracket!r} indexes something that is not a list")
        if re.fullmatch(r"-?\d+", bracket):
            index = int(bracket)
            if not -len(current) <= index < len(current):
                raise ResolutionError(f"{path!r}: index {index} is out of range")
            current = current[index]
            continue
        criteria = [part.split("=", 1) for part in bracket.split(",")]
        matches = [
            item
            for item in current
            if isinstance(item, dict)
            and all(key in item and _match_value(item[key], value) for key, value in criteria)
        ]
        if len(matches) != 1:
            raise ResolutionError(
                f"{path!r}: [{bracket}] matched {len(matches)} rows, expected exactly 1"
            )
        current = matches[0]
    return current


def release_value(field: str, release: dict[str, Any], *, submission: bool) -> str:
    """One ``[[release:...]]`` substitution.

    ``archive_statement`` is the only derived field. Before the release is minted there is no
    DOI to print, and inventing a placeholder token would put a string into the text that a
    reviewer could mistake for a citation; the draft therefore states plainly that the DOI is
    registered at submission, and ``--submission`` refuses to render at all until it exists.
    """
    if field != "archive_statement":
        value = release.get(field)
        if value is None:
            raise ResolutionError(f"release.json has no value for {field!r}")
        return str(value)
    doi = release.get("version_doi")
    version = release.get("version", "")
    if doi:
        return f"archived at {release.get('archive', 'Zenodo')} as version {version}, doi:{doi}"
    if submission:
        raise ResolutionError(
            "release.json has no version_doi: mint the archived release before building a "
            "submission draft (see docs/RELEASE.md)"
        )
    return (
        f"archived at {release.get('archive', 'Zenodo')} as version {version} on release; "
        "the version DOI is inserted at submission"
    )


def render(
    source: str,
    results: Path,
    *,
    release: Path | dict[str, Any] | None = None,
    submission: bool = False,
) -> str:
    """Substitute every marker in ``source`` with its value from ``results``."""
    cache: dict[str, Any] = {}
    problems: list[str] = []
    if release is None:
        release = DEFAULT_RELEASE
    if isinstance(release, Path):
        release = json.loads(release.read_text(encoding="utf-8")) if release.exists() else {}

    def substitute_release(match: re.Match[str]) -> str:
        try:
            return release_value(match.group(1), release, submission=submission)
        except ResolutionError as exc:
            problems.append(str(exc))
            return match.group(0)

    def substitute(match: re.Match[str]) -> str:
        filename, path, spec = match.group(1), match.group(2), match.group(3)
        if filename not in cache:
            file_path = results / filename
            if not file_path.exists():
                problems.append(f"{filename} is missing from {results}")
                return match.group(0)
            cache[filename] = json.loads(file_path.read_text(encoding="utf-8"))
        try:
            value = resolve(cache[filename], path)
        except ResolutionError as exc:
            problems.append(str(exc))
            return match.group(0)
        if spec and re.fullmatch(r"sci\d+", spec):
            try:
                return scientific(value, int(spec[3:]))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                problems.append(f"{path!r}: cannot format {value!r} as {spec!r} ({exc})")
                return match.group(0)
        if spec:
            try:
                return format(value, spec)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                problems.append(f"{path!r}: cannot format {value!r} as {spec!r} ({exc})")
                return match.group(0)
        return str(value)

    # Markers inside HTML comments are documentation of the syntax, not citations: the file
    # explains its own notation in a comment at the top, and that example must not be
    # resolved (there is no results file called "<file>").
    pieces = _COMMENT.split(source)
    rendered = "".join(
        piece
        if _COMMENT.fullmatch(piece)
        else RELEASE_MARKER.sub(substitute_release, MARKER.sub(substitute, piece))
        for piece in pieces
    )
    if problems:
        raise ResolutionError(
            "the manuscript cites values that cannot be resolved:\n  " + "\n  ".join(problems)
        )
    # Medical Physics asks for figure captions beneath each figure and listed again
    # after the references. build_pdf places the first; this appends the second,
    # generated from paper/README.md so they are written in exactly one place. Only
    # the document that has a reference list gets a caption list after it.
    if "## References" in rendered:
        import build_pdf  # noqa: PLC0415

        captions = build_pdf.caption_list_block(PAPER_DIR / "README.md")
        rendered = rendered.rstrip("\n") + "\n\n" + captions
    return rendered


def build(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    results: Path = DEFAULT_RESULTS,
    *,
    submission: bool = False,
) -> str:
    """Render the manuscript and write it. Returns the rendered text."""
    text = render(source.read_text(encoding="utf-8"), results, submission=submission)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Without an explicit --source, every document in DOCUMENTS is rendered.
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the built file differs from a fresh render",
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="render for submission: fail unless the archived release DOI has been minted",
    )
    args = parser.parse_args(argv)

    if args.source is not None or args.output is not None:
        source = args.source or DEFAULT_SOURCE
        output = args.output or PAPER_DIR / "build" / source.name
        jobs = [(source, output)]
    else:
        jobs = [
            (PAPER_DIR / f"{name}.md", PAPER_DIR / "build" / f"{name}.md") for name in DOCUMENTS
        ]

    status = 0
    for source, output in jobs:
        status = max(
            status,
            _render_one(source, output, args.results, check=args.check, submission=args.submission),
        )
    return status


def _render_one(
    source: Path, output: Path, results: Path, *, check: bool, submission: bool = False
) -> int:
    """Render one document, or verify that its built copy is current."""
    try:
        text = render(source.read_text(encoding="utf-8"), results, submission=submission)
    except ResolutionError as exc:
        print(exc, file=sys.stderr)
        return 2

    if check:
        if not output.exists():
            print(f"{output} has never been built", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != text:
            print(f"{output} is out of date: rebuild it", file=sys.stderr)
            return 1
        print(f"{output} is up to date")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
