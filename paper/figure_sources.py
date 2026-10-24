"""Record which results file each generated figure was drawn from, and check it is still that one.

Why
---
Figure 10 was drawn from ``results/real_liver.json`` when the real-data arm had seven methods.
The arm grew to nine, every number in the text and in Table 3 moved with it, and the figure did
not: its generator crashed on the new labels, nobody re-ran it, and the PDF carried a Spearman
correlation of -0.29 beside a text that said -0.08 for six weeks. Nothing in the build noticed,
because placing a figure only needs the file to exist.

A modification time will not do: a fresh clone gives every file the same one. What is recorded
instead is the SHA-256 of the file the figure was drawn from, so a source that has changed since
makes the figure demonstrably stale.

A figure has two kinds of source, and recording only the first is not enough
----------------------------------------------------------------------------
The numbers are one. The *drawing code* is the other, and leaving it out let a second stale
figure through: the body text was converted to -ize spelling, ``denoiq_core/figures.py`` was
converted with it, and Figure 5 went on printing "channelised Hotelling (CHO)" in its legend
because nothing redrew it and nothing noticed. The results had not changed, so a results-only
record was satisfied. Both are recorded now.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
RECORD = PAPER / "figures" / "SOURCES.json"

#: Repositories other than this one that draw a figure in this paper, by the name used as the
#: prefix of a recorded key. Figure 9 is drawn by ldct-io, which the manuscript cites by DOI,
#: and it was the one figure nothing here watched: it had no entry at all, so neither its data
#: nor its drawing code was checked, and it reached a journal 12.3 in wide with a result
#: sentence painted into it. A figure drawn elsewhere still has to be watched from here.
COMPANION_REPOS = {
    "ldct-io": Path(os.environ.get("LDCT_IO_REPO") or r"D:\DevGit\ldct-io"),
}


def _key(source: Path) -> str:
    """How a code file is keyed: relative to this repository, or ``<companion>:<path>``."""
    source = source.resolve()
    try:
        return source.relative_to(REPO).as_posix()
    except ValueError:
        pass
    for name, root in COMPANION_REPOS.items():
        try:
            return f"{name}:{source.relative_to(root.resolve()).as_posix()}"
        except (ValueError, OSError):
            continue
    raise ValueError(
        f"{source} is in neither this repository nor a companion listed in COMPANION_REPOS; "
        "add it there rather than recording a path that only exists on one machine"
    )


def _resolve(key: str) -> Path | None:
    """The file a recorded key names, or None when its companion is not on this machine."""
    if ":" not in key:
        return REPO / key
    name, _, relative = key.partition(":")
    root = COMPANION_REPOS.get(name)
    return None if root is None or not root.exists() else root / relative


def digest(path: Path) -> str:
    """SHA-256 of a file's content, as the record stores it.

    Line endings are normalised first. This repository has ``core.autocrlf = true`` and no
    ``.gitattributes``, so it stores LF and checks out CRLF on Windows: hashing the bytes on
    disk records a digest of the checkout rather than of the content, and every figure then
    reads as stale on Linux. CI went red exactly that way, on the commit that introduced this
    check. Every file recorded here is JSON or Python, so this is always safe.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def record(figure: Path, results: list[Path] | None = None, code: list[Path] | None = None) -> None:
    """Note that ``figure`` was drawn from these results, by this code, as they are now.

    Results are keyed by bare name because they are looked up inside ``results/``; code is keyed
    by its path relative to the repository, because it is spread across packages.
    """
    entries = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    entries[figure.name] = {
        "results": {
            source.name: digest(source) for source in sorted(results or [], key=lambda p: p.name)
        },
        "code": {
            _key(source): digest(source) for source in sorted(code or [], key=lambda p: p.name)
        },
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stale(results: Path) -> list[str]:
    """Figures whose recorded source digest no longer matches the file on disk."""
    if not RECORD.exists():
        return []
    entries = json.loads(RECORD.read_text(encoding="utf-8"))
    problems: list[str] = []
    for figure, sources in entries.items():
        # An entry written before code was recorded is a flat map of results; read it as one
        # rather than crashing, so an old checkout reports what it can.
        results_sources = sources.get("results", sources) if "code" in sources else sources
        for name, recorded in results_sources.items():
            path = results / name
            if not path.exists():
                problems.append(f"{figure} was drawn from {name}, which is missing")
            elif digest(path) != recorded:
                problems.append(
                    f"{figure} was drawn from an older {name}; redraw it before the numbers in "
                    "the figure and the numbers in the text disagree"
                )
        for name, recorded in sources.get("code", {}).items():
            path = _resolve(name)
            if path is None:
                # The companion repository is not on this machine. Say so rather than passing
                # in silence: the figure is unchecked here, and CI elsewhere is not the place
                # this gets noticed. unchecked() is what the gate reports as a warning.
                continue
            if not path.exists():
                problems.append(f"{figure} was drawn by {name}, which is missing")
            elif digest(path) != recorded:
                problems.append(
                    f"{figure} was drawn by an older {name}; redraw it before the figure and the "
                    "text disagree about a label"
                )
    return problems


def unchecked() -> list[str]:
    """Recorded generators that live in a companion repository absent from this machine.

    The gate reports these as warnings. A figure drawn elsewhere is still this paper's figure,
    and "nothing complained" has to be distinguishable from "nothing looked".
    """
    if not RECORD.exists():
        return []
    entries = json.loads(RECORD.read_text(encoding="utf-8"))
    out: list[str] = []
    for figure, sources in entries.items():
        for name in sources.get("code", {}):
            if ":" in name and _resolve(name) is None:
                out.append(
                    f"{figure} is drawn by {name}, whose repository is not on this machine, "
                    "so it was not checked against what drew it"
                )
    return out
