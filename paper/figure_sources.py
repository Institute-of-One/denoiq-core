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
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
RECORD = PAPER / "figures" / "SOURCES.json"


def digest(path: Path) -> str:
    """SHA-256 of a file, as the record stores it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            source.resolve().relative_to(REPO).as_posix(): digest(source)
            for source in sorted(code or [], key=lambda p: p.name)
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
            path = REPO / name
            if not path.exists():
                problems.append(f"{figure} was drawn by {name}, which is missing")
            elif digest(path) != recorded:
                problems.append(
                    f"{figure} was drawn by an older {name}; redraw it before the figure and the "
                    "text disagree about a label"
                )
    return problems
