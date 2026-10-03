"""Record which results file each generated figure was drawn from, and check it is still that one.

Why
---
Figure 10 was drawn from ``results/real_liver.json`` when the real-data arm had seven methods.
The arm grew to nine, every number in the text and in Table 3 moved with it, and the figure did
not: its generator crashed on the new labels, nobody re-ran it, and the PDF carried a Spearman
correlation of -0.29 beside a text that said -0.08 for six weeks. Nothing in the build noticed,
because placing a figure only needs the file to exist.

A modification time will not do: a fresh clone gives every file the same one. What is recorded
instead is the SHA-256 of the results file the figure was drawn from, so a results file that has
changed since makes the figure demonstrably stale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RECORD = PAPER / "figures" / "SOURCES.json"


def digest(path: Path) -> str:
    """SHA-256 of a file, as the record stores it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(figure: Path, sources: list[Path]) -> None:
    """Note that ``figure`` was drawn from ``sources`` as they are now."""
    entries = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    entries[figure.name] = {
        source.name: digest(source) for source in sorted(sources, key=lambda p: p.name)
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
        for name, recorded in sources.items():
            path = results / name
            if not path.exists():
                problems.append(f"{figure} was drawn from {name}, which is missing")
            elif digest(path) != recorded:
                problems.append(
                    f"{figure} was drawn from an older {name}; redraw it before the numbers in "
                    "the figure and the numbers in the text disagree"
                )
    return problems
