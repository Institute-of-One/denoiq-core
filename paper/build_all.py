"""Rebuild every paper artefact in dependency order, and leave nothing older behind.

Why this exists
---------------
``paper/build/`` had accumulated four generations of the same document under three naming
schemes. ``manuscript.pdf`` was from one day, ``manuscript_v2.docx`` from the same day by a
different renderer, and ``manuscript_v2.pdf`` and ``supplementary_v2.pdf`` were two months old
and carried a different title, a different conclusion and seven arms instead of nine. Opening the
wrong one meant reviewing a paper that no longer exists, and nothing said which was which.

So the build owns the directory. Everything is regenerated from ``results/`` in dependency order,
and any file in ``paper/build/`` that this run did not write is removed. A stale artefact cannot
survive a build, which is the only arrangement under which "open the newest one" is safe advice.

    python paper/build_all.py              # rebuild everything, delete what is stale
    python paper/build_all.py --check      # fail if anything is stale or missing; write nothing

The order matters: results consolidate first, then the Markdown resolves markers against them,
then the renderers read that Markdown, then the submission set copies the renderers' output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
BUILD = PAPER / "build"

#: In dependency order. Each is run as its own process so that one failing cannot leave a
#: half-imported module behind for the next.
STEPS: list[tuple[str, list[str]]] = [
    ("consolidate the real-data runs", ["build_real_liver_results.py"]),
    ("redraw the real-data scatter", ["make_real_liver_figure.py"]),
    ("resolve the manuscript's markers", ["build_manuscript.py"]),
    ("render the review PDF", ["build_pdf.py"]),
    ("render the supplement PDF", ["build_pdf.py", "--document", "supplementary"]),
    ("assemble the submission set", ["make_submission_files.py"]),
]

#: Written by the steps above, directly or through them. Anything else under build/ is stale.
EXPECTED = {
    "manuscript.md",
    "supplementary.md",
    "manuscript.pdf",
    "supplementary.pdf",
}


def run(step: str, argv: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    result = subprocess.run([sys.executable, str(PAPER / argv[0]), *argv[1:]], cwd=PAPER.parent)
    if result.returncode != 0:
        raise SystemExit(f"{argv[0]} failed; nothing further was rebuilt")


def sweep(*, dry_run: bool) -> list[str]:
    """Remove build artefacts this build does not produce. Returns what was (or would be) gone."""
    removed: list[str] = []
    for path in sorted(BUILD.iterdir()):
        if path.is_dir():
            continue  # submission/ is owned by make_submission_files.py, which clears it itself
        if path.name in EXPECTED:
            continue
        removed.append(path.name)
        if not dry_run:
            path.unlink()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale artefacts and rebuild nothing; for a pre-submission gate",
    )
    args = parser.parse_args(argv)

    if args.check:
        if not BUILD.exists():
            print(f"{BUILD} does not exist; run python paper/build_all.py")
            return 1
        stale = sweep(dry_run=True)
        missing = sorted(name for name in EXPECTED if not (BUILD / name).exists())
        if stale or missing:
            if stale:
                print("stale artefacts in paper/build/ (this build does not produce them):")
                for name in stale:
                    print(f"  {name}")
            if missing:
                print("missing:")
                for name in missing:
                    print(f"  {name}")
            print("\nrun python paper/build_all.py")
            return 1
        print("paper/build/ holds exactly what this build produces")
        return 0

    BUILD.mkdir(parents=True, exist_ok=True)
    for step, argv_step in STEPS:
        run(step, argv_step)

    removed = sweep(dry_run=False)
    print()
    if removed:
        print("removed stale artefacts:")
        for name in removed:
            print(f"  {name}")
    print("\npaper/build/ now holds only this build's output, and submission/ beside it.")
    print("The submission set is what gets uploaded; manuscript.pdf is for reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
