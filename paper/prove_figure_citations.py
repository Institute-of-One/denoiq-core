"""Break the figure-citation check on purpose, and require it to notice.

A checker that has never failed is not known to work. This one had never failed correctly: it
searched for "Figure 3" one figure at a time, could not see the 3 in "Figures 2 and 3", and
reported a correct manuscript as citing no figure 3 -- which is how the Physica Medica
submission went out with a red gate that nobody could act on, because the finding was false.

Each case rewrites one sentence of the real manuscript and states what the check must say about
figure 3. Other failures (the oversized gallery, for one) are ignored here on purpose: this
proves one check, and a proof that passes because something else failed proves nothing.

    python paper/prove_figure_citations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER))

import presubmission_check as gate  # noqa: E402

#: The sentence in the manuscript that cites figures 2 and 3 together.
REAL = "Figures 2 and 3 give the"

#: Each case: what replaces that sentence opening, whether figure 3 must be reported missing,
#: whether a comma-continued citation must be refused, and why the case exists.
CASES = (
    (REAL, False, False, "the manuscript as written: a plural citation names both figures"),
    ("Figure 2 gives the", True, False, "figure 3 really is uncited; the check must say so"),
    # Both are reported, and that is right. The citation stops at the comma, so figure 3 has
    # no citation the parser can read; saying only "unparseable" would leave the author to
    # guess which figure was dropped. This expectation started out as comma-only and was
    # corrected by running it -- the checker was right and the proof was wrong.
    (
        "Figures 2, 3 give the",
        True,
        True,
        "a comma continuation is refused, and the figure after it is then uncited",
    ),
    ("Figures 2-3 give the", False, False, "a range names every figure between its ends"),
    ("Figures 2 & 3 give the", False, False, "an ampersand joins a list as 'and' does"),
    ("Figures 3 and 2 give the", False, False, "order is not this check's business"),
    (
        "<!-- Figures 2 and 3 give the -->",
        True,
        False,
        "a citation inside a comment is an editorial note, not a citation",
    ),
)


def _figure_three_missing(errors: list[str]) -> bool:
    return any("fig3_task_vs_dose.png is registered but the prose never names" in e for e in errors)


def _comma_refused(errors: list[str]) -> bool:
    return any("continued by a comma" in e for e in errors)


def main() -> int:
    # The source, not the build: the build appends the caption list, and "**Figure 3.** Task
    # d' against relative dose" satisfies any citation pattern, so a proof run against the
    # build can never see figure 3 go uncited. That is also why check_figures reads the source.
    source = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    if source.count(REAL) != 1:
        print(f"the sentence this proof rewrites is not in the manuscript source: {REAL!r}")
        return 2

    failures = 0
    for replacement, expect_missing, expect_comma, why in CASES:
        gate.errors.clear()
        gate.warnings.clear()
        gate.check_figures(source.replace(REAL, replacement))
        missing, comma = _figure_three_missing(gate.errors), _comma_refused(gate.errors)
        ok = missing == expect_missing and comma == expect_comma
        failures += not ok
        print(f"  {'ok  ' if ok else 'MISS'} {replacement!r}")
        print(
            f"       expected missing={expect_missing} comma={expect_comma}; "
            f"got missing={missing} comma={comma} -- {why}"
        )

    gate.errors.clear()
    gate.warnings.clear()
    if failures:
        print(
            f"\n{failures} of {len(CASES)} cases did not behave as stated: the check is not "
            "proven and must not be trusted"
        )
        return 1
    print(f"\nall {len(CASES)} cases behaved as stated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
