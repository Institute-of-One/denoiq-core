"""Which journal this manuscript is currently built for, and what that journal's limits are.

Why this file exists
--------------------
The word limit lived in the test suite as one number with a comment naming Physica Medica.
Physica Medica rejected the manuscript before review on 2026-10-21, the paper was rebuilt around
the decision it hands a reader, and the suite went on enforcing the rule of a journal nobody is
submitting to. Two bad answers were available: cut the paper to fit a dead rule, or raise the
number until the suite went quiet. Both hide the real state, which is that the venue has not
been chosen yet.

So the target is declared here, in one place, and the two readers of it behave differently on
purpose:

* the test suite SKIPS the limit while the target is unset, because a limit nobody has chosen
  is not a fact about this manuscript and a red suite during revision trains people to ignore it;
* ``paper/presubmission_check.py`` FAILS while the target is unset, because the one moment the
  question must be answered is the moment before the file is uploaded.

This is the pattern ``build_docx.LINE_NUMBERS`` already uses for line numbering: the switch is
the single place that decides, and the checks follow the switch rather than asserting one
journal's rule for all time.

Setting it
----------
Put the journal's name in ``NAME`` and its limits in ``LIMITS`` below, read from the journal's
own author guide rather than from memory. Where a journal expresses its limit in pages rather
than words, record the word equivalent the guide states and say so in the comment.
"""

from __future__ import annotations

#: The journal this manuscript is built for. None means the venue has not been chosen.
#: Was "Physica Medica" until 2026-10-21, when it was rejected before review.
#: Chosen 2026-10-22 by the author: AI Horizons (IOP Publishing), as a Research Paper.
NAME: str | None = "AI Horizons"

#: Body word limits, excluding abstract, figure captions and references, as each journal's own
#: guide states them. A journal is added here when its guide has been read, not before. None
#: means the journal states no limit for the type being submitted -- which is not the same as
#: the venue being unchosen, and the checks distinguish the two.
LIMITS: dict[str, int | None] = {
    # Full length article: the guide refuses a submission over the limit automatically,
    # without the Editor-in-Chief's permission.
    "Physica Medica": 10_000,
    # Research paper. Checked 2026-09-01 for IORN-007's submission to the same publisher.
    "Physics in Medicine & Biology": 10_000,
    # Research Papers are "articles of unlimited length that report original research of
    # outstanding quality" (About AI Horizons, read 2026-10-22). The other types are limited --
    # Letters and Mini Reviews 4,000 words, Perspectives 3,000 -- and this is a Research Paper.
    # No limit is not a licence to be long: it only means the suite cannot enforce one here.
    "AI Horizons": None,
}

#: What else the journal asks for, recorded when its pages were read rather than from memory.
#: AI Horizons, read 2026-10-22 at
#: https://publishingsupport.iopscience.iop.org/journals/ai-horizons/about-ai-horizons/
#:
#: * submission: ScholarOne at https://mc04.manuscriptcentral.com/aih-eit
#: * format: "format in any way you choose"; IOP does the typesetting, and no reference style
#:   is mandated, so the Vancouver list this manuscript already carries is acceptable
#: * peer review: single-anonymous or double-anonymous, chosen by the author
#: * charges: "No charges are required for the authors from 2026 to 2031". The marketing email
#:   of 2026-10-20 says only "currently waived", which is weaker; re-read the About page at
#:   submission and keep a dated copy of whichever wording is live then
#: * abstract: no limit is stated anywhere on that page. 429 words is long by any other
#:   journal's standard, so it is cut to about 300 regardless -- an unstated limit is not an
#:   absent one, it is one the portal may apply without warning
NOTES = "AI Horizons: Research Paper, unlimited length, format-free, no APC 2026-2031"


#: Whether the submission PDF should carry its own line numbers.
#:
#: This was a style decision in build_pdf: on for --submission in "jmi" style, which is SPIE
#: Journal of Medical Imaging's convention and nobody else's. PMB-122419 was submitted to an IOP
#: journal through ScholarOne with the manuscript's own numbering, and the proof the portal built
#: carried two columns of line numbers, because ScholarOne numbers what it renders. One journal's
#: rule asserted for all of them is the same defect the word limit had.
#:
#: AI Horizons is IOP, on the same ScholarOne family (mc04.manuscriptcentral.com/aih-eit), and
#: asks for no particular format. The portal is therefore expected to number its own proof, so
#: the uploaded PDF does not. That is an inference from PMB rather than a published rule, so the
#: proof PDF is read at submission before anything is sent.
LINE_NUMBERS: dict[str, bool] = {
    "SPIE Journal of Medical Imaging": True,
    "Medical Physics": True,  # returns a submission that has none, before review
    "Physica Medica": False,
    "Physics in Medicine & Biology": False,
    "AI Horizons": False,
}


def word_limit() -> int | None:
    """The body word limit for the declared target, or None while none is declared."""
    return None if NAME is None else LIMITS[NAME]


def line_numbers(default: bool) -> bool:
    """Whether to number the lines, for the declared target; ``default`` while none is."""
    return default if NAME is None else LINE_NUMBERS.get(NAME, default)
