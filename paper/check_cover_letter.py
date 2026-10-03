"""Check every checkable claim in the cover letter against the live state.

A cover letter is the one document in a submission that nothing else validates. The manuscript's
numbers are resolved from ``results/`` and the build fails if they drift; the letter is prose
typed once and read by an editor who has no way to tell. This programme has already sent a letter
claiming a public repository that was local-only, and kept a retracted number in a letter through
two revision rounds.

So the letter is checked the way the manuscript is: every number in it must match the results
file it came from, every URL must answer, every DOI must resolve, and the affiliation must be the
one Crossref carries.

    python paper/check_cover_letter.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
LETTER = PAPER / "cover_letter.txt"
RESULTS = REPO / "results"

#: The affiliation exactly as Crossref carries it on the published papers. Written here so a
#: letter cannot quietly use a different one; the postcode goes after the city.
AFFILIATION = "Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan"
ORCID = "0000-0001-9211-1071"
EMAIL = "yamamoto@lisit.jp"


def resolve(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        value = value[part]
    return value


def reachable(url: str) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": f"cover-letter-check ({EMAIL})"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        return 200 <= exc.code < 400
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if not LETTER.exists():
        print(f"{LETTER} does not exist")
        return 1
    letter = LETTER.read_text(encoding="utf-8")
    liver = json.loads((RESULTS / "real_liver.json").read_text(encoding="utf-8"))
    release = json.loads((PAPER / "release.json").read_text(encoding="utf-8"))
    ldct = json.loads((PAPER / "ldct_io_release.json").read_text(encoding="utf-8"))
    title = next(
        line[2:].strip()
        for line in (PAPER / "manuscript.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("# ")
    )

    problems: list[str] = []

    def claim(description: str, ok: bool) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {description}")
        if not ok:
            problems.append(description)

    print("the letter says:")
    claim("the manuscript's own title", title in letter)
    claim(
        f"nine methods ({resolve(liver, 'held_out.n_methods')})",
        "nine methods" in letter and resolve(liver, "held_out.n_methods") == 9,
    )
    claim(
        f"a thousand held-out pairs ({resolve(liver, 'held_out.n_pairs')})",
        "thousand held-out pairs" in letter and resolve(liver, "held_out.n_pairs") == 1000,
    )
    claim(
        f"four held-out cases ({resolve(liver, 'held_out.n_test_cases')})",
        "four cases" in letter and resolve(liver, "held_out.n_test_cases") == 4,
    )
    claim(
        f"twelve liver cases ({resolve(liver, 'all_cases.n_cases')})",
        "twelve Siemens liver cases" in letter and resolve(liver, "all_cases.n_cases") == 12,
    )
    claim(
        f"best PSNR ranked seventh of nine ({resolve(liver, 'held_out.psnr_winner_task_rank')})",
        "seventh of nine" in letter and resolve(liver, "held_out.psnr_winner_task_rank") == 7,
    )
    claim(
        f"no arm exceeded the ceiling ({resolve(liver, 'all_exceedances')} exceedances)",
        "no arm exceeded" in letter and resolve(liver, "all_exceedances") == 0,
    )
    claim(
        f"no processed arm above the noise ({resolve(liver, 'fabrication.n_above_the_noise')})",
        resolve(liver, "fabrication.n_above_the_noise") == 0,
    )
    crossing = resolve(liver, "guidance.crossing.unprocessed")
    claim(f"the crossing is {crossing:.3f} of routine", f"{crossing:.3f}" in letter)
    needing = resolve(liver, "guidance.processing_penalty.n_needing_more_dose")
    total = resolve(liver, "guidance.processing_penalty.n_processed")
    claim(
        f"seven of eight processed arms need more exposure ({needing} of {total})",
        "seven of the eight" in letter and (needing, total) == (7, 8),
    )
    parameters = resolve(liver, "capacity.large.parameters")
    claim(
        f"the adversarial arm at 1.85 M parameters ({parameters})",
        "1.85 M parameters" in letter and 1_800_000 < parameters < 1_900_000,
    )

    print("\nidentifiers:")
    for label, doi in (("denoiq-core", release["version_doi"]), ("ldct-io", ldct["version_doi"])):
        claim(f"{label} cites {doi}", doi is not None and doi in letter)
        if doi:
            claim(f"{label} DOI resolves", reachable(f"https://doi.org/{doi}"))
    for url in re.findall(r"github\.com/[\w.-]+/[\w.-]+", letter):
        claim(f"{url} answers", reachable(f"https://{url}"))
    claim("the TCIA collection DOI", "10.7937/9npb-2637" in letter)

    print("\nthe author block:")
    claim("the Crossref affiliation, city before postcode", AFFILIATION in letter)
    claim("the ORCID", ORCID in letter)
    claim("the correspondence address", EMAIL in letter)

    print("\nclaims that must stay true:")
    claim("it states there is no preprint", "no preprint" in letter.lower())
    # Saying the code is public "now, not upon acceptance" is the point, so a bare search for
    # the words flags the sentence that makes the claim this check exists to protect. Strip the
    # denials first, then look for the promise.
    without_denials = re.sub(
        r"\b(?:not|rather than|never)\s+(?:upon|on)\s+(?:acceptance|request)", " ", letter, flags=re.I
    )
    claim(
        "it does not promise code upon acceptance",
        not re.search(
            r"(?:available|released|provided|shared)[^.]{0,40}(?:upon acceptance|on request"
            r"|upon reasonable request)",
            without_denials,
            re.I,
        ),
    )
    claim("it names no reviewer", not re.search(r"suggest(ed)? reviewer", letter, re.I))

    print()
    if problems:
        print(f"{len(problems)} claim(s) in the cover letter are not supported:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("every checkable claim in the cover letter holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
