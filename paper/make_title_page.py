"""Build the separate title page Medical Physics requires as its own file.

The submission form refuses to proceed without at least one Title Page file. Its contents
are generated from the manuscript rather than typed, so the title, author block, counts
and statements cannot drift from the paper they describe.

    python paper/make_title_page.py    # -> paper/build/title_page.docx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
BUILT = PAPER / "build" / "manuscript.md"
OUT_MD = PAPER / "build" / "title_page.md"
OUT_DOCX = PAPER / "build" / "title_page.docx"

RUNNING_TITLE = "Denoising under a data-processing ceiling"


def main() -> int:
    if not BUILT.exists():
        print("build/manuscript.md missing - run paper/build_manuscript.py first")
        return 2
    b = BUILT.read_text(encoding="utf-8")
    body = re.sub(r"<!--.*?-->", "", b, flags=re.S)

    title = next(ln[2:].strip() for ln in body.splitlines() if ln.startswith("# "))
    abstract = body.split("## Abstract")[1].split("**Keywords")[0]
    keywords = body.split("**Keywords:**")[1].split("\n\n")[0].strip().replace("\n", " ")
    refs = len(re.findall(r"^\d+\. ", body.split("## References")[1], re.M))
    figures = len({int(m) for m in re.findall(r"\*\*Figure (\d+)\.\*\*", body)})
    tables = len({int(m) for m in re.findall(r"\*\*Table (\d+)\.\*\*", body)}) + 2  # 2 generated

    page = f"""**Title**

{title}

**Running title**

{RUNNING_TITLE}

**Author**

Shuji Yamamoto, PhD

Institute of One, LISIT Co., Ltd., Tokyo, Japan

ORCID 0000-0001-9211-1071

**Corresponding author**

Shuji Yamamoto, Institute of One, LISIT Co., Ltd., Tokyo, Japan.
Email: yamamoto@lisit.jp

**Article type**

Research Article

**Counts**

Abstract {len(abstract.split())} words. Manuscript {len(body.split())} words.
{figures} figures, {tables} tables, {refs} references. One supplementary document.

**Keywords**

{keywords}

**Conflict of interest**

S.Y. is the Representative Director (CEO) of LISIT Co., Ltd. and Chief Executive Officer of
TexelCraft OU; Institute of One is the open-research initiative of LISIT Co., Ltd. Neither
company sells or licenses any product related to the subject of this manuscript, and the work
used no client or patient data. Two further interests bear on the subject rather than on
finance: the software the study runs on was written by the author, who is therefore both the
implementer and the assessor, as the manuscript's Limitations state; and the manuscript argues
for openly referenced validation while the author's research initiative is founded on releasing
code and data openly. There are no other competing interests.

**Funding**

This research received no external funding.

**Ethics**

Not applicable. The study involved no human participants and no animal subjects, and collected
no data. The real-data arm uses de-identified public images from LDCT-and-Projection-data, The
Cancer Imaging Archive, under CC BY 4.0.

**Data availability**

Code, tests, generated results and figures are openly available at
https://github.com/Institute-of-One/denoiq-core under the MIT licence and archived on Zenodo,
version DOI 10.5281/zenodo.21733389 (concept DOI 10.5281/zenodo.21733388). The measured images
are the LDCT-and-Projection-data collection in The Cancer Imaging Archive, DOI
10.7937/9npb-2637, used under CC BY 4.0.

**Use of generative AI**

Declared in Section 2.9 of the manuscript and in the Acknowledgments. No numerical result came
from the model; no AI system is an author.
"""
    OUT_MD.write_text(page, encoding="utf-8")

    try:
        import pypandoc  # noqa: PLC0415
    except ImportError:
        print(f"wrote {OUT_MD}; install pypandoc to produce the .docx")
        return 0
    pypandoc.convert_file(str(OUT_MD), "docx", outputfile=str(OUT_DOCX))
    print(f"wrote {OUT_DOCX} ({OUT_DOCX.stat().st_size // 1024} KB)")
    print(f"  abstract {len(abstract.split())} words, manuscript {len(body.split())} words")
    print(f"  {figures} figures, {tables} tables, {refs} references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
