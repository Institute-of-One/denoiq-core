# Medical Physics (AAPM / Wiley) — Submission Kit — IORN-005 (CT-Noise_Core)

Target journal: **Medical Physics**, AAPM, published by Wiley. Article type: **Research
Article**. Manuscript source: `paper/manuscript.md`; build with
`python paper/build_manuscript.py`, then `paper/build_docx.py` for the upload file.

**Run `python paper/presubmission_check.py` before building anything to send.** It reports
front-matter drift, reference numbering and citation both ways, AMA author counts, table
columns and captions, unregistered or uncited figures, figures too wide to survive
reduction, control characters, unresolved markers, duplicate captions and stale builds.
It must print `ready to submit`.

## Why this venue

Reference 4 is Yu et al., *Med Phys* 2023, "Need for objective task-based evaluation of
deep learning-based denoising methods", in SPECT. **This manuscript is the CT answer to
a call this journal published.** References 8 and 9 are Eulig et al. and Nelson et al.,
both *Med Phys* 2024, both benchmarking deep low-dose CT denoisers on downstream tasks
rather than on fidelity — the genre is established here, recently, repeatedly. The
classification risk that dogs the companion paper does not apply: this is unambiguously
a research study, with prespecified hypotheses, effect sizes with intervals, one
hypothesis partially refuted, and two independent data arms.

**Cost.** Medical Physics is hybrid. The subscription route carries no APC and the author
has no funder mandating open access, so no charge applies unless OnlineOpen is chosen.

## Requirements verified

- **Structured abstract**, Background / Purpose / Methods / Results / Conclusions,
  maximum 500 words — **ours: 479**.
- **References numbered in order of first citation**, AMA 10th style; up to six authors
  listed in full, seven or more give the first three and *et al.* — **ours: 23, verified
  in order, all cited, none listed uncited**.
- Title in **sentence case** — ours already is.
- Figures: **9 main + 1 supplementary**. Each is registered in `build_pdf.FIGURE_FILES`,
  captioned once in `paper/README.md`, and named in the prose, which is what anchors its
  placement. Each caption appears beneath its figure **and** in a list after the
  references, as the form requires.
- Generative AI is declared in the **Methods** (Section 2.9), which is where Wiley asks
  for it, as well as in Disclosures. The form's first radio option is the correct answer.

## Form fields (copy–paste)

**Title:** Denoising under a data-processing ceiling: fidelity–task divergence in real
low-dose CT and in a controlled synthetic matrix

**Article type:** Research Article

**Author:** Shuji Yamamoto — sole & corresponding author
- Affiliation: **Institute of One, LISIT Co., Ltd., Tokyo, Japan**
- Email: yamamoto@lisit.jp · ORCID: 0000-0001-9211-1071

**Keywords:** low-dose CT; image denoising; deep learning; task-based image quality;
model observer; detectability; fidelity–task divergence

**Abstract:** paste from `paper/build/manuscript.md` (479 words, resolved). **Not** from
`paper/manuscript.md`, which still carries the `[[results:...]]` markers.

## Suggested reviewers

Addresses must be looked up from published records, never generated. The shortlist used
for the companion paper (Racine, CHUV; Siewerdsen, Johns Hopkins; Persson, KTH) applies
here too and is recorded in `taskiq-core/paper/jimaging_submission_kit.md` with sources.

Closer to this manuscript specifically, and all authors of work it cites:

| Candidate | Why | Where to find the address |
|---|---|---|
| Marc Kachelrieß, DKFZ | Senior author of reference 8, the *Med Phys* 2024 benchmark of deep low-dose CT denoisers | corresponding-author details of doi:10.1002/mp.17379 |
| Brandon J. Nelson / Rongping Zeng, FDA CDRH | Reference 9; model-observer evaluation of CT denoising | doi:10.1002/mp.16901 |
| Abhinav K. Jha, Washington University | Senior author of reference 4, the paper this manuscript answers | doi:10.1002/mp.16407 |

Suggesting the authors of the work you engage with is normal and is not a conflict. Note
that Section 1.1 states what this study adds beyond a benchmark, which is the question
those authors will ask first.

**Excluded reviewers: none.**

## Statements

| Field | Answer |
|---|---|
| Conflicts of interest | Yes — S.Y. is CEO of LISIT Co., Ltd. and TexelCraft OÜ; Institute of One is LISIT's open-research initiative. Neither sells a product related to the subject. The author also wrote `denoiq-core` and `taskiq-core`, the software the study runs on, which the Limitations state |
| Generative AI | Yes — Claude via Claude Code, for code, tests, figure scripts and prose. No numerical result came from the model: every number is emitted by executed code into `results/` and the test suite fails if the manuscript and those files disagree. Every reference checked against Crossref |
| Prior publication | None. No conference paper, no thesis, and **no preprint** — arXiv requires an endorser the author does not have, and medRxiv refused the affiliation |
| Funding | None |
| Ethics / IRB | Not applicable. No human participants, no animal subjects, no data collected for this study. The real-data arm uses de-identified public images from LDCT-and-Projection-data (TCIA, CC BY 4.0) |
| Data availability | Code and results: GitHub + Zenodo. Measured data: TCIA, DOI 10.7937/9npb-2637 |

## Cover letter

```text
Dear Editors of Medical Physics,

Please consider our manuscript, "Denoising under a data-processing ceiling: fidelity-task
divergence in real low-dose CT and in a controlled synthetic matrix," as a Research
Article.

Two claims are routinely made about denoising in medical imaging: that it improves image
quality, evidenced by fidelity metrics, and that it therefore permits dose reduction. The
first is measurable and usually true. The second does not follow, because denoising is a
function of the image it is given: the data-processing inequality and the Neyman-Pearson
lemma bound what any processing of an input can attain. What has not been quantified is
how far apart the two claims actually fall, and whether the gap survives an acquisition
nobody controlled.

We measure it on two arms. In a controlled synthetic matrix with an analytic ceiling,
fidelity change and task change were correlated at Spearman rho = -0.62, 88 per cent of
processed evaluations were divergent, and the effect differed systematically between
observers: the same processing improved the non-prewhitening estimate while leaving the
prewhitening one no better, which is consistent with redistribution of existing
information rather than creation of new information. One of three prespecified hypotheses was partially refuted
and is reported as such. On twelve Siemens liver cases from LDCT-and-Projection-data,
with a synthetic lesion of known size and contrast inserted into real parenchyma and a
case-disjoint held-out split, the rank correlation between PSNR and detectability across seven methods
was -0.29, and the method with the best PSNR of all seven ranked sixth on the task.
A larger configuration of the same residual CNN - 86 times the parameters, trained on more
patches for more epochs - improved its validation loss and gave the best PSNR in the study
while lowering detectability. Capacity was not varied alone, so what that shows is that
the divergence persisted under a substantially larger and longer-trained network of this
architecture, which weakens the first objection such a result invites without excluding
it. Nothing exceeded the closed-form ceiling in any
real-data comparison.

We believe this belongs in Medical Physics because the journal asked for it. Yu et al.
(Med Phys 2023;50:4122) argued the need for objective task-based evaluation of
deep-learning denoising, in SPECT; this is the CT answer, measured against a bound rather
than against other methods. Eulig et al. (2024;51:8776) and Nelson et al. (2024;51:978)
established the genre here recently, and Greffier et al., Fan et al., Tivnan et al. and Toia
et al. show that CT deep-learning reconstruction has been evaluated task-based already and can
raise detectability under some conditions - which is what makes the claim here a narrow one. What this study adds to a benchmark is the ceiling: a
benchmark ranks methods against each other, while an analytic bound turns the question
into how much of the information already present survives.

The work involves no human participants, no animal subjects and no data collected for this
study; the real-data arm uses de-identified public images from The Cancer Imaging Archive
under CC BY 4.0, so no ethics approval or informed consent applies. There is no preprint
and the manuscript is not under consideration elsewhere. Generative AI was used as a tool
and is disclosed in the manuscript; no numerical result came from it, and no AI system is
an author.

Competing interests, disclosed in full: the author is Representative Director (CEO) of
LISIT Co., Ltd. and Chief Executive Officer of TexelCraft OU, and Institute of One is the
open-research initiative of LISIT Co., Ltd. Neither company sells or licenses any product
related to the subject of this manuscript, and the work used no client or patient data. Two
further interests bear on the subject rather than on finance. The software the study runs
on - denoiq-core and taskiq-core - was written by the author, so the author is both the
implementer and the assessor; the manuscript states this in its Limitations. And the
manuscript argues for openly referenced validation while the author's research initiative is
founded on releasing code and data openly, which is an ideological position related to the
subject and is declared as such. There are no other competing interests and no funders.

Yours sincerely,
Shuji Yamamoto, PhD
Institute of One, LISIT Co., Ltd., Tokyo, Japan
yamamoto@lisit.jp - ORCID 0000-0001-9211-1071
```
