<!--
  MANUSCRIPT SOURCE — do not type numbers into this file.

  Every quantity is written as a marker resolved from results/ at build time:

      [[results:<file>.json:<path>|<format>]]

  Formats are Python format specs, plus `sciN` for journal-style scientific notation.
  Build with `python paper/build_manuscript.py` (output: paper/build/manuscript.md).
  tests/test_manuscript_consistency.py rebuilds it on every CI run and fails if the
  committed build is out of date, if a marker does not resolve, if a bare number has
  been typed into the prose, or if the terminology guards below are violated.

  TERMINOLOGY, fixed for the whole paper (the test suite enforces):
    * "analytic ideal linear (prewhitening) observer on the unprocessed input" = the ceiling,
      computed in closed form. Shorthand after Section 2.1: "the input's analytic ceiling".
    * "likelihood-ratio ideal observer" = the theoretical optimum the bound is stated for.
    * "held-out prewhitening linear observer" (PW) = what is MEASURED on processed images.
      Never call this an ideal observer.
    * The floor is "operational" / "prespecified" / "task-specific", never a boundary of
      zero information. Never write "chance-level" for a d' that is not ~0.
    * The phenomenon is "fidelity-task divergence"; an individual arm is "divergent".

  TITLE is fixed and must match CITATION.cff, .zenodo.json, README and the PDF metadata.

  Target journal: SPIE Journal of Medical Imaging. This file is the content of record; the
  submission setting (Times-metric serif, wide leading, superscript citations, "Fig. N",
  SPIE section numbering, JMI back matter) is applied by `paper/build_pdf.py --style jmi`,
  which is the default. `--submission` additionally refuses to build until the archived
  release DOI in `paper/release.json` has been minted.
-->

# Denoising under a data-processing ceiling: fidelity–task divergence in real low-dose CT and in a controlled synthetic matrix

**Shuji Yamamoto**
Institute of One, LISIT Co., Ltd., Tokyo, Japan
yamamoto@lisit.jp · ORCID 0000-0001-9211-1071

## Abstract

**Background.** Denoising is widely held to improve image quality, evidenced by
reference-based fidelity metrics, and thereby to permit dose reduction. The second claim does
not follow from the first: processing is a function of the image it is given and cannot add
task information to it.

**Purpose.** To quantify when fidelity gains correspond to gains or losses in task
detectability, how those effects depend on observer efficiency and on the detectability
available in the input, and whether they survive an acquisition nobody controlled.

**Methods.** Two arms. *Controlled*: signal-known-exactly / background-known-exactly detection
of a low-contrast disk on synthetic phantoms over a matrix of dose, noise correlation length
and lesion configuration, with three deterministic denoisers (Gaussian, total variation,
non-local means) and three observers — a prewhitening linear observer (PW), a channelised
Hotelling observer (CHO), and a non-prewhitening observer with an eye filter (NPWE) used as a
stylized surrogate for limited prewhitening efficiency. Each processed arm is referenced to the
unprocessed arm of its own condition and to that input's **analytic** ideal-observer
detectability — the ceiling the data-processing inequality [1] and the Neyman–Pearson lemma [2]
place on any processing of it. Observer estimates are cross-fitted; uncertainty is by bootstrap
over whole realisations; comparisons are Holm-adjusted. The matrix ran over
10 realisations
(1,696,000 scored trials).
*Real data*: 12 Siemens liver cases from LDCT-and-Projection-data,
vendor reconstructions of the routine and quarter-dose acquisitions, a lesion of known size and
contrast inserted into real parenchyma, and a residual CNN trained on
8 cases and evaluated on the 4 it
never saw, at two capacities spanning 86× in parameters.

**Results.** *Controlled*: ΔSSIM and Δ`d'`(PW) correlated at Spearman ρ =
-0.62
(88.2% of processed evaluations divergent, with ΔSSIM >
0 and Δ`d'`(PW) < 0). Denoising helped the inefficient observer more than the efficient one:
`B` = Δ`d'`(NPWE) − Δ`d'`(PW) = +1.13. The
information-floor hypothesis was **partially refuted**: below the floor fidelity rose while the
task estimate fell, but failure patterns did not concentrate there, and excess lesion-like
responses were absent throughout. *Real data*: across 7 methods the
rank correlation between PSNR and `d'` was ρ =
-0.29; the method with the best PSNR ranked
6 of 7 on the task; and
0 arms exceeded the closed-form ceiling in any real-data comparison.
Raising network capacity 86-fold improved validation loss and
gave the best PSNR in the study, 28.77 dB, while lowering
detectability from 5.31 to 4.96.

**Conclusions.** Denoising effects are observer-dependent: processing may improve performance
for an inefficient observer without increasing the task information available in the input.
Fidelity gains alone do not establish task preservation, and on real low-dose CT they rank
methods close to inversely to the task. The divergence is not an artefact of the controlled
model in which it was isolated, nor of a network too small to be representative.

**Keywords:** denoising; task-based image quality; model observer; detectability;
fidelity–task divergence; observer efficiency; operational information floor.

## 1. Introduction

Two claims are routinely made about image denoising in medical imaging. The first is that it
improves image quality, evidenced by fidelity metrics. The second, following from the first, is
that it permits dose reduction. The first claim is measurable and usually true. The second does
not follow from it, and this paper is about the gap between them.

The gap has a precise shape. Denoising is a function of the image it is given, so hypothesis
`H`, input image `X` and processed image `Y = g(X)` form a Markov chain `H → X → Y`. Two
consequences follow, and they should be distinguished rather than merged. First, the
data-processing inequality [1] gives `I(H;Y) ≤ I(H;X)`: processing cannot increase the mutual
information between the image and the truth. Second, for binary detection, the Neyman–Pearson
lemma [2] implies that the likelihood-ratio test on `X` attains an ROC curve that no test based
only on `Y` can dominate; in particular its AUC is an upper bound. Mutual information and AUC
are not interchangeable, and no monotone map between them is assumed here: they are two
distinct consequences of the same Markov structure, and it is the second that this study
measures against.

Neither consequence says what denoising *does*. A bound on what is recoverable is not a
statement about whether a given observer recovers it, and real observers are not ideal. A
denoiser that suppresses noise where an inefficient observer is overwhelmed does for that
observer what it cannot do for itself, and the gain is real even though no information was
added. Conversely, processing that improves a picture's resemblance to the truth may remove
precisely the structure a detection task depends on. Which of these happens, when, and by how
much is not settled by the theorem, and that is the question this paper answers
quantitatively.

We also separate a second question that is often conflated with the first. A study can state
how much detectability its task requires. Where the input's own ideal-observer detectability
falls below that requirement, no post-processing can bring the requirement back into reach:
that follows from the bound, applied at that condition. We use "information floor" as a
concise operational term for the contour at which the input ideal-observer detectability falls
below a prespecified task requirement. It is not a zero-information boundary. Conditions below
it retain measurable task information; they simply do not meet the requirement that was chosen.
What processing can still do below the floor is produce a cleaner-looking image, which is why
visual plausibility there does not establish that the required detectability survived.

### 1.1 Relation to previous work

That denoising can degrade task performance while improving fidelity is established
empirically. Li et al. [3] assessed deep denoising on binary signal detection with model
observers and connected the result to the data-processing inequality; Yu et al. [4] reported
for myocardial perfusion SPECT that denoising improved RMSE and SSIM while frequently
degrading detection performance; Li et al. [5] proposed a framework for mapping the nonlinear
system and noise response of such algorithms; and task-informed training [6] shows the
trade-off can be shifted but not escaped. Hallucination in reconstruction has been
characterised in terms of a task-relevant null space [7]. In CT specifically, Eulig et al.
[8] benchmarked deep low-dose CT denoisers on the downstream detection and diagnosis of
lesions rather than on fidelity alone, and Nelson et al. [9] found that a network trained on
adult images changes low-contrast detectability differently on paediatric-sized phantoms. The
relation between task-based image quality and dose is treated comprehensively by Barrett et
al. [10].

What is not established quantitatively is *how* these effects vary: how strongly fidelity
gains predict task changes across imaging conditions, how the effect depends on the efficiency
of the observer doing the looking, and how failure patterns distribute relative to the
detectability available in the input. A benchmark also answers a different question from the one asked here: it ranks methods
against each other, whereas an analytic ceiling bounds what *any* processing of a given input
can attain, so the question becomes how much of the information already present survives.
This study measures all three on one controlled matrix,
with an analytic reference that removes the usual ambiguity about whether an apparent loss is
real or an artefact of the estimator, and with repeated independent realisations so that
effect sizes come with intervals.

### 1.2 Prespecified hypotheses

We tested three prespecified hypotheses. First, gains in reference-based fidelity would not
reliably predict changes in task detectability. Second, the effect of denoising would depend
on observer efficiency, with larger benefits for the non-prewhitening observer than for the
prewhitening observer. Third, below a prespecified input-detectability criterion, visually
plausible outputs would remain possible despite failure to preserve the required task
performance.

That no processed arm exceeds its input's ceiling is **not** among them: that is a theorem, and
its role here is measurement-chain validation and leakage control (Sections 2.5 and 3.4).

### 1.3 Contributions

1. Quantification of fidelity–task divergence across denoisers and imaging conditions, with
   effect sizes and intervals from repeated independent realisations.
2. Quantification of observer-dependent denoising benefits, comparing a prewhitening observer,
   a channelised Hotelling observer and a non-prewhitening observer on identical images.
3. Stratification of failure patterns — erasure, excess lesion-like responses, task degradation
   — by an operational information floor.
4. Leakage-controlled measurement against an analytic ceiling, with a positive control asserted
   to fail when the class label is made available to the processing.
5. Confirmation of the divergence and of the ceiling on real low-dose CT — twelve liver cases,
   a case-disjoint held-out split, and a learned denoiser at two capacities spanning
   86x in parameters — establishing that the effect is not a property of
   the synthetic model in which it was isolated.
6. An open, deterministic implementation in which every reported number is regenerated from
   machine-readable outputs.

## 2. Methods

### 2.1 Task, observers, and what "input" means

The task is signal-known-exactly / background-known-exactly (SKE/BKE) detection of a
low-contrast disk on a uniform background, the standard paradigm of objective, task-based
assessment [11,12]. Trials, phantoms, model observers and the ROC
machinery are reused from `taskiq-core` [13] (version
0.4.0) rather than reimplemented; the trial
generator returns, with the image stacks, the **analytic** noise power spectrum (NPS) of the
noise it generated, which is what allows an observer to be evaluated in closed form with
nothing estimated. Geometry: 64 ×
64 pixels at
0.5 mm, lesion radius
3.0 mm except where the lesion sweep varies it,
1,200 trials per class per arm in the three sweeps and fewer at the atlas settings.

**What "input" and "unprocessed" denote.** This study concerns post-processing of a defined
input image. The unprocessed images are synthetic images formed directly from the acquisition
model — not detector projection data, and not the output of a reconstruction. We make no claim
about information gained or lost in reconstruction, which is a separate map with its own bound.
Throughout, `X` is the image handed to the denoiser; "raw" appears only as a compact axis or
legend label for that same image.

Three observers are estimated from images and scored out of fold (Section 2.5):

1. The **held-out prewhitening linear observer** (PW): the class-mean difference prewhitened by
   the measured NPS. This is the efficient observer of the study, and the one every
   processed-image comparison uses.
2. A **channelised Hotelling observer** (CHO) on Laguerre–Gauss channels [14], an intermediate
   observer: tractable because a channel covariance can be estimated where a pixel covariance
   cannot.
3. A **non-prewhitening observer with a Burgess eye filter** (NPWE) [15], used as a stylized
   surrogate for limited noise-prewhitening efficiency. No human observer study was performed
   here, and NPWE is not offered as a validated model of a human reader.

Separately, and only for the unprocessed input, we compute the **analytic ideal linear
(prewhitening) observer** from the true signal and the analytic NPS. That quantity is exact for
the experiment it describes and carries no sampling error; it is the input's *ceiling*, and it
is the only observer here that is not estimated from images.

### 2.2 Relative acquisition model

Contrast and noise follow three documented proportionalities relative to a reference setting:
photon count `N ∝ mAs·kV²`, noise standard deviation `σ ∝ 1/√N`, and subject contrast
`c ∝ kV^(−p)` with `p` = 1.5. The
reference setting is 120 kV,
100 mAs, with reference noise
30 and reference contrast
20 in the same arbitrary intensity units. Two
consequences are visible in the results: for a fixed signal profile `d' ∝ c/σ`, so
`d' ∝ √mAs · kV^(1−p)`, and the mAs at which a *fixed* detectability requirement is met scales
as `mAs_floor ∝ kV^(2p−2)`, which for this `p` is linear in kV.

Dose is taken proportional to mAs; it also rises with kV in reality, and that dependence is
left out, so **every dose comparison in this paper is made at fixed kV**. On the kV axis we
report kV itself and never a dose ratio. This is a normalised model, not a calibration: it
carries no measured dose-to-noise relation for any device and no absolute exposure values.

### 2.3 Denoisers, and what they are told

Three deterministic classical denoisers: a Gaussian filter of fixed width
1.5 pixels, total
variation with weight
0.4 × the noise
standard deviation, and non-local means with `h` =
0.6 × the noise standard
deviation. A small residual convolutional network in the style of DnCNN [16] is evaluated
separately (Section 3.6) and is not part of the primary matrix.

**Where the noise level comes from, and why it does not break the bound.** For total variation
and non-local means the noise standard deviation used to set the parameter is the *true*
simulation value for that acquisition setting: a setting-informed (oracle) parameterization,
stated rather than left implicit, since it is more than a blind method would have. What matters
for `H → X → Y` is that the processing is conditionally independent of the hypothesis given the
input: the parameter depends only on the acquisition setting, which is fixed within a condition
and identical for its signal-present and signal-absent arms. No denoiser receives the class
label, the lesion location, the noise realisation, or any other trial's pixels, and each image
is processed independently — all asserted in the test suite.

### 2.4 Experimental design and independent realisations

The matrix has four domains, described in Table 1. An **input condition** is one acquisition
and phantom setting; an **arm** is one condition evaluated with one processing, including the
unprocessed one; an **evaluation** is one arm in one realisation. Conditions are not shared
between domains, so arms need no deduplication: the dose, noise-correlation and lesion domains
each generate their own conditions, and the atlas domain adds settings on the kV–mAs plane used
for the floor demonstration, where a single denoiser is applied rather than all three. That is
why the processed-arm count is 55 rather than
three times the 21 input conditions.

The whole matrix was run over 10 independent
realisations. The first is the seed of the earlier single-realisation study, retained so that
the previous result remains inspectable; the others follow from it by a fixed stride recorded in
the configuration, so the list is a property of the code rather than of a session.
Signal-present and signal-absent trials come from one seeded stream per realisation and are
independent by construction. Every realisation runs the identical matrix, which makes arms
pairable across realisations, and per-realisation results are written alongside the aggregate.
We report 76 unique arms evaluated across
10 independent realisations —
760 arm–realisation evaluations — and never
as a single inflated condition count. The representative realisation used for the example
images is the first seed; Figure 1 shows the signal-present and signal-absent images at one
setting, for the unprocessed input and each classical denoiser on a common display window.

### 2.5 Cross-fitted estimation, and the ceiling comparison

On the unprocessed input the ideal linear observer is closed-form. On processed images neither
the noise spectrum nor the effective signal is known analytically — a non-linear denoiser has no
transfer function — so both are estimated, and the estimate must not be allowed to score itself.

Estimation is **cross-fitted** over `K` =
5 folds. Fold membership is by trial index
modulo `K`; the trials are i.i.d. draws from one seeded stream, so a positional partition is
already a random one and needs no second random number to record. For each fold, the effective
signal `Ŝ` (the class-mean difference) and the noise spectrum are estimated from the other folds
inside a central 24 ×
24 pixel window, the prewhitening template
`Ŵ = Ŝ/NPS` is formed, and the held-out fold is scored with it. Pooling the out-of-fold scores
gives `d'` from the pooled-variance separation and AUC from the Mann–Whitney statistic, using
every trial exactly once while no score comes from a template that saw its own image. The
measured NPS is made invertible by filling the DC bin from its neighbours, adding a ridge of
0.01 × the mean power, and clamping
at 1×10⁻⁶ × the peak; these
steps stabilise the estimate but may reduce its efficiency relative to the unknown optimum. The
same scheme estimates NPWE; the CHO uses the split estimator of the underlying package, which is
also held out.

The ceiling comparison, reported as validation in Section 3.4, tests

`AUC_PW(g(X)) ≤ AUC_ceiling(X) + m`, with `m = z·SE + 1/(n₁n₀)`,

`z` = `1.96`, `SE` the Hanley–McNeil standard error [17] of the scored AUC, and the second term
one quantisation step of the Mann–Whitney statistic. The comparison is made at every arm, and it
is reported twice: at the armwise margin above, and at a *simultaneous* margin in which `z` is
widened by Bonferroni to a family-wise level of `0.05` over all arm–realisation comparisons. The
claim of interest concerns processing, so the simultaneous statement is made over the processed
arms; the unprocessed arms are self-comparisons of an input against its own ceiling and are
reported separately, since an identity map cannot create information and any excess there
measures the estimator and the finite sample. Alongside both we report the largest excess
anywhere against the margin at that arm, and we summarise the AUC-saturated arms — those whose
analytic ceiling AUC lies within ten quantisation steps of 1 — separately, since there the
comparison is limited by the resolution of the rank statistic rather than by information.

**Estimator efficiency.** Applied to unprocessed images the estimator must recover most of the
analytic ceiling; otherwise the comparison would be satisfied for statistical rather than
informational reasons. We report the distribution of that recovery ratio, and Section 3.7
compares cross-fitting with the single `50/50` split used previously.

### 2.6 Primary endpoints

For each processed arm, with the unprocessed arm of the same condition and realisation as its
reference:

`ΔSSIM = SSIM(processed) − SSIM(input)`, and `Δd'_O = d'_O(processed) − d'_O(input)` for each
observer `O` in {PW, CHO, NPWE}.

**Endpoint 1 — fidelity–task divergence.** The Spearman correlation between ΔSSIM and Δ`d'`(PW)
across evaluations, and the **divergence rate**: the fraction of evaluations with ΔSSIM > 0 and
Δ`d'`(PW) < 0. Both are reported overall, per denoiser, and stratified by floor stratum. An
individual arm is called *divergent*; the phenomenon is *fidelity–task divergence*.

**Endpoint 2 — observer-dependent benefit.** `B = Δd'(NPWE) − Δd'(PW)`, the extent to which
processing helped the inefficient observer more than the efficient one, reported per denoiser
with intervals, together with the fraction of evaluations that improved NPWE while not improving
PW, and the observer ordering with CHO in between.

**Endpoint 3 — floor-stratified failure patterns.** Each input condition is assigned a stratum
by its analytic ceiling: *above floor*, *marginal* (within the gauge's marginal factor of the
requirement) and *below floor*. Strata are compared on divergence rate, contrast erasure rate,
excess lesion-like response rate, observer-dependent benefit, task degradation and verdict
distribution. The floor stratifies conditions by the detectability available in the input; it is
not claimed to *cause* the failures.

Erasure is contrast recovery below the gauge's threshold, and an excess response is a
lesion-like matched-filter rate above the gauge's allowance **and** above the same rate measured
on the unprocessed input of that condition — in signal-absent trials the object is uniform but
the acquired image is not, and noise alone produces such responses. Both definitions, and the
matched-filter normalisation behind them, are restated in Supplementary Methods.

### 2.7 Statistical analysis

Arms are not independent observations: all arms of a realisation share its noise stream, and the
arms of a condition share its images. Intervals therefore come from a **cluster bootstrap that
resamples whole realisations** with replacement (4000 replicates,
seeded), carrying every arm of a drawn realisation along. Point estimates are the statistic on
the full data; intervals are percentile intervals; two-sided bootstrap p-values are reported
beside them and Holm-adjusted within each family (denoisers within an endpoint, stratum
contrasts within theirs). Proportions are additionally reported with a Wilson interval as the
conventional unclustered reference, which is the narrower of the two. Conclusions rest on effect
sizes and intervals rather than on p-values. Every statistic in this manuscript is computed by
the analysis module and written to `results/statistics.json`; none is typed.

### 2.8 The operational floor and the gauge

The floor is the contour where the input's analytic ceiling `d'` crosses a prespecified
requirement, here the Rose criterion [18] at
`d'` = 5. It is used in this
paper as a stratifying variable, and it is not a zero-information boundary.

The study also carries an auditable rule-based translation of the measured quantities into a
green / amber / red verdict with the rule that fired attached. Its thresholds are task-specific
author-set values, not clinically validated criteria; the complete rule set is given in
Supplementary Methods, and verdict counts per stratum are reported in Section 3.5.

## 3. Results

### 3.1 Study design and evaluated conditions (Table 1)

Table 1 gives the matrix. The primary analysis comprises
21 input conditions and
76 unique arms
(21 unprocessed,
55 processed), each evaluated in
10 independent realisations, giving
760 arm–realisation evaluations and
1,696,000 scored image trials. Endpoints are
computed on the 550 processed evaluations, each
paired with the unprocessed arm of its own condition and realisation.

Figures 2 and 3 give the dose response of the representative realisation: Figure 2 the
reference-based fidelity (SSIM and PSNR) against relative dose, and Figure 3 the task `d'` of the
three estimated observers against relative dose, with the analytic ideal-observer ceiling of the
unprocessed input.

### 3.2 Fidelity gains frequently diverge from task performance (Figure 4)

Across all processed evaluations, ΔSSIM and Δ`d'`(PW) were correlated at Spearman ρ =
-0.62 (`95 %` CI
-0.63 to
-0.60). Mean ΔSSIM
was +0.252
(+0.252 to
+0.253) while mean Δ`d'`(PW) was
-0.832
(-0.846 to
-0.819): fidelity improved on
average, the task estimate did not.

88.2% of evaluations were divergent —
fidelity up, task estimate down — with a clustered `95 %` interval of
86.7% to
89.6%. The rate differed by
denoiser:
100.0%
for the Gaussian filter,
62.4% for
total variation and
99.4% for
non-local means — an ordering that follows filter strength rather than filter family, with
total variation the most conservative of the three at these settings. Figure 4 shows every
evaluation in the ΔSSIM–Δ`d'` plane; the quadrant in which fidelity improves while the task
estimate falls is where most of the distribution lies.

### 3.3 Denoising benefits depend on observer efficiency (Figure 5, Table 2)

On identical images the three observers responded differently to the same processing. Averaged
over all evaluations, Δ`d'`(PW) was
-0.83
(-0.85 to
-0.82), Δ`d'`(CHO)
-0.14
(-0.14 to
-0.14) and Δ`d'`(NPWE)
+0.30
(+0.29 to
+0.30) — an ordering
that follows observer efficiency, with the intermediate observer in between.

The observer-dependent benefit was `B` =
+1.13
(+1.12 to
+1.14), and
77.8%
(75.8% to
79.6%) of
evaluations improved the non-prewhitening observer while not improving the prewhitening one.
Table 2 gives the per-denoiser effects with intervals and Holm-adjusted p-values; Figure 5 plots
them with realisation-level uncertainty. The benefit also varied with noise correlation length —
the condition that separates an observer that can prewhiten from one that cannot: `B` =
+0.77 at zero
correlation against
+5.23 at the
longest correlation length studied.

### 3.4 Performance relative to the data-processing ceiling (Figure 6)

Across 760 arm–realisation evaluations the mean
excess of the cross-fitted prewhitening AUC over the analytic ceiling of its own input was
-0.032, and the largest was
1.64×10⁻⁴.

The claim the design supports is about processing, and it is stated simultaneously over the
whole family: with the armwise margin widened by Bonferroni to a family-wise level of `0.05`
across all 760 comparisons
(`z` = 3.99),
**0 of the
550 processed-arm
comparisons exceeded the ceiling margin**. The same holds at the unadjusted armwise margin
(0 of
550).

One comparison did exceed its margin, and it is reported separately because it is not a
statement about processing. It occurred on an **unprocessed** arm — a self-comparison of the
input against its own analytic ceiling, where the processing is the identity and cannot create
information, so the exceedance measures the estimator and the finite sample rather than a
violated bound. That arm is AUC-saturated:
its analytic ceiling AUC is
0.9999933 and the estimator
achieved perfect separation
(1.0000000), an excess of
6.75×10⁻⁶ in AUC — of the order of ten
misordered pairs in `1.44` million. At perfect separation the Hanley–McNeil standard error is
identically zero, so the margin collapses to a single quantisation step
(6.94×10⁻⁷ in AUC, one misordered pair
out of `1.44` million) and no widening of `z` rescues it: the family-wise margin leaves this
same 1 exceedance
for exactly that reason. This is a known degeneracy of the closed-form interval at the top of
the scale, it is a property of a finite sample rather than of the bound, and it is why the
saturated regime is analysed separately.

180 evaluations were AUC-saturated in the sense
defined in Section 2.5. Over the remaining
580 the largest excess was
1.64×10⁻⁴, the mean
-0.040, and there were
0 exceedances, so nothing here rests
on the saturated regime. Figure 6 plots processed against ceiling AUC with an inset
over the unsaturated range.

This is validation, not a finding: the theorem concerns the likelihood-ratio ideal observer, and
what is checked here is the measurement chain. Two readings must be kept apart. A true ideal
observer would preserve performance under an invertible transformation and would sit on the
identity line; the observer plotted is an estimated linear one and need not attain that equality
after a non-linear transformation, so points below the line are not by themselves evidence of
information loss. The unprocessed arms are plotted too and also fall slightly below the line —
that offset is the estimator's own cost, quantified in Section 3.7.

### 3.5 Failure patterns across the operational information floor (Figures 7–8)

Figure 7 is the kV–mAs atlas with the operational floor drawn as the contour where the input's
analytic ceiling crosses the requirement; Figure 8 and Supplementary Table S3 give the
stratified comparison.

The third hypothesis was supported in one respect and refuted in another, and the refutation is
the more interesting half.

**Supported: plausible output without preserved detectability.** Below the floor, processing
raised SSIM by +0.210
on average while Δ`d'`(PW) was
-0.334, and
83.3%
(81.3% to
85.4%) of those
evaluations were divergent. Visually improved output therefore coexisted, throughout that
stratum, with an input that did not meet the requirement and a processed estimate that fell
further below it.

**Refuted: failures do not concentrate below the floor.** We had anticipated that failure
patterns would be worse below the floor. They were not. The divergent fraction was *higher*
above the floor —
100.0% against
83.3%, a difference
of
-16.7%
(-18.7%
to
-14.6%)
— and so was mean task degradation:
30.3% of the input's
detectability above the floor against
14.7% below it
(difference
-15.6%,
-16.5%
to
-14.9%).
The mechanism is straightforward once seen: *relative* loss is largest where there was most to
lose. A high-detectability input has detectability for a filter to discard, and these filters
discard it; a low-detectability input has little, and proportionally little is removed. The same
ordering appears in the observer-dependent benefit, which was
+2.59 above the floor
against +0.64 below it.

Contrast erasure did not differ appreciably between strata
(3.8% above against
5.1% below; difference
+1.3%,
-1.0%
to
+3.6%,
Holm-adjusted p =
1.00),
and excess lesion-like responses were absent in every stratum
(0.0% below the
floor). With these denoisers, at this lesion-like response threshold, the failure mode is erasure
and dilution of true contrast rather than fabrication of false structure — a result that should
not be generalised to generative or learned methods, whose failure modes may differ.

The gauge's verdicts reflect the strata by construction more than by discovery: above the floor
70 of
130 evaluations were green and
5 red on erasure or task loss,
while below the floor all
390 were red — an input that fails
the requirement is itself a red rule, so what is informative there is the reason attached rather
than the colour.

### 3.6 The same question on real low-dose CT

The controlled matrix isolates mechanisms by fixing everything. The obvious question about any
conclusion drawn from it is whether the conclusion survives an acquisition nobody controlled, so
the same measurement was repeated on real data with no change to the framework.

**Data and design.** Twelve Siemens liver cases from LDCT-and-Projection-data (The Cancer Imaging
Archive, `CC BY 4.0`), vendor reconstructions of both the routine and the simulated quarter-dose
acquisition. A lesion of known size and contrast is inserted into real parenchyma, so that the
task has a ground truth the acquisition itself cannot supply. The learned denoiser is trained on
quarter-dose / full-dose patch pairs from 8 cases and evaluated on
the 4 it never saw (1000 pairs). **The split is by
case, not by slice**: slices from one patient are not independent, and a network tested on
another slice of a liver it trained on is being tested on its own training set. Normalisation at
training matches normalisation at inference, since a network trained in absolute HU and deployed
through a normalising wrapper is not the network that was trained. The network never sees a
lesion — its targets are full-dose reconstructions of ordinary anatomy, which is what a
denoiser is actually given — and the lesion exists only in the evaluation.

**Result.** Figure 9 and Table 3 give the held-out comparison against the closed-form ceiling
`d'` = 8.09:

**Table 3.** The seven arms on the held-out real low-dose CT split, ordered by task detectability. `d'` is against the closed-form ceiling of the unprocessed input; PSNR is against the full-dose reconstruction. The two networks differ only in capacity.

| method | `d'` | of ceiling | PSNR (dB) |
|---|---|---|---|
| `tv 1x noise` | 6.09 | 0.75 | 28.07 |
| unprocessed | 6.03 | 0.75 | 23.81 |
| `nlm 0.8x noise` | 5.50 | 0.68 | 27.50 |
| CNN, 21385 parameters | 5.31 | 0.66 | 28.74 |
| `gaussian 0.75 mm` | 5.10 | 0.63 | 28.04 |
| CNN, 1849633 parameters | 4.96 | 0.61 | **28.77** |
| `gaussian 1.00 mm` | 4.82 | 0.60 | 27.85 |

Three things follow.

*Fidelity and task rank the methods differently, and if anything inversely.* Across the
7 arms the rank correlation between PSNR and `d'` is Spearman
ρ = -0.29
(`p` = 0.53). The arm with the best PSNR of
all 7 ranks 6 of
7 on the task. Selecting a denoiser by fidelity on these data would have
selected close to the worst available option for detection.

*Nothing exceeded the ceiling.* 0 of
7 arms on the held-out split, and 0 across every
real-data arm in this study — the held-out split, the 12-case run, and
both operating points of Section 3.6.1. The bound the controlled matrix was built to test is not
an artefact of the controlled matrix.

*Capacity does not reverse the ordering; it deepens it.* The large network has
86 times the parameters of the small one
(1849633 against 21385), was trained on
3000 patches per case against
800, and reached a better validation loss. It bought
28.77 dB against 28.74 — the best
PSNR in the study — and a `d'` of 4.96 against
5.31. Raising capacity improved the objective the network was
trained on and moved the task in the other direction. This bears directly on the scope of the
claim: the divergence reported here is not an artefact of a network too small to be
representative, which is the first objection such a result invites.

Two caveats are stated here rather than left for a reader to find. The large network's validation
loss reached its minimum well before the last of its
60 epochs and drifted upward thereafter, so the
saved model is not the best one the run produced; the gap is a small fraction of the validation
loss and far too little to move `d'`. And +4.96 dB of PSNR over the unprocessed
input for 86x the capacity is itself informative: the
training target is a full-dose *reconstruction*, which carries noise of its own that no network
can predict, so mean-squared error against it saturates well before the image does.

#### 3.6.1 A second operating point

Every number above rests on one acquisition. The same measurement at a second and very different
one — chest at 10% dose, where the noise standard
deviation is 8.2 times the liver's and
the closed-form ceiling falls from 9.21 to
2.64 — produced
0 exceedances of its own ceiling. The bound holds
where the task is nearly impossible as well as where it is comfortable.

### 3.7 Sensitivity and implementation validation

**Estimator efficiency.** On unprocessed images the cross-fitted estimator recovered a median
95% of the analytic ceiling `d'`
(interquartile range 86% to
97%), with a minimum of
49% over all evaluations and
93% over conditions at or
above the floor. The lowest recoveries occur at the lowest doses, where the template is estimated
from images in which the signal is weakest — the regime in which the estimator, not the bound, is
the limiting factor, which is why it is reported rather than summarised away.

**Cross-fitting against a single split.** Supplementary Table S2 compares the cross-fitted
estimator with the single `50/50` split used in the earlier single-realisation analysis at matched
conditions; cross-fitting recovers more of the ceiling at every dose, and the qualitative
conclusions are unchanged.

**Closed form.** In white noise the analytic observer must satisfy `d' = ‖s‖₂/σ`. Over the
combinations of noise level and lesion radius in Supplementary Table S1 the maximum relative
deviation was 1.32×10⁻¹⁶, against a criterion of 1 %
fixed in advance.

**Leakage positive control.** A processor handed the class label — adding signal to the
signal-present stack only — is detected by the ceiling comparison as an exceedance beyond the
margin. That control is asserted in the test suite, and it is what gives the negative result of
Section 3.4 its content.

## 4. Discussion

The ceiling is a theorem, and demonstrating a theorem numerically proves nothing about the
theorem. What the demonstration establishes is that the *measurement chain* — phantoms,
denoisers, observers, estimators — is consistent with it, which is a precondition for trusting
anything else the chain reports. A reproducible excess beyond a properly calibrated uncertainty
bound signals truth leakage, estimator bias, model mismatch, or an implementation error; the
positive control shows the comparison has the sensitivity to see one.

The findings sit on top of that. The first is that reference-based fidelity is a poor predictor
of task change: the rank correlation between ΔSSIM and Δ`d'`(PW) was negative, and a large
fraction of evaluations improved the picture while the task estimate fell. This is not an
artefact of a badly chosen fidelity metric — SSIM is what much of the literature reports and
what many learned methods are optimised against — but a consequence of the two quantities
measuring different things.

The second is that the effect of denoising is observer-dependent in a systematic, quantifiable
way. The same processing on the same images improved the non-prewhitening observer while leaving
the prewhitening observer no better, with the channelised observer in between. That ordering
follows observer efficiency, and it is the mechanism behind an apparent paradox in the
literature: a denoiser can genuinely raise a reader's performance without any information being
added, because the reader was not using all of it. The corollary is that a reported task
improvement is a statement about the observer as much as about the algorithm, and a study that
does not say which observer it used has not reported an effect size.

The third prespecified hypothesis was **partially refuted**, and the part that failed is as
informative as the part that held. What held is the coexistence it predicted: below the
operational requirement, fidelity still improved — by
+0.210 in SSIM — while
the task estimate fell, so a visually plausible output there does not establish that the required
detectability survived. What failed is the expectation that the *failure patterns* would
concentrate below the floor. Mean task degradation was larger above it
(30.3% against
14.7%), which is what
a relative loss must do when the input has less to lose; the contrast-erasure rate did not differ
appreciably between the strata; and excess lesion-like responses were absent in every stratum, so
these denoisers did not invent structure anywhere in the matrix.

The floor should therefore be read for what it is: a boundary of *attainable required
performance*, derived from the input and the requirement alone. It says that no processing of an
input below it can bring the requirement back into reach. It does not predict which denoiser will
erase contrast, or where lesion-like responses will appear — those are properties of the
processing, not of the boundary, and in this matrix they were not stratified by it. Reading the
floor as a predictor of denoiser-specific failure modes would over-claim; reading it as a limit on
what any processing can deliver is what the bound supports.

This is where the practical consequence about dose lies. A denoiser may
support dose reduction only insofar as the reduced-dose input still contains enough task
information to meet the required performance criterion for the intended observer. Establishing
that for a clinical protocol needs what this study deliberately does not have: a defined clinical
task, a specified required performance, the intended observers including humans, and the
scanner's own measured dose-to-noise calibration. Nothing here licenses a dose threshold for any
device.

The gauge is an auditable rule-based translation of the measured quantities, not a clinically
validated instrument. Its value is that a verdict naming the rule that fired, on numbers that are
written down, can be argued with; an impression that the images look acceptable cannot.

### 4.1 Limitations

The phantom is a disk on a uniform background with stationary Gaussian noise and the signal is
known exactly; anatomical background variability, non-stationary noise from iterative
reconstruction, and search tasks are outside the design. The observer evaluated on processed data
is a cross-fitted prewhitening linear observer, not the likelihood-ratio ideal observer of the
processed data; the CHO is estimated by a different held-out scheme than the other two. The Rose
criterion `d'` = 5 is one chosen
operational criterion, not a universal information boundary, and every gauge threshold is
task-specific: clinical deployment would require re-specifying and validating all of them. No
human observer study was performed, so the non-prewhitening observer stands only as a stylized
surrogate for limited prewhitening efficiency. The classical denoisers are parameterized with the
true noise level of the acquisition setting, which is more than a blind method would know. The
learned denoiser is small, CPU-trained, evaluated at one setting, and does not represent the
state of the art. The acquisition model is relative and analytic: it locates conditions on an
axis, not on a scanner. Ten realisations bound sampling variability but do not make the matrix
exhaustive, and the denoiser parameters were fixed a priori rather than swept. Finally, the scope
is post-processing of a defined input image; reconstruction from projection data is a separate
map, subject to its own bound, and is not studied here.

## 5. Conclusion

Under a fixed data-processing ceiling, denoising produced measurable and systematic
observer-dependent benefits: the same processing improved an inefficient observer while leaving
an efficient one no better, which is redistribution of existing information rather than creation
of new information. Reference-based fidelity gains frequently diverged from task performance, and
the divergence was not confined to extreme conditions. An operational information floor — a
prespecified requirement on the input's detectability, not a boundary of zero information —
identified the regime in which visual plausibility cannot establish that the required
detectability was preserved. It bounds what processing can attain; it did not, in this matrix,
predict which failure mode processing would produce, and neither erasure nor fabrication was
stratified by it. Across every condition studied these denoisers removed contrast rather than
inventing it, and no excess lesion-like response was observed.

## Disclosures

The author declares no financial or commercial conflicts of interest relevant to this work. The
controlled arm uses no measured data of any kind. The real-data arm uses de-identified public
images from LDCT-and-Projection-data, distributed by The Cancer Imaging Archive under `CC BY 4.0`;
no data were collected for this study, no proprietary software was used, and the work required no
additional ethical approval.

**AI-assisted tools.** Generative AI tools were used for language editing, code review, and
consistency checking during development of the software and manuscript. All study design
decisions, implementations, numerical results, interpretations, and final text were independently
reviewed and approved by the author, who assumes full responsibility for the work.

## Code and Data Availability

The controlled arm is generated analytically from documented seeds. The real-data arm uses the
public LDCT-and-Projection-data collection (TCIA, `CC BY 4.0`, DOI `10.7937/9npb-2637`); the code
that reads it, inserts the lesion and scores the result is `ldct-io`, and the consolidated
outputs are committed as `results/real_liver.json`. Every number reported here is produced by
the code below rather than transcribed. The software is `denoiq-core`
0.1.0 with `taskiq-core`
0.4.0, MIT licensed, Python `3.10` to `3.12`. The
repository — source, tests, generated results, figures and this manuscript's build — is public at
https://github.com/Institute-of-One/denoiq-core and is available at review time; the exact version reported here is
archived at Zenodo as version 0.1.0, doi:10.5281/zenodo.21733389.

Realisation seeds are the 10 values recorded in
`results/statistics.json`; the bootstrap seed is
20260731 with 4000 replicates.
Four commands regenerate everything:

1. `denoiq_core.experiment.run_primary()` — the multi-realisation matrix, endpoints and
   statistics.
2. `denoiq_core.experiment.run_all()` — the representative-realisation artefacts and the atlas.
3. `paper/make_figures.py` — Figs. 1 to 8 and the supplementary figures.
4. `paper/build_real_liver_results.py` — consolidates the real-data runs and their derived
   statistics into `results/real_liver.json`.
5. `paper/build_manuscript.py` — resolves every number in this manuscript from `results/`.

`pytest` runs the consistency suite, which re-derives the reported endpoints, effect sizes and
intervals from the same files, checks the design counts and the seed list, and fails if the
committed manuscript is out of date, if a number has been typed into the prose, or if the
observer and floor terminology has drifted in the text or in a figure label.

## Acknowledgments

This work received no external funding.

## References

<!-- Bibliographic details verified against publisher or Crossref records; the numbers below are
     bibliographic, not results-derived. -->

1. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, Hoboken, New Jersey (2006). (Data-processing inequality, Ch. 2.)
2. J. Neyman and E. S. Pearson, "On the problem of the most efficient tests of statistical hypotheses," *Philos. Trans. R. Soc. Lond. A* **231**, 289–337 (1933) [doi:10.1098/rsta.1933.0009].
3. K. Li, W. Zhou, H. Li, and M. A. Anastasio, "Assessing the impact of deep neural network-based image denoising on binary signal detection tasks," *IEEE Trans. Med. Imaging* **40**(9), 2295–2305 (2021) [doi:10.1109/TMI.2021.3076810].
4. Z. Yu, M. A. Rahman, R. Laforest, et al., "Need for objective task-based evaluation of deep learning-based denoising methods: a study in the context of myocardial perfusion SPECT," *Med. Phys.* **50**(7), 4122–4137 (2023) [doi:10.1002/mp.16407].
5. J. Li, W. Wang, M. Tivnan, J. W. Stayman, and G. J. Gang, "Performance assessment framework for neural network denoising," *Proc. SPIE* **12031**, 1203114 (2022) [doi:10.1117/12.2612732].
6. K. Li, H. Li, and M. A. Anastasio, "Investigating the use of signal detection information in supervised learning-based image denoising with consideration of task-shift," *J. Med. Imaging* **11**(5), 055501 (2024) [doi:10.1117/1.JMI.11.5.055501].
7. S. Bhadra, V. A. Kelkar, F. J. Brooks, and M. A. Anastasio, "On hallucinations in tomographic image reconstruction," *IEEE Trans. Med. Imaging* **40**(11), 3249–3260 (2021) [doi:10.1109/TMI.2021.3077857].
8. E. Eulig, B. Ommer, and M. Kachelrieß, "Benchmarking deep learning-based low-dose CT image denoising algorithms," *Med. Phys.* **51**(12), 8776–8788 (2024) [doi:10.1002/mp.17379].
9. B. J. Nelson, P. Kc, A. Badal, L. Jiang, S. C. Masters, and R. Zeng, "Pediatric evaluations for deep learning CT denoising," *Med. Phys.* **51**(2), 978–990 (2024) [doi:10.1002/mp.16901].
10. H. H. Barrett, K. J. Myers, C. Hoeschen, M. A. Kupinski, and M. P. Little, "Task-based measures of image quality and their relation to radiation dose and patient risk," *Phys. Med. Biol.* **60**(2), R1–R75 (2015) [doi:10.1088/0031-9155/60/2/R1].
11. H. H. Barrett and K. J. Myers, *Foundations of Image Science*, Wiley, Hoboken, New Jersey (2004).
12. H. H. Barrett, "Objective assessment of image quality: effects of quantum noise and object variability," *J. Opt. Soc. Am. A* **7**(7), 1266–1278 (1990) [doi:10.1364/JOSAA.7.001266].
13. S. Yamamoto, "taskiq-core: task-based image quality on synthetic phantoms," Zenodo (2026) [doi:10.5281/zenodo.21422924].
14. K. J. Myers and H. H. Barrett, "Addition of a channel mechanism to the ideal-observer model," *J. Opt. Soc. Am. A* **4**(12), 2447–2457 (1987) [doi:10.1364/JOSAA.4.002447].
15. A. E. Burgess, "Visual signal detection with two-component noise: low-pass spectrum effects," *J. Opt. Soc. Am. A* **16**(3), 694–704 (1999) [doi:10.1364/JOSAA.16.000694].
16. K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising," *IEEE Trans. Image Process.* **26**(7), 3142–3155 (2017) [doi:10.1109/TIP.2017.2662206].
17. J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology* **143**(1), 29–36 (1982) [doi:10.1148/radiology.143.1.7063747].
18. A. Rose, "The sensitivity performance of the human eye on an absolute scale," *J. Opt. Soc. Am.* **38**(2), 196–208 (1948) [doi:10.1364/JOSA.38.000196].
