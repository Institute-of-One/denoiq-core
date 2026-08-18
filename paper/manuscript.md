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
non-local means) and three observers: prewhitening linear (PW), channelised Hotelling
(CHO), and non-prewhitening with an eye filter (NPWE). Each processed arm is referenced to the
unprocessed arm of its own condition and to that input's **analytic** ideal-observer
detectability — the ceiling the data-processing inequality [1] and the Neyman–Pearson lemma [2]
place on any processing of it. Estimates are cross-fitted, uncertainty is by bootstrap over
realisations, and comparisons are Holm-adjusted. The matrix ran over
[[results:statistics.json:design.n_seeds]] realisations
([[results:statistics.json:design.scored_image_trials|,]] scored trials).
*Real data*: [[results:real_liver.json:all_cases.n_cases]] Siemens liver cases from LDCT-and-Projection-data,
vendor reconstructions of the routine and quarter-dose acquisitions, a synthetic lesion of known size and
contrast inserted into real parenchyma, and a residual CNN trained on
[[results:real_liver.json:held_out.n_train_cases]] cases and evaluated on the [[results:real_liver.json:held_out.n_test_cases]] it
never saw, at two capacities spanning [[results:real_liver.json:capacity.parameter_ratio|.0f]]× in parameters.

**Results.** *Controlled*: ΔSSIM and Δ`d'`(PW) correlated at Spearman ρ =
[[results:statistics.json:divergence.spearman_delta_ssim_delta_d_pw.value|.2f]]
([[results:statistics.json:divergence.divergence_rate.value|.1%]] of processed evaluations divergent, with ΔSSIM >
0 and Δ`d'`(PW) < 0). Denoising helped the inefficient observer more than the efficient one:
`B` = Δ`d'`(NPWE) − Δ`d'`(PW) = [[results:statistics.json:observer_dependence.overall.benefit.value|+.2f]]. The
information-floor hypothesis was **partially refuted**: below the floor fidelity rose while the
task estimate fell, but failure patterns did not concentrate there, and excess lesion-like
responses were absent throughout. *Real data*: across [[results:real_liver.json:held_out.n_methods]] methods the
rank correlation between PSNR and `d'` was ρ =
[[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.rho|+.2f]]; the method with the best PSNR ranked
[[results:real_liver.json:held_out.psnr_winner_task_rank]] of [[results:real_liver.json:held_out.n_methods]] on the task; and
[[results:real_liver.json:all_exceedances]] arms exceeded the closed-form ceiling in any real-data comparison.
A larger configuration of the same architecture -- [[results:real_liver.json:capacity.parameter_ratio|.0f]]x the parameters, trained on more patches for more epochs -- improved validation loss and gave the best PSNR in the study, [[results:real_liver.json:capacity.large.psnr|.2f]] dB, while lowering detectability from [[results:real_liver.json:capacity.small.d_prime|.2f]] to [[results:real_liver.json:capacity.large.d_prime|.2f]].

**Conclusions.** Denoising effects differed by observer, in a direction consistent with differences in prewhitening efficiency: processing improved the estimate for the non-prewhitening observer while leaving the prewhitening observer no better, which would be redistribution of existing information rather than creation of new information. Fidelity gains alone do not establish task preservation, and on real low-dose CT they ranked methods close to inversely to the task. The divergence appeared in both arms and persisted under a substantially larger and longer-trained network of the same architecture.

**Keywords:** low-dose CT; image denoising; deep learning; task-based image quality;
model observer; detectability; fidelity–task divergence.

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
   [[results:real_liver.json:capacity.parameter_ratio|.0f]]x in parameters — establishing that the effect is not a property of
   the synthetic model in which it was isolated.
6. An open, deterministic implementation in which every reported number is regenerated from
   machine-readable outputs.

## 2. Methods

### 2.1 Task, observers, and what "input" means

The task is signal-known-exactly / background-known-exactly (SKE/BKE) detection of a
low-contrast disk on a uniform background, the standard paradigm of objective, task-based
assessment [11,12]. Trials, phantoms, model observers and the ROC
machinery are reused from `taskiq-core` [13] (version
[[results:statistics.json:provenance.taskiq_core]]) rather than reimplemented; the trial
generator returns, with the image stacks, the **analytic** noise power spectrum (NPS) of the
noise it generated, which is what allows an observer to be evaluated in closed form with
nothing estimated. Geometry: [[results:dose_sweep.json:config.phantom.size]] ×
[[results:dose_sweep.json:config.phantom.size]] pixels at
[[results:dose_sweep.json:config.phantom.spacing]] mm, lesion radius
[[results:dose_sweep.json:config.phantom.radius_mm]] mm except where the lesion sweep varies it,
[[results:statistics.json:design.trials_per_class_main_sweeps|,]] trials per class per arm in the three sweeps and fewer at the atlas settings.

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
3. A **non-prewhitening observer with a Burgess eye filter** (NPWE) [15], used as a
   stylized surrogate for limited noise-prewhitening efficiency. No human observer study
   was performed here, and NPWE is not offered as a validated model of a human reader.

Separately, and only for the unprocessed input, we compute the **analytic ideal linear
(prewhitening) observer** from the true signal and the analytic NPS. That quantity is exact for
the experiment it describes and carries no sampling error; it is the input's *ceiling*, and it
is the only observer here that is not estimated from images.

### 2.2 Relative acquisition model

Contrast and noise follow three documented proportionalities relative to a reference setting:
photon count `N ∝ mAs·kV²`, noise standard deviation `σ ∝ 1/√N`, and subject contrast
`c ∝ kV^(−p)` with `p` = [[results:dose_sweep.json:config.model.contrast_exponent]]. The
reference setting is [[results:dose_sweep.json:config.model.kv_ref|.0f]] kV,
[[results:dose_sweep.json:config.model.mas_ref|.0f]] mAs, with reference noise
[[results:dose_sweep.json:config.model.sigma_ref|.0f]] and reference contrast
[[results:dose_sweep.json:config.model.c_ref|.0f]] in the same arbitrary intensity units. Two
consequences are visible in the results: for a fixed signal profile `d' ∝ c/σ`, so
`d' ∝ √mAs · kV^(1−p)`, and the mAs at which a *fixed* detectability requirement is met scales
as `mAs_floor ∝ kV^(2p−2)`, which for this `p` is linear in kV.

Dose is taken proportional to mAs; it also rises with kV in reality, and that dependence is
left out, so **every dose comparison in this paper is made at fixed kV**. On the kV axis we
report kV itself and never a dose ratio. This is a normalised model, not a calibration: it
carries no measured dose-to-noise relation for any device and no absolute exposure values.

### 2.3 Denoisers, and what they are told

Three deterministic classical denoisers: a Gaussian filter of fixed width
[[results:dose_sweep.json:config.denoisers[label=gaussian(1.5 px)].params.sigma]] pixels, total
variation with weight
[[results:dose_sweep.json:config.denoisers[label=TV(0.4 sd)].params.weight]] × the noise
standard deviation, and non-local means with `h` =
[[results:dose_sweep.json:config.denoisers[label=NLM(0.6 sd)].params.h]] × the noise standard
deviation. A small residual convolutional network in the style of DnCNN [16] is evaluated
separately (Section 3.6) and is not part of the primary matrix.

The Gaussian filter is `scipy.ndimage.gaussian_filter` with `mode=[[results:real_liver.json:spec.implementation.gaussian]]`; total variation is `skimage.restoration.denoise_tv_chambolle` applied as [[results:real_liver.json:spec.implementation.tv]]; non-local means is `skimage.restoration.denoise_nl_means` with [[results:real_liver.json:spec.implementation.nlm]]. Library versions are recorded with the provenance of every run.

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
why the processed-arm count is [[results:statistics.json:design.processed_arms]] rather than
three times the [[results:statistics.json:design.input_conditions]] input conditions.

The whole matrix was run over [[results:statistics.json:design.n_seeds]] independent
realisations. The first is the seed of the earlier single-realisation study, retained so that
the previous result remains inspectable; the others follow from it by a fixed stride recorded in
the configuration, so the list is a property of the code rather than of a session.
Signal-present and signal-absent trials come from one seeded stream per realisation and are
independent by construction. Every realisation runs the identical matrix, which makes arms
pairable across realisations, and per-realisation results are written alongside the aggregate.
We report [[results:statistics.json:design.unique_arms]] unique arms evaluated across
[[results:statistics.json:design.n_seeds]] independent realisations —
[[results:statistics.json:design.arm_seed_evaluations]] arm–realisation evaluations — and never
as a single inflated condition count. The representative realisation used for the example
images is the first seed; Figure 1 shows the signal-present and signal-absent images at one
setting, for the unprocessed input and each classical denoiser on a common display window.

### 2.5 Cross-fitted estimation, and the ceiling comparison

On the unprocessed input the ideal linear observer is closed-form. On processed images neither
the noise spectrum nor the effective signal is known analytically — a non-linear denoiser has no
transfer function — so both are estimated, and the estimate must not be allowed to score itself.

Estimation is **cross-fitted** over `K` =
[[results:dose_sweep.json:config.eval_config.n_folds]] folds. Fold membership is by trial index
modulo `K`; the trials are i.i.d. draws from one seeded stream, so a positional partition is
already a random one and needs no second random number to record. For each fold, the effective
signal `Ŝ` (the class-mean difference) and the noise spectrum are estimated from the other folds
inside a central [[results:dose_sweep.json:config.eval_config.roi_size]] ×
[[results:dose_sweep.json:config.eval_config.roi_size]] pixel window, the prewhitening template
`Ŵ = Ŝ/NPS` is formed, and the held-out fold is scored with it. Pooling the out-of-fold scores
gives `d'` from the pooled-variance separation and AUC from the Mann–Whitney statistic, using
every trial exactly once while no score comes from a template that saw its own image. The
measured NPS is made invertible by filling the DC bin from its neighbours, adding a ridge of
[[results:dose_sweep.json:config.eval_config.nps_ridge_fraction]] × the mean power, and clamping
at [[results:dose_sweep.json:config.eval_config.nps_floor_fraction|sci0]] × the peak; these
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
resamples whole realisations** with replacement ([[results:statistics.json:n_boot]] replicates,
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
`d'` = [[results:dose_sweep.json:config.criteria.d_prime_threshold|.0f]]. It is used in this
paper as a stratifying variable, and it is not a zero-information boundary.

The study also carries an auditable rule-based translation of the measured quantities into a
green / amber / red verdict with the rule that fired attached. Its thresholds are task-specific
author-set values, not clinically validated criteria; the complete rule set is given in
Supplementary Methods, and verdict counts per stratum are reported in Section 3.5.

### 2.10 The real low-dose CT arm

**Images.** Twelve Siemens liver cases from LDCT-and-Projection-data, using the vendor
reconstructions of both the routine-dose and the simulated quarter-dose acquisition. Nothing is
re-reconstructed. The quarter-dose noise field is taken as the difference between the two
reconstructions of the same anatomy, so the noise carried into every trial is the acquisition's
own and not a model of it.

**Lesion insertion.** A patient scan has no ground truth: the lesions in it were found by a
reader, at a contrast nobody measured. A **synthetic** lesion of known size and amplitude is
therefore inserted into real parenchyma, which keeps the background that makes the task hard and
supplies the truth that makes it measurable. The lesion is a disk of
[[results:real_liver.json:spec.lesion.diameter_mm|.0f]] mm diameter and [[results:real_liver.json:spec.lesion.contrast_hu|.0f]] HU
contrast, with a Gaussian edge of [[results:real_liver.json:spec.lesion.edge_sigma_mm|.1f]] mm, rendered on a grid
supersampled [[results:real_liver.json:spec.lesion.supersample]]-fold so its own edge is not a one-pixel staircase,
and added **after** reconstruction. The one assumption this makes is stated rather than left
implicit: an inserted lesion does not carry the reconstruction's own response to a real lesion of
that contrast, so the arm measures detection of a known additive signal in real anatomy and
real noise, not detection of pathology.

**Where lesions are placed.** Candidate sites are regions of interest of
[[results:real_liver.json:spec.lesion.roi_px]] pixels whose mean lies between [[results:real_liver.json:spec.sites.hu_range_low|.0f]] and
[[results:real_liver.json:spec.sites.hu_range_high|.0f]] HU and whose standard deviation is below
[[results:real_liver.json:spec.sites.max_sd_hu|.0f]] HU, drawn from the
[[results:real_liver.json:spec.sites.slice_halfwidth]] slices either side of the middle of each case. Up to
[[results:real_liver.json:spec.sites.max_per_case]] sites are used per case and a case contributing fewer than
[[results:real_liver.json:spec.sites.min_per_case]] is dropped. Signal-present and signal-absent trials are built at
the same sites from the same background, so the pair differs only by the lesion.

**Denoisers.** The classical arms use the implementations of Section 2.3, parameterized in
physical units for this data: Gaussian filters of [[results:real_liver.json:spec.arms.gaussian_small_mm|.2f]] mm and
[[results:real_liver.json:spec.arms.gaussian_large_mm|.2f]] mm, total variation at
[[results:real_liver.json:spec.arms.tv_weight_x_noise|.0f]]× and non-local means at
[[results:real_liver.json:spec.arms.nlm_h_x_noise|.1f]]× the measured noise standard deviation.

**Network.** A residual convolutional denoiser in the DnCNN formulation:
[[results:real_liver.json:spec.cnn.formulation]]. Two configurations are evaluated — the smaller with
[[results:real_liver.json:spec.cnn.small.depth]] layers of [[results:real_liver.json:spec.cnn.small.width]] channels and
[[results:real_liver.json:spec.cnn.small.kernel]]×[[results:real_liver.json:spec.cnn.small.kernel]] kernels
([[results:real_liver.json:capacity.small.parameters]] parameters), the larger with
[[results:real_liver.json:spec.cnn.large.depth]] layers of [[results:real_liver.json:spec.cnn.large.width]] channels and
[[results:real_liver.json:spec.cnn.large.kernel]]×[[results:real_liver.json:spec.cnn.large.kernel]] kernels
([[results:real_liver.json:capacity.large.parameters]] parameters). Both are trained on
[[results:real_liver.json:spec.cnn.patch_px]]×[[results:real_liver.json:spec.cnn.patch_px]] quarter-dose / routine-dose patch pairs with
[[results:real_liver.json:spec.cnn.optimiser]] at a learning rate of [[results:real_liver.json:spec.cnn.learning_rate]], the smaller for
[[results:real_liver.json:capacity.small.epochs]] epochs at batch [[results:real_liver.json:capacity.small.batch]] on
[[results:real_liver.json:capacity.small.patches_per_case]] patches per case, the larger for
[[results:real_liver.json:capacity.large.epochs]] epochs at batch [[results:real_liver.json:capacity.large.batch]] on
[[results:real_liver.json:capacity.large.patches_per_case]] patches per case. **The two therefore differ in three
respects at once** — parameters, training data and training length — and are not a controlled
comparison of capacity.

**What the network is and is not shown.** Training pairs are normalised exactly as inference
normalises them, by each image's own mean and estimated noise level; a network trained in
absolute HU and deployed through a normalising wrapper is not the network that was trained. The
split is by case: the network sees [[results:real_liver.json:held_out.n_train_cases]] cases and is evaluated on the
[[results:real_liver.json:held_out.n_test_cases]] it never saw, because slices from one patient are not independent
and a network tested on another slice of a liver it trained on is being tested on its own
training set. The network never sees a lesion — its targets are routine-dose reconstructions of
ordinary anatomy, which is what a denoiser is actually given — so the lesion exists only in the
evaluation.

### 2.9 Use of generative AI

Generative AI (Claude, Anthropic, through the Claude Code command-line tool) was used as
a tool in preparing this work: scaffolding and refactoring the released software,
drafting unit tests, writing the figure and analysis scripts, and drafting and revising
manuscript prose. It was not used to design the study, to choose the endpoints, or to
decide what the results mean.

No numerical result came from the model. Every number, table and figure in this
manuscript is emitted by executed code into machine-readable files under `results/`, and
the text resolves against those files at build time; the test suite fails if the two
disagree, so a value cannot be typed into the prose. Every reference was checked against
its Crossref record before being cited. The author designed the study, re-executed every
result and verified all figures, equations and claims against the code, and is solely
accountable for the content. No AI system is an author. This disclosure follows ICMJE and
COPE guidance and is repeated under Disclosures.

## 3. Results

### 3.1 Study design and evaluated conditions (Table 1)

Table 1 gives the matrix. The primary analysis comprises
[[results:statistics.json:design.input_conditions]] input conditions and
[[results:statistics.json:design.unique_arms]] unique arms
([[results:statistics.json:design.unprocessed_arms]] unprocessed,
[[results:statistics.json:design.processed_arms]] processed), each evaluated in
[[results:statistics.json:design.n_seeds]] independent realisations, giving
[[results:statistics.json:design.arm_seed_evaluations]] arm–realisation evaluations and
[[results:statistics.json:design.scored_image_trials|,]] scored image trials. Endpoints are
computed on the [[results:statistics.json:n_arm_seed_evaluations]] processed evaluations, each
paired with the unprocessed arm of its own condition and realisation.

Figures 2 and 3 give the dose response of the representative realisation: Figure 2 the
reference-based fidelity (SSIM and PSNR) against relative dose, and Figure 3 the task `d'` of the
three estimated observers against relative dose, with the analytic ideal-observer ceiling of the
unprocessed input.

### 3.2 Fidelity gains frequently diverge from task performance (Figure 4)

Across all processed evaluations, ΔSSIM and Δ`d'`(PW) were correlated at Spearman ρ =
[[results:statistics.json:divergence.spearman_delta_ssim_delta_d_pw.value|.2f]] (`95 %` CI
[[results:statistics.json:divergence.spearman_delta_ssim_delta_d_pw.ci_low|.2f]] to
[[results:statistics.json:divergence.spearman_delta_ssim_delta_d_pw.ci_high|.2f]]). Mean ΔSSIM
was [[results:statistics.json:divergence.mean_delta_ssim.value|+.3f]]
([[results:statistics.json:divergence.mean_delta_ssim.ci_low|+.3f]] to
[[results:statistics.json:divergence.mean_delta_ssim.ci_high|+.3f]]) while mean Δ`d'`(PW) was
[[results:statistics.json:divergence.mean_delta_d_pw.value|+.3f]]
([[results:statistics.json:divergence.mean_delta_d_pw.ci_low|+.3f]] to
[[results:statistics.json:divergence.mean_delta_d_pw.ci_high|+.3f]]): fidelity improved on
average, the task estimate did not.

[[results:statistics.json:divergence.divergence_rate.value|.1%]] of evaluations were divergent —
fidelity up, task estimate down — with a clustered `95 %` interval of
[[results:statistics.json:divergence.divergence_rate.ci_low|.1%]] to
[[results:statistics.json:divergence.divergence_rate.ci_high|.1%]]. The rate differed by
denoiser:
[[results:statistics.json:divergence.by_denoiser["gaussian(1.5 px)"].divergence_rate.value|.1%]]
for the Gaussian filter,
[[results:statistics.json:divergence.by_denoiser["TV(0.4 sd)"].divergence_rate.value|.1%]] for
total variation and
[[results:statistics.json:divergence.by_denoiser["NLM(0.6 sd)"].divergence_rate.value|.1%]] for
non-local means — an ordering that follows filter strength rather than filter family, with
total variation the most conservative of the three at these settings. Figure 4 shows every
evaluation in the ΔSSIM–Δ`d'` plane; the quadrant in which fidelity improves while the task
estimate falls is where most of the distribution lies.

### 3.3 Denoising benefits depend on observer efficiency (Figure 5, Table 2)

On identical images the three observers responded differently to the same processing. Averaged
over all evaluations, Δ`d'`(PW) was
[[results:statistics.json:observer_dependence.overall.delta_d_pw.value|+.2f]]
([[results:statistics.json:observer_dependence.overall.delta_d_pw.ci_low|+.2f]] to
[[results:statistics.json:observer_dependence.overall.delta_d_pw.ci_high|+.2f]]), Δ`d'`(CHO)
[[results:statistics.json:observer_dependence.overall.delta_d_cho.value|+.2f]]
([[results:statistics.json:observer_dependence.overall.delta_d_cho.ci_low|+.2f]] to
[[results:statistics.json:observer_dependence.overall.delta_d_cho.ci_high|+.2f]]) and Δ`d'`(NPWE)
[[results:statistics.json:observer_dependence.overall.delta_d_npwe.value|+.2f]]
([[results:statistics.json:observer_dependence.overall.delta_d_npwe.ci_low|+.2f]] to
[[results:statistics.json:observer_dependence.overall.delta_d_npwe.ci_high|+.2f]]) — an ordering
that follows observer efficiency, with the intermediate observer in between.

The observer-dependent benefit was `B` =
[[results:statistics.json:observer_dependence.overall.benefit.value|+.2f]]
([[results:statistics.json:observer_dependence.overall.benefit.ci_low|+.2f]] to
[[results:statistics.json:observer_dependence.overall.benefit.ci_high|+.2f]]), and
[[results:statistics.json:observer_dependence.npwe_improved_pw_did_not.value|.1%]]
([[results:statistics.json:observer_dependence.npwe_improved_pw_did_not.ci_low|.1%]] to
[[results:statistics.json:observer_dependence.npwe_improved_pw_did_not.ci_high|.1%]]) of
evaluations improved the non-prewhitening observer while not improving the prewhitening one.
Table 2 gives the per-denoiser effects with intervals and Holm-adjusted p-values; Figure 5 plots
them with realisation-level uncertainty. The benefit also varied with noise correlation length —
the condition that separates an observer that can prewhiten from one that cannot: `B` =
[[results:statistics.json:observer_dependence.by_correlation["0"].benefit.value|+.2f]] at zero
correlation against
[[results:statistics.json:observer_dependence.by_correlation["1"].benefit.value|+.2f]] at the
longest correlation length studied.

### 3.4 Performance relative to the data-processing ceiling (Figure 6)

Across [[results:statistics.json:ceiling.n_evaluations]] arm–realisation evaluations the mean
excess of the cross-fitted prewhitening AUC over the analytic ceiling of its own input was
[[results:statistics.json:ceiling.mean_excess|.3f]], and the largest was
[[results:statistics.json:ceiling.max_excess|sci2]].

The claim the design supports is about processing, and it is stated simultaneously over the
whole family: with the armwise margin widened by Bonferroni to a family-wise level of `0.05`
across all [[results:statistics.json:ceiling.family_wise.n_comparisons]] comparisons
(`z` = [[results:statistics.json:ceiling.family_wise.z|.2f]]),
**[[results:statistics.json:ceiling.family_wise.n_exceedances_processed_arms]] of the
[[results:statistics.json:ceiling.family_wise.n_processed_comparisons]] processed-arm
comparisons exceeded the ceiling margin**. The same holds at the unadjusted armwise margin
([[results:statistics.json:ceiling.n_violations_processed_arms]] of
[[results:statistics.json:ceiling.family_wise.n_processed_comparisons]]).

One comparison did exceed its margin, and it is reported separately because it is not a
statement about processing. It occurred on an **unprocessed** arm — a self-comparison of the
input against its own analytic ceiling, where the processing is the identity and cannot create
information, so the exceedance measures the estimator and the finite sample rather than a
violated bound. That arm is AUC-saturated:
its analytic ceiling AUC is
[[results:statistics.json:ceiling.violation_detail[0].ceiling_auc|.7f]] and the estimator
achieved perfect separation
([[results:statistics.json:ceiling.violation_detail[0].auc|.7f]]), an excess of
[[results:statistics.json:ceiling.violation_detail[0].excess|sci2]] in AUC — of the order of ten
misordered pairs in `1.44` million. At perfect separation the Hanley–McNeil standard error is
identically zero, so the margin collapses to a single quantisation step
([[results:statistics.json:ceiling.violation_detail[0].margin|sci2]] in AUC, one misordered pair
out of `1.44` million) and no widening of `z` rescues it: the family-wise margin leaves this
same [[results:statistics.json:ceiling.family_wise.n_exceedances_unprocessed_arms]] exceedance
for exactly that reason. This is a known degeneracy of the closed-form interval at the top of
the scale, it is a property of a finite sample rather than of the bound, and it is why the
saturated regime is analysed separately.

[[results:statistics.json:ceiling.n_saturated]] evaluations were AUC-saturated in the sense
defined in Section 2.5. Over the remaining
[[results:statistics.json:ceiling.unsaturated.n_evaluations]] the largest excess was
[[results:statistics.json:ceiling.unsaturated.max_excess|sci2]], the mean
[[results:statistics.json:ceiling.unsaturated.mean_excess|.3f]], and there were
[[results:statistics.json:ceiling.unsaturated.n_violations]] exceedances, so nothing here rests
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
raised SSIM by [[results:statistics.json:floor_strata["below floor"].mean_delta_ssim.value|+.3f]]
on average while Δ`d'`(PW) was
[[results:statistics.json:divergence.by_stratum["below floor"].mean_delta_d_pw.value|+.3f]], and
[[results:statistics.json:floor_strata["below floor"].divergence_rate.value|.1%]]
([[results:statistics.json:floor_strata["below floor"].divergence_rate.ci_low|.1%]] to
[[results:statistics.json:floor_strata["below floor"].divergence_rate.ci_high|.1%]]) of those
evaluations were divergent. Visually improved output therefore coexisted, throughout that
stratum, with an input that did not meet the requirement and a processed estimate that fell
further below it.

**Refuted: failures do not concentrate below the floor.** We had anticipated that failure
patterns would be worse below the floor. They were not. The divergent fraction was *higher*
above the floor —
[[results:statistics.json:floor_strata["above floor"].divergence_rate.value|.1%]] against
[[results:statistics.json:floor_strata["below floor"].divergence_rate.value|.1%]], a difference
of
[[results:statistics.json:stratum_contrasts["divergence_rate: below floor - above floor"].value|+.1%]]
([[results:statistics.json:stratum_contrasts["divergence_rate: below floor - above floor"].ci_low|+.1%]]
to
[[results:statistics.json:stratum_contrasts["divergence_rate: below floor - above floor"].ci_high|+.1%]])
— and so was mean task degradation:
[[results:statistics.json:floor_strata["above floor"].task_degradation.value|.1%]] of the input's
detectability above the floor against
[[results:statistics.json:floor_strata["below floor"].task_degradation.value|.1%]] below it
(difference
[[results:statistics.json:stratum_contrasts["task_degradation: below floor - above floor"].value|+.1%]],
[[results:statistics.json:stratum_contrasts["task_degradation: below floor - above floor"].ci_low|+.1%]]
to
[[results:statistics.json:stratum_contrasts["task_degradation: below floor - above floor"].ci_high|+.1%]]).
The mechanism is straightforward once seen: *relative* loss is largest where there was most to
lose. A high-detectability input has detectability for a filter to discard, and these filters
discard it; a low-detectability input has little, and proportionally little is removed. The same
ordering appears in the observer-dependent benefit, which was
[[results:statistics.json:floor_strata["above floor"].benefit.value|+.2f]] above the floor
against [[results:statistics.json:floor_strata["below floor"].benefit.value|+.2f]] below it.

Contrast erasure did not differ appreciably between strata
([[results:statistics.json:floor_strata["above floor"].erasure_rate.value|.1%]] above against
[[results:statistics.json:floor_strata["below floor"].erasure_rate.value|.1%]] below; difference
[[results:statistics.json:stratum_contrasts["erasure_rate: below floor - above floor"].value|+.1%]],
[[results:statistics.json:stratum_contrasts["erasure_rate: below floor - above floor"].ci_low|+.1%]]
to
[[results:statistics.json:stratum_contrasts["erasure_rate: below floor - above floor"].ci_high|+.1%]],
Holm-adjusted p =
[[results:statistics.json:stratum_contrasts.p_holm["erasure_rate: below floor - above floor"]|.2f]]),
and excess lesion-like responses were absent in every stratum
([[results:statistics.json:floor_strata["below floor"].excess_response_rate.value|.1%]] below the
floor). With these denoisers, at this lesion-like response threshold, the failure mode is erasure
and dilution of true contrast rather than fabrication of false structure — a result that should
not be generalised to generative or learned methods, whose failure modes may differ.

The gauge's verdicts reflect the strata by construction more than by discovery: above the floor
[[results:statistics.json:floor_strata["above floor"].verdicts.green]] of
[[results:statistics.json:floor_strata["above floor"].n_arm_seed]] evaluations were green and
[[results:statistics.json:floor_strata["above floor"].verdicts.red]] red on erasure or task loss,
while below the floor all
[[results:statistics.json:floor_strata["below floor"].n_arm_seed]] were red — an input that fails
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
quarter-dose / full-dose patch pairs from [[results:real_liver.json:held_out.n_train_cases]] cases and evaluated on
the [[results:real_liver.json:held_out.n_test_cases]] it never saw ([[results:real_liver.json:held_out.n_pairs]] pairs). **The split is by
case, not by slice**: slices from one patient are not independent, and a network tested on
another slice of a liver it trained on is being tested on its own training set. Normalisation at
training matches normalisation at inference, since a network trained in absolute HU and deployed
through a normalising wrapper is not the network that was trained. The network never sees a
lesion — its targets are full-dose reconstructions of ordinary anatomy, which is what a
denoiser is actually given — and the lesion exists only in the evaluation.

**Result.** Figure 9 and Table 3 give the held-out comparison against the closed-form ceiling
`d'` = [[results:real_liver.json:held_out.ceiling|.2f]]:

**Table 3.** The seven arms on the held-out real low-dose CT split, ordered by task detectability. `d'` is against the closed-form ceiling of the unprocessed input; PSNR is against the full-dose reconstruction. The two networks share an architecture, a training split and a seed. The larger differs in three respects at once -- parameters, training patches per case and epochs -- so this is not a controlled comparison of capacity alone.

| method | `d'` | of ceiling | PSNR (dB) |
|---|---|---|---|
| `tv 1x noise` | [[results:real_liver.json:held_out.by_method.tv.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.tv.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.tv.psnr|.2f]] |
| unprocessed | [[results:real_liver.json:held_out.by_method.unprocessed.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.unprocessed.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.unprocessed.psnr|.2f]] |
| `nlm 0.8x noise` | [[results:real_liver.json:held_out.by_method.nlm.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.nlm.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.nlm.psnr|.2f]] |
| CNN, [[results:real_liver.json:capacity.small.parameters]] parameters | [[results:real_liver.json:held_out.by_method.cnn_small.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_small.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_small.psnr|.2f]] |
| `gaussian 0.75 mm` | [[results:real_liver.json:held_out.by_method.gauss075.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gauss075.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gauss075.psnr|.2f]] |
| CNN, [[results:real_liver.json:capacity.large.parameters]] parameters | [[results:real_liver.json:held_out.by_method.cnn_large.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_large.ratio|.2f]] | **[[results:real_liver.json:held_out.by_method.cnn_large.psnr|.2f]]** |
| `gaussian 1.00 mm` | [[results:real_liver.json:held_out.by_method.gauss100.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gauss100.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gauss100.psnr|.2f]] |

Three things follow.

*Fidelity and task rank the methods differently, and if anything inversely.* Across the
[[results:real_liver.json:held_out.n_methods]] arms the rank correlation between PSNR and `d'` is Spearman
ρ = [[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.rho|+.2f]]
(`p` = [[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.p_value|.2f]]). The arm with the best PSNR of
all [[results:real_liver.json:held_out.n_methods]] ranks [[results:real_liver.json:held_out.psnr_winner_task_rank]] of
[[results:real_liver.json:held_out.n_methods]] on the task. Selecting a denoiser by fidelity on these data would have
selected close to the worst available option for detection.

*Nothing exceeded the ceiling.* [[results:real_liver.json:held_out.n_exceeding_ceiling]] of
[[results:real_liver.json:held_out.n_methods]] arms on the held-out split, and [[results:real_liver.json:all_exceedances]] across every
real-data arm in this study — the held-out split, the [[results:real_liver.json:all_cases.n_cases]]-case run, and
both operating points of Section 3.6.1. The bound the controlled matrix was built to test is not
an artefact of the controlled matrix.

*A larger network, trained longer on more data, does not reverse the ordering.* The large network has
[[results:real_liver.json:capacity.parameter_ratio|.0f]] times the parameters of the small one
([[results:real_liver.json:capacity.large.parameters]] against [[results:real_liver.json:capacity.small.parameters]]), was trained on
[[results:real_liver.json:capacity.large.patches_per_case]] patches per case against
[[results:real_liver.json:capacity.small.patches_per_case]], and reached a better validation loss. It bought
[[results:real_liver.json:capacity.large.psnr|.2f]] dB against [[results:real_liver.json:capacity.small.psnr|.2f]] — the best
PSNR in the study — and a `d'` of [[results:real_liver.json:capacity.large.d_prime|.2f]] against
[[results:real_liver.json:capacity.small.d_prime|.2f]]. The configuration that scored better on the objective it was trained against scored worse on the task. This bears on the scope of the claim, though less strongly than it may look. Capacity was not varied alone: the larger configuration also saw [[results:real_liver.json:capacity.large.patches_per_case]] training patches per case against [[results:real_liver.json:capacity.small.patches_per_case]], and ran for [[results:real_liver.json:capacity.large.epochs]] epochs against [[results:real_liver.json:capacity.small.epochs]], so parameters, training data and training length moved together. What the comparison shows is that the divergence persisted when the network was made substantially larger and trained longer within this architecture. That weakens the first objection such a result invites -- that the network was too small to be representative -- without excluding it, and it says nothing about architectures not tried here.

Two caveats are stated here rather than left for a reader to find. The large network's validation
loss reached its minimum well before the last of its
[[results:real_liver.json:capacity.large.epochs]] epochs and drifted upward thereafter, so the
saved model is not the best one the run produced; the gap is a small fraction of the validation
loss and far too little to move `d'`. And [[results:real_liver.json:capacity.large.delta_psnr|+.2f]] dB of PSNR over the unprocessed
input for [[results:real_liver.json:capacity.parameter_ratio|.0f]]x the capacity is itself informative: the
training target is a full-dose *reconstruction*, which carries noise of its own that no network
can predict, so mean-squared error against it saturates well before the image does.

#### 3.6.1 A second operating point

Every number above rests on one acquisition. The same measurement at a second and very different
one — chest at [[results:real_liver.json:operating_points.chest.dose|.0%]] dose, where the noise standard
deviation is [[results:real_liver.json:operating_points.noise_ratio_chest_over_liver|.1f]] times the liver's and
the closed-form ceiling falls from [[results:real_liver.json:operating_points.liver.ceiling|.2f]] to
[[results:real_liver.json:operating_points.chest.ceiling|.2f]] — produced
[[results:real_liver.json:operating_points.chest.n_exceeding_ceiling]] exceedances of its own ceiling. The bound holds
where the task is nearly impossible as well as where it is comfortable.

### 3.7 Sensitivity and implementation validation

**Estimator efficiency.** On unprocessed images the cross-fitted estimator recovered a median
[[results:statistics.json:ceiling.estimator_recovery.median|.0%]] of the analytic ceiling `d'`
(interquartile range [[results:statistics.json:ceiling.estimator_recovery.q1|.0%]] to
[[results:statistics.json:ceiling.estimator_recovery.q3|.0%]]), with a minimum of
[[results:statistics.json:ceiling.estimator_recovery.min|.0%]] over all evaluations and
[[results:statistics.json:ceiling.estimator_recovery.above_floor_min|.0%]] over conditions at or
above the floor. The lowest recoveries occur at the lowest doses, where the template is estimated
from images in which the signal is weakest — the regime in which the estimator, not the bound, is
the limiting factor, which is why it is reported rather than summarised away.

**Cross-fitting against a single split.** Supplementary Table S2 compares the cross-fitted
estimator with the single `50/50` split used in the earlier single-realisation analysis at matched
conditions; cross-fitting recovers more of the ceiling at every dose, and the qualitative
conclusions are unchanged.

**Closed form.** In white noise the analytic observer must satisfy `d' = ‖s‖₂/σ`. Over the
combinations of noise level and lesion radius in Supplementary Table S1 the maximum relative
deviation was [[results:closed_form.json:max_relative_error|sci2]], against a criterion of 1 %
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
the prewhitening observer no better, with the channelised observer in between. That ordering is consistent with differences in prewhitening efficiency, and offers a reading of an apparent paradox in the literature: a denoiser could genuinely raise a reader's performance without information being added, if the reader was not using all of it. The three observers differ in template, channel model and noise handling as well as in prewhitening efficiency, so this is a reading of the ordering rather than a controlled attribution. The corollary is that a reported task
improvement is a statement about the observer as much as about the algorithm, and a study that
does not say which observer it used has not reported an effect size.

The third prespecified hypothesis was **partially refuted**, and the part that failed is as
informative as the part that held. What held is the coexistence it predicted: below the
operational requirement, fidelity still improved — by
[[results:statistics.json:floor_strata["below floor"].mean_delta_ssim.value|+.3f]] in SSIM — while
the task estimate fell, so a visually plausible output there does not establish that the required
detectability survived. What failed is the expectation that the *failure patterns* would
concentrate below the floor. Mean task degradation was larger above it
([[results:statistics.json:floor_strata["above floor"].task_degradation.value|.1%]] against
[[results:statistics.json:floor_strata["below floor"].task_degradation.value|.1%]]), which is what
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
criterion `d'` = [[results:dose_sweep.json:config.criteria.d_prime_threshold|.0f]] is one chosen
operational criterion, not a universal information boundary, and every gauge threshold is
task-specific: clinical deployment would require re-specifying and validating all of them. No
human observer study was performed, so the non-prewhitening observer stands only as a
stylized surrogate for limited prewhitening efficiency. The classical denoisers are parameterized with the
true noise level of the acquisition setting, which is more than a blind method would know. The
learned denoiser is small, CPU-trained, evaluated at one setting, and does not represent the
state of the art. The acquisition model is relative and analytic: it locates conditions on an
axis, not on a scanner. Ten realisations bound sampling variability but do not make the matrix
exhaustive, and the denoiser parameters were fixed a priori rather than swept. Finally, the scope
is post-processing of a defined input image; reconstruction from projection data is a separate
map, subject to its own bound, and is not studied here.

## 5. Conclusion

Under a fixed data-processing ceiling, the effect of denoising differed systematically between observers: the same processing improved the non-prewhitening estimate while leaving the prewhitening one no better. That is consistent with redistribution of existing information rather than creation of new information, which the ceiling forbids in any case. Reference-based fidelity gains frequently diverged from task performance, and
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
[[results:statistics.json:provenance.denoiq_core]] with `taskiq-core`
[[results:statistics.json:provenance.taskiq_core]], MIT licensed, Python `3.10` to `3.12`. The
repository — source, tests, generated results, figures and this manuscript's build — is public at
[[release:repository]] and is available at review time; the exact version reported here is
[[release:archive_statement]].

Realisation seeds are the [[results:statistics.json:design.n_seeds]] values recorded in
`results/statistics.json`; the bootstrap seed is
[[results:statistics.json:bootstrap_seed]] with [[results:statistics.json:n_boot]] replicates.
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
