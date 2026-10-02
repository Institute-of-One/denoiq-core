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

# Mathematical characterization of task detectability limits in AI-denoised CT: implications for task-based dose optimization

**Shuji Yamamoto**
Institute of One, LISIT Co., Ltd., Tokyo, Japan
yamamoto@lisit.jp · ORCID 0000-0001-9211-1071

## Abstract

**Background.** Denoising improves reference-based fidelity metrics and is widely held thereby to
permit dose reduction. That does not follow: processing cannot add task information.

**Purpose.** To characterize task detectability limits in denoised CT by separating the
detectability an input offers an ideal observer from that a specified observer achieves, and
thereby compare the exposure each method needs.

**Methods.** Signal- and background-known-exactly detection of a low-contrast disk on phantoms
with three classical denoisers and three observers, and on
[[results:real_liver.json:all_cases.n_cases]] liver cases from LDCT-and-Projection-data with a
synthetic lesion in abdominal soft tissue. Each arm is referenced to its unprocessed input and its
analytic ceiling [1,2]. Residual CNNs trained on
[[results:real_liver.json:held_out.n_train_cases]] cases were evaluated on
[[results:real_liver.json:held_out.n_test_cases]] never seen, at two capacities, each paired with a
counterpart differing only in an adversarial objective. Exposure was compared on a dose simulation
rescaling the collection's measured noise, against `d'` ≥
[[results:real_liver.json:guidance.requirement|.0f]].

**Results.** [[results:statistics.json:divergence.divergence_rate.value|.1%]] of evaluations
improved fidelity while detectability fell; denoising helped the inefficient observer more
([[results:statistics.json:observer_dependence.overall.benefit.value|+.2f]]). On real data PSNR
ranked [[results:real_liver.json:held_out.n_methods]] methods against `d'` at ρ
[[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.rho|+.2f]], the best-PSNR method ranking
[[results:real_liver.json:held_out.psnr_winner_task_rank]] of
[[results:real_liver.json:held_out.n_methods]] on the task, and
[[results:real_liver.json:all_exceedances]] arms exceeded the ceiling. The adversarial arms showed
no detected increase in lesion-shaped response. Unprocessed images met the criterion to
[[results:real_liver.json:guidance.crossing.unprocessed|.3f]] of routine exposure;
[[results:real_liver.json:guidance.processing_penalty.n_needing_more_dose]] of
[[results:real_liver.json:guidance.processing_penalty.n_processed]] processed arms needed more.

**Conclusions.** Separating the input's ceiling from achieved detectability allows exposure
requirements to be compared against a defined criterion. Fidelity gains alone did not establish
task preservation. Clinical use requires scanner-specific calibration and human-reader
validation.

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
detectability available in the input.

Task-based evaluation of CT denoising and deep-learning reconstruction is neither new nor
uniformly negative, and the scope of what follows is set by that. Greffier et al. characterised
two generations of a deep-learning reconstruction against noise power spectrum, task transfer
function and a detectability index across dose [11,12]; Fan et al. evaluated a deep-CNN
reconstruction with a channelised Hotelling observer on the ACR phantom [13]; and Tivnan et al.
optimised a tunable network against low-contrast lesion detectability directly, rather than
against fidelity [14]. These report conditions under which such methods raise detectability, not
merely fidelity. Toia et al. put the two kinds of observer side by side on one algorithm: dose
reductions of up to ninety per cent were judged non-inferior by twenty-four human readers,
and up to seventy per cent by a task-based model observer [15].

The question here is therefore not whether denoising can help a task — it can — but how far that
depends on which observer is asked and on how much detectability the input carried, and whether
a fidelity gain is evidence of either.

A benchmark also answers a different question: it ranks methods against each other, whereas an
analytic ceiling bounds what *any* processing of a given input can attain, so the question
becomes how much of the information already present survives. This study measures all three on
one controlled matrix, with an analytic reference that removes the usual ambiguity about whether
an apparent loss is real or an artefact of the estimator, and with repeated realisations so that
effect sizes come with intervals.

### 1.2 Prespecified hypotheses

We tested three prespecified hypotheses. First, gains in reference-based fidelity would not
reliably predict changes in task detectability. Second, the effect of denoising would depend
on observer efficiency, with larger benefits for the non-prewhitening observer than for the
prewhitening observer. Third, below a prespecified input-detectability criterion, visually
plausible outputs would remain possible despite failure to preserve the required task
performance.

That no processed arm exceeds its input's ceiling is *not* among them: that is a theorem, and
its role here is measurement-chain validation and leakage control (Sections 2.5 and 3.4).

Two parts of this paper are not prespecified. The adversarial arms (Sections 2.9 and 3.6.1) were
added after the first results, because the denoisers originally tested cannot fabricate and a null
from them alone would be uninformative; and the conversion of the floor into an exposure (Section
3.7) analyses data already collected. Both are reported as analyses, not as hypothesis tests.

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
   a case-disjoint held-out split, and a learned denoiser at two configurations spanning
   [[results:real_liver.json:capacity.parameter_ratio|.0f]]× in parameters — showing that the same divergence was also
   observed with synthetic lesions inserted into real low-dose CT backgrounds.
6. An open, deterministic implementation in which every reported number is regenerated from
   machine-readable outputs.

## 2. Methods

### 2.1 Detection task, observers, and the reference input

The task is signal-known-exactly / background-known-exactly (SKE/BKE) detection of a
low-contrast disk on a uniform background, the standard paradigm of objective, task-based
assessment [16,17]. Trials, phantoms, model observers and the ROC
machinery are reused from `taskiq-core` [18] (version
[[results:statistics.json:provenance.taskiq_core]]) rather than reimplemented; the trial
generator returns, with the image stacks, the analytic noise power spectrum (NPS) of the
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

1. The *held-out prewhitening linear observer* (PW): the class-mean difference prewhitened by
   the measured NPS. This is the efficient observer of the study, and the one every
   processed-image comparison uses.
2. A *channelised Hotelling observer* (CHO) on Laguerre–Gauss channels [19], an intermediate
   observer: tractable because a channel covariance can be estimated where a pixel covariance
   cannot.
3. A *non-prewhitening observer with a Burgess eye filter* (NPWE) [20], used as a
   stylized surrogate for limited noise-prewhitening efficiency. No human observer study
   was performed here, and NPWE is not offered as a validated model of a human reader.

Separately, and only for the unprocessed input, we compute the *analytic ideal linear
(prewhitening) observer* from the true signal and the analytic NPS. That quantity is exact for
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
left out, so every dose comparison in this paper is made at fixed kV. On the kV axis we
report kV itself and never a dose ratio. This is a normalised model, not a calibration: it
carries no measured dose-to-noise relation for any device and no absolute exposure values.

### 2.3 Denoisers

Three deterministic classical denoisers: a Gaussian filter of fixed width
[[results:dose_sweep.json:config.denoisers[label=gaussian(1.5 px)].params.sigma]] pixels, total
variation with weight
[[results:dose_sweep.json:config.denoisers[label=TV(0.4 sd)].params.weight]] × the noise
standard deviation, and non-local means with `h` =
[[results:dose_sweep.json:config.denoisers[label=NLM(0.6 sd)].params.h]] × the noise standard
deviation. A small residual convolutional network in the style of DnCNN [21] is evaluated
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

The matrix has four domains, described in Table 1. An *input condition* is one acquisition
and phantom setting; an *arm* is one condition evaluated with one processing, including the
unprocessed one; an *evaluation* is one arm in one realisation. Conditions are not shared
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

### 2.5 Cross-fitted estimation and the ceiling comparison

On the unprocessed input the ideal linear observer is closed-form. On processed images neither
the noise spectrum nor the effective signal is known analytically — a non-linear denoiser has no
transfer function — so both are estimated, and the estimate must not be allowed to score itself.

Estimation is *cross-fitted* over `K` =
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

`z` = `1.96`, `SE` the Hanley–McNeil standard error [22] of the scored AUC, and the second term
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
informational reasons. We report the distribution of that recovery ratio, and Section 3.8
compares cross-fitting with the single `50/50` split used previously.

### 2.6 Primary endpoints

For each processed arm, with the unprocessed arm of the same condition and realisation as its
reference:

`ΔSSIM = SSIM(processed) − SSIM(input)`, and `Δd'_O = d'_O(processed) − d'_O(input)` for each
observer `O` in {PW, CHO, NPWE}.

**Endpoint 1 — fidelity–task divergence.** The Spearman correlation between ΔSSIM and Δ`d'`(PW)
across evaluations, and the *divergence rate*: the fraction of evaluations with ΔSSIM > 0 and
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
lesion-like matched-filter rate above the gauge's allowance *and* above the same rate measured
on the unprocessed input of that condition — in signal-absent trials the object is uniform but
the acquired image is not, and noise alone produces such responses. Both definitions, and the
matched-filter normalisation behind them, are restated in Supplementary Methods.

### 2.7 Statistical analysis

Arms are not independent observations: all arms of a realisation share its noise stream, and the
arms of a condition share its images. Intervals therefore come from a cluster bootstrap that
resamples whole realisations with replacement ([[results:statistics.json:n_boot]] replicates,
seeded), carrying every arm of a drawn realisation along. Point estimates are the statistic on
the full data; intervals are percentile intervals; two-sided bootstrap p-values are reported
beside them and Holm-adjusted within each family (denoisers within an endpoint, stratum
contrasts within theirs). Proportions are additionally reported with a Wilson interval as the
conventional unclustered reference, which is the narrower of the two. Conclusions rest on effect
sizes and intervals rather than on p-values. Every statistic in this manuscript is computed by
the analysis module and written to `results/statistics.json`; none is typed.

### 2.8 The operational information floor

The floor is the contour where the input's analytic ceiling `d'` crosses a prespecified
requirement, here the Rose criterion [23] at
`d'` = [[results:dose_sweep.json:config.criteria.d_prime_threshold|.0f]]. It is a requirement a
task imposes, not a boundary of zero information, and it is used in this paper as a stratifying
variable.

The study also carries an auditable rule-based translation of the measured quantities into a
green / amber / red verdict with the rule that fired attached. Its thresholds are task-specific
author-set values, not clinically validated criteria; the complete rule set is given in
Supplementary Methods, and verdict counts per stratum are reported in Section 3.5.

### 2.9 The real low-dose CT arm

**Images.** Twelve Siemens liver cases from LDCT-and-Projection-data, using the vendor
reconstructions of both the routine-dose and the simulated quarter-dose acquisition. Nothing is
re-reconstructed. The quarter-dose noise field is taken as the difference between the two
reconstructions of the same anatomy, so the noise carried into every trial is the acquisition's
own and not a model of it.

**Lesion insertion.** A patient scan has no ground truth: the lesions in it were found by a
reader, at a contrast nobody measured. A synthetic lesion of known size and amplitude is
therefore inserted into real abdominal soft tissue, which keeps the background that makes the task
hard and supplies the truth that makes it measurable. The lesion is a disk of
[[results:real_liver.json:spec.lesion.diameter_mm|.0f]] mm diameter and [[results:real_liver.json:spec.lesion.contrast_hu|.0f]] HU
contrast, with a Gaussian edge of [[results:real_liver.json:spec.lesion.edge_sigma_mm|.1f]] mm, rendered on a grid
supersampled [[results:real_liver.json:spec.lesion.supersample]]-fold so its own edge is not a one-pixel staircase,
and added *after* reconstruction. The one assumption this makes is stated rather than left
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
the same sites from the same background, so the two members of a pair differ in two respects and
in no others: the lesion, which is present in one and absent in the other, and the noise, which is
an independent measured realisation drawn for each member from a site other than the one supplying
the background. The anatomy is common to the pair and cancels in their difference; the noise does
not, and that is what the detectability expression below is built on.

That rule admits more than parenchyma: rendering the sites shows bowel, mesentery, vessel and
body-wall structure alongside liver. Measured as the standard deviation of a site after a blur
that removes the quantum noise, the structure inside an admitted site has a median of
[[results:real_liver.json:site_sensitivity.structure_sd_median|.1f]] HU and exceeds the lesion's own
[[results:real_liver.json:site_sensitivity.lesion_depth_hu|.0f]] HU depth in
[[results:real_liver.json:site_sensitivity.structure_over_lesion_fraction|.0%]] of them. This does
not bias `d'`, since the background is common to the pair and the noise spectrum comes from paired
differences, but the denoisers are non-linear, so a ranking measured on structured tissue need not
hold on parenchyma. Section 3.6.2 tightens the rule and reports what changes.

**The observer, on anatomical background.** A prewhitening matched filter needs the noise
power spectrum of the images it scores. On a uniform phantom that spectrum is the noise; on
abdominal anatomy an image-by-image estimate would be dominated by the anatomy, which is not noise and
does not repeat between trials. The paired construction of the preceding paragraph is what makes
the estimate possible: because the signal-present and signal-absent members of a pair are built at
the same site from the same background, their difference removes the anatomy and leaves the noise.
The noise power spectrum is therefore estimated as the two-dimensional spectrum of those paired
differences after their mean is removed, and the anatomy enters neither the template nor the
covariance.

The template is formed on training folds and applied to held-out ones. Trials are assigned to
[[results:real_liver.json:spec.observer.folds]] folds by [[results:real_liver.json:spec.observer.fold_assignment]]. Within each training
fold the signal is estimated as the difference of the present and absent means rather than assumed
known, the noise power spectrum is estimated from the paired differences as above, and the
template is the inverse transform of the estimated signal spectrum divided by that noise spectrum.
The division is regularised in three steps, because an estimated spectrum has a zeroed DC bin and
small high-frequency values that would otherwise dominate the template: the DC bin is replaced by
the mean of its four neighbours, a ridge of [[results:real_liver.json:spec.observer.nps_ridge_x_mean]] of the spectrum's
mean is added, and a floor of [[results:real_liver.json:spec.observer.nps_floor_x_max]] of its maximum is applied. Each
held-out trial is then scored by the inner product of its image with the template. A template that
had seen the trial it scores would report the fit rather than the detectability, which is why the
folds are held out.

**Detectability.** The two members of a pair share a background, so the per-trial difference of
their scores removes it and carries two independent noise draws. Its variance is twice that of a
single image, and the detectability of one image is therefore

> *d*′ = √2 · mean(Δ) / sd(Δ),

with Δ the per-trial difference of the present and absent scores. Omitting the √2 understates *d*′ by
[[results:real_liver.json:spec.observer.understatement_without_sqrt_two_percent]] per cent, which is larger than the differences between denoisers this arm is measuring. The
same quantity is computed for the unprocessed quarter-dose images and for every denoised arm, from
the same trials, so the comparison across arms is paired throughout.

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
[[results:real_liver.json:capacity.large.patches_per_case]] patches per case. The two therefore differ in three
respects at once — parameters, training data and training length — and are not a controlled
comparison of capacity.

The validation loss quoted below is measured on a [[results:real_liver.json:spec.cnn.validation_fraction|.0%]] split of the patches drawn from the
[[results:real_liver.json:held_out.n_train_cases]] training cases, held out from the gradient. The held-out cases take no part in
training or in model selection: they are read once, after training has finished. This is the
same discipline Section 2.5 applies to the observers, for the same reason — a quantity that
selects a model cannot also be evidence about it.

**What the network is and is not shown.** Training pairs are normalised exactly as inference
normalises them, by each image's own mean and estimated noise level; a network trained in
absolute HU and deployed through a normalising wrapper is not the network that was trained. The
split is by case: the network sees [[results:real_liver.json:held_out.n_train_cases]] cases and is evaluated on the
[[results:real_liver.json:held_out.n_test_cases]] it never saw, because slices from one patient are not independent
and a network tested on another slice of a liver it trained on is being tested on its own
training set. The network never sees a lesion — its targets are routine-dose reconstructions of
ordinary anatomy, which is what a denoiser is actually given — so the lesion exists only in the
evaluation.

**The adversarial arm.** Every denoiser above regresses towards the mean, so none of them can
fabricate and a null result on fabrication from them alone would be uninformative. At each
capacity a second network is therefore trained that differs in the objective and in nothing
else: the same architecture built from the same seed, the same patches, the same epochs, the
same batch size and the same learning rate, with the generator minimising
`mse_weight × MSE + adv_weight × L_adv`, where `L_adv` is the least-squares adversarial term of
Mao et al. [24] against a small patch critic. The three weights
(`mse_weight` = [[results:real_liver.json:objective.small.adversarial_config.mse_weight|.2f]],
`adv_weight` = [[results:real_liver.json:objective.small.adversarial_config.adv_weight|.1f]], critic
learning rate [[results:real_liver.json:objective.small.adversarial_config.discriminator_lr|.0e]]) were fixed
by a recorded sweep on a subset before this arm was run, and are reported with that sweep rather
than chosen from the result. The epoch count is not raised to the sweep's: holding it at the
fidelity arm's value is what keeps the comparison controlled.

Least squares rather than the log loss, because the instabilities of the log loss are a property
of that loss rather than of the question, and bounded gradients keep a seeded run comparable
with a rerun. What is claimed is a seeded run with a recorded weight hash, not bitwise equality
across machines.

The bound's premise is preserved rather than assumed. At inference the generator is a
deterministic function of its input image and of nothing else — it takes no latent, no class
label, no lesion location and no other trial's pixels — so `H` → `X` → `Y` holds and the
data-processing inequality bounds it exactly as it bounds a Gaussian filter. A generator that
sampled from a latent would not be covered by that argument, which is why this one does not; a
test scores the same image twice and requires the same output.

**Fabrication, with the anatomy cancelled.** The prespecified criterion of Section 2.8 counts
lesion-shaped matched-filter responses in signal-absent images. On a uniform phantom every such
response was put there by the noise or by the processing. On real anatomy it is not: abdominal
soft tissue carries disc-like structure at around twice this lesion's contrast, so the rate is already near
one before anything is processed and cannot rise. The threshold is not retuned. Instead the
measurement is repeated on what the processing *added*, which this design can isolate because it
knows the background exactly: the lesion-free, noise-free anatomy is passed through the same
method and subtracted, so the anatomy cancels and every remaining response is one the processing
produced. Structure a method invents from the anatomy alone is identical in both classes and
cancels too; it carries no information about the hypothesis and so cannot breach the ceiling,
which is why this is the right measure for the claim and why it is not a complete account of
what a denoiser might put on a screen. The measure's sensitivity is established by injection
rather than assumed, against a denoiser built to stamp a lesion where the noise suggested one.

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
paired with the unprocessed arm of its own condition and realisation. Figures 2 and 3 give the
dose response of the representative realisation, in fidelity and in task `d'` respectively.

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
consistent with differences in observer efficiency, with the intermediate observer in between.

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

Stated simultaneously over the whole family, with the armwise margin widened by Bonferroni to a
family-wise level of `0.05`,
[[results:statistics.json:ceiling.family_wise.n_exceedances_processed_arms]] of the
[[results:statistics.json:ceiling.family_wise.n_processed_comparisons]] processed-arm comparisons
exceeded the ceiling margin, and the same holds at the unadjusted margin. One exceedance occurred,
on an *unprocessed* arm, where the processing is the identity and cannot create information: it is
a degeneracy of the closed-form interval at perfect separation, where the Hanley–McNeil standard
error is identically zero and the margin collapses to one quantisation step. Excluding the
[[results:statistics.json:ceiling.n_saturated]] AUC-saturated evaluations, the remaining
[[results:statistics.json:ceiling.unsaturated.n_evaluations]] gave a largest excess of
[[results:statistics.json:ceiling.unsaturated.max_excess|sci2]] and
[[results:statistics.json:ceiling.unsaturated.n_violations]] exceedances, so nothing rests on the
saturated regime. Supplementary Methods give the margin construction, the saturation criterion and
the exceeding arm in full.

This validates the measurement chain rather than finding anything: the observer plotted is an
estimated linear one, which need not sit on the identity line after a non-linear transformation,
so points below it are not evidence of information loss. The unprocessed arms fall slightly below
it too, and that offset is the estimator's own cost, quantified in Section 3.8.

### 3.5 Failure patterns across the operational information floor (Figures 7–8)

Figure 7 is the kV–mAs atlas with the operational floor drawn as the contour where the input's
analytic ceiling crosses the requirement; Figure 8 and Supplementary Table S3 give the
stratified comparison.

The third hypothesis was supported in one respect and refuted in another.

**Supported: plausible output without preserved detectability.** Below the floor, processing
raised SSIM by [[results:statistics.json:floor_strata["below floor"].mean_delta_ssim.value|+.3f]]
on average while Δ`d'`(PW) was
[[results:statistics.json:divergence.by_stratum["below floor"].mean_delta_d_pw.value|+.3f]], and
[[results:statistics.json:floor_strata["below floor"].divergence_rate.value|.1%]]
([[results:statistics.json:floor_strata["below floor"].divergence_rate.ci_low|.1%]] to
[[results:statistics.json:floor_strata["below floor"].divergence_rate.ci_high|.1%]]) of those
evaluations were divergent: visually improved output coexisted, throughout that stratum, with an
input that did not meet the requirement and a processed estimate that fell further below it.

**Refuted: failures do not concentrate below the floor.** The divergent fraction was *higher*
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
The same ordering appears in the observer-dependent benefit, which was
[[results:statistics.json:floor_strata["above floor"].benefit.value|+.2f]] above the floor
against [[results:statistics.json:floor_strata["below floor"].benefit.value|+.2f]] below it. We
offer no mechanism for the ordering: the stratum with more detectability available also shows the
larger relative loss, but these data do not establish why.

Contrast erasure did not differ appreciably between strata
([[results:statistics.json:floor_strata["above floor"].erasure_rate.value|.1%]] above against
[[results:statistics.json:floor_strata["below floor"].erasure_rate.value|.1%]] below, Holm-adjusted
p = [[results:statistics.json:stratum_contrasts.p_holm["erasure_rate: below floor - above floor"]|.2f]]),
and excess lesion-like responses were absent in every stratum. With these denoisers, at this
response threshold, the failure mode is erasure and dilution of true contrast rather than
fabrication of false structure — a result that should not be generalised to generative or learned
methods, whose failure modes may differ. Supplementary Table S3 gives the stratified counts, the
interval estimates and the gauge's verdict breakdown, which reflects the strata by construction:
an input that fails the requirement is itself a red rule, so below the floor the informative part
is the reason attached rather than the colour.

### 3.6 The same question on real low-dose CT

The controlled matrix isolates mechanisms by fixing everything. The obvious question about any
conclusion drawn from it is whether the conclusion survives an acquisition nobody controlled, so
the same measurement was repeated on real data with no change to the framework.

**Data and design.** Twelve Siemens liver cases from LDCT-and-Projection-data (The Cancer Imaging
Archive, `CC BY 4.0`), vendor reconstructions of both the routine and the simulated quarter-dose
acquisition. A lesion of known size and contrast is inserted into real abdominal soft tissue, so
that the task has a ground truth the acquisition itself cannot supply. The learned denoiser is trained on
quarter-dose / full-dose patch pairs from [[results:real_liver.json:held_out.n_train_cases]] cases and evaluated on
the [[results:real_liver.json:held_out.n_test_cases]] it never saw ([[results:real_liver.json:held_out.n_pairs]] pairs). The split is by
case, not by slice: slices from one patient are not independent, and a network tested on
another slice of a liver it trained on is being tested on its own training set. Normalisation at
training matches normalisation at inference, since a network trained in absolute HU and deployed
through a normalising wrapper is not the network that was trained. The network never sees a
lesion — its targets are full-dose reconstructions of ordinary anatomy, which is what a
denoiser is actually given — and the lesion exists only in the evaluation.

**Result.** Figure 9 draws one held-out site, lesion present and absent, unprocessed and after
four denoisers: the panels grow cleaner from left to right as the detectability above them falls.
Figure 10 and Table 3 give the held-out comparison against the closed-form ceiling
`d'` = [[results:real_liver.json:held_out.ceiling|.2f]]:

**Table 3.** The [[results:real_liver.json:held_out.n_methods]] arms on the held-out real low-dose CT split, ordered by task detectability. `d'` is against the closed-form ceiling of the unprocessed input; PSNR is against the full-dose reconstruction. *Added* is the rate of lesion-shaped structure the method put into a lesion-free image with the anatomy cancelled, and is read against the unprocessed row, which is the rate the noise itself carries; *recovery* is the fraction of a real lesion's contrast that survives. The four networks share an architecture and a training split; within each capacity the two differ only in the objective, so that pair is controlled. Across capacities they are not: the larger differs in parameters, training patches per case and epochs at once. The exposure each arm needs to meet the prespecified requirement is given in Section 3.7.

| method | `d'` | of ceiling | PSNR (dB) | added | recovery |
|---|---|---|---|---|---|
| `tv 1x noise` | [[results:real_liver.json:held_out.by_method.tv.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.tv.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.tv.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.tv.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.tv.contrast_recovery|.3f]] |
| unprocessed | [[results:real_liver.json:held_out.by_method.unprocessed.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.unprocessed.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.unprocessed.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.unprocessed.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.unprocessed.contrast_recovery|.3f]] |
| GAN, [[results:real_liver.json:capacity.small.parameters]] parameters | [[results:real_liver.json:held_out.by_method.gan_small.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gan_small.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gan_small.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.gan_small.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.gan_small.contrast_recovery|.3f]] |
| `nlm 0.8x noise` | [[results:real_liver.json:held_out.by_method.nlm.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.nlm.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.nlm.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.nlm.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.nlm.contrast_recovery|.3f]] |
| CNN, [[results:real_liver.json:capacity.small.parameters]] parameters | [[results:real_liver.json:held_out.by_method.cnn_small.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_small.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_small.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_small.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.cnn_small.contrast_recovery|.3f]] |
| `gaussian 0.75 mm` | [[results:real_liver.json:held_out.by_method.gauss075.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gauss075.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gauss075.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.gauss075.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.gauss075.contrast_recovery|.3f]] |
| CNN, [[results:real_liver.json:capacity.large.parameters]] parameters | [[results:real_liver.json:held_out.by_method.cnn_large.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_large.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_large.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.cnn_large.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.cnn_large.contrast_recovery|.3f]] |
| `gaussian 1.00 mm` | [[results:real_liver.json:held_out.by_method.gauss100.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gauss100.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gauss100.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.gauss100.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.gauss100.contrast_recovery|.3f]] |
| GAN, [[results:real_liver.json:capacity.large.parameters]] parameters | [[results:real_liver.json:held_out.by_method.gan_large.d_prime|.2f]] | [[results:real_liver.json:held_out.by_method.gan_large.ratio|.2f]] | [[results:real_liver.json:held_out.by_method.gan_large.psnr|.2f]] | [[results:real_liver.json:held_out.by_method.gan_large.added_structure_rate|.3f]] | [[results:real_liver.json:held_out.by_method.gan_large.contrast_recovery|.3f]] |

Three things follow.

*Fidelity carries no usable information about the ranking.* Across the
[[results:real_liver.json:held_out.n_methods]] arms the rank correlation between PSNR and `d'` is Spearman
ρ = [[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.rho|+.2f]]
(`p` = [[results:real_liver.json:held_out.spearman_psnr_vs_d_prime.p_value|.2f]]) — not distinguishable from
zero, and with [[results:real_liver.json:held_out.n_methods]] arms this study is not powered to
distinguish a weak association from none. The claim here is therefore the absence of a usable
one, not the presence of an inverse one. What is not in doubt is the consequence: the arm with
the best PSNR of all [[results:real_liver.json:held_out.n_methods]] ranks
[[results:real_liver.json:held_out.psnr_winner_task_rank]] of
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
[[results:real_liver.json:capacity.small.d_prime|.2f]]: the configuration that scored better on the
objective it was trained against scored worse on the task. Capacity was not varied alone, since
parameters, training patches and epochs moved together, so what this shows is that the divergence
persisted when the network was made substantially larger and trained longer within this
architecture. That weakens the objection that the network was too small to be representative
without excluding it, and says nothing about architectures not tried here. Supplementary Methods
give the training curves, including the large network's validation minimum well before its last
epoch, and the saturation of mean-squared error against a target that carries noise of its own.

Repeating the measurement at a second and very different acquisition — chest at
[[results:real_liver.json:operating_points.chest.dose|.0%]] dose, with
[[results:real_liver.json:operating_points.noise_ratio_chest_over_liver|.1f]] times the liver's noise and
a ceiling of [[results:real_liver.json:operating_points.chest.ceiling|.2f]] against
[[results:real_liver.json:operating_points.liver.ceiling|.2f]] — produced
[[results:real_liver.json:operating_points.chest.n_exceeding_ceiling]] exceedances of its own ceiling,
so the bound holds where the task is nearly impossible as well as where it is comfortable.

#### 3.6.1 A denoiser that could fabricate, and did not

Every arm above regresses towards the mean: a Gaussian filter, total variation, non-local means
and a network trained on mean-squared error remove contrast rather than inventing it, so the
measurement that would catch fabrication has never been aimed at anything capable of it, and a
null result from them alone would be uninformative.

The adversarial arms answer that objection. At each capacity a second network was trained that
differs from the first in the objective and in nothing else — same architecture, patches, epochs,
batch size, learning rate and seed — with a least-squares adversarial term [24] added, so it is
rewarded for producing images a critic cannot tell from full-dose ones rather than for being
close to them pixel by pixel. Its weight was fixed by a recorded sweep before the arm was run.
The bound's premise survives: at inference the generator is a deterministic function of its input
image and of nothing else — no latent, no class label, no lesion location — so `H` → `X` → `Y`
holds and the data-processing inequality bounds it as it bounds a Gaussian filter.

The measure did not detect fabrication. Counting lesion-shaped structure that processing put into a lesion-free
image, with the anatomy cancelled by passing the lesion-free, noise-free background through the
same method and subtracting,
[[results:real_liver.json:fabrication.n_above_the_noise]] of
[[results:real_liver.json:fabrication.n_processed]] processed arms exceeded the rate the noise itself
carries ([[results:real_liver.json:fabrication.noise_itself|.3f]]); the largest of any processed arm was
[[results:real_liver.json:fabrication.max_processed|.3f]], and the adversarial arm at
[[results:real_liver.json:capacity.large.parameters]] parameters reached
[[results:real_liver.json:fabrication.by_method.gan_large|.3f]]. What that arm did instead was erase: its
contrast recovery, [[results:real_liver.json:fabrication.lowest_contrast_recovery.contrast_recovery|.3f]], is
the lowest of every method in the study, and its `d'` is
[[results:real_liver.json:objective.large.adversarial.d_prime|.2f]] against
[[results:real_liver.json:objective.large.mse.d_prime|.2f]] for the same network trained on fidelity. The
detector's sensitivity is not assumed: aimed at a denoiser constructed to stamp a lesion where
the noise suggested one, it fires on the majority of images while reading below
[[results:real_liver.json:fabrication.max_processed|.2f]] for an honest smoother.

Within the operating points examined, and within the reach of a measure blind to structure
invented from the anatomy alone, optimising a denoiser for appearance produced no detectable
excess of lesion-shaped response. What it produced was more erasure of a real lesion, behind a
more convincing picture.

#### 3.6.2 Does the ranking depend on where the lesions were put?

The admission rule of Section 2.9 passes structured tissue as well as parenchyma, so the held-out
evaluation was repeated with one criterion added — the structure inside a site capped at
[[results:real_liver.json:site_sensitivity.structure_threshold_hu|.0f]] HU, below half the lesion's
depth — and everything else fixed, including the networks, which were not retrained, and the pairs
contributed by each case. Matching those matters: the strict rule admits far fewer sites in the
quietest case, so comparing the rules on their natural counts would compare a change of case mix
as much as a change of site.

The structure inside a site falls from a median of
[[results:real_liver.json:site_sensitivity.structure_sd_median|.1f]] HU to
[[results:real_liver.json:site_sensitivity.structure_sd_median_strict|.1f]] HU against a nearly
unchanged ceiling ([[results:real_liver.json:site_sensitivity.ceiling|.2f]] against
[[results:real_liver.json:site_sensitivity.ceiling_strict|.2f]]), so the two sets carry comparable
noise. The ranking by `d'` agrees at Spearman ρ =
[[results:real_liver.json:site_sensitivity.spearman_rho|.2f]]
(`p` = [[results:real_liver.json:site_sensitivity.spearman_p|.4f]]), with
[[results:real_liver.json:site_sensitivity.n_positions_changed]] of
[[results:real_liver.json:site_sensitivity.n_methods]] positions changed, and those two adjacent.
The three best and the three worst arms are the same under both rules, the best-PSNR arm ranks
fifth of [[results:real_liver.json:site_sensitivity.n_methods]] on the task under both, and under
both the best-performing arm exceeds the unprocessed input by a margin this design does not
resolve. The conclusions of Sections 3.6 and 3.7 do not depend on the admission rule.

One thing does change, and no mechanism is offered for it: every arm's efficiency falls on the
low-structure sites, the unprocessed input from
[[results:real_liver.json:site_sensitivity.by_method.unprocessed.published|.2f]] to
[[results:real_liver.json:site_sensitivity.by_method.unprocessed.strict|.2f]]. A less structured
background yields a *lower* fraction of the available detectability, which these data record
without explaining.

### 3.7 The floor, converted into an exposure

Sections 3.1–3.6 measure. None of them tells a reader what to set. This section converts the
operational floor of Section 2.8 into the quantity the decision is actually made in, using the
same held-out pairs and the same prespecified requirement, `d'` =
[[results:real_liver.json:guidance.requirement|.0f]].

The dose axis is a simulation anchored on a measured noise realisation, and the distinction
matters. Because this collection's low-dose series is a reconstruction of the same projections
with noise inserted, the difference of the two reconstructions is a measured realisation of the
noise that dose reduction costs at one fraction, α, with a real spectrum on real anatomy.
Scaling it by `k`(β) = √((1/β − 1)/(1/α − 1)) produces a realisation at any other fraction β.
The spectrum and the anatomy are therefore measured; the amplitude at β is modelled. The scaling
assumes the inserted noise is independent of the noise already present and that its power follows
`1/β − 1`, and no independent acquisition at β exists to check it, so only the row at β = α rests
on an observed amplitude. The trials are built once at α and recombined linearly, which is exact
given the scaling.

**Where the task is lost.** The unprocessed input meets the requirement down to
[[results:real_liver.json:guidance.crossing.unprocessed|.3f]] of the routine protocol. Of the
[[results:real_liver.json:guidance.processing_penalty.n_processed]] processed arms,
[[results:real_liver.json:guidance.processing_penalty.n_needing_more_dose]] need *more* exposure than that
to meet the same requirement — up to
[[results:real_liver.json:guidance.processing_penalty.worst_percent|+.0f]] % for
`[[results:real_liver.json:guidance.crossing.worst_processed_label]]` — and the best of them differs from
doing nothing by [[results:real_liver.json:guidance.processing_penalty.best_percent|+.0f]] %, which is
within what this design resolves. No arm measured here extended the exposure floor by an amount
this study can distinguish from none.

**What the picture claims.** A gain of Δ dB in PSNR is the fidelity of an exposure with
`10^(Δ/10)` times less noise power, so every processed image implies an exposure. At the nominal
fraction the implied exposure reaches
[[results:real_liver.json:guidance.overstatement.at_nominal_max|.2f]]× the one actually delivered
(`[[results:real_liver.json:guidance.overstatement.at_nominal_max_label]]`), and the discrepancy grows as
the exposure falls: at [[results:real_liver.json:guidance.overstatement.worst_below_crossing_dose|.3f]] of
routine, `[[results:real_liver.json:guidance.overstatement.worst_below_crossing_label]]` has the fidelity of
[[results:real_liver.json:guidance.overstatement.worst_below_crossing_apparent|.3f]] of routine —
[[results:real_liver.json:guidance.overstatement.worst_below_crossing|.1f]]× its own exposure — while
delivering `d'` = [[results:real_liver.json:guidance.overstatement.worst_below_crossing_d_prime|.2f]] against
the [[results:real_liver.json:guidance.requirement|.0f]] required. The flattery is greatest where the task
has already failed.

**What an ideal-observer floor omits.** A floor computed from an ideal observer is not reachable.
The achieved fraction of the ceiling on the unprocessed input at the nominal exposure is
[[results:real_liver.json:guidance.efficiency.unprocessed_at_nominal|.3f]], and exposure enters
and on this axis the ceiling crosses the requirement at
[[results:real_liver.json:guidance.efficiency.ideal_observer_floor|.3f]] of the routine protocol while the
unprocessed input crosses it at
[[results:real_liver.json:guidance.efficiency.reachable_floor_measured|.3f]] — a factor
[[results:real_liver.json:guidance.efficiency.floor_ratio|.2f]], or
[[results:real_liver.json:guidance.efficiency.floor_ratio_percent|.0f]] % more exposure than an
ideal-observer calculation would ask for. With the requirement as the denominator rather than the
ideal floor, the same fact reads: an ideal-observer calculation supplies
[[results:real_liver.json:guidance.efficiency.ideal_floor_fraction_of_required|.0%]] of the exposure the
task needs. The two percentages describe one fact and are not interchangeable.

Holding the efficiency constant and solving for the crossing gives
[[results:real_liver.json:guidance.efficiency.reachable_floor_constant_efficiency|.3f]], within
[[results:real_liver.json:guidance.efficiency.constant_efficiency_agreement_percent|.1f]] % of the
measured one, which is the evidence for the conversion. It is local, because efficiency varies
along the dose axis, and it is specific to this axis: a site working in the total noise of its own
images uses `d'` ∝ √`D`, under which the same efficiency implies
[[results:real_liver.json:guidance.efficiency.sqrt_law_floor_factor|.2f]] instead. The convention
has to travel with the number.

**How far the scaling law may be carried.** `d'` ∝ 1/`k`(β) holds exactly for the ceiling, whose
template is the noise's own: over the [[results:real_liver.json:guidance.scaling_law.dose_range_fold|.0f]]-fold
range measured, `d'`·`k` varies by [[results:real_liver.json:guidance.scaling_law.ceiling_spread|.1%]]. It holds
for no achievable observer: the same quantity varies by up to
[[results:real_liver.json:guidance.scaling_law.worst_observer_spread|.0%]]
(`[[results:real_liver.json:guidance.scaling_law.worst_observer_label]]`), because observer efficiency is
itself a function of exposure. The textbook `d'` ∝ √dose rule therefore *overstates* what a
denoised image keeps when the exposure is cut, and an efficiency measured at one exposure
extrapolates only near it. The crossings above are read from the measured curve for that reason.

The conversions are implemented in `denoiq_core.guidance` for a reader's own protocol and task.
The absolute exposures here are specific to this lesion and a background treated as known, and
are not a clinical recommendation.

### 3.8 Sensitivity and implementation validation

**Estimator efficiency.** On unprocessed images the cross-fitted estimator recovered a median
[[results:statistics.json:ceiling.estimator_recovery.median|.0%]] of the analytic ceiling `d'`
(interquartile range [[results:statistics.json:ceiling.estimator_recovery.q1|.0%]] to
[[results:statistics.json:ceiling.estimator_recovery.q3|.0%]]), with a minimum of
[[results:statistics.json:ceiling.estimator_recovery.min|.0%]] over all evaluations and
[[results:statistics.json:ceiling.estimator_recovery.above_floor_min|.0%]] over conditions at or
above the floor. The lowest recoveries occur at the lowest doses, where the signal the template
is estimated from is weakest, and the estimator rather than the bound is the limiting factor.
Supplementary Table S2 compares the cross-fitted estimator with a single `50/50` split at matched
conditions; cross-fitting recovers more of the ceiling at every dose.

**Closed form.** In white noise the analytic observer must satisfy `d' = ‖s‖₂/σ`. Over the
combinations of noise level and lesion radius in Supplementary Table S1 the maximum relative
deviation was [[results:closed_form.json:max_relative_error|sci2]], against a criterion of 1 %
fixed in advance.

**Leakage positive control.** A processor handed the class label — adding signal to the
signal-present stack only — is detected by the ceiling comparison as an exceedance beyond the
margin. That control is asserted in the test suite, and it is what gives the negative result of
Section 3.4 its content.

## 4. Discussion

The contribution of this work is a framework that separates two quantities usually reported as
one: the detectability an input makes available to an ideal observer, and the detectability a
specified observer actually achieves after processing. Holding them apart is what turns a
statement about image quality into a statement about an exposure, because the requirement a task
imposes can then be placed on the second quantity rather than the first, and the exposure that
meets it compared across processing methods.

The ceiling itself is a theorem, so the agreement reported in Section 3.4 establishes not the
theorem but that the measurement chain is consistent with it, which is a precondition for
trusting anything else the chain reports; the leakage control shows the comparison has the
sensitivity to see a violation.

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

The third prespecified hypothesis was partially refuted, and the part that failed is as
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

The clinical relevance of this framework lies in linking image processing to a specified
diagnostic task and performance requirement. For radiologists, it motivates assessing whether
improved image appearance preserves the features needed for the intended decision. For
radiographers and medical physicists, it provides a basis for jointly evaluating acquisition
exposure and post-processing against task-specific performance criteria. The intended patient
benefit is to avoid unnecessary exposure while retaining information needed for diagnosis. These
are translational objectives, not outcomes demonstrated here: the present study evaluated model
observers and synthetic lesions, including lesions inserted into real CT backgrounds. Clinical
application requires scanner-specific calibration, representative diagnostic tasks, and
validation with human readers.

The gauge is an auditable rule-based translation of the measured quantities, not a clinically
validated instrument. Its value is that a verdict naming the rule that fired, on numbers that are
written down, can be argued with; an impression that the images look acceptable cannot.

### 4.1 Limitations

The phantom is a disk on a uniform background with stationary Gaussian noise and the signal is
known exactly; non-stationary noise from iterative reconstruction and search tasks are outside the
design. The observer evaluated on processed data is a cross-fitted prewhitening linear observer,
not the likelihood-ratio ideal observer of the processed data. The Rose
criterion `d'` = [[results:dose_sweep.json:config.criteria.d_prime_threshold|.0f]] is one chosen
operational criterion, not a universal information boundary, and every gauge threshold is
task-specific: clinical deployment would require re-specifying and validating all of them. No
human observer study was performed, so the non-prewhitening observer stands only as a
stylized surrogate for limited prewhitening efficiency. The lesion is inserted, not native: what
is measured is detection of a specified low-contrast signal, not of hepatocellular carcinoma,
haemangioma or any other native lesion, whose contrast, margin and size distributions differ.
Whether the ordering reported here transfers to those is a question for readers and for clinical
validation, not one this measurement answers. The classical denoisers are parameterized with the
true noise level of the acquisition setting, which is more than a blind method would know. The
learned denoisers do not represent the state of the art. The acquisition model is relative and
analytic: it locates conditions on an axis, not on a scanner. Ten realisations bound sampling
variability but do not make the matrix exhaustive. Finally, the scope is post-processing of a
defined input image; reconstruction from projection data is a separate map, subject to its own
bound, and is not studied here.

## 5. Conclusion

This study characterizes task detectability limits in denoised CT by separating the input's
ideal-observer ceiling from the performance achieved by specified observers. Controlled
experiments and synthetic-lesion evaluations on real CT backgrounds showed observer-dependent
denoising effects and divergence between fidelity and detectability. Within the evaluated dose
simulation, task thresholds provided a basis for comparing exposure requirements across
processing methods. These findings support evaluating acquisition and post-processing jointly
against a defined detection requirement. Translation to clinical dose optimization requires
scanner-specific calibration and human-reader validation, with the ultimate aim of preserving
diagnostic information while avoiding unnecessary patient exposure.

## Disclosures

The author declares no financial or commercial conflicts of interest relevant to this work. The
controlled arm uses no measured data of any kind. The real-data arm uses de-identified public
images from LDCT-and-Projection-data, distributed by The Cancer Imaging Archive under `CC BY 4.0`;
no data were collected for this study, no proprietary software was used, and the work required no
additional ethical approval.

**Use of generative AI.** Generative AI (Claude, Anthropic, through the Claude Code command-line tool) was used as
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
accountable for the content. No AI system is an author. This disclosure follows ICMJE and COPE guidance.

## Code and Data Availability

The controlled arm is generated analytically from documented seeds. The real-data arm uses the
public LDCT-and-Projection-data collection (TCIA, `CC BY 4.0`, DOI `10.7937/9npb-2637`).

**Two packages are needed, and neither reproduces the paper alone.** The controlled arm, the
endpoints, the statistics and the figures are `denoiq-core`, below. The code that reads the
images, inserts the lesion, trains the four networks, realises the observer of Section 2.9 and
scores the result is `ldct-io`, at <https://github.com/Institute-of-One/ldct-io>, MIT licensed,
[[ldct_io:archive_statement]]. Its entry points are
`examples/liver_cnn.py` for the held-out comparison, `examples/dose_decision.py` for Section 3.7
and `examples/site_sensitivity.py` for Section 3.6.2; `cross_fitted_paired` is the observer and
`PRESETS` the four network configurations. Its per-run outputs are committed here under
`paper/results/` and consolidated into `results/real_liver.json`. Every number reported here is
produced by the code rather than transcribed. The software is `denoiq-core`
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
11. J. Greffier, D. Dabli, J. Frandon, et al., "Comparison of two versions of a deep learning image reconstruction algorithm on CT image quality and dose reduction: a phantom study," *Med. Phys.* **48**(10), 5743–5755 (2021) [doi:10.1002/mp.15180].
12. J. Greffier, S. Si-Mohamed, J. Frandon, et al., "Impact of an artificial intelligence deep-learning reconstruction algorithm for CT on image quality and potential dose reduction: a phantom study," *Med. Phys.* **49**(8), 5052–5063 (2022) [doi:10.1002/mp.15807].
13. M. Fan, Z. Zhou, T. Vrieze, et al., "Efficient evaluation of low-contrast detectability of deep-CNN-based CT reconstruction using channelized Hotelling observer on the ACR accreditation phantom," in *Medical Imaging 2022: Physics of Medical Imaging*, Proc. SPIE (2022) [doi:10.1117/12.2612414].
14. M. Tivnan, T. Lee, R. Zhang, et al., "Task-driven CT image quality optimization for low-contrast lesion detectability with tunable neural networks," in *Medical Imaging 2023: Physics of Medical Imaging*, Proc. SPIE (2023) [doi:10.1117/12.2653936].
15. G. Toia, D. Zamora, M. Singleton, et al., "Detectability of small low-attenuation lesions with deep learning CT image reconstruction: a 24-reader phantom study," *AJR Am. J. Roentgenol.* **220**(2), 283–295 (2023) [doi:10.2214/AJR.22.28407].
16. H. H. Barrett and K. J. Myers, *Foundations of Image Science*, Wiley, Hoboken, New Jersey (2004).
17. H. H. Barrett, "Objective assessment of image quality: effects of quantum noise and object variability," *J. Opt. Soc. Am. A* **7**(7), 1266–1278 (1990) [doi:10.1364/JOSAA.7.001266].
18. S. Yamamoto, "taskiq-core: task-based image quality on synthetic phantoms," Zenodo (2026) [doi:10.5281/zenodo.21422924].
19. K. J. Myers and H. H. Barrett, "Addition of a channel mechanism to the ideal-observer model," *J. Opt. Soc. Am. A* **4**(12), 2447–2457 (1987) [doi:10.1364/JOSAA.4.002447].
20. A. E. Burgess, "Visual signal detection with two-component noise: low-pass spectrum effects," *J. Opt. Soc. Am. A* **16**(3), 694–704 (1999) [doi:10.1364/JOSAA.16.000694].
21. K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising," *IEEE Trans. Image Process.* **26**(7), 3142–3155 (2017) [doi:10.1109/TIP.2017.2662206].
22. J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology* **143**(1), 29–36 (1982) [doi:10.1148/radiology.143.1.7063747].
23. A. Rose, "The sensitivity performance of the human eye on an absolute scale," *J. Opt. Soc. Am.* **38**(2), 196–208 (1948) [doi:10.1364/JOSA.38.000196].
24. X. Mao, Q. Li, H. Xie, et al., "Least squares generative adversarial networks," in *2017 IEEE International Conference on Computer Vision (ICCV)*, 2813–2821 (2017) [doi:10.1109/ICCV.2017.304].
