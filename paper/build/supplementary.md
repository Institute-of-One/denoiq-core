<!--
  SUPPLEMENTARY MATERIAL — same rules as the manuscript: no typed numbers, every quantity a
  marker resolved from results/ at build time.

      python paper/build_manuscript.py            # renders this file and the manuscript
      python paper/build_pdf.py --document supplementary
-->

# Supplementary material

**Mathematical characterization of task detectability limits in AI-denoised CT: implications for task-based dose optimization**

Shuji Yamamoto · Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan

## S1. Supplementary Methods: the gauge, and the two failure measures

### S1.1 The verdict rules in full

The gauge is an auditable rule-based translation of the measured quantities; its thresholds are
task-specific values fixed in advance, not clinically validated criteria. Let `C` be the input's
analytic ceiling `d'`, `T` and `T_input` the cross-fitted prewhitening `d'` on the processed and
unprocessed images, `S` the SSIM of the processed images against the noise-free object, `F` and
`F_input` the lesion-like response rates, and `ρ` the contrast recovery.

| level | rule | reads as |
|---|---|---|
| red | `C` < 5.0 | the input does not meet the prespecified requirement, and no processing can restore it |
| red | `S` ≥ 0.7 and `T` ≤ 1.0 | high fidelity with inadequate task performance |
| red | `F` > 0.2 and `F` > `F_input` | excess lesion-like responses over the input |
| red | `ρ` < 0.5 | erasure of true contrast |
| amber | `C` < 1.2 × 5.0 | marginal: a small dose change crosses the floor |
| amber | 1 − `T`/`T_input` > 0.2 | severe task degradation relative to the input |
| green | none of the above | — |

Any red rule outranks any amber rule; within a level every rule that fired contributes a reason,
in the order listed. The adequacy threshold on `T` is a threshold for *inadequate* performance,
not for chance performance: `d'` = 1.0
corresponds to an AUC of about `0.76` for an equal-variance Gaussian decision variable, whereas
chance would be `d'` = `0`, AUC `0.5`.

### S1.2 Lesion-like responses and contrast recovery

In signal-absent trials the underlying object is spatially uniform, although the acquired image
contains noise; noise alone therefore produces lesion-like structure, and any measure of
"invented" structure must be read against the unprocessed input rather than absolutely.

A matched-filter amplitude map is formed with the known lesion profile, mean-subtracted and
normalized so that an image containing exactly one true lesion reads `1.0` at that location. The
map is evaluated over all positions (a full 2-D correlation in `same` mode with zero padding;
the lesion is centred and the image is 64 pixels
across, so boundary effects fall outside the lesion support). For each image the maximum over
positions is taken, and the *lesion-like response rate* is the fraction of signal-absent images
whose maximum reaches 0.5 of a true lesion's
amplitude — a round fraction chosen in advance rather than tuned. The same statistic is computed
on the unprocessed input of the same condition, and the gauge fires only on an excess over it.

*Contrast recovery* is the projection of the class-mean difference image onto the same
mean-subtracted lesion profile, divided by that profile's squared norm: the amplitude of the
surviving lesion in units of the true contrast. It is 1 when contrast is preserved and 0 when it
is erased; values above 1 mean the processing amplified the mean difference. Values are not
clipped.

## S2. Closed-form validation of the analytic observer

### S2.1 Result (Table S1)

In white noise the analytic ideal linear (prewhitening) observer satisfies `d' = ‖s‖₂/σ`
exactly. Both columns of Table S1 are computed analytically and neither is estimated from image
samples — the left from the identity, the right by the implementation from the analytic noise
power spectrum — so the residual is numerical rather than statistical, and the comparison tests
the implementation rather than a sample.

## S3. Estimator sensitivity

### S3.1 Cross-fitting against a single split (Table S2)

Table S2 compares the cross-fitted estimator used throughout this study with the single 50/50
split used in the earlier single-realization analysis, at matched conditions of the dose sweep in
the representative realization. Recovery is the estimated `d'` as a fraction of the analytic
ceiling of the same input; recovery slightly above `100 %` at the highest doses is sampling noise
in the `d'` estimate, not a breached bound, since the ceiling test is armwise, in AUC, and carries
its own margin (Section 3.4).

## S4. The learned denoiser used in the synthetic demonstration

This network is the synthetic-arm demonstration network and is distinct from the real-data CNNs
described in Section 2.9.

The convolutional network is a small residual (DnCNN-style) model trained deterministically on
CPU on synthetic pairs spanning dose, with a checkpoint whose parameter hash is recorded in
`results/cnn_training.json`. It is blind to the noise level at inference: each image is
normalized by a noise standard deviation estimated from that image alone. It is evaluated at one
prespecified low-dose setting outside the primary matrix and contributes to no primary endpoint.

Figure S1 is the red-lamp console at three settings of the demonstration, with the network as
the processing: the unprocessed input, the processed image, both verdicts and the reason each
earned. The interactive version of the same record is `paper/figures/redlamp_console.html` in
the repository.

## S5. Floor-stratified failure patterns

### S5.1 Full comparison (Table S3)

Table S3 gives every stratified quantity of Section 3.5 with its clustered interval: divergence
rate, erasure rate, excess lesion-like response rate, observer-dependent benefit, mean task
degradation and the verdict distribution, for each of the three strata.

## S6. Per-denoiser endpoints

### S6.1 Divergence endpoints in full (Table S4)

Table S4 gives, for each denoiser, the Spearman correlation between ΔSSIM and Δ`d'`(PW), the
divergence rate, the mean ΔSSIM and Δ`d'` for all three observers, the observer-dependent
benefit, and the Holm-adjusted p-values within each family.

## S7. The four networks of the real low-dose CT arm

These are the networks of Sections 3.6 and 3.6.1, and are distinct from the demonstration
network of Section S4.

Two capacities, and at each capacity two objectives. The small configuration is
21385 parameters trained on
800 patches per case for
16 epochs at batch
32; the large is
1849633 parameters on
3000 patches per case for
60 epochs at batch
64, a ratio of
86× in parameters. Capacity, training data
and training length move together between the two, so they do not isolate capacity.

Within a capacity they do isolate the objective. The adversarial network is built from the same
seed with the same architecture and trained on the same patches, for the same epochs, at the
same batch size and learning rate; only the loss differs, by the least-squares adversarial term
of Mao et al. [24] against a small patch critic. Its weights —
`mse_weight` = 0.05,
`adv_weight` = 1.0,
critic learning rate
1e-03 — were fixed
by a sweep recorded in `results/adv_weight_sweep.json` before this arm was run. A first sweep at
the fidelity arm's pixelwise weight found the adversarial term doing almost nothing across a
five-hundred-fold range of weights, because the pixelwise term is in units of the input's own
noise while the least-squares term is bounded when the critic is confused; the weights above are
where the term bites, and the effect is flat over more than an order of magnitude around them,
so the arm does not sit on a knife edge.

The checkpoint kept for each run is its last epoch rather than its lowest-validation one. The
best validation loss reached was 5.642 for
the small fidelity arm and 5.000 for the
large one; for the adversarial arms the full per-epoch history of the generator loss, the critic
loss and the validation mean-squared error is recorded in
`paper/results/liver_cnn_small_gan.json` and `paper/results/liver_cnn_large_gan.json`. Every
checkpoint's SHA-256 is in the same files.

## S8. The ceiling comparison: margin, saturation, and the one exceedance

The comparison of Section 3.4 is made against a margin rather than against the ceiling itself,
because an estimated AUC has sampling error and a bound compared without one would be violated by
noise alone. The armwise margin is the Hanley–McNeil standard error of the estimated AUC scaled
by `z`; the family-wise statement widens it by Bonferroni over all
760 comparisons, giving `z` =
3.99 at a family-wise level of `0.05`. At the
largest excess observed the margin was
7.00×10⁻⁴ in AUC.

An evaluation is treated as saturated when its analytic ceiling AUC exceeds
`1 − 10/(n_present · n_absent)`, which is the regime in which the closed-form interval degenerates:
at perfect separation the Hanley–McNeil standard error is identically zero, the margin collapses
to a single quantization step, and no widening of `z` recovers it.
180 of
760 evaluations met that rule and are analysed
separately for this reason and not because excluding them helps: over the remaining
580 the largest excess was
1.64×10⁻⁴ and the mean
-0.040.

The single exceedance occurred on an **unprocessed** arm, where the processing is the identity and
cannot create information, so what it measures is the estimator and the finite sample rather than
a violated bound. Its analytic ceiling AUC is
0.9999933, the estimator achieved
1.0000000, and the excess is
6.75×10⁻⁶ in AUC against a margin of
6.94×10⁻⁷ — of the order of ten
misordered pairs in `1.44` million. The family-wise margin leaves this same
1 exceedance, for
exactly that reason. Across the processed arms, which are the ones the claim is about,
0 of
550 exceeded the margin.
