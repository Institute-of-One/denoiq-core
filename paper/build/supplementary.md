<!--
  SUPPLEMENTARY MATERIAL — same rules as the manuscript: no typed numbers, every quantity a
  marker resolved from results/ at build time.

      python paper/build_manuscript.py            # renders this file and the manuscript
      python paper/build_pdf.py --document supplementary
-->

# Supplementary material

**Denoising under a data-processing ceiling: observer-dependent benefits, fidelity–task divergence, and an information floor**

Shuji Yamamoto · Institute of One, LISIT Co., Ltd., Tokyo, Japan

## S1. Supplementary Methods: the gauge, and the two failure measures

### S1.1 The verdict rules in full

The gauge is an auditable rule-based translation of the measured quantities; its thresholds are
task-specific values fixed in advance, not clinically validated criteria. Let `C` be the input's
analytic ceiling `d'`, `T` and `T_input` the cross-fitted prewhitening `d'` on the processed and
unprocessed images, `S` the SSIM of the processed images against the noise-free object, `F` and
`F_input` the lesion-like response rates, and `ρ` the contrast recovery.

| level | rule | reads as |
|---|---|---|
| **red** | `C` < 5.0 | the input does not meet the prespecified requirement, and no processing can restore it |
| **red** | `S` ≥ 0.7 and `T` ≤ 1.0 | high fidelity with inadequate task performance |
| **red** | `F` > 0.2 and `F` > `F_input` | excess lesion-like responses over the input |
| **red** | `ρ` < 0.5 | erasure of true contrast |
| **amber** | `C` < 1.2 × 5.0 | marginal: a small dose change crosses the floor |
| **amber** | 1 − `T`/`T_input` > 0.2 | severe task degradation relative to the input |
| **green** | none of the above | — |

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
normalised so that an image containing exactly one true lesion reads `1.0` at that location. The
map is evaluated over all positions (a full 2-D correlation in `same` mode with zero padding;
the lesion is centred and the image is 64 pixels
across, so boundary effects fall outside the lesion support). For each image the maximum over
positions is taken, and the **lesion-like response rate** is the fraction of signal-absent images
whose maximum reaches 0.5 of a true lesion's
amplitude — a round fraction chosen in advance rather than tuned. The same statistic is computed
on the unprocessed input of the same condition, and the gauge fires only on an excess over it.

**Contrast recovery** is the projection of the class-mean difference image onto the same
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
split used in the earlier single-realisation analysis, at matched conditions of the dose sweep in
the representative realisation. Recovery is the estimated `d'` as a fraction of the analytic
ceiling of the same input; recovery slightly above `100 %` at the highest doses is sampling noise
in the `d'` estimate, not a breached bound, since the ceiling test is armwise, in AUC, and carries
its own margin (Section 3.4).

## S4. The learned denoiser

The convolutional network is a small residual (DnCNN-style) model trained deterministically on
CPU on synthetic pairs spanning dose, with a checkpoint whose parameter hash is recorded in
`results/cnn_training.json`. It is blind to the noise level at inference: each image is
normalised by a noise standard deviation estimated from that image alone. It is evaluated at one
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
