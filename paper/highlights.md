<!--
  HIGHLIGHTS — Physica Medica requires 3 to 5 bullets, each at most 85 characters including
  spaces, submitted as a separate editable file with "highlights" in the file name.

  The limit is checked by tests/test_manuscript_consistency.py, because a bullet one character
  over is the kind of thing that is noticed by an editorial office rather than by an author.

  Numbers here are deliberately few. A highlight that needs a condition attached is not a
  highlight, and the conditions are in the abstract.

  Every number is a marker, resolved against results/ at build time, for the same reason the
  manuscript's are: this file said the best-PSNR denoiser "ranked fifth of nine" while the
  abstract, the body and the cover letter all said 7 of 9. Fifth is the value under the
  stricter admission rule of Section 3.6.2, and it had been typed here by hand. The highlights
  are the first thing a reviewer reads, so they were the worst place in the submission for it.

  The 85-character limit is checked after the markers resolve, not before.
-->

# Highlights

- Fidelity gains did not predict task detectability on real low-dose CT
- The best-PSNR denoiser ranked [[results:real_liver.json:held_out.psnr_winner_task_rank]] of [[results:real_liver.json:held_out.n_methods]] on the detection task
- No arm exceeded the input's ideal-observer ceiling, adversarial ones included
- An ideal-observer dose floor supplies [[results:real_liver.json:guidance.efficiency.ideal_floor_fraction_of_required|.0%]] of the exposure a real observer needs
- Exposure requirements can be compared across methods against a stated criterion
