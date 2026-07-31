# results/

Every number this study reports lives here. Nothing in the manuscript, the figures or the
tables is computed anywhere else, and nothing is typed by hand:
`tests/test_manuscript_consistency.py` re-derives the manuscript's numbers from these files
on every CI run, and `paper/build_manuscript.py` refuses to build a manuscript that cites
something absent from them.

Regenerate all of it with:

```bash
python -c "from denoiq_core.experiment import run_all; run_all()"
```

Everything is deterministic: the same command on the same code produces byte-identical files
(`tests/test_determinism.py` asserts it).

| file | what it holds |
|---|---|
| `dose_sweep.json` / `.csv` | mAs sweep at fixed kV × every denoiser: fidelity, three observers, the ceiling test, fabrication, erasure, the lamp verdict |
| `texture_sweep.json` / `.csv` | the same, over noise correlation length at fixed dose |
| `signal_sweep.json` / `.csv` | the same, over lesion radius and contrast |
| `redlamp_atlas.json` | the kV–mAs atlas: raw ideal `d'` across the plane, the information-floor contour, and the green/amber/red grid (analytic — no Monte Carlo) |
| `redlamp_demo.json` / `.csv` | Monte-Carlo demonstration points on that atlas |
| `closed_form.json` / `.csv` | Table 1: the ideal observer against `d' = ‖s‖/σ` in white noise |
| `task_gains.json` / `.csv` | Table 2: signed change in each observer's `d'` per denoiser |
| `summary.json` | the headline numbers, including whether the ceiling held everywhere |
| `taskbench.json` / `.csv` | one acquisition setting in detail (`examples/run_denoise_taskbench.py`) |
| `taskbench_cnn.json` / `.csv` | the same with the optional CNN row — kept separate so that everything else reproduces without torch |
| `cnn_training.json` | the training run behind that checkpoint: loss history, the SHA-256 of the weights, and a sanity check that the network denoises without erasing the lesion |
| `redlamp_console.json` | the three console scenes: the input's analytic ceiling, the held-out task estimates, fidelity, contrast recovery, lesion-like response rate and both verdicts with their reasons. Figure 8 is drawn from this file and the tiles beside it |

Each file carries a `provenance` block with the `denoiq-core` and `taskiq-core` versions that
produced it, and a `quick` flag. **A file with `"quick": true` came from the fast smoke-test
sweep and is not publishable**; the manuscript-consistency test rejects it.

Reproducing the two CNN files needs the `[dl]` extra:

```bash
python -m denoiq_core.train --epochs 40 --out checkpoints/cnn.pt
python examples/run_denoise_taskbench.py --cnn checkpoints/cnn.pt
```

The checkpoint itself is not tracked (see `.gitignore`) — the weights are regenerable from
the seed, and their hash is recorded in `cnn_training.json` so that a reader can confirm they
regenerated the same ones.
