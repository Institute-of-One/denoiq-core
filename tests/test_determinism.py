"""Same seed, same numbers — and same bytes.

A study whose numbers move between runs cannot be checked by anyone, including its author.
Everything here is generated from a seed: the phantoms, the trials, the observer splits, the
figures, and (with the ``[dl]`` extra) the training of the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from denoiq_core.bound import check_denoiser
from denoiq_core.denoisers import denoise
from denoiq_core.evaluate import evaluate_condition
from denoiq_core.physics import make_trials


def test_trials_are_reproducible_from_the_seed():
    a = make_trials(120.0, 25.0, n_trials=16, seed=42)
    b = make_trials(120.0, 25.0, n_trials=16, seed=42)
    assert np.array_equal(a.present, b.present)
    assert np.array_equal(a.absent, b.absent)
    assert np.array_equal(a.signal, b.signal)


def test_different_seeds_give_different_noise():
    a = make_trials(120.0, 25.0, n_trials=16, seed=42)
    b = make_trials(120.0, 25.0, n_trials=16, seed=43)
    assert not np.array_equal(a.absent, b.absent)
    assert np.array_equal(a.signal, b.signal)  # the signal is deterministic, not random


@pytest.mark.parametrize("method,params", [("gaussian", {"sigma": 1.5}), ("tv", {"weight": 20.0})])
def test_denoising_is_bit_identical(method, params):
    trials = make_trials(120.0, 25.0, n_trials=8, seed=44)
    first = denoise(trials.present, method, **params)
    second = denoise(trials.present, method, **params)
    assert np.array_equal(first, second)


def test_evaluation_is_bit_identical():
    trials = make_trials(120.0, 25.0, n_trials=64, seed=45)
    first = evaluate_condition(trials, "gaussian", {"sigma": 1.5})
    second = evaluate_condition(trials, "gaussian", {"sigma": 1.5})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_ceiling_test_is_bit_identical():
    trials = make_trials(120.0, 25.0, n_trials=64, seed=46)
    first = check_denoiser(trials, "tv", {"weight": 20.0})
    second = check_denoiser(trials, "tv", {"weight": 20.0})
    assert first.to_dict() == second.to_dict()


def test_a_whole_sweep_is_reproducible(tmp_path):
    """Two independent quick runs must write identical results files."""
    from denoiq_core.experiment import run_all

    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    run_all(quick=True, directory=first)
    run_all(quick=True, directory=second)

    names = sorted(p.name for p in first.iterdir())
    assert names, "the quick run wrote nothing"
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_figures_are_byte_identical(tmp_path):
    """A figure that changes without its data changing cannot be checked into a repository."""
    from denoiq_core.experiment import run_all
    from denoiq_core.figures import make_all_figures

    results = tmp_path / "results"
    results.mkdir()
    run_all(quick=True, directory=results)

    first = tmp_path / "fig_a"
    second = tmp_path / "fig_b"
    make_all_figures(results=results, figures=first)
    make_all_figures(results=results, figures=second)
    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_the_console_figure_is_reproducible_from_its_record(tmp_path):
    """Supplementary Figure S1 is drawn from results/redlamp_console.json, not screenshotted.

    That is what keeps it honest: a screenshot silently survives a change of wording or of
    measurement, and this one cannot. Skipped when the record has not been generated.
    """
    import json

    from denoiq_core.figures import figureS1_console_panel

    repo = Path(__file__).resolve().parent.parent
    record = repo / "results" / "redlamp_console.json"
    tiles = repo / "paper" / "figures" / "console"
    if not (record.exists() and tiles.exists()):
        pytest.skip("no console record: run python paper/redlamp_console.py")

    console = json.loads(record.read_text(encoding="utf-8"))
    first = figureS1_console_panel(tmp_path / "a.png", console, tiles)
    second = figureS1_console_panel(tmp_path / "b.png", console, tiles)
    assert first.read_bytes() == second.read_bytes()

    committed = repo / "paper" / "figures" / "figS1_redlamp_console.png"
    if committed.exists() and _same_render_stack(repo):
        assert committed.read_bytes() == first.read_bytes(), (
            "paper/figures/figS1_redlamp_console.png is out of date with the console record — "
            "rebuild it with python paper/make_figures.py"
        )


def _same_render_stack(repo: Path) -> bool:
    """Whether this machine draws the same bytes as the machine that committed the figures.

    A PNG's bytes depend on the matplotlib and FreeType versions that drew it, so a byte
    comparison across environments tests the environment, not the figure. The versions used are
    recorded in paper/figures/PROVENANCE.json when the figures are generated; where they differ,
    the comparison is skipped and run-to-run determinism — the property that actually matters —
    is still asserted above.
    """
    sys.path.insert(0, str(repo / "paper"))
    import make_figures

    provenance = repo / "paper" / "figures" / "PROVENANCE.json"
    if not provenance.exists():
        return False
    recorded = json.loads(provenance.read_text(encoding="utf-8"))
    current = make_figures.render_stack()
    if recorded == current:
        return True
    pytest.skip(f"figures were drawn with {recorded}; this environment has {current}")


def test_cnn_training_is_reproducible(tmp_path):
    """With the [dl] extra: the same seed must give a checkpoint with the same hash."""
    pytest.importorskip("torch")
    from denoiq_core.train import TrainConfig, train

    config = TrainConfig(
        mas_values=(100.0, 25.0), n_images_per_level=8, epochs=2, batch_size=4, seed=5
    )
    first = train(config, out=tmp_path / "a.pt")
    second = train(config, out=tmp_path / "b.pt")
    assert first.sha256 == second.sha256
    assert first.train_loss == second.train_loss


def test_cnn_inference_is_reproducible_and_batch_independent(tmp_path):
    """The network's output for one image must not depend on what it was batched with.

    Exact equality is too strong a demand of float32 convolutions — the kernels take
    different code paths for different batch shapes — so the tolerance here is a few parts
    per million of the image scale, which is rounding, not information crossing between
    trials.
    """
    pytest.importorskip("torch")
    from denoiq_core.train import TrainConfig, train

    config = TrainConfig(mas_values=(100.0,), n_images_per_level=8, epochs=1, batch_size=4, seed=6)
    result = train(config, out=tmp_path / "cnn.pt")

    trials = make_trials(120.0, 25.0, n_trials=6, seed=47)
    together = denoise(trials.present, "cnn", checkpoint=str(result.checkpoint))
    again = denoise(trials.present, "cnn", checkpoint=str(result.checkpoint))
    apart = np.stack(
        [denoise(plane, "cnn", checkpoint=str(result.checkpoint)) for plane in trials.present]
    )
    assert np.array_equal(together, again)
    scale = float(np.abs(together).max())
    assert np.max(np.abs(together - apart)) < 1e-5 * max(scale, 1.0)
