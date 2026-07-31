r"""Deterministic training of the optional CNN denoiser.

Requires the ``[dl]`` extra. Run it as a script::

    python -m denoiq_core.train --epochs 30 --out checkpoints/cnn.pt

and it writes the checkpoint, records its SHA-256 and the loss history in
``results/cnn_training.json``, and prints both.

Two properties are non-negotiable
---------------------------------
**Determinism.** Fixed seeds, ``torch.use_deterministic_algorithms(True)``, single-threaded
batching with a seeded numpy shuffle. Training twice gives the same weights, hence the same
denoised images, hence the same numbers in ``results/``. Without that, the ceiling test could
be re-rolled until it passed.

**No leakage.** The training pairs are generated from their own seed range, disjoint from the
seeds any evaluation uses, and the network never sees a label, a signal location, or a noise
level. Both matter for the same reason: a network that had seen the evaluation trials — or
been told where the lesion is — could beat the ceiling, and it would prove nothing except
that the experiment was wired wrong.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from denoiq_core.cnn import (
    DEFAULT_CNN,
    CNNConfig,
    build_cnn,
    denoise_stack,
    estimate_noise_sd,
    save_checkpoint,
    torch_available,
)
from denoiq_core.experiment import provenance, results_dir, write_json
from denoiq_core.physics import (
    DEFAULT_MODEL,
    DEFAULT_PHANTOM,
    AcquisitionModel,
    PhantomSpec,
    make_trials,
)

__all__ = ["TrainConfig", "TrainResult", "main", "make_training_pairs", "train"]

#: Training seeds live here and evaluation seeds do not. Any overlap would be leakage.
TRAIN_SEED_BASE = 900_000_000


@dataclass(frozen=True)
class TrainConfig:
    """Everything the training run needs, and nothing it is allowed to learn from.

    Attributes
    ----------
    mas_values:
        Dose levels the training set spans. The network must work across dose, because it is
        applied across dose and is never told which one it is looking at.
    n_images_per_level:
        Training images per dose level, half signal-present and half signal-absent. The
        network is not told which is which — it only ever sees (noisy, clean) pairs.
    epochs, batch_size, learning_rate:
        Plain Adam training. No schedule, no augmentation, no early stopping: fewer knobs,
        fewer ways for the result to depend on one.
    val_fraction:
        Held-out fraction of the pairs, used only to report a validation loss.
    seed:
        Seeds the weights, the shuffling and the phantom generation (offset by
        :data:`TRAIN_SEED_BASE`).

    """

    mas_values: tuple[float, ...] = (200.0, 100.0, 50.0, 25.0, 12.0, 6.0)
    kv: float = 120.0
    n_images_per_level: int = 96
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 12345
    cnn: CNNConfig = DEFAULT_CNN
    model: AcquisitionModel = DEFAULT_MODEL
    phantom: PhantomSpec = DEFAULT_PHANTOM
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the training configuration."""
        return {
            "mas_values": list(self.mas_values),
            "kv": float(self.kv),
            "n_images_per_level": int(self.n_images_per_level),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "val_fraction": float(self.val_fraction),
            "seed": int(self.seed),
            "cnn": self.cnn.to_dict(),
            "model": self.model.to_dict(),
            "phantom": self.phantom.to_dict(),
            **self.meta,
        }


@dataclass(frozen=True)
class TrainResult:
    """Where the weights are, what they hash to, and how the loss went.

    ``sha256`` is the hash of the **parameters** (:func:`denoiq_core.cnn.weights_sha256`), not
    of the checkpoint file: it is stable across torch's archive format and is therefore the
    thing to compare when asking whether two training runs produced the same network.
    """

    checkpoint: Path
    sha256: str
    train_loss: list[float]
    val_loss: list[float]
    config: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record, ready for ``results/``."""
        from denoiq_core.cnn import checkpoint_sha256

        return {
            # As posix, so the record does not change with the operating system that wrote it.
            "checkpoint": self.checkpoint.as_posix(),
            "sha256": self.sha256,
            "checkpoint_file_sha256": (
                checkpoint_sha256(self.checkpoint) if self.checkpoint.exists() else None
            ),
            "train_loss": [float(x) for x in self.train_loss],
            "val_loss": [float(x) for x in self.val_loss],
            "final_train_loss": float(self.train_loss[-1]) if self.train_loss else float("nan"),
            "final_val_loss": float(self.val_loss[-1]) if self.val_loss else float("nan"),
            "config": self.config.to_dict(),
        }


def make_training_pairs(config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(noisy, clean)`` training pairs across dose levels.

    Both members of a pair are normalised the way the network will see them at inference time
    — offset by the image mean, scaled by the noise standard deviation **estimated from the
    noisy image** — so that training and inference cannot disagree about units, and so that
    the network is never handed the true noise level by the back door.

    Returns
    -------
    tuple
        ``(noisy, clean)``, each ``(n, ny, nx)`` and already normalised.

    """
    noisy_list: list[np.ndarray] = []
    clean_list: list[np.ndarray] = []
    per_class = max(2, config.n_images_per_level // 2)

    for i, mas in enumerate(config.mas_values):
        trials = make_trials(
            config.kv,
            mas,
            n_trials=per_class,
            seed=TRAIN_SEED_BASE + config.seed + 7919 * i,
            model=config.model,
            phantom=config.phantom,
        )
        background = float(config.phantom.background)
        clean_present = background + trials.signal
        clean_absent = np.full(trials.shape, background, dtype=np.float64)
        for noisy, clean in (
            (trials.present, clean_present),
            (trials.absent, clean_absent),
        ):
            for plane in noisy:
                scale = estimate_noise_sd(plane)
                offset = float(plane.mean())
                noisy_list.append((plane - offset) / scale)
                clean_list.append((clean - offset) / scale)

    return np.asarray(noisy_list), np.asarray(clean_list)


def train(
    config: TrainConfig | None = None, *, out: str | Path = "checkpoints/cnn.pt"
) -> TrainResult:
    """Train the residual denoiser deterministically and write the checkpoint.

    Parameters
    ----------
    config:
        Training configuration; the default :class:`TrainConfig` when omitted.
    out:
        Where to write the checkpoint.

    Returns
    -------
    TrainResult

    """
    config = config or TrainConfig()
    if not torch_available():
        raise ImportError(
            'training needs the optional [dl] extra: pip install -e ".[dl]". The classical '
            "results, the ceiling and the red lamp do not require it."
        )
    import torch

    torch.manual_seed(int(config.seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    noisy, clean = make_training_pairs(config)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(noisy.shape[0])
    noisy, clean = noisy[order], clean[order]

    n_val = max(1, int(round(noisy.shape[0] * config.val_fraction)))
    x_val = torch.from_numpy(noisy[:n_val]).unsqueeze(1).to(torch.float32)
    y_val = torch.from_numpy(clean[:n_val]).unsqueeze(1).to(torch.float32)
    x_train = torch.from_numpy(noisy[n_val:]).unsqueeze(1).to(torch.float32)
    y_train = torch.from_numpy(clean[n_val:]).unsqueeze(1).to(torch.float32)

    model = build_cnn(config.cnn, seed=config.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_loss: list[float] = []
    val_loss: list[float] = []
    n = x_train.shape[0]
    for epoch in range(int(config.epochs)):
        model.train()
        # Seeded per-epoch shuffle: the whole run is a function of config.seed alone.
        indices = np.random.default_rng(config.seed + epoch).permutation(n)
        running = 0.0
        for start in range(0, n, int(config.batch_size)):
            batch = indices[start : start + int(config.batch_size)]
            if batch.size == 0:  # pragma: no cover - defensive
                continue
            xb = x_train[batch]
            yb = y_train[batch]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * batch.size
        train_loss.append(running / n)
        model.eval()
        with torch.no_grad():
            val_loss.append(float(loss_fn(model(x_val), y_val).item()))

    checkpoint = Path(out)
    sha = save_checkpoint(
        model,
        checkpoint,
        extra={
            "config": config.to_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
    )
    return TrainResult(
        checkpoint=checkpoint,
        sha256=sha,
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
    )


def _sanity_check(result: TrainResult) -> dict[str, Any]:
    """Does the trained network actually denoise? Reported next to the loss.

    A residual-loss number is not evidence of anything on its own. This applies the network to
    a fresh experiment (its own seed range, so still no leakage) and reports the noise
    standard deviation before and after, plus the recovered lesion contrast — the two ways it
    could be cheating (doing nothing, or erasing the lesion) are both visible here.
    """
    from denoiq_core.cnn import load_checkpoint
    from denoiq_core.redlamp import contrast_recovery

    model, _ = load_checkpoint(result.checkpoint)
    config = result.config
    trials = make_trials(
        config.kv,
        config.mas_values[len(config.mas_values) // 2],
        n_trials=32,
        seed=TRAIN_SEED_BASE // 2 + 1,
        model=config.model,
        phantom=config.phantom,
    )
    present = denoise_stack(trials.present, model)
    absent = denoise_stack(trials.absent, model)
    return {
        "raw_noise_sd": float(np.std(trials.absent)),
        "denoised_noise_sd": float(np.std(absent)),
        "contrast_recovery": float(contrast_recovery(present, absent, trials.signal)),
        "check_mas": float(config.mas_values[len(config.mas_values) // 2]),
        "check_seed": TRAIN_SEED_BASE // 2 + 1,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: train, checkpoint, and record in ``results/``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    parser.add_argument("--images-per-level", type=int, default=TrainConfig().n_images_per_level)
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--depth", type=int, default=DEFAULT_CNN.depth)
    parser.add_argument("--width", type=int, default=DEFAULT_CNN.width)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/cnn.pt"))
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="directory for cnn_training.json (default: results/ beside the package)",
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        epochs=args.epochs,
        n_images_per_level=args.images_per_level,
        seed=args.seed,
        cnn=replace(DEFAULT_CNN, depth=args.depth, width=args.width),
    )
    result = train(config, out=args.out)
    payload = result.to_dict()
    payload["provenance"] = provenance()
    payload["sanity_check"] = _sanity_check(result)
    path = write_json(payload, "cnn_training.json", directory=args.results or results_dir())

    print(f"checkpoint: {result.checkpoint}")
    print(f"weights:    sha256 {result.sha256}")
    print(
        f"final loss: train {payload['final_train_loss']:.5f}  val {payload['final_val_loss']:.5f}"
    )
    print(
        "sanity:     noise sd {raw_noise_sd:.2f} -> {denoised_noise_sd:.2f}, "
        "contrast recovery {contrast_recovery:.2f}".format(**payload["sanity_check"])
    )
    print(f"recorded:   {path}")
    print(json.dumps({"sha256": result.sha256}, indent=None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
