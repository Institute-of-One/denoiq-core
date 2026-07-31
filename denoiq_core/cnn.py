r"""A small residual CNN denoiser — optional, and deliberately unremarkable.

Requires the ``[dl]`` extra (``pip install -e ".[dl]"``). **Nothing else in this package
needs it**: the classical denoisers, all three observers, the data-processing ceiling, the
red-lamp gauge, every sweep and every figure run without torch. Importing this module is the
only thing that adds ``"cnn"`` to :func:`denoiq_core.denoisers.available_methods`.

Why a network is here at all
----------------------------
Not to be state of the art. The claim under test is that *no* post-processing can exceed the
information in the raw data, and a claim of that shape is not strengthened by a bigger
network — it is tested by including one at all, so that "but a neural network would be
different" has an answer with a number attached. A DnCNN-style residual network with a
handful of layers is enough to produce visibly clean images, high SSIM, and a task
performance that still sits under the ceiling.

Blind operation
---------------
The network must be a function of the image alone, or the ceiling test measures nothing (see
:mod:`denoiq_core.bound`). So it never receives the noise level as a parameter: the input is
normalised by a noise standard deviation **estimated from the image itself**
(:func:`estimate_noise_sd`, the Immerkaer Laplacian estimator), and the same scale is used to
undo the normalisation. Training then spans dose levels rather than being pinned to one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import convolve

from denoiq_core.denoisers import register_denoiser

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

__all__ = [
    "CNNConfig",
    "build_cnn",
    "checkpoint_sha256",
    "denoise_stack",
    "estimate_noise_sd",
    "load_checkpoint",
    "save_checkpoint",
    "torch_available",
    "weights_sha256",
]


def torch_available() -> bool:
    """Whether the ``[dl]`` extra is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'the CNN path needs the optional [dl] extra: pip install -e ".[dl]". Every '
            "classical denoiser, observer, bound and figure in this package runs without it."
        ) from exc
    return torch


#: Immerkaer's Laplacian mask. ``sum(mask^2) = 36``, hence the 1/6 in the estimator.
_LAPLACIAN = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def estimate_noise_sd(image: np.ndarray) -> float:
    r"""Noise standard deviation estimated from one image, using nothing else.

    Immerkaer's estimator: convolve with a Laplacian-like mask that annihilates smooth
    structure, then

    .. math::  \hat\sigma = \frac{1}{6}\sqrt{\frac{\pi}{2}}\,
               \frac{1}{(W-2)(H-2)}\sum |I * L|

    (the mean-absolute form, which is less sensitive to the lesion edge than the sum of
    squares). It is a function of the pixels alone — no ground truth, no acquisition
    parameters — which is what keeps the network an honest post-processing step.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    response = convolve(img, _LAPLACIAN, mode="nearest")[1:-1, 1:-1]
    scale = np.sqrt(np.pi / 2.0) / 6.0
    sd = float(scale * np.mean(np.abs(response)))
    return max(sd, 1e-12)


@dataclass(frozen=True)
class CNNConfig:
    """Architecture of the residual denoiser.

    Attributes
    ----------
    depth:
        Number of convolutional layers. The first maps 1 channel to ``width``, the last maps
        ``width`` back to 1, the ones in between are ``width``-to-``width``.
    width:
        Channels per hidden layer. Small on purpose: the study is CPU-only and reproducible,
        and a wider network changes the picture quality, not the ceiling.
    kernel_size:
        Convolution kernel, odd, with ``same`` padding so the output keeps its shape.
    residual:
        ``True`` learns the *noise* and subtracts it (the DnCNN formulation), which trains far
        faster at this size than learning the clean image directly.

    """

    depth: int = 6
    width: int = 24
    kernel_size: int = 3
    residual: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable record of the architecture."""
        return {
            "depth": int(self.depth),
            "width": int(self.width),
            "kernel_size": int(self.kernel_size),
            "residual": bool(self.residual),
        }


DEFAULT_CNN = CNNConfig()


def build_cnn(config: CNNConfig = DEFAULT_CNN, *, seed: int = 0) -> nn.Module:
    """A DnCNN-style residual network, initialised deterministically from ``seed``.

    Same seed, same weights, on any machine with the same torch version — which is what makes
    ``tests/test_determinism.py`` able to assert bit-equality of a training run.
    """
    torch = _require_torch()
    from torch import nn as torch_nn

    if config.depth < 2:
        raise ValueError(f"depth must be >= 2 (an input and an output layer), got {config.depth}")
    if config.kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {config.kernel_size}")

    torch.manual_seed(int(seed))
    pad = config.kernel_size // 2
    layers: list[Any] = [
        torch_nn.Conv2d(1, config.width, config.kernel_size, padding=pad, padding_mode="replicate"),
        torch_nn.ReLU(inplace=True),
    ]
    for _ in range(config.depth - 2):
        layers += [
            torch_nn.Conv2d(
                config.width,
                config.width,
                config.kernel_size,
                padding=pad,
                padding_mode="replicate",
                bias=False,
            ),
            torch_nn.BatchNorm2d(config.width),
            torch_nn.ReLU(inplace=True),
        ]
    layers.append(
        torch_nn.Conv2d(config.width, 1, config.kernel_size, padding=pad, padding_mode="replicate")
    )

    class ResidualDenoiser(torch_nn.Module):  # type: ignore[misc]  # torch is untyped here
        """Predicts the noise field and subtracts it (or predicts the image directly)."""

        def __init__(self) -> None:
            super().__init__()
            self.body = torch_nn.Sequential(*layers)
            self.residual = config.residual

        def forward(self, x: Any) -> Any:
            """Denoise a normalised batch ``(n, 1, ny, nx)``."""
            out = self.body(x)
            return x - out if self.residual else out

    net = ResidualDenoiser()
    net.config = config  # kept on the module so save_checkpoint can record the architecture
    return net


def weights_sha256(model: nn.Module) -> str:
    """SHA-256 of the model's parameters themselves, in sorted key order.

    Not the hash of the checkpoint *file*: a torch archive is a zip container, and two
    archives holding identical tensors can differ in their container bytes (the member names
    depend on the path written to). Hashing the tensors answers the question that matters —
    *are these the same weights?* — and is stable across torch's serialisation details, which
    is what lets ``tests/test_determinism.py`` assert that a seeded training run is
    reproducible.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    model: nn.Module, path: str | Path, *, extra: dict[str, Any] | None = None
) -> str:
    """Save weights plus architecture, and return :func:`weights_sha256` of the model.

    The hash is what goes in ``results/``: it is how a reader can tell whether the weights
    behind a published number are the weights they have.
    """
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = getattr(model, "config", DEFAULT_CNN)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "extra": dict(extra or {}),
        },
        path,
    )
    return weights_sha256(model)


def checkpoint_sha256(path: str | Path) -> str:
    """SHA-256 of a checkpoint *file*, for integrity of a specific artefact.

    Use :func:`weights_sha256` to ask whether two runs produced the same network; use this to
    ask whether a file arrived intact.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns the network in eval mode and the payload's ``extra`` dictionary (training
    configuration, loss history, the sha256 of the file it was trained from, and so on).
    """
    torch = _require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = CNNConfig(**payload["config"])
    model = build_cnn(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("extra", {}))


def denoise_stack(
    images: np.ndarray,
    model: nn.Module,
    *,
    batch_size: int = 64,
    background: float | None = None,
) -> np.ndarray:
    """Apply the network to an image or a stack, one image's worth of information at a time.

    Each image is normalised by its **own** estimated noise level and its own mean before it
    reaches the network, and denormalised afterwards. Nothing about the batch is shared: the
    result for image *i* is identical whether it is processed alone or alongside a thousand
    others, which is required for the trials to stay independent (and is asserted in
    ``tests/test_denoisers.py``).
    """
    torch = _require_torch()
    arr = np.asarray(images, dtype=np.float64)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected a 2-D image or a 3-D stack, got shape {arr.shape}")

    offsets = np.array(
        [float(plane.mean()) if background is None else float(background) for plane in arr]
    )
    scales = np.array([estimate_noise_sd(plane) for plane in arr])
    normalised = (arr - offsets[:, None, None]) / scales[:, None, None]

    model.eval()
    out = np.empty_like(normalised)
    with torch.no_grad():
        for start in range(0, normalised.shape[0], int(batch_size)):
            stop = start + int(batch_size)
            batch = torch.from_numpy(normalised[start:stop]).unsqueeze(1).to(torch.float32)
            out[start:stop] = model(batch).squeeze(1).numpy().astype(np.float64)

    result = out * scales[:, None, None] + offsets[:, None, None]
    return result[0] if single else result


# --------------------------------------------------------------------------------------
# registration into the unified denoiser interface
# --------------------------------------------------------------------------------------

_CACHE: dict[str, tuple[nn.Module, dict[str, Any]]] = {}


def _cnn_denoiser(images: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """The ``method="cnn"`` entry of :func:`denoiq_core.denoisers.denoise`."""
    checkpoint = params.get("checkpoint")
    if checkpoint is None:
        raise ValueError(
            "method='cnn' needs checkpoint=<path>; train one with "
            "python -m denoiq_core.train (it writes results/cnn_training.json)"
        )
    key = str(Path(checkpoint).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_checkpoint(key)
    model, _ = _CACHE[key]
    return denoise_stack(images, model, batch_size=int(params.get("batch_size", 64)))


if torch_available():  # pragma: no branch - trivial
    register_denoiser(
        "cnn",
        _cnn_denoiser,
        {"checkpoint": None, "batch_size": 64},
        stack_aware=True,
    )
