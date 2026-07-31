r"""Classical denoisers behind one interface — and the rule they all obey.

Every denoiser here is a map :math:`g` from an image to an image. It sees the pixels and
nothing else: not the signal, not its location, not the noise level that generated it, not
which class the image belongs to. That restriction is the whole point. The data-processing
inequality (:mod:`denoiq_core.bound`) bounds what any such map can do for a detection task,
and the bound is only meaningful if the map is genuinely a function of the image alone —
a "denoiser" allowed to peek at the truth could beat it trivially, and would be measuring
the experimenter rather than the algorithm.

The denoisers are deliberately ordinary — a Gaussian filter, total variation, non-local
means, optionally BM3D. None of them is a straw man and none is the point; they are here so
that the ceiling can be demonstrated against something real, and so the classical results
stand on their own without a deep-learning dependency.

Determinism
-----------
None of these draws a random number. ``denoise`` on the same input with the same parameters
returns bit-identical output, which is what makes ``tests/test_determinism.py`` a
meaningful test rather than a tolerance check.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle

__all__ = [
    "DENOISER_DEFAULTS",
    "available_methods",
    "denoise",
    "denoiser_label",
    "has_bm3d",
    "register_denoiser",
]


#: Default parameters per method. Every value is a *modelling choice*, not a tuned optimum:
#: the study sweeps conditions rather than hyper-parameters, and the ceiling holds for any
#: choice. Lengths are in pixels unless the name says otherwise.
DENOISER_DEFAULTS: dict[str, dict[str, Any]] = {
    "none": {},
    "gaussian": {"sigma": 1.0},
    "tv": {"weight": 0.1, "max_num_iter": 200},
    "nlm": {"h": 1.0, "patch_size": 5, "patch_distance": 6},
    "bm3d": {"sigma_psd": 1.0},
}


def has_bm3d() -> bool:
    """Whether the optional ``bm3d`` package is importable."""
    try:
        import bm3d  # noqa: F401
    except ImportError:
        return False
    return True


def available_methods() -> tuple[str, ...]:
    """Method names usable in this environment, in a stable order.

    ``"bm3d"`` appears only when the optional package is installed and ``"cnn"`` only once
    :mod:`denoiq_core.cnn` has been imported (which needs the ``[dl]`` extra), so a sweep can
    iterate over this and produce the same numbers everywhere else regardless.
    """
    return tuple(name for name in DENOISER_DEFAULTS if name != "bm3d" or has_bm3d())


def register_denoiser(
    name: str,
    fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    defaults: dict[str, Any],
    *,
    stack_aware: bool = False,
) -> None:
    """Add a denoiser to the unified interface at run time.

    Used by :mod:`denoiq_core.cnn` to register the optional network under the same
    ``denoise(image, method, **params)`` call as the classical filters — so the CNN is not a
    parallel code path with its own conventions, and the ceiling test cannot accidentally
    treat it differently.

    Parameters
    ----------
    name:
        The ``method`` string callers will pass to :func:`denoise`.
    fn:
        ``fn(image, params) -> image``, taking the merged parameter dictionary.
    defaults:
        Default parameters, which also define the accepted parameter names.
    stack_aware:
        ``True`` if ``fn`` handles a whole ``(n, ny, nx)`` stack itself (a network wants
        batches). It must still process each image independently.

    """
    if name in DENOISER_DEFAULTS and name not in _REGISTERED:
        raise ValueError(f"{name!r} is a built-in denoiser and cannot be replaced")
    DENOISER_DEFAULTS[name] = dict(defaults)
    _DISPATCH[name] = fn
    _REGISTERED.add(name)
    if stack_aware:
        _STACK_AWARE.add(name)


def denoiser_label(method: str, params: dict[str, Any] | None = None) -> str:
    """Short, stable label for results files and figure legends, e.g. ``"gaussian(sigma=1)"``."""
    merged = dict(DENOISER_DEFAULTS.get(method, {}))
    merged.update(params or {})
    if not merged:
        return method
    inner = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(merged.items()))
    return f"{method}({inner})"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def denoise(image: np.ndarray, method: str = "gaussian", **params: Any) -> np.ndarray:
    """Denoise one image ``(ny, nx)`` or a stack ``(n, ny, nx)``; shape is preserved.

    Parameters
    ----------
    image:
        A 2-D image or a 3-D stack of them. A stack is processed slice by slice, exactly as
        if each slice had been passed on its own — no information crosses between trials,
        which would otherwise let a denoiser average the noise of an ensemble it would never
        have in practice (and would break the per-trial independence the observers assume).
    method:
        One of :func:`available_methods`. ``"none"`` returns a copy, which makes the raw
        condition just another entry in a sweep rather than a special case.
    **params:
        Method parameters, overriding :data:`DENOISER_DEFAULTS`.

    Returns
    -------
    np.ndarray
        ``float64``, same shape as the input.

    Raises
    ------
    ValueError
        On an unknown method, a non-2-D/3-D input, non-finite pixels, or an unexpected
        parameter name (a silently ignored typo would change nothing and be attributed to
        the algorithm).
    ImportError
        If ``method="bm3d"`` and the optional package is not installed.

    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or a 3-D stack, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains non-finite values")

    if method == "cnn" and method not in DENOISER_DEFAULTS:
        # Lazy registration: importing denoiq_core.cnn is what adds the network to this
        # interface, and it must not happen at package import time — torch is optional.
        import denoiq_core.cnn  # noqa: F401
    if method not in DENOISER_DEFAULTS:
        raise ValueError(f"unknown denoiser {method!r}; available: {', '.join(DENOISER_DEFAULTS)}")
    if method == "bm3d" and not has_bm3d():
        raise ImportError(
            "method='bm3d' needs the optional 'bm3d' package: pip install bm3d. Every other "
            "denoiser, observer and bound in this package runs without it."
        )

    merged = dict(DENOISER_DEFAULTS[method])
    unknown = set(params) - set(merged)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) for {method!r}: {sorted(unknown)}; accepted: {sorted(merged)}"
        )
    merged.update(params)

    fn = _DISPATCH[method]
    if arr.ndim == 2 or method in _STACK_AWARE:
        out = fn(arr, merged)
    else:
        out = np.stack([fn(plane, merged) for plane in arr], axis=0)

    out = np.asarray(out, dtype=np.float64)
    if out.shape != arr.shape:  # pragma: no cover - defensive
        raise ValueError(f"denoiser {method!r} changed the shape: {arr.shape} -> {out.shape}")
    return out


# --------------------------------------------------------------------------------------
# per-method implementations (2-D only; `denoise` handles stacking)
# --------------------------------------------------------------------------------------


def _none(plane: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return plane.copy()


def _gaussian(plane: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    sigma = float(params["sigma"])
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"gaussian sigma must be finite and >= 0, got {sigma!r}")
    return gaussian_filter(plane, sigma=sigma, mode="nearest")


def _tv(plane: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    weight = float(params["weight"])
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"tv weight must be finite and > 0, got {weight!r}")
    # TV is scale-sensitive and the images carry a large flat background (water ~ 0 HU on a
    # +1000 scale, or whatever `background` the trials used). Subtracting the image mean
    # before filtering and adding it back makes `weight` mean the same thing regardless of
    # the background level; it is an exact identity for TV, which is translation-invariant.
    offset = float(plane.mean())
    filtered = denoise_tv_chambolle(
        plane - offset,
        weight=weight,
        max_num_iter=int(params["max_num_iter"]),
        channel_axis=None,
    )
    return np.asarray(filtered, dtype=np.float64) + offset


def _nlm(plane: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    h = float(params["h"])
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"nlm h must be finite and > 0, got {h!r}")
    offset = float(plane.mean())
    filtered = denoise_nl_means(
        plane - offset,
        h=h,
        sigma=h,
        patch_size=int(params["patch_size"]),
        patch_distance=int(params["patch_distance"]),
        fast_mode=True,
        channel_axis=None,
        preserve_range=True,
    )
    return np.asarray(filtered, dtype=np.float64) + offset


def _bm3d(plane: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    import bm3d as _bm3d  # local import: optional dependency

    sigma_psd = float(params["sigma_psd"])
    if not np.isfinite(sigma_psd) or sigma_psd <= 0.0:
        raise ValueError(f"bm3d sigma_psd must be finite and > 0, got {sigma_psd!r}")
    offset = float(plane.mean())
    filtered = _bm3d.bm3d(plane - offset, sigma_psd=sigma_psd)
    return np.asarray(filtered, dtype=np.float64) + offset


_DISPATCH: dict[str, Callable[[np.ndarray, dict[str, Any]], np.ndarray]] = {
    "none": _none,
    "gaussian": _gaussian,
    "tv": _tv,
    "nlm": _nlm,
    "bm3d": _bm3d,
}

#: Names added at run time by :func:`register_denoiser`, and which of them take whole stacks.
_REGISTERED: set[str] = set()
_STACK_AWARE: set[str] = set()
