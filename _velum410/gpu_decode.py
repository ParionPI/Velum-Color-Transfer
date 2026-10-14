"""Bounded CUDA implementation of the legacy anchored sRGB decode.

The caller owns the CPU fallback.  This module only returns a complete result
when every tile has succeeded; a failed call never exposes a partial array.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import gpu_mapping as gm
from .legacy25 import oklab


_TILE_FALLBACK = 262_144
_MIN_FREE_BYTES = 256 * 1024 * 1024


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _state() -> dict[str, Any] | None:
    # gpu_mapping deliberately keeps this ContextVar private, but this helper
    # is a sibling implementation and must enrich the same per-transfer audit.
    return gm._STATE.get()


def _fail(state: dict[str, Any], reason: str, strict: bool) -> None:
    state["decode_backend"] = "cpu"
    state["decode_fallback_reason"] = reason
    if strict:
        raise gm.BackendUnavailableError(f"CUDA anchored decode unavailable: {reason}")


def _valid(name: str, value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite float32 data with shape HxWx3")
    return array


def _mat3(torch: Any, matrix: np.ndarray, like: Any) -> Any:
    # ``_mul3`` computes row-vector ``value @ matrix.T`` explicitly, matching
    # the NumPy legacy implementation's ``lab @ _M2_INV.T`` convention.
    return torch.as_tensor(matrix.astype(np.float32), device=like.device, dtype=torch.float32)


def _mul3(torch: Any, value: Any, matrix: Any) -> Any:
    # Explicit scalar products preserve the intended channelwise 3x3 path and
    # avoid torch.matmul (and any process-wide TF32 setting changes).
    return torch.stack(
        (
            value[..., 0] * matrix[0, 0] + value[..., 1] * matrix[0, 1] + value[..., 2] * matrix[0, 2],
            value[..., 0] * matrix[1, 0] + value[..., 1] * matrix[1, 1] + value[..., 2] * matrix[1, 2],
            value[..., 0] * matrix[2, 0] + value[..., 1] * matrix[2, 1] + value[..., 2] * matrix[2, 2],
        ),
        dim=-1,
    )


def _decode_lab(torch: Any, lab: Any, m2_inv: Any, m1_inv: Any) -> Any:
    lms_root = _mul3(torch, lab, m2_inv)
    lms = lms_root * lms_root * lms_root
    linear = _mul3(torch, lms, m1_inv)
    sign = torch.sign(linear)
    absolute = torch.abs(linear)
    low = absolute * 12.92
    high = 1.055 * torch.pow(absolute, 1.0 / 2.4) - 0.055
    return sign * torch.where(absolute <= 0.0031308, low, high)


def _tile_pixels() -> int:
    limit = gm._tile_pixels()
    return max(1, limit)


def try_anchored_decode(
    mapped: np.ndarray, baseline_lab: np.ndarray, baseline_rgb: np.ndarray
) -> np.ndarray | None:
    """Try GPU decode, returning ``None`` so the caller can run CPU decode."""
    state = _state()
    info = gm.current_info() if state is None else dict(state)
    requested = str(info.get("requested", "cpu")).lower()
    strict = requested == "gpu"
    started = time.perf_counter()
    if state is not None:
        state.setdefault("decode_backend", "cpu")
        state.setdefault("decode_fallback_reason", None)
        state["decode_seconds"] = 0.0
    # Mapping may have selected CPU because of an explicit CPU mode, a small
    # image, or a probe failure.  Do not cold-probe CUDA a second time.
    if info.get("effective") != "gpu":
        if strict:
            _fail(state or {"decode_backend": "cpu"}, "mapping_not_gpu", True)
        return None
    arrays = (np.asarray(mapped), np.asarray(baseline_lab), np.asarray(baseline_rgb))
    if arrays[0].ndim != 3 or arrays[0].shape[2] != 3:
        if strict:
            _fail(state or {"decode_backend": "cpu"}, "invalid_shape", True)
        return None
    shape = tuple(int(x) for x in arrays[0].shape)
    try:
        if min(shape)<=0 or shape[1]>_tile_pixels(): raise ValueError('empty_or_overwide_image')
        mapped_a = _valid("mapped", arrays[0], shape)
        lab_a = _valid("baseline_lab", arrays[1], shape)
        rgb_a = _valid("baseline_rgb", arrays[2], shape)
        torch, probe_reason = gm._probe()
        if torch is None:
            raise RuntimeError(probe_reason)
        free, _total = torch.cuda.mem_get_info()
        pixels = shape[0] * shape[1]
        tile_limit = _tile_pixels()
        if int(free) < max(_MIN_FREE_BYTES, min(pixels, tile_limit) * 3 * 4 * 8):
            raise RuntimeError("insufficient_cuda_memory")
        device = torch.device("cuda")
        m2_inv = _mat3(torch, oklab._M2_INV, torch.empty((), device=device, dtype=torch.float32))
        m1_inv = _mat3(torch, oklab._M1_INV, torch.empty((), device=device, dtype=torch.float32))
        output = np.empty(shape, dtype=np.float32)
        rows = max(1, min(shape[0], tile_limit // max(shape[1], 1)))
        for y0 in range(0, shape[0], rows):
            y1 = min(shape[0], y0 + rows)
            m = torch.as_tensor(np.ascontiguousarray(mapped_a[y0:y1]), device=device, dtype=torch.float32)
            b = torch.as_tensor(np.ascontiguousarray(lab_a[y0:y1]), device=device, dtype=torch.float32)
            base = torch.as_tensor(np.ascontiguousarray(rgb_a[y0:y1]), device=device, dtype=torch.float32)
            decoded = base + (_decode_lab(torch, m, m2_inv, m1_inv) - _decode_lab(torch, b, m2_inv, m1_inv))
            if decoded.dtype != torch.float32 or tuple(decoded.shape) != (y1 - y0, shape[1], 3):
                raise RuntimeError("GPU anchored decode returned invalid tile")
            tile = decoded.cpu().numpy()
            torch.cuda.synchronize()
            if tile.dtype != np.float32 or not np.all(np.isfinite(tile)):
                raise RuntimeError("GPU anchored decode returned non-finite tile")
            output[y0:y1] = tile
        if not np.all(np.isfinite(output)):
            raise RuntimeError("GPU anchored decode output is non-finite")
        if state is not None:
            state.update(decode_backend="gpu", decode_fallback_reason=None,
                         decode_seconds=float(time.perf_counter() - started))
        return output
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if state is not None:
            state["decode_seconds"] = float(time.perf_counter() - started)
            _fail(state, _reason(exc), strict)
        elif strict:
            raise gm.BackendUnavailableError(f"CUDA anchored decode unavailable: {_reason(exc)}")
        return None


__all__ = ["try_anchored_decode"]
