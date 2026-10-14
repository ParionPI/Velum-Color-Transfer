"""Optional, bounded CUDA evaluator for the legacy continuous mapping.

This module deliberately has no torch import at module import time.  The CPU
evaluator remains the owner of fallback behaviour; this helper only returns a
complete GPU result when every tile has succeeded.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
import time
from typing import Any, Iterator

import numpy as np


_MIN_PIXELS = 262_144
_DEFAULT_TILE_PIXELS = 262_144
_PROBE_LOCK = threading.Lock()
_PROBE_DONE = False
_PROBE_TORCH: Any = None
_PROBE_REASON = "not_probed"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "velum_gpu_mapping_state", default=None
)


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested CUDA evaluation cannot run."""


def _new_state(requested: str) -> dict[str, Any]:
    return {
        "requested": requested,
        "effective": "not_used",
        "fallback_reason": None,
        "device": None,
    }


def _probe() -> tuple[Any, str]:
    """Lazily return ``(torch, reason)`` for a CUDA-capable torch install."""
    global _PROBE_DONE, _PROBE_TORCH, _PROBE_REASON
    if _PROBE_DONE:
        return _PROBE_TORCH, _PROBE_REASON
    with _PROBE_LOCK:
        if _PROBE_DONE:
            return _PROBE_TORCH, _PROBE_REASON
        try:
            import torch  # type: ignore

            if not bool(torch.cuda.is_available()):
                reason = "cuda_unavailable"
                torch_obj = None
            else:
                # Initialisation is intentionally tiny and does not alter any
                # process-wide torch flags or the current device.
                torch.empty(1, device="cuda", dtype=torch.float64)
                reason = "available"
                torch_obj = torch
        except Exception as exc:
            torch_obj = None
            reason = f"{type(exc).__name__}: {str(exc)[:240]}"
        _PROBE_TORCH, _PROBE_REASON, _PROBE_DONE = torch_obj, reason, True
    return _PROBE_TORCH, _PROBE_REASON


def current_info() -> dict[str, Any]:
    """Return a copy of the current call's diagnostics without probing CUDA."""
    state = _STATE.get()
    return dict(state) if state is not None else _new_state("cpu")


@contextmanager
def session(requested: str = "auto") -> Iterator[dict[str, Any]]:
    """Bind one mapping call to an isolated ``auto``, ``cpu`` or ``gpu`` mode."""
    mode = str(requested).strip().lower()
    if mode not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unknown GPU mapping backend: {requested!r}")
    state = _new_state(mode)
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def _state_for_call() -> dict[str, Any]:
    state = _STATE.get()
    if state is None:
        # Private callers outside a public transfer are safely CPU-only.
        state = _new_state("cpu")
    return state


def _tile_pixels() -> int:
    raw = os.environ.get("VELUM_GPU_TILE_PIXELS", str(_DEFAULT_TILE_PIXELS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TILE_PIXELS
    return max(1, min(value, _DEFAULT_TILE_PIXELS))


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _fail(state: dict[str, Any], reason: str, strict: bool) -> None:
    state["fallback_reason"] = reason
    state["effective"] = "not_used" if strict else "cpu"
    if strict:
        raise BackendUnavailableError(f"CUDA mapping unavailable: {reason}")


def _validate(src_lab: np.ndarray, beta: np.ndarray, exponents: Any) -> tuple[int, int]:
    src = np.asarray(src_lab)
    coefficients = np.asarray(beta)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("src_lab must have shape HxWx3")
    if not np.issubdtype(src.dtype, np.floating) or not np.all(np.isfinite(src)):
        raise ValueError("src_lab must be finite floating-point data")
    if coefficients.shape != (56, 3) or not np.issubdtype(coefficients.dtype, np.floating):
        raise ValueError("beta must have shape (56, 3) and floating-point dtype")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("beta must be finite")
    if len(exponents) != 56:
        raise ValueError("exponents must contain 56 terms")
    return int(src.shape[0]), int(src.shape[1])


def _evaluate(torch: Any, src_lab: np.ndarray, beta: np.ndarray,
              exponents: Any, state: dict[str, Any]) -> np.ndarray:
    started = time.perf_counter()
    h, w = src_lab.shape[:2]
    tile_pixels = _tile_pixels()
    rows = max(1, tile_pixels // max(w, 1))
    # Avoid an unexpectedly large temporary when a very narrow image is used.
    rows = min(rows, h)
    device = torch.device("cuda")
    coefficients = torch.as_tensor(np.ascontiguousarray(beta), device=device,
                                    dtype=torch.float64)
    output = np.empty((h, w, 3), dtype=np.float64)
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        lab32 = torch.as_tensor(np.ascontiguousarray(src_lab[y0:y1]),
                                device=device)
        # Keep the source float32 divide semantics before promoting to float64.
        z0 = lab32[..., 0].to(torch.float64).reshape(-1)
        z1 = (lab32[..., 1] / 0.25).to(torch.float64).reshape(-1)
        z2 = (lab32[..., 2] / 0.25).to(torch.float64).reshape(-1)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float64)
        xx = torch.arange(w, device=device, dtype=torch.float64)
        x = (xx / max(w - 1, 1) * 2.0 - 1.0).expand(y1 - y0, w).reshape(-1)
        y = (yy / max(h - 1, 1) * 2.0 - 1.0).unsqueeze(1).expand(y1 - y0, w).reshape(-1)
        vectors = (z0, z1, z2, x, y)
        powers = tuple((v, v * v, v * v * v) for v in vectors)
        pred = torch.zeros((vectors[0].numel(), 3), device=device,
                           dtype=torch.float64)
        ones = torch.ones_like(z0)
        for index, exponent in enumerate(exponents):
            term = ones
            for axis, power in enumerate(exponent):
                if power:
                    if term is ones:
                        term = powers[axis][power - 1]
                    else:
                        term = term * powers[axis][power - 1]
            pred.add_(term.unsqueeze(1) * coefficients[index])
        tile = pred.cpu().numpy()
        if tile.shape != ((y1 - y0) * w, 3) or not np.all(np.isfinite(tile)):
            raise RuntimeError("GPU mapping returned non-finite or invalid tile")
        output[y0:y1] = tile.reshape(y1 - y0, w, 3)
    state["seconds"] = float(time.perf_counter() - started)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("GPU mapping output is non-finite")
    return output


def try_evaluate(src_lab: np.ndarray, beta: np.ndarray,
                 exponents: Any) -> np.ndarray | None:
    """Attempt CUDA evaluation, returning ``None`` for non-strict fallback."""
    state = _state_for_call()
    mode = state["requested"]
    if mode == "cpu":
        state["effective"] = "cpu"
        state["device"] = None
        state["fallback_reason"] = None
        return None
    if os.environ.get("VELUM_DISABLE_GPU") == "1":
        _fail(state, "disabled_by_environment", mode == 'gpu')
        return None
    h, w = _validate(src_lab, beta, exponents)
    pixels = h * w
    if h == 0 or w == 0 or w > _tile_pixels():
        _fail(state, 'empty_or_overwide_image', mode == 'gpu')
        return None
    if mode == "auto" and pixels < _MIN_PIXELS:
        state.update(effective="cpu", device=None, fallback_reason="small_image")
        return None
    try:
        torch, probe_reason = _probe()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _fail(state, _reason(exc), mode == "gpu")
        return None
    if torch is None:
        _fail(state, probe_reason, mode == "gpu")
        return None
    try:
        # A conservative estimate prevents avoidable allocator pressure.
        free, _total = torch.cuda.mem_get_info()
        estimate = min(pixels, _tile_pixels()) * 32 * 8
        if int(free) < max(256 * 1024 * 1024, estimate * 4):
            raise RuntimeError("insufficient_cuda_memory")
        state["device"] = str(torch.cuda.get_device_name())
        result = _evaluate(torch, np.asarray(src_lab), np.asarray(beta), exponents, state)
        state["effective"] = "gpu"
        state["fallback_reason"] = None
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _fail(state, _reason(exc), mode == "gpu")
        return None


__all__ = ["BackendUnavailableError", "current_info", "session", "try_evaluate", "_probe"]
