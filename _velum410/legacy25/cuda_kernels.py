"""Optional CUDA kernels for Velum's per-pixel colour mathematics.

PyTorch is imported lazily and every public function returns ``None`` when CUDA is
unavailable. CPU remains the reference implementation. The CUDA path uses float64 for
CAM16 because J ranks are sensitive to last-bit tie changes; lowering CAM16 to float32
was measured to shift final pixels by up to 7/255.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import numpy as np

_MIN_PIXELS = int(os.environ.get("VELUM_CUDA_MIN_PIXELS", 1536 * 1024))
_MAX_CHUNK_PIXELS = int(os.environ.get("VELUM_CUDA_CHUNK_PIXELS", 4 * 1024 * 1024))
_TORCH = None
_CUDA_OK: bool | None = None
_BACKEND: ContextVar[str] = ContextVar("velum_backend", default="auto")


class BackendUnavailableError(RuntimeError):
    """Raised when the user explicitly requires a backend that cannot run."""


def _torch():
    global _TORCH, _CUDA_OK
    if _CUDA_OK is False:
        return None
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch
            _CUDA_OK = bool(torch.cuda.is_available())
        except Exception:
            _CUDA_OK = False
    return _TORCH if _CUDA_OK else None


def _required_torch() -> Any:
    """Typed view used only after `should_use` established CUDA availability."""
    return cast(Any, _torch())


def cuda_available() -> bool:
    """Whether a usable PyTorch CUDA device is available."""
    return _torch() is not None


def resolve_backend(requested: str) -> str:
    """Resolve auto/cpu/gpu to the backend used for CUDA-capable stages."""
    mode = str(requested or "auto").strip().lower()
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown compute backend: {requested!r}")
    if mode == "cpu":
        return "cpu"
    if cuda_available():
        return "gpu"
    if mode == "gpu":
        raise BackendUnavailableError(
            "GPU 模式不可用：未检测到可用的 NVIDIA CUDA / PyTorch CUDA 环境。"
            "请检查显卡驱动、CUDA 版 PyTorch 和 torch.cuda.is_available()。")
    return "cpu"


@contextmanager
def backend(requested: str):
    """Bind one grade call to an explicit, thread-safe compute backend."""
    effective = resolve_backend(requested)
    token = _BACKEND.set(effective)
    try:
        yield effective
    finally:
        _BACKEND.reset(token)


def should_use(array: np.ndarray) -> bool:
    mode = _BACKEND.get()
    if mode == "cpu":
        return False
    # An explicit/auto-resolved GPU backend prioritises CUDA as requested. Outside a
    # grade backend context, retain the historical size threshold for library callers.
    if mode == "gpu":
        return _torch() is not None
    return np.asarray(array).shape[0] * np.asarray(array).shape[1] >= _MIN_PIXELS \
        and _torch() is not None


def active_backend() -> str:
    return _BACKEND.get()


def srgb_to_jmh(rgb01: np.ndarray, constants: dict) -> np.ndarray | None:
    """Float64 CAM16 forward on CUDA; algebraically matches the CPU specialisation."""
    if not should_use(rgb01):
        return None
    torch = _required_torch()
    try:
        dev = torch.device("cuda")
        matrix = torch.as_tensor(constants["to_cone"], device=dev,
                                 dtype=torch.float64)
        src = np.ascontiguousarray(rgb01)
        h, w = src.shape[:2]
        rows = max(1, _MAX_CHUNK_PIXELS // max(w, 1))
        result = np.empty((h, w, 3), np.float32)
        for y0 in range(0, h, rows):
            x = torch.as_tensor(src[y0:y0 + rows], device=dev,
                                dtype=torch.float64).clamp_(0.0, 1.0)
            linear = torch.where(x <= 0.04045, x / 12.92,
                                 ((x + 0.055) / 1.055) ** 2.4)
            cone = linear @ matrix.T
            magnitude = (float(constants["f_l"]) * cone.abs() / 100.0) ** 0.42
            adapted = torch.sign(cone) * 400.0 * magnitude / (magnitude + 27.13) + 0.1
            r, g, b = adapted.unbind(-1)
            a = r - 12.0 * g / 11.0 + b / 11.0
            bb = (r + g - 2.0 * b) / 9.0
            achromatic = ((2.0 * r + g + b / 20.0 - 0.305)
                          * float(constants["n_bb"])
                          )
            J = 100.0 * (achromatic.clamp_min(0.0) / float(constants["a_w"])) \
                ** float(constants["exp_j"])
            hue_rad = torch.atan2(bb, a)
            hue = torch.rad2deg(hue_rad).remainder(360.0)
            eccentricity = 0.25 * (torch.cos(hue_rad + 2.0) + 3.8)
            denom = r + g + (21.0 / 20.0) * b
            denom = torch.where(denom.abs() < 1e-12,
                                torch.full_like(denom, 1e-12), denom)
            t = (float(constants["t_gain"]) * eccentricity
                 * torch.hypot(a, bb) / denom)
            alpha = t.clamp_min(0.0) ** 0.9 * float(constants["alpha_gain"])
            C = alpha * torch.sqrt(J / 100.0)
            out = torch.stack((J, C * float(constants["m_gain"]), hue), -1)
            result[y0:y0 + rows] = out.to(torch.float32).cpu().numpy()
        return result
    except Exception:
        return None


def jmh_to_srgb(jmh: np.ndarray, constants: dict) -> np.ndarray | None:
    """Float64 CAM16 inverse on CUDA, specialised to Velum's fixed conditions."""
    if not should_use(jmh):
        return None
    torch = _required_torch()
    try:
        dev = torch.device("cuda")
        x = torch.as_tensor(np.ascontiguousarray(jmh), device=dev,
                            dtype=torch.float64)
        J = x[..., 0]
        M = x[..., 1].clamp_min(0.0)
        hue = x[..., 2].remainder(360.0)
        C = M / float(constants["m_gain"])
        alpha = C / torch.sqrt(J.clamp_min(1e-12) / 100.0)
        t = (alpha / float(constants["alpha_gain"])) ** (1.0 / 0.9)
        hr = torch.deg2rad(hue)
        sin_h, cos_h = torch.sin(hr), torch.cos(hr)
        eccentricity = 0.25 * (torch.cos(hr + 2.0) + 3.8)
        A = float(constants["a_w"]) * (J / 100.0) \
            ** (1.0 / float(constants["exp_j"]))
        p1 = float(constants["t_gain"]) * eccentricity / t.clamp_min(1e-12)
        p2 = A / float(constants["n_bb"]) + 0.305
        p3 = 21.0 / 20.0
        numerator = p2 * (2.0 + p3) * (460.0 / 1403.0)

        safe_sin = torch.where(sin_h.abs() < 1e-12,
                               torch.full_like(sin_h, 1e-12), sin_h)
        safe_cos = torch.where(cos_h.abs() < 1e-12,
                               torch.full_like(cos_h, 1e-12), cos_h)
        use_b = sin_h.abs() >= cos_h.abs()
        b_first = numerator / (
            p1 / safe_sin
            + (2.0 + p3) * (220.0 / 1403.0) * cos_h / safe_sin
            - 27.0 / 1403.0 + p3 * (6300.0 / 1403.0))
        a_first = numerator / (
            p1 / safe_cos + (2.0 + p3) * (220.0 / 1403.0)
            - ((27.0 / 1403.0) - p3 * (6300.0 / 1403.0))
            * sin_h / safe_cos)
        a = torch.where(use_b, b_first * cos_h / safe_sin, a_first)
        b = torch.where(use_b, b_first, a_first * sin_h / safe_cos)
        zero = t <= 0.0
        a = torch.where(zero, torch.zeros_like(a), a)
        b = torch.where(zero, torch.zeros_like(b), b)

        compressed = torch.stack((
            (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0,
            (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0,
            (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0,
        ), -1)
        u = compressed - 0.1
        ratio = 27.13 * u.abs() / (400.0 - u.abs()).clamp_min(1e-12)
        adapted_cone = torch.sign(u) * 100.0 / float(constants["f_l"]) \
            * ratio ** (1.0 / 0.42)

        inverse_matrix = torch.linalg.inv(torch.as_tensor(
            constants["to_cone"], device=dev, dtype=torch.float64))
        linear = adapted_cone @ inverse_matrix.T
        srgb = torch.where(linear <= 0.0031308, linear * 12.92,
                           1.055 * linear.clamp_min(0.0) ** (1.0 / 2.4) - 0.055)
        return srgb.to(torch.float32).cpu().numpy()
    except Exception:
        return None


def evaluate_mapping(src_lab: np.ndarray, beta: np.ndarray,
                     exponents: tuple[tuple[int, ...], ...]) -> np.ndarray | None:
    """Evaluate the degree-3 mapping on CUDA, preserving float64 arithmetic."""
    if not should_use(src_lab):
        return None
    torch = _required_torch()
    try:
        dev = torch.device("cuda")
        h, w = src_lab.shape[:2]
        b = torch.as_tensor(np.ascontiguousarray(beta), device=dev,
                            dtype=torch.float64)
        rows = max(1, _MAX_CHUNK_PIXELS // max(w, 1))
        result = np.empty((h, w, 3), np.float64)
        for y0 in range(0, h, rows):
            y1 = min(y0 + rows, h)
            lab = torch.as_tensor(np.ascontiguousarray(src_lab[y0:y1]),
                                  device=dev, dtype=torch.float64)
            yy = torch.linspace(y0 / max(h - 1, 1) * 2.0 - 1.0,
                                (y1 - 1) / max(h - 1, 1) * 2.0 - 1.0,
                                y1 - y0, device=dev, dtype=torch.float64)
            xx = torch.linspace(-1.0, 1.0, w, device=dev, dtype=torch.float64)
            y, x = torch.meshgrid(yy, xx, indexing="ij")
            z = torch.stack((lab[..., 0], lab[..., 1] / 0.25,
                             lab[..., 2] / 0.25, x, y), -1).reshape(-1, 5)
            # Accumulate terms directly: no N×56 feature matrix and bounded memory.
            pred = torch.zeros((z.shape[0], 3), device=dev, dtype=torch.float64)
            for i, exponent in enumerate(exponents):
                term = torch.ones(z.shape[0], device=dev, dtype=torch.float64)
                for axis, power in enumerate(exponent):
                    if power:
                        term.mul_(z[:, axis] ** power)
                pred.add_(term[:, None] * b[i])
            result[y0:y1] = pred.reshape(y1 - y0, w, 3).cpu().numpy()
        return result
    except Exception:
        return None


def project_to_convex_hull(points: np.ndarray,
                           hull: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact Euclidean projection onto a 2-D convex polygon on CUDA.

    Uses the same float32 cross-product membership test, tolerance and nearest-segment
    formula as the CPU implementation. It is worthwhile only for large, many-vertex
    hulls; small polygons remain faster on CPU.
    """
    array = np.asarray(points, np.float32)
    polygon = np.asarray(hull, np.float32)
    if (active_backend() != "gpu" or _torch() is None
            or array[..., 0].size < 512 * 1024 or polygon.shape[0] < 32):
        return None
    torch = _required_torch()
    try:
        dev = torch.device("cuda")
        shape = array.shape
        flat = array.reshape(-1, 2)
        a = torch.as_tensor(polygon, device=dev)
        v = torch.roll(a, shifts=-1, dims=0) - a
        edge_dot = (a * v).sum(1)
        edge_len2 = (v * v).sum(1).clamp_min(1e-20)
        result = np.empty_like(flat)
        moved = np.empty(flat.shape[0], bool)
        block = 131072
        for start in range(0, flat.shape[0], block):
            q = torch.as_tensor(flat[start:start + block], device=dev)
            cross = (v[:, 0][None, :] * (q[:, 1, None] - a[:, 1][None, :])
                     - v[:, 1][None, :] * (q[:, 0, None] - a[:, 0][None, :]))
            inside = torch.all(cross >= -1e-6, 1) | torch.all(cross <= 1e-6, 1)
            output = q.clone()
            ids = torch.where(~inside)[0]
            if ids.numel():
                x = q[ids]
                t = (((x @ v.T) - edge_dot) / edge_len2).clamp(0.0, 1.0)
                candidate = a[None, :, :] + t[:, :, None] * v[None, :, :]
                nearest = torch.argmin(((x[:, None, :] - candidate) ** 2).sum(2), 1)
                output[ids] = candidate[
                    torch.arange(ids.numel(), device=dev), nearest]
            end = start + q.shape[0]
            result[start:end] = output.cpu().numpy()
            moved[start:end] = (~inside).cpu().numpy()
        return result.reshape(shape), moved.reshape(shape[:-1])
    except Exception:
        return None
