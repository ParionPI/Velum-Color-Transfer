"""Oklab perceptual colour space + sRGB transfer functions.

Everything here is float64/float32 and vectorised over (..., 3) arrays.
Oklab is used because L is well decoupled from (a, b) and the space is far more
perceptually uniform than RGB / HSV / CIELAB in the blue region, which is what
lets us push colour aggressively without visible tearing.
"""

from __future__ import annotations

import numpy as np

# linear sRGB -> LMS  (Bjorn Ottosson's Oklab)
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

# LMS' (cube-rooted) -> Lab
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB EOTF. Sign-safe so slightly-negative values survive round trips."""
    c = np.asarray(c, dtype=np.float32)
    s = np.sign(c)
    a = np.abs(c)
    lo = a / 12.92
    hi = ((a + 0.055) / 1.055) ** 2.4
    return (s * np.where(a <= 0.04045, lo, hi)).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF. Sign-safe."""
    c = np.asarray(c, dtype=np.float32)
    s = np.sign(c)
    a = np.abs(c)
    lo = a * 12.92
    hi = 1.055 * (a ** (1.0 / 2.4)) - 0.055
    return (s * np.where(a <= 0.0031308, lo, hi)).astype(np.float32)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ _M1.T.astype(np.float32)
    lms = np.cbrt(lms)  # np.cbrt is correct for negatives
    return (lms @ _M2.T.astype(np.float32)).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    lms = lab @ _M2_INV.T.astype(np.float32)
    lms = lms ** 3
    return (lms @ _M1_INV.T.astype(np.float32)).astype(np.float32)


def srgb_to_oklab(rgb01: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb01))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


# ---------------------------------------------------------------------------
# torch twins, for the GPU-resident pipeline.
#
# Identical maths to the numpy functions above; the matrices are the SAME
# constants, so a tensor round-trip matches the numpy path to fp32. These are
# used only when the Oklab image is kept resident on a CUDA device across stages;
# the numpy versions remain the reference and the fallback.
# ---------------------------------------------------------------------------
def srgb_to_linear_t(c):
    import torch
    s = torch.sign(c)
    a = torch.abs(c)
    lo = a / 12.92
    hi = ((a + 0.055) / 1.055) ** 2.4
    return s * torch.where(a <= 0.04045, lo, hi)


def linear_to_srgb_t(c):
    import torch
    s = torch.sign(c)
    a = torch.abs(c)
    lo = a * 12.92
    hi = 1.055 * (a ** (1.0 / 2.4)) - 0.055
    return s * torch.where(a <= 0.0031308, lo, hi)


def _mat_t(M, like):
    import torch
    return torch.as_tensor(M, dtype=like.dtype, device=like.device)


def linear_to_oklab_t(rgb):
    import torch
    lms = rgb @ _mat_t(_M1.T, rgb)
    lms = torch.sign(lms) * torch.abs(lms) ** (1.0 / 3.0)   # cbrt, sign-safe
    return lms @ _mat_t(_M2.T, rgb)


def oklab_to_linear_t(lab):
    lms = lab @ _mat_t(_M2_INV.T, lab)
    lms = lms ** 3
    return lms @ _mat_t(_M1_INV.T, lab)


def srgb_to_oklab_t(rgb01):
    return linear_to_oklab_t(srgb_to_linear_t(rgb01))


def oklab_to_srgb_t(lab):
    return linear_to_srgb_t(oklab_to_linear_t(lab))


def chroma_t(lab):
    import torch
    return torch.hypot(lab[..., 1], lab[..., 2])


def hue_t(lab):
    import torch
    return torch.atan2(lab[..., 2], lab[..., 1])


def chroma(lab: np.ndarray) -> np.ndarray:
    return np.hypot(lab[..., 1], lab[..., 2])


def hue(lab: np.ndarray) -> np.ndarray:
    return np.arctan2(lab[..., 2], lab[..., 1])


def lch_to_lab(L: np.ndarray, C: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1).astype(np.float32)
