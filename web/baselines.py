"""Reference implementations of the existing modes, for honest comparison.

Implemented locally rather than imported from `color-matcher` because that
package ships inside the ComfyUI embedded interpreter, whose PIL build conflicts
with the local one. These are faithful ports of the same maths.

References
----------
- Reinhard et al. 2001, "Color Transfer between Images", IEEE CG&A
  https://doi.org/10.1109/38.946629
- hm-mkl-hm pipeline inspired by color-matcher (MIT licence), reimplemented
  independently from first principles without copying any library source code.
  https://github.com/hahnec/color-matcher
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# CIELAB used by color-matcher's reinhard (via its own rgb2lab), matching the
# original Reinhard et al. 2001 "Color Transfer between Images" formulation
# which works in Ruderman's decorrelated lab space.
# ---------------------------------------------------------------------------
_RGB2LMS = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444],
], dtype=np.float64)

_LMS2LAB = (
    np.array([
        [1.0 / np.sqrt(3.0), 0.0, 0.0],
        [0.0, 1.0 / np.sqrt(6.0), 0.0],
        [0.0, 0.0, 1.0 / np.sqrt(2.0)],
    ])
    @ np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -2.0],
        [1.0, -1.0, 0.0],
    ])
)

_LMS2RGB = np.linalg.inv(_RGB2LMS)
_LAB2LMS = np.linalg.inv(_LMS2LAB)


def _rgb_to_ruderman(rgb: np.ndarray) -> np.ndarray:
    lms = rgb.astype(np.float64) @ _RGB2LMS.T
    lms = np.log10(np.maximum(lms, 1e-6))
    return lms @ _LMS2LAB.T


def _ruderman_to_rgb(lab: np.ndarray) -> np.ndarray:
    lms = lab @ _LAB2LMS.T
    lms = np.power(10.0, lms)
    return lms @ _LMS2RGB.T


def reinhard_lab(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Classic Reinhard mean/std transfer in Ruderman lab.

    Per channel:  x' = (x - mu_s) * (sigma_r / sigma_s) + mu_r
    A global affine map -- which is precisely why it cannot reorder tonality,
    and why an unbounded sigma ratio can invent saturation the reference never had.
    """
    s = _rgb_to_ruderman(np.clip(src, 1e-6, 1.0))
    r = _rgb_to_ruderman(np.clip(ref, 1e-6, 1.0))

    out = np.empty_like(s)
    for c in range(3):
        sc, rc = s[..., c], r[..., c]
        ss, rs = sc.std(), rc.std()
        gain = (rs / ss) if ss > 1e-9 else 1.0
        out[..., c] = (sc - sc.mean()) * gain + rc.mean()

    return np.clip(_ruderman_to_rgb(out), 0.0, 1.0).astype(np.float32)


def reinhard_gains(src: np.ndarray, ref: np.ndarray) -> dict:
    """Expose the per-channel sigma ratios reinhard would apply (diagnostics)."""
    s = _rgb_to_ruderman(np.clip(src, 1e-6, 1.0))
    r = _rgb_to_ruderman(np.clip(ref, 1e-6, 1.0))
    out = {}
    for c, nm in enumerate(("L", "alpha(y-b)", "beta(r-g)")):
        ss, rs = float(s[..., c].std()), float(r[..., c].std())
        out[nm] = dict(src_std=ss, ref_std=rs,
                       gain=(rs / ss if ss > 1e-9 else 1.0),
                       src_mean=float(s[..., c].mean()),
                       ref_mean=float(r[..., c].mean()))
    return out


# ---------------------------------------------------------------------------
# hm-mkl-hm (color-matcher's "histogram matching + MKL" composite)
# Faithful reimplementation so the Web bench can show it without importing the
# ComfyUI-bundled package (whose PIL build clashes with the local one).
# ---------------------------------------------------------------------------
def _channel_cdf_match(src_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
    """1-D histogram matching of one channel via CDF look-up.

    Standard algorithm: build the CDF of both channels, then for each input
    level find the output level with the same cumulative probability. This is
    what gives hm its greedy exact-match character -- and its banding, because
    a narrow reference CDF maps many source levels onto one output level.
    """
    src = np.clip(src_ch, 0.0, 1.0)
    ref = np.clip(ref_ch, 0.0, 1.0)
    nb = 256
    s_hist, _ = np.histogram(src.ravel(), bins=nb, range=(0, 1))
    r_hist, _ = np.histogram(ref.ravel(), bins=nb, range=(0, 1))
    s_cdf = np.cumsum(s_hist).astype(np.float64)
    r_cdf = np.cumsum(r_hist).astype(np.float64)
    s_cdf /= max(s_cdf[-1], 1e-12)
    r_cdf /= max(r_cdf[-1], 1e-12)
    r_cdf = np.maximum.accumulate(r_cdf)   # ensure monotone for inverse

    grid = (np.arange(nb) + 0.5) / nb
    # inverse CDF of the reference, evaluated at the source's CDF levels
    inv = np.interp(s_cdf, r_cdf, grid)
    lut = np.clip(inv, 0.0, 1.0).astype(np.float32)
    idx = np.clip((src * nb).astype(np.int32), 0, nb - 1)
    return lut[idx].reshape(src_ch.shape).astype(np.float32)


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel histogram matching in sRGB."""
    out = np.empty_like(src, dtype=np.float32)
    for c in range(3):
        out[..., c] = _channel_cdf_match(src[..., c], ref[..., c])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    M = (M + M.T) * 0.5
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _psd_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    M = (M + M.T) * 0.5
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def mkl_transfer(src: np.ndarray, ref: np.ndarray,
                 max_gain: float = 8.0) -> np.ndarray:
    """Monge-Kantorovich Linear optimal transport in sRGB.

    x' = (x - mu_s) @ T.T + mu_r  with T = Cs^-1/2 (Cs^1/2 Cr Cs^1/2)^1/2 Cs^-1/2

    Unlike reinhard this matches the joint covariance (correlation between
    channels included), but like reinhard it is one global affine map, so it
    also cannot route different colours to different destinations.
    """
    s = src.reshape(-1, 3).astype(np.float64)
    r = ref.reshape(-1, 3).astype(np.float64)
    mu_s, mu_r = s.mean(0), r.mean(0)
    cov_s = np.cov(s, rowvar=False) + np.eye(3) * 1e-6
    cov_r = np.cov(r, rowvar=False) + np.eye(3) * 1e-6

    Cs_sqrt = _psd_sqrt(cov_s)
    Cs_inv_sqrt = _psd_inv_sqrt(cov_s)
    T = Cs_inv_sqrt @ _psd_sqrt(Cs_sqrt @ cov_r @ Cs_sqrt) @ Cs_inv_sqrt
    T = (T + T.T) * 0.5
    U, S, Vt = np.linalg.svd(T)
    S = np.clip(S, 0.0, max_gain)
    T = U @ np.diag(S) @ Vt

    out = (s - mu_s) @ T.T + mu_r
    return np.clip(out.reshape(src.shape), 0.0, 1.0).astype(np.float32)


def hm_mkl_hm(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The ComfyUI Easy-Use 'hm-mkl-hm' mode, reimplemented locally."""
    a = histogram_match(src, ref)
    b = mkl_transfer(a, ref)
    return histogram_match(b, ref)
