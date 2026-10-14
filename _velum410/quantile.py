"""Velum 4.05: fitted-L mean-residual calibration; original 3.53 OFF path retained.

Lightness: the saved J values and their dynamic range are the reference
image's J distribution, delivered through a continuous inverse-CDF quantile
ladder; the rank order is the source image's. Chroma: free 3D sliced
transport in CAM16-UCS, gamut-projected at fixed J along the neutral axis.

Differences from the locked experiment baseline (velum-quantile-tone-v1):
only one deliberate change - XYZ->sRGB uses the exact inverse of the
forward sRGB matrix instead of colour-science's default approximation
(white round-trip 1.00004358 -> 2.2e-16). No other constants touched.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import colour
from numba import get_num_threads, njit, prange
from .color import RGBEncoding, ViewingConditions, D65, srgb_matrix
from .appearance import Appearance
from .core import palette, palette_counts, pack
from .relative import lightness_ucs
from .distribution import weighted_quantile_match
from .support import JointSupport
from .local_support import LocalReferenceSupport
from .deband import DebandOptions

_SRGB_INV = np.linalg.inv(srgb_matrix())
# Machine-epsilon-scaled gamut tolerance: absorbs float noise at the sRGB
# boundary (e.g. pure black returning -1e-16). Not an aesthetic constant;
# genuine out-of-gamut chroma is orders of magnitude larger.
_GAMUT_TOL = 4096 * np.finfo(np.float64).eps
PROJECTION_BISECTIONS = 24
TRANSPORT_ROTATIONS = 32


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Exact inverse of RGBEncoding.from_icc(None) TRCs. NO clipping:
    negative channels stay negative (12.92*x branch is sign-preserving and
    monotone), so the gamut boundary check in project() sees the truth.
    Clipping here would silently pass out-of-gamut targets as valid."""
    linear = np.asarray(linear, np.float64)
    low = linear <= 0.0031308
    safe = np.where(low, 1.0, linear)          # never raise negative**power
    return np.where(low, 12.92 * linear, 1.055 * safe ** (1 / 2.4) - 0.055)


def rgb_float(features: np.ndarray) -> np.ndarray:
    """CAM16-UCS (J', a', b') -> nonlinear sRGB float via the exact matrix."""
    jmh = colour.CAM16UCS_to_JMh_CAM16(np.asarray(features, np.float64))
    spec = colour.CAM_Specification_CAM16(J=jmh[:, 0], M=jmh[:, 1], h=jmh[:, 2])
    xyz = colour.CAM16_to_XYZ(spec, D65 * 100, 64, 20) / 100.0
    return _srgb_encode(xyz @ _SRGB_INV.T)


def directions(count: int = TRANSPORT_ROTATIONS) -> list[np.ndarray]:
    ds = [np.eye(3)[i] for i in range(3)]
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(count - 3):
        z = 1 - 2 * (i + 0.5) / (count - 3)
        r = np.sqrt(max(0.0, 1 - z * z))
        theta = i * golden
        ds.append(np.array([r * np.cos(theta), r * np.sin(theta), z]))
    return ds


@njit(cache=True, nogil=True, parallel=True)
def _apply_slice(x, mapped, projection, direction):
    for i in prange(len(x)):
        shift = mapped[i] - projection[i]
        for channel in range(3):
            x[i, channel] = x[i, channel] + shift * direction[channel]


def sliced3(source, reference, source_weights, reference_weights,
            count: int = TRANSPORT_ROTATIONS) -> np.ndarray:
    """Free 3D weighted sliced transport; identity when the clouds match."""
    x = np.array(source, dtype=np.float64, copy=True)
    for d in directions(count):
        p = x @ d
        m = weighted_quantile_match(p, source_weights, reference @ d,
                                    reference_weights)
        _apply_slice(x, m, p, d)
    return x


def quantile_j(source_j, source_counts, reference_j, reference_counts):
    """Continuous inverse-CDF ladder: source J rank -> reference J value.

    Exact identity when the source and reference J distributions coincide
    (np.interp returns knot values exactly).
    """
    su = (np.cumsum(source_counts) - source_counts / 2) / np.sum(source_counts)
    ru = (np.cumsum(reference_counts) - reference_counts / 2) / np.sum(reference_counts)
    out = np.interp(su, np.r_[0.0, ru, 1.0], np.r_[reference_j[0], reference_j, reference_j[-1]])
    out[0] = reference_j[0]
    out[-1] = reference_j[-1]
    return out


def project_float(features: np.ndarray, cam: Appearance):
    """Original fixed-J gamut realization, stopped BEFORE 8-bit encoding.

    Returns (floating RGB in feature order, per-color chroma scale alpha in
    [0, 1]). J is never changed here; alpha < 1 only where the unscaled
    chroma leaves sRGB.
    """
    features = np.asarray(features, np.float64)
    if features.ndim != 2 or features.shape[1] != 3 or not np.all(np.isfinite(features)):
        raise ValueError('Expected finite N x 3 CAM16-UCS targets.')
    if not len(features):
        return np.empty((0, 3), np.float64), np.empty(0, np.float64)
    rgb = rgb_float(features)
    valid = np.all((rgb >= -_GAMUT_TOL) & (rgb <= 1 + _GAMUT_TOL), axis=1)
    alpha = np.ones(len(features))
    invalid = np.flatnonzero(~valid)
    if len(invalid):
        selected = features[invalid]
        jp = selected[:, 0]
        j = jp / (1.7 - 0.007 * jp)
        neutral = cam.neutral_ab(j)
        delta = selected[:, 1:] - neutral
        lo = np.zeros(len(invalid))
        hi = np.ones(len(invalid))
        for _ in range(PROJECTION_BISECTIONS):
            mid = (lo + hi) / 2
            trial = np.c_[jp, neutral + mid[:, None] * delta]
            x = rgb_float(trial)
            ok = np.all((x >= -_GAMUT_TOL) & (x <= 1 + _GAMUT_TOL), axis=1)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        rgb[invalid] = rgb_float(np.c_[jp, neutral + lo[:, None] * delta])
        alpha[invalid] = lo
    if not np.all(np.isfinite(rgb)):
        raise AssertionError('Projection returned nonfinite RGB.')
    if np.any(rgb < -2e-7) or np.any(rgb > 1 + 2e-7):
        raise AssertionError(f'Projection left sRGB: [{rgb.min()}, {rgb.max()}]')
    return rgb, alpha


def project(features: np.ndarray, cam: Appearance):
    """Legacy public encoder: identical to 3.53 when debanding is disabled."""
    rgb, alpha = project_float(features, cam)
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), alpha


@dataclass
class QuantileResult:
    rgb: np.ndarray            # uint8 output image, source layout
    colors: np.ndarray | None  # off-only atom representation; enabled output is per-pixel
    order: np.ndarray          # source-J sort order (palette -> ladder)
    counts: np.ndarray         # source palette pixel counts, palette order
    target_j: np.ndarray | None  # off-only reference ladder; not a final-J claim when enabled
    alpha: np.ndarray | None   # off-only per-atom gamut scale
    stats: dict


def transfer(source, reference, source_encoding=None, reference_encoding=None,
             viewing=ViewingConditions(), deband=DebandOptions()) -> QuantileResult:
    """Quantile-J + free chroma transfer. source may be uint8 sRGB or float64
    linear (with RGBEncoding.linear_srgb()); reference must be uint8 sRGB."""
    if not isinstance(deband, DebandOptions):
        raise TypeError('deband must be a DebandOptions instance.')
    if source_encoding is None:
        source_encoding = (RGBEncoding.linear_srgb() if source.dtype == np.float64
                           else RGBEncoding.from_icc(None))
    reference_encoding = reference_encoding or RGBEncoding.from_icc(None)
    started = perf_counter()
    scolors, sidx, scounts, _ = palette(source)
    rcolors, rcounts = palette_counts(reference)
    t1 = perf_counter()
    cam = Appearance(viewing)
    # Both callers require absolute UCS chroma. The previous describe() calls
    # computed (and discarded) a neutral-axis solve, then repeated raw().
    source_xyz = source_encoding.xyz(scolors)
    sj, sab = cam.raw(source_xyz)
    if not deband.active:
        del source_xyz
    rj, rab = cam.raw(reference_encoding.xyz(rcolors))
    so = np.argsort(sj, kind='stable')
    ro = np.argsort(rj, kind='stable')
    t2 = perf_counter()
    ref_features = np.c_[lightness_ucs(rj[ro]), rab[ro]]
    free = sliced3(np.c_[lightness_ucs(sj[so]), sab[so]],
                   ref_features,
                   scounts[so], rcounts[ro])
    ladder = quantile_j(sj[so], scounts[so], rj[ro], rcounts[ro])
    target = free.copy()
    target[:, 0] = lightness_ucs(ladder)
    correction_started = perf_counter()
    support = JointSupport(ref_features, rcounts[ro])
    corrected, support_changed = support.correct(target)
    local_started = perf_counter()
    local = LocalReferenceSupport(ref_features, workers=get_num_threads())
    corrected, local_changed, local_audit = local.correct(corrected)
    local_audit['seconds'] = perf_counter() - local_started
    local_audit['corrected_pixel_fraction'] = float(np.average(local_changed, weights=scounts[so]))
    support_changed |= local_changed
    displacement = np.linalg.norm(corrected[:, 1:] - target[:, 1:], axis=1)
    target = corrected
    correction_seconds = perf_counter() - correction_started
    t3 = perf_counter()
    deband_seconds = 0.0
    deband_audit = {'skipped': True, 'amount': 0.0, 'reason': 'exact_bypass'}
    if deband.active:
        float_colors, alpha = project_float(target, cam)
        by_float = np.empty_like(float_colors)
        by_float[so] = float_colors
        projection_finished = perf_counter()
        deband_started = perf_counter()
        from .deband import _apply_palette
        source_rgb = _srgb_encode((source_xyz / 100.0) @ _SRGB_INV.T)
        del source_xyz
        full_float, deband_audit = _apply_palette(source_rgb, by_float, sidx, source.shape, deband)
        deband_seconds = perf_counter() - deband_started
        encode_started = perf_counter()
        output = np.rint(np.clip(full_float, 0, 1) * 255).astype(np.uint8)
        encode_seconds = perf_counter() - encode_started
        colors = None
        packed_output = pack(output.reshape(-1, 3))
    else:
        colors, alpha = project(target, cam)
        by = np.empty_like(colors)
        by[so] = colors
        output = by[sidx].reshape(source.shape)
        projection_finished = perf_counter()
        encode_seconds = 0.0
        packed_output = pack(by)
    t4 = perf_counter()
    if np.count_nonzero(np.diff(ladder) < 0):
        raise AssertionError('Quantile ladder is not nondecreasing.')
    stats = {'version': '4.05',
             'deband': deband_audit,
             'result_representation': 'per_pixel' if deband.active else 'source_atoms',
             'final_J_order_guaranteed': False,
             'baseline_J_ladder_order_preserved': True,
             'gamut_statistics_scope': '3.53 baseline before optional legacy mapping',
             'reference_support': {
                 'method': '3.51 convex section followed by actual local-reference graph',
                 'local_support': local_audit,
                 'rank': support.rank, 'faces': len(support.planes),
                 'corrected_pixel_fraction': float(np.average(support_changed, weights=scounts[so])),
                 'weighted_chroma_displacement': float(np.average(displacement, weights=scounts[so])),
                 'seconds': correction_seconds,
                 'unchanged_locally_supported_targets': True,
                  'scope': '3.53 baseline targets before optional legacy deband; not reasserted afterwards',
                 'reference_density_guaranteed': False,
                 'final_reference_support_guaranteed': False},
             'source_pixels': int(scounts.sum()),
             'reference_pixels': int(rcounts.sum()),
             'source_unique_colors': len(scolors),
             'reference_unique_colors': len(rcolors),
               'output_unique_colors': int(len(np.unique(packed_output))),
             'source_encoding': source_encoding.label,
             'reference_encoding': reference_encoding.label,
             'viewing': {'white': 'D65', 'surround': 'Average',
                         'LA': viewing.adapting_luminance,
                         'Yb': viewing.background_luminance},
             'transport_rotations': TRANSPORT_ROTATIONS,
             'projection_bisections': PROJECTION_BISECTIONS,
             'ladder_inversions_float': int(np.count_nonzero(np.diff(ladder) < 0)),
             'compressed_pixel_fraction': float(
                 scounts[so][alpha < 1].sum() / scounts.sum()),
             'weighted_mean_alpha': float(
                 (scounts[so] * alpha).sum() / scounts.sum()),
             'target_chroma_mass': float(
                 (scounts[so] * np.linalg.norm(target[:, 1:], axis=1)).sum()),
             'seconds': {'palette': t1 - started, 'appearance_sort': t2 - t1,
                          'transport_ladder': t3 - t2, 'projection_scatter': projection_finished - t3,
                          'deband': deband_seconds, 'encode_after_deband': encode_seconds,
                         'core_total': t4 - started}}
    return QuantileResult(output, colors, so, scounts,
                          None if deband.active else ladder,
                          None if deband.active else alpha, stats)
