"""Continuous Mapping Spectrum: intrinsic, non-smearing debanding.

This module never filters the final image.  It fits a low-curvature mapping from
SOURCE features (L,a,b,x,y) to the complete Velum result, then admits the part of
the complete mapping that this smooth model cannot explain in proportion to the
SOURCE's absolute gradient energy.

The baseline is therefore a mapping, not a blurred image.  Large-scale colour
trends (for example green -> magenta across a wall) remain expressible through
the spatial terms x/y, while thin bands and JPEG-shaped wiggles require curvature
the model is explicitly penalised for using.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np


# A compact degree<=3 basis.  It is expressive enough for broad spatial/tonal
# trends but deliberately unable to encode pixel-scale contours.
_EXPONENTS: tuple[tuple[int, ...], ...] = tuple(
    e for e in itertools.product(range(4), repeat=5) if sum(e) <= 3
)

# Cache everything that depends only on the polynomial basis. The old fit rebuilt
# the 15 derivative feature matrices (5 diagonal + 10 mixed Hessian entries) for
# every image even though the monomial exponents never change.
_E = np.asarray(_EXPONENTS, np.int8)


def _curvature_gram(z: np.ndarray) -> np.ndarray:
    """Hessian Frobenius Gram matrix without materialising 15 N×56 matrices.

    For monomial phi_e(z)=prod z_i^e_i, a second derivative is another monomial
    times a scalar. The Gram element between basis terms e,f is therefore that
    scalar product times the MEAN of a monomial z^p. We need moments only up to
    degree 6 in each of five dimensions; compute the 7^5 possible moment keys on
    demand and assemble the tiny 56×56 matrix directly.

    This is algebraically identical to summing d2.T@d2 in the old code, not an
    approximation. It removes 15 derivative matrices and their matrix products.
    """
    z = np.asarray(z, np.float64)
    # Powers and lazily cached mixed moments.
    zp = np.stack([z ** p for p in range(7)], axis=-1)
    moments: dict[tuple[int, ...], float] = {}

    def moment(p):
        key = tuple(int(x) for x in p)
        v = moments.get(key)
        if v is None:
            x = np.ones(z.shape[0], np.float64)
            for q in range(5):
                if key[q]:
                    x *= zp[:, q, key[q]]
            v = float(x.mean())
            moments[key] = v
        return v

    curv = np.zeros((len(_EXPONENTS), len(_EXPONENTS)), np.float64)
    E = _E.astype(np.int16)
    for j in range(5):
        for k in range(j, 5):
            if j == k:
                coeff = E[:, j] * (E[:, j] - 1)
                shift = np.zeros(5, np.int16)
                shift[j] = 2
                weight = 1.0
            else:
                coeff = E[:, j] * E[:, k]
                shift = np.zeros(5, np.int16)
                shift[j] = shift[k] = 1
                weight = 2.0
            nz = np.flatnonzero(coeff)
            for ii, a in enumerate(nz):
                for b in nz[ii:]:
                    p = E[a] + E[b] - 2 * shift
                    val = weight * float(coeff[a] * coeff[b]) * moment(p)
                    curv[a, b] += val
                    if a != b:
                        curv[b, a] += val
    return curv


def _features(z: np.ndarray) -> np.ndarray:
    """Polynomial features for z=(L,a_scaled,b_scaled,x,y)."""
    cols = []
    for e in _EXPONENTS:
        c = np.ones(z.shape[0], np.float64)
        for j, p in enumerate(e):
            if p:
                c *= z[:, j] ** p
        cols.append(c)
    return np.stack(cols, 1)


STRUCTURE_FIELDS = frozenset({"gradient_energy", "dog_guided", "dog_raw"})


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(x, -1, (size, size), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide: np.ndarray, src: np.ndarray,
                   radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Single-channel guided filter used by the DoG structure-field option."""
    mean_g = _box_mean(guide, radius)
    mean_p = _box_mean(src, radius)
    var_g = np.maximum(_box_mean(guide * guide, radius) - mean_g * mean_g, 0.0)
    cov_gp = _box_mean(guide * src, radius) - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return (_box_mean(a, radius) * guide + _box_mean(b, radius)).astype(np.float32)


def _gradient_energy(L: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 9.0)


def _dog_energy(L: np.ndarray, guided: bool) -> np.ndarray:
    blur = cv2.GaussianBlur(L, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    energy = np.abs(L - blur).astype(np.float32)
    if guided:
        energy = np.clip(_guided_filter(L, energy), 0.0, None)

    # Keep the historical structure anchors comparable across selectable operators.
    # This is a numerical range conversion, not spatial blending: the candidate's
    # edge layout is preserved while its 1st/99th percentiles match gradient energy.
    reference = _gradient_energy(L)
    lo, hi = np.percentile(energy, [1.0, 99.0])
    ref_lo, ref_hi = np.percentile(reference, [1.0, 99.0])
    aligned = (energy - lo) / max(float(hi - lo), 1e-12)
    return (aligned * max(float(ref_hi - ref_lo), 1e-12) + ref_lo).astype(np.float32)


def source_structure(src_lab: np.ndarray, sigma: float = 3.0,
                     field: str = "gradient_energy") -> np.ndarray:
    """Continuous SOURCE structure field used to release transfer residuals.

    This is deliberately the historical gradient-energy field, not a new proxy.
    It is the only one of v1's four modes retained because it is the proven visual
    baseline. The other modes were removed after direct measurement showed that on
    flat content gradient/highpass/range differed by <0.004 and produced
    bit-identical output -- exposing them as choices implied control that did not
    exist.

    The two absolute anchors that set the flat/structured transition are a known,
    measured debt; two self-calibrating replacements were tried and both regressed
    the frozen baseline. See the comment at the transition for the numbers and why
    a correct fix needs a different structure operator rather than a renormalisation.

    Spatial smoothing applies only to this control field, never to image pixels.
    """
    if field not in STRUCTURE_FIELDS:
        raise ValueError(f"unknown deband structure field: {field!r}")
    L = src_lab[..., 0].astype(np.float32)
    if field == "gradient_energy":
        energy = _gradient_energy(L)
    elif field == "dog_guided":
        energy = _dog_energy(L, guided=True)
    else:
        energy = _dog_energy(L, guided=False)

    # THESE TWO ANCHORS ARE A KNOWN, MEASURED DEBT -- DO NOT "TIDY" THEM BLIND
    # 0.015 and 0.30 are not derived, and they do assume an absolute contrast scale,
    # so they are the weakest numbers left in this module. Two self-calibrating
    # replacements were implemented and measured against the frozen baseline, and
    # BOTH lost quality, so both were reverted:
    #
    #   median + mean-absolute-deviation : assumes at least half the image is flat.
    #       On target's night scene the median sits in the noise floor, genuinely
    #       flat pixels were classified as structured, and AUTO banding went
    #       14467 -> 125262.
    #   energy / mean(energy) : contrast-invariant and much closer, fixing target and
    #       improving 17 metrics, but still lost 8 -- notably dot/auto banding
    #       4686 -> 5890 and witch/auto hue error 4.01 -> 7.41 degrees.
    #
    # Both attempts share one flaw: a single global statistic cannot express "how
    # much local variation counts as an edge", because that is a property of the
    # local neighbourhood, not of the whole image. Replacing these anchors properly
    # therefore needs a different structure operator, not a different normalisation
    # of this one -- a larger change than a constant cleanup. Until then the proven
    # values stay, because shipping a measured regression to remove a constant would
    # trade real quality for tidiness.
    noise_floor, structure_level = 0.015, 0.30
    t = np.clip((energy - noise_floor) /
                (structure_level - noise_floor), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return cv2.GaussianBlur(t, (0, 0), max(float(sigma), 0.4))

# Frozen v1.9 basis scaling. A gamut-derived replacement was measured separately
# and regressed four AUTO metrics; retain the proven basis until the fit and its
# regulariser can be re-derived together rather than changing one coordinate scale.
_OKLAB_AB_SCALE = 0.25


def _z_field(src_lab: np.ndarray) -> np.ndarray:
    h, w = src_lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([
        src_lab[..., 0],
        src_lab[..., 1] / _OKLAB_AB_SCALE,
        src_lab[..., 2] / _OKLAB_AB_SCALE,
        xx / max(w - 1, 1) * 2.0 - 1.0,
        yy / max(h - 1, 1) * 2.0 - 1.0,
    ], -1).astype(np.float64)


def _fit_low_curvature_coefficients(src_lab: np.ndarray,
                                     target_lab: np.ndarray,
                                     amount: float,
                                     max_fit_side: int,
                                     max_samples: int):
    """Solve only the 56×3 coefficient matrix and return fit metadata.

    Split from full-resolution evaluation so callers that only need candidate
    quality (auto tuning, diagnostics) do not build a megapixel prediction.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    h, w = src_lab.shape[:2]
    side = max(h, w)
    if side > max_fit_side:
        scale = max_fit_side / side
        fw, fh = max(16, round(w * scale)), max(16, round(h * scale))
        src_fit = cv2.resize(src_lab, (fw, fh), interpolation=cv2.INTER_AREA)
        tgt_fit = cv2.resize(target_lab, (fw, fh), interpolation=cv2.INTER_AREA)
    else:
        src_fit, tgt_fit = src_lab, target_lab

    z = _z_field(src_fit).reshape(-1, 5)
    y = tgt_fit.reshape(-1, 3).astype(np.float64)
    n = z.shape[0]
    if n > max_samples:
        step = int(np.ceil(n / max_samples))
        ids = np.arange(0, n, step, dtype=np.intp)[:max_samples]
        z, y = z[ids], y[ids]

    phi = _features(z)
    normal = phi.T @ phi / phi.shape[0]
    rhs = phi.T @ y / phi.shape[0]

    zc = (z[np.linspace(0, z.shape[0] - 1, 4096, dtype=np.intp)]
          if z.shape[0] > 4096 else z)
    curv = _curvature_gram(zc)

    # Frozen v1.9 schedule. Trace normalisation is dimensionally cleaner, but an
    # isolated full-suite gate regressed 15 AUTO metrics. Keep this measured debt
    # until the polynomial basis and regularisation objective are redesigned as one.
    lam = 3e-4 * (10.0 ** (3.0 * a) - 1.0) / (10.0 ** 3.0 - 1.0)
    ridge = 1e-7 * np.eye(normal.shape[0])
    beta = np.linalg.solve(normal + lam * curv + ridge, rhs)
    return beta, src_fit, tgt_fit, {"curvature_lambda": float(lam),
                                    "basis_terms": len(_EXPONENTS)}


def _z_chunk(flat_lab: np.ndarray, h: int, w: int, start: int, stop: int) -> np.ndarray:
    """Construct exactly _z_field(src_lab).reshape(-1, 5)[start:stop].

    Global integer coordinates and all floating operations match _z_field.
    Coordinates are NOT normalized per chunk. Colour division remains in the
    input dtype, then stack promotes to float64, as in the original function.
    """
    pixel = np.arange(start, stop, dtype=np.int64)
    yy, xx = np.divmod(pixel, w)
    lab = flat_lab[start:stop]
    return np.stack([
        lab[:, 0],
        lab[:, 1] / _OKLAB_AB_SCALE,
        lab[:, 2] / _OKLAB_AB_SCALE,
        xx / max(w - 1, 1) * 2.0 - 1.0,
        yy / max(h - 1, 1) * 2.0 - 1.0,
    ], -1).astype(np.float64, copy=False)


def _evaluate_mapping(src_lab: np.ndarray, beta: np.ndarray,
                      chunk: int = 65536) -> np.ndarray:
    from .. import gpu_mapping
    gpu = gpu_mapping.try_evaluate(src_lab, beta, _EXPONENTS)
    if gpu is not None:
        return gpu
    h, w = src_lab.shape[:2]

    # 4.07 compiles only feature assembly, NOT the reduction or power operation.
    # Keep GPU routing, matrix shapes, global coordinates and BLAS as before.
    from .evaluation_features import evaluation_features
    if h * w >= 4 * 1024 * 1024:
        return _evaluate_mapping_cpu_tiled(src_lab, beta)
    # Keep the original matrix multiplication shapes and feature arithmetic.
    # Only the lifetime of the five-dimensional coordinate buffer changes.
    flat = src_lab.reshape(-1, 3)
    pred = np.empty((flat.shape[0], 3), np.float64)
    for i in range(0, flat.shape[0], chunk):
        stop = min(i + chunk, flat.shape[0])
        z = _z_chunk(flat, h, w, i, stop)
        pred[i:stop] = evaluation_features(z) @ beta
    return pred.reshape(h, w, 3)


def _evaluate_mapping_cpu_tiled(src_lab: np.ndarray, beta: np.ndarray,
                                tile_rows: int = 256) -> np.ndarray:
    """Memory-bounded CPU fallback with global coordinates preserved exactly."""
    from .evaluation_features import evaluation_features
    h, w = src_lab.shape[:2]
    pred = np.empty((h, w, 3), np.float64)
    for y0 in range(0, h, tile_rows):
        y1 = min(y0 + tile_rows, h)
        lab = src_lab[y0:y1]
        yy, xx = np.mgrid[y0:y1, 0:w]
        z = np.stack([
            lab[..., 0],
            lab[..., 1] / _OKLAB_AB_SCALE,
            lab[..., 2] / _OKLAB_AB_SCALE,
            xx / max(w - 1, 1) * 2.0 - 1.0,
            yy / max(h - 1, 1) * 2.0 - 1.0,
        ], -1).astype(np.float64).reshape(-1, 5)
        pred[y0:y1] = (evaluation_features(z) @ beta).reshape(y1 - y0, w, 3)
    return pred


def _calibrate_mapping(pred: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
    """Correct only the DC lightness residual; retain the accepted a/b calibration.

    delta = argmin_d mean((pred_L + d - target_L)**2).
    A constant shift preserves all derivatives of the fitted L function instead
    of re-stretching its percentiles. It matches the GLOBAL base mean, not each
    output pixel or every region. No old per-pixel J values are restored.
    """
    bias = (np.mean(target_lab[..., 0], dtype=np.float64)
            - np.mean(pred[..., 0], dtype=np.float64))
    pred[..., 0] += bias
    for c in (1, 2):
        p10, p90 = np.percentile(pred[..., c], [10.0, 90.0])
        t10, t90 = np.percentile(target_lab[..., c], [10.0, 90.0])
        if p90 - p10 > 1e-8:
            pred[..., c] = t10 + (pred[..., c] - p10) * (
                (t90 - t10) / (p90 - p10))
    return pred


def _flatness(structure: np.ndarray) -> np.ndarray:
    """Flat-region selector used by automatic deband tuning.

    The selector answers one question: which pixels are flat enough in the SOURCE
    that any chroma high-frequency the grade adds there must be banding rather than
    detail. `structure` is a smoothstepped field on [0, 1], so a single threshold on
    it is the whole definition.

    An earlier version added a second definition on top: when fewer than 64 pixels
    passed the threshold it silently switched to "the flattest 20% of the image".
    That is a patch, not a rule -- it redefined flatness whenever the real
    definition was inconvenient, and an image with genuinely no flat area would be
    handed a fabricated one. Both numbers are gone. A weighted mean is defined
    whenever any weight is non-zero, and `_flat_defect_gain` /
    `_defect_curve_evaluator` already return 0 for an all-zero weight, so the
    degenerate case needs no branch.

    The remaining 0.05 is measured debt: it is the decision boundary itself, shared
    with the structure field's own anchors, and cannot be removed by reweighting --
    only by redesigning the defect objective together with the structure operator.
    """
    return (structure < 0.05).astype(np.float32)


def _flat_roughnesses(src_lab: np.ndarray, out_lab: np.ndarray,
                      structure: np.ndarray) -> tuple[float, float]:
    """Source/output roughness in the SAME Oklab units; no denominator floor."""
    weight = _flatness(structure)
    total = float(weight.sum())

    def rough(lab):
        ab = lab[..., 1:3].astype(np.float32)
        local = cv2.GaussianBlur(ab, (0, 0), 3.0)
        amplitude = np.sqrt(np.sum((ab - local) ** 2, axis=-1))
        return float(np.sum(amplitude * weight) / total) if total > 0.0 else 0.0

    return rough(src_lab), rough(out_lab)


def _flat_defect_gain(src_lab: np.ndarray, out_lab: np.ndarray,
                      structure: np.ndarray) -> float | None:
    """Diagnostic ratio only. A zero source amplitude makes this ratio undefined."""
    source, output = _flat_roughnesses(src_lab, out_lab, structure)
    return output / source if source > 0.0 else None


def _defect_curve_evaluator(src_lab: np.ndarray,
                            base: np.ndarray,
                            full_lab: np.ndarray,
                            structure: np.ndarray):
    """Return an output-roughness evaluator and the source roughness in Oklab units.

    Why this replaces repeated image composition
    --------------------------------------------
    ``compose(a)`` used to materialise a full-resolution Oklab image and
    ``_flat_defect_gain`` then Gaussian-blurred that image for every candidate.
    Auto mode tests 21 coarse plus up to 11 fine candidates, so this meant about
    64 full-resolution Gaussian blurs and 32 large image allocations. Profiling on
    the 800x1236 regression case attributed 1.67 s to those evaluations alone.

    The same curve has an exact algebraic form. With ``q = 1 - structure``:

        freedom(a) = smoothstep(1 - a q)
                   = 1 - 3 a² q² + 2 a³ q³

        out(a) = full + a² C2 + a³ C3
        C2 = -3 (full - base) q²
        C3 =  2 (full - base) q³

    The high-pass operator used by the defect metric, ``H(x)=x-blur(x)``, is
    linear, therefore:

        H(out(a)) = H(full) + a² H(C2) + a³ H(C3)

    We compute those three coefficient fields ONCE (three blurs), weight them by
    flatness, and evaluate every candidate with scalar polynomial arithmetic. This
    is mathematically the same objective -- no approximation, changed threshold,
    reduced image, or fitted shortcut -- but removes 60 blurs.
    """
    weight = _flatness(structure)
    total = float(weight.sum())
    scale = 1.0 / total if total > 0.0 else 0.0

    src_ab = src_lab[..., 1:3].astype(np.float32)
    src_hp = src_ab - cv2.GaussianBlur(src_ab, (0, 0), 3.0)
    source_roughness = float(np.sum(np.linalg.norm(src_hp, axis=-1) * weight) * scale)

    q = (1.0 - structure).astype(np.float32)[..., None]
    residual = (full_lab[..., 1:3] - base[..., 1:3]).astype(np.float32)
    c0 = full_lab[..., 1:3].astype(np.float32)
    c2 = -3.0 * residual * (q * q)
    c3 = 2.0 * residual * (q * q * q)

    def highpass(x: np.ndarray) -> np.ndarray:
        return x - cv2.GaussianBlur(x, (0, 0), 3.0)

    # Keep planar contiguous coefficient fields. The fused pointwise scorer writes
    # one reusable float32 amplitude buffer; NumPy retains the original reduction.
    def split_highpass(x):
        # The former generator called highpass(x) twice, once for each channel.
        # Cache the identical result: same arithmetic, three coefficient blurs not six.
        h = highpass(x)
        return np.ascontiguousarray(h[..., 0].ravel()), np.ascontiguousarray(h[..., 1].ravel())
    h0a, h0b = split_highpass(c0)
    h2a, h2b = split_highpass(c2)
    h3a, h3b = split_highpass(c3)
    flat_weight = np.ascontiguousarray(weight.ravel())
    from .auto_score import make_scratch, score_amplitudes
    buf_norm = np.empty_like(h0a)
    scratch = make_scratch(h0a.size)

    def evaluate(a: float) -> float:
        a2 = np.float32(float(a) * float(a))
        a3 = np.float32(float(a) * float(a) * float(a))
        score_amplitudes(h0a, h0b, h2a, h2b, h3a, h3b,
                         flat_weight, a2, a3, buf_norm, scratch=scratch)
        return float(buf_norm.sum()) * scale

    return evaluate, source_roughness


def apply_continuous_spectrum(src_lab: np.ndarray,
                              full_lab: np.ndarray,
                              amount: float | None,
                              auto: bool = True,
                              structure_field: str = "gradient_energy",
                              ) -> tuple[np.ndarray, dict]:
    """Apply Continuous Mapping Spectrum v1.5.

    The expensive low-curvature basis is fitted ONCE at maximum regularisation.
    `amount` then only controls how much of the full-transfer residual is released:

        out = base + freedom(amount, source_structure) * (full - base)

    This decoupling is the main v1.5 simplification. v1 refit the 56-term system for
    every amount even though the user's actual intent was the residual freedom; the
    two effects were entangled and autotuning required repeated solves.

    AUTO mode uses a coarse-to-fine search over residual blends, measures the
    source-flat defect amplification curve G(a), and selects its knee:

        argmax_a  (G(0)-G(a))/(G(0)-min G) - a

    The first term is the fraction of this IMAGE'S achievable defect reduction;
    the second is the regularisation paid for it. A 21-point 0.05 coarse pass finds
    the neighbourhood, then an 0.01 pass refines within +/-0.05. The result therefore
    has 0.01 precision without paying for 101 full-range evaluations. This uses no
    cross-image absolute target (which the historical model often cannot reach).

    Manual amount=0 is an exact bypass and performs no fit.

    Performance note: the historical search materialised and Gaussian-blurred a
    full-resolution image for every candidate (~64 blurs). ``freedom(a)`` is a
    cubic polynomial and the high-pass operator is linear, so v1.6 evaluates the
    EXACT same curve from three precomputed high-pass coefficient fields. On the
    800x1236 regression case the stage falls from 5.28 s to ~1.4-2.0 s with the
    chosen amount and output unchanged to fp32 precision.
    """
    requested = float(np.clip(0.0 if amount is None else amount, 0.0, 1.0))
    if not auto and requested <= 0.0:
        return full_lab, {"skipped": True, "amount": 0.0, "auto": False}

    # Solve once, evaluate once at full resolution. Auto below is exact on the
    # delivered pixels; its repeated full-image work has been algebraically reduced
    # to a cubic high-pass polynomial by _defect_curve_evaluator.
    beta, _src_fit, _full_fit, info = _fit_low_curvature_coefficients(
        src_lab, full_lab, 1.0, max_fit_side=720, max_samples=60000)
    # Retain the private four-value API, but do not hold unused fitting images
    # through the subsequent full-resolution evaluation and automatic search.
    del _src_fit, _full_fit
    base = _evaluate_mapping(src_lab, beta)
    base = _calibrate_mapping(base, full_lab).astype(np.float32)
    structure = source_structure(src_lab, field=structure_field)
    def compose(a: float) -> np.ndarray:
        freedom = 1.0 - a * (1.0 - structure)
        freedom = freedom * freedom * (3.0 - 2.0 * freedom)
        return (base + freedom[..., None] * (full_lab - base)).astype(np.float32)

    audit_candidates: list[float] | None = None
    if auto:
        # Build the defect curve once as a cubic high-pass polynomial. This is
        # exactly equivalent to composing and blurring every candidate, but needs
        # three coefficient blurs plus one source blur; see _defect_curve_evaluator.
        # UHD images contain millions of redundant control samples. Keep the accepted
        # full-resolution decision at and below 1K; only true UHD delivery uses the
        # deterministic 2x lattice. The final base, structure and composition remain
        # full resolution.
        if min(src_lab.shape[:2]) >= 1536:
            search_src = np.ascontiguousarray(src_lab[::2, ::2])
            search_base = np.ascontiguousarray(base[::2, ::2])
            search_full = np.ascontiguousarray(full_lab[::2, ::2])
            search_structure = np.ascontiguousarray(structure[::2, ::2])
            info["auto_search_stride"] = 2
        else:
            search_src, search_base = src_lab, base
            search_full, search_structure = full_lab, structure
            info["auto_search_stride"] = 1
        roughness, source_roughness = _defect_curve_evaluator(
            search_src, search_base, search_full, search_structure)
        info['auto_scorer_backend'] = 'numpy_no_jit'
        coarse = np.linspace(0.0, 1.0, 21)
        coarse_roughness = np.asarray([
            roughness(float(candidate))
            for candidate in coarse
        ], np.float64)
        # In 4.02 E=max(R_out/max(R_src,1e-8)-1,0). For positive R_src above
        # that floor, normalization of E cancels the denominator exactly.
        # Work directly with E=max(R_out-R_src,0): same structural-excess target,
        # finite even when R_src=0, and no arbitrary minimum source amplitude.
        # The small-source regime intentionally uses the ACTUAL source amplitude;
        # it is not claimed numerically identical to the old floored definition.
        excess = np.maximum(coarse_roughness - source_roughness, 0.0)
        achievable = float(excess[0] - excess.min())
        if achievable <= 0.0:
            # No sampled candidate reduces positive source-relative excess.
            a = 0.0
            chosen_roughness = float(coarse_roughness[0])
            candidates = coarse
            rough_values = coarse_roughness
        else:
            coarse_score = ((excess[0] - excess) / achievable - coarse)
            centre = float(coarse[int(np.argmax(coarse_score))])
            lo, hi = max(0.0, centre - 0.05), min(1.0, centre + 0.05)
            fine = np.round(np.arange(lo, hi + 0.0001, 0.01), 2)
            fine_roughness = np.asarray([
                roughness(float(candidate))
                for candidate in fine
            ], np.float64)
            fine_excess = np.maximum(fine_roughness - source_roughness, 0.0)
            fine_score = ((excess[0] - fine_excess) / achievable - fine)
            k = int(np.argmax(fine_score))
            a = float(fine[k])
            chosen_roughness = float(fine_roughness[k])
            # Keep the audit curve self-describing and ordered. Duplicate coarse /
            # fine samples are collapsed; this is diagnostic metadata only.
            samples = {round(float(x), 2): float(y)
                       for x, y in zip(coarse, coarse_roughness)}
            samples.update({round(float(x), 2): float(y)
                            for x, y in zip(fine, fine_roughness)})
            candidates = np.asarray(sorted(samples), np.float64)
            rough_values = np.asarray([samples[float(x)] for x in candidates], np.float64)
        curve = [float(x) / source_roughness if source_roughness > 0.0 else None for x in rough_values]
        rough_curve = [float(x) for x in rough_values]
        audit_candidates = [float(x) for x in candidates]
    else:
        a = requested
        # The chosen amount is known; no full-resolution defect search needed.
        chosen_roughness = None
        curve = None

    out = compose(a)
    if not auto:
        source_roughness, chosen_roughness = _flat_roughnesses(src_lab, out, structure)
        rough_curve = None
    chosen_gain = chosen_roughness / source_roughness if source_roughness > 0.0 else None
    freedom = 1.0 - a * (1.0 - structure)
    freedom = freedom * freedom * (3.0 - 2.0 * freedom)
    info.update({
        "amount": a,
        "requested_amount": requested,
        "auto": bool(auto),
        "flat_defect_gain": float(chosen_gain) if chosen_gain is not None else None,
        "roughness_source": float(source_roughness),
        "roughness_output": float(chosen_roughness),
        "roughness_excess": max(float(chosen_roughness - source_roughness), 0.0),
        "auto_roughness_curve": rough_curve,
        "auto_objective": "normalized max(output_roughness-source_roughness,0) improvement minus strength",
        "source_amplitude_floor": None,
        "auto_candidates": audit_candidates,
        "auto_gain_curve": curve,
        "structure_mean": float(structure.mean()),
        "freedom_mean": float(freedom.mean()),
        "method": "v1.5_low_curvature_spatial_residual",
        "structure_field": structure_field,
    })
    return out, info
