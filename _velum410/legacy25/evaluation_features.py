"""Evaluation-only degree-three feature assembly, preserving NumPy arithmetic.

Do not replace the fitting features or the BLAS reduction. NumPy still computes
the powers (notably cubes); the compiled loop only packs/multiplies the same
monomials in the same axis order. No fastmath, FMA, Horner, or precision change.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def _assemble(powers):
    n = powers.shape[1]
    out = np.empty((n, 56), np.float64)
    for i in prange(n):
        a, a2, a3 = powers[0, i], powers[1, i], powers[2, i]
        b, b2, b3 = powers[3, i], powers[4, i], powers[5, i]
        c, c2, c3 = powers[6, i], powers[7, i], powers[8, i]
        s, s2, s3 = powers[9, i], powers[10, i], powers[11, i]
        t, t2, t3 = powers[12, i], powers[13, i], powers[14, i]
        # Lexicographic itertools.product(range(4), repeat=5), sum <= 3.
        out[i, 0] = 1.0
        out[i, 1] = t
        out[i, 2] = t2
        out[i, 3] = t3
        out[i, 4] = s
        out[i, 5] = s*t
        out[i, 6] = s*t2
        out[i, 7] = s2
        out[i, 8] = s2*t
        out[i, 9] = s3
        out[i, 10] = c
        out[i, 11] = c*t
        out[i, 12] = c*t2
        out[i, 13] = c*s
        out[i, 14] = (c*s)*t
        out[i, 15] = c*s2
        out[i, 16] = c2
        out[i, 17] = c2*t
        out[i, 18] = c2*s
        out[i, 19] = c3
        out[i, 20] = b
        out[i, 21] = b*t
        out[i, 22] = b*t2
        out[i, 23] = b*s
        out[i, 24] = (b*s)*t
        out[i, 25] = b*s2
        out[i, 26] = b*c
        out[i, 27] = (b*c)*t
        out[i, 28] = (b*c)*s
        out[i, 29] = b*c2
        out[i, 30] = b2
        out[i, 31] = b2*t
        out[i, 32] = b2*s
        out[i, 33] = b2*c
        out[i, 34] = b3
        out[i, 35] = a
        out[i, 36] = a*t
        out[i, 37] = a*t2
        out[i, 38] = a*s
        out[i, 39] = (a*s)*t
        out[i, 40] = a*s2
        out[i, 41] = a*c
        out[i, 42] = (a*c)*t
        out[i, 43] = (a*c)*s
        out[i, 44] = a*c2
        out[i, 45] = a*b
        out[i, 46] = (a*b)*t
        out[i, 47] = (a*b)*s
        out[i, 48] = (a*b)*c
        out[i, 49] = a*b2
        out[i, 50] = a2
        out[i, 51] = a2*t
        out[i, 52] = a2*s
        out[i, 53] = a2*c
        out[i, 54] = a2*b
        out[i, 55] = a3
    return out


def evaluation_features(z):
    """Return exactly the existing float64 N-by-56 evaluation feature matrix."""
    if z.ndim != 2 or z.shape[1] != 5 or z.dtype != np.float64:
        raise ValueError('Evaluation coordinates must be float64 with shape (N, 5).')
    powers = np.empty((15, z.shape[0]), np.float64)
    for axis in range(5):
        for power in range(1, 4):
            # Preserve ndarray ** integer dispatch too: **2 may use np.square,
            # unlike generic np.power(..., 2, out=...), which can differ by 1 ULP.
            powers[axis*3 + power - 1] = z[:, axis] ** power
    return _assemble(powers)
