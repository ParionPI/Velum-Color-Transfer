"""Float64 RGB -> common D65 XYZ -> CAM16 J. No inverse transform or gamut clamp.

Untagged input is IEC sRGB. Embedded RGB matrix/TRC ICC profiles are evaluated
without 8-bit conversion. ICC constants/curves follow ICC.1, not image tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

D65 = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
M16 = np.array([[0.401288, 0.650173, -0.051461],
                [-0.250268, 1.204414, 0.045854],
                [-0.002079, 0.048952, 0.953127]])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                    [-0.7502, 1.7135, 0.0367],
                    [0.0389, -0.0685, 1.0296]])


def srgb_matrix() -> np.ndarray:
    """Normalized primary matrix from IEC sRGB primaries and D65 white."""
    primaries = np.array([[0.64, 0.30, 0.15],
                          [0.33, 0.60, 0.06],
                          [0.03, 0.10, 0.79]])
    return primaries * np.linalg.solve(primaries, D65)


def srgb_curve(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _fixed(data: bytes, offset: int, count: int) -> np.ndarray:
    return np.asarray(struct.unpack_from('>' + 'i' * count, data, offset)) / 65536.0


def _icc_curve(data: bytes, x: np.ndarray) -> np.ndarray:
    """ICC curveType and parametricCurveType, including their specified branches."""
    signature = data[:4]
    if signature == b'curv':
        count = struct.unpack_from('>I', data, 8)[0]
        if count == 0:
            return x.copy()
        if count == 1:
            return x ** (struct.unpack_from('>H', data, 12)[0] / 256.0)
        table = np.asarray(struct.unpack_from('>' + 'H' * count, data, 12)) / 65535.0
        return np.interp(x, np.linspace(0, 1, count), table)
    if signature != b'para':
        raise ValueError(f'Unsupported ICC TRC type {signature!r}; no silent conversion.')
    function = struct.unpack_from('>H', data, 8)[0]
    counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
    if function not in counts:
        raise ValueError(f'Unsupported ICC parametric curve {function}.')
    p = _fixed(data, 12, counts[function])
    if function == 0:
        return x ** p[0]
    g, a, b = p[:3]
    if a <= 0:
        raise ValueError('Invalid ICC parametric curve: a must be positive.')
    y = np.empty_like(x)
    if function in (1, 2):
        mask = x >= -b / a
        offset = p[3] if function == 2 else 0.0
        y[mask] = (a * x[mask] + b) ** g + offset
        y[~mask] = offset
    else:
        c, d = p[3:5]
        mask = x >= d
        e, f = p[5:7] if function == 4 else (0.0, 0.0)
        y[mask] = (a * x[mask] + b) ** g + e
        y[~mask] = c * x[~mask] + f
    return y


@dataclass(frozen=True)
class RGBEncoding:
    """RGB decoding LUT and matrix to relative D65 XYZ (white Y=1)."""
    matrix: np.ndarray
    curves: np.ndarray
    label: str
    icc: bytes | None = None
    linear_float: bool = False

    @classmethod
    def linear_srgb(cls) -> 'RGBEncoding':
        return cls(srgb_matrix(), np.tile(np.arange(256)/255, (3,1)),
                   'linear sRGB float64', None, True)

    @classmethod
    def from_icc(cls, profile: bytes | None) -> 'RGBEncoding':
        x = np.arange(256, dtype=np.float64) / 255.0
        if not profile:
            return cls(srgb_matrix(), np.tile(srgb_curve(x), (3, 1)), 'sRGB (untagged)')
        if len(profile) < 132 or profile[36:40] != b'acsp':
            raise ValueError('Invalid ICC profile.')
        if profile[16:24] != b'RGB XYZ ':
            raise ValueError('Only RGB matrix/TRC ICC with XYZ PCS is supported.')
        tags = {}
        for i in range(struct.unpack_from('>I', profile, 128)[0]):
            tag, offset, size = struct.unpack_from('>4sII', profile, 132 + 12 * i)
            if offset + size > len(profile):
                raise ValueError('ICC tag exceeds profile size.')
            tags[tag] = profile[offset:offset + size]
        # LUT transforms can override colorants; never silently substitute a matrix.
        if any(tag in tags for tag in (b'A2B0', b'A2B1', b'A2B2', b'D2B0', b'D2B1', b'D2B2', b'D2B3')):
            raise ValueError('LUT-based ICC profiles are not supported; no palette-altering fallback.')
        required = (b'rXYZ', b'gXYZ', b'bXYZ', b'rTRC', b'gTRC', b'bTRC')
        if any(tag not in tags for tag in required):
            raise ValueError('ICC profile lacks RGB matrix/TRC tags.')
        if any(tags[tag][:4] != b'XYZ ' for tag in required[:3]):
            raise ValueError('ICC colorant has an invalid XYZ type.')
        matrix_pcs = np.column_stack([_fixed(tags[tag], 8, 3) for tag in required[:3]])
        curves = np.stack([_icc_curve(tags[tag], x) for tag in required[3:]])
        # ICC colorant tags are already adapted to PCS white. Adapt PCS -> D65
        # once, not twice via chad. Header illuminant is the encoded PCS white.
        pcs_white = _fixed(profile, 68, 3)
        adaptation = np.linalg.solve(BRADFORD,
            np.diag((BRADFORD @ D65) / (BRADFORD @ pcs_white)) @ BRADFORD)
        matrix = adaptation @ matrix_pcs
        if not np.all(np.isfinite(curves)) or not np.all(np.isfinite(matrix)):
            raise ValueError('ICC transform produced nonfinite values.')
        if np.any(np.diff(curves, axis=1) < 0):
            raise ValueError('ICC TRCs must be monotone.')
        digest = hashlib.sha256(profile).hexdigest()[:12]
        return cls(matrix, curves, f'ICC matrix/TRC ({digest})', profile)

    def linear(self, rgb: np.ndarray) -> np.ndarray:
        if self.linear_float:
            rgb = np.asarray(rgb, dtype=np.float64)
            if not np.all(np.isfinite(rgb)) or np.any(rgb < 0) or np.any(rgb > 1):
                raise ValueError('Linear float RGB must be finite and in [0,1]; no clamp.')
            return rgb
        if rgb.dtype != np.uint8:
            raise ValueError('Nonlinear RGB encoding requires uint8 values.')
        return np.column_stack([self.curves[c, rgb[:, c]] for c in range(3)])

    def xyz(self, rgb: np.ndarray) -> np.ndarray:
        linear = self.linear(rgb)
        # Explicit three-term reduction: independent of BLAS threading/block size.
        return (linear[:, 0, None] * self.matrix[:, 0]
                + linear[:, 1, None] * self.matrix[:, 1]
                + linear[:, 2, None] * self.matrix[:, 2]) * 100.0


@dataclass(frozen=True)
class ViewingConditions:
    """Shared CAM16 Average surround; LA in cd/m², Yb relative to Yw=100.

    64/20 are declared viewing conditions, not fitted transfer parameters.
    They may be changed, but source and reference always share the same values.
    """
    adapting_luminance: float = 64.0
    background_luminance: float = 20.0

    def __post_init__(self):
        if (not np.isfinite(self.adapting_luminance) or self.adapting_luminance <= 0
                or not np.isfinite(self.background_luminance) or self.background_luminance <= 0):
            raise ValueError('CAM16 LA and Yb must be finite and positive.')


class CAM16:
    """Only compute J: no unnecessary hue/chroma or CAM16 inverse operation."""
    def __init__(self, viewing: ViewingConditions = ViewingConditions()):
        self.viewing = viewing
        la, yb = viewing.adapting_luminance, viewing.background_luminance
        n = yb / 100.0
        self.exponent = 0.69 * (1.48 + np.sqrt(n))
        k = 1 / (5 * la + 1)
        self.fl = 0.2 * k**4 * 5 * la + 0.1 * (1 - k**4)**2 * np.cbrt(5 * la)
        # Average surround F=1. For LA>0 the standard D lies in (0,1): no clamp.
        d = 1 - np.exp((-la - 42) / 92) / 3.6
        white_rgb = M16 @ (D65 * 100)
        self.adaptation = d * 100 / white_rgb + 1 - d
        self.aw = self._achromatic(white_rgb[None, :])[0]

    def _achromatic(self, rgb: np.ndarray) -> np.ndarray:
        adapted = rgb * self.adaptation
        p = (self.fl * np.abs(adapted) / 100.0) ** 0.42
        response = np.sign(adapted) * 400 * p / (27.13 + p)
        # +0.1 response offsets exactly cancel -0.305. Nbb cancels in A/Aw.
        return 2 * response[:, 0] + response[:, 1] + response[:, 2] / 20

    def j_from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        rgb = (xyz[:, 0, None] * M16[:, 0]
               + xyz[:, 1, None] * M16[:, 1]
               + xyz[:, 2, None] * M16[:, 2])
        a = self._achromatic(rgb)
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError('Input is outside the real-valued CAM16 J domain; not clamped.')
        return 100 * (a / self.aw) ** self.exponent

    def j(self, rgb: np.ndarray, encoding: RGBEncoding, block_size: int = 65536) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError('CAM16 input must have shape (N,3), dtype uint8.')
        if block_size < 1:
            raise ValueError('block_size must be positive (memory scheduling only).')
        result = np.empty(len(rgb), dtype=np.float64)
        for start in range(0, len(rgb), block_size):
            stop = start + block_size
            result[start:stop] = self.j_from_xyz(encoding.xyz(rgb[start:stop]))
        return result
