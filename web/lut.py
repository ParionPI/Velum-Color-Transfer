"""Small, bounded 3-D LUT fitter for the web display result.

The public grid is indexed ``[blue, green, red, channel]``.  Consequently
the last (red) coordinate is the fastest coordinate in a C-order flattening,
which is the ordering used by the .cube format.
"""

from __future__ import annotations

import time
import json
import re
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


_MAX_SAMPLES = 200_000


def _check_image(value: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != 3 or not a.shape[0] or not a.shape[1]:
        raise ValueError(f"{name} must be a non-empty HxWx3 array")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.min(a) < 0.0 or np.max(a) > 1.0:
        raise ValueError(f"{name} values must be in [0, 1]")
    return a


def _corners(x: np.ndarray, size: int):
    p = np.clip(x, 0.0, 1.0) * (size - 1)
    lo = np.floor(p).astype(np.int32)
    hi = np.minimum(lo + 1, size - 1)
    f = p - lo
    return lo, hi, f


def _basis(x: np.ndarray, size: int):
    lo, hi, f = _corners(x[:, [2, 1, 0]], size)
    weights = np.empty((len(x), 8), np.float64)
    columns = np.empty((len(x), 8), np.int32)
    for k, (ib, ig, ir) in enumerate(((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))):
        weights[:, k] = ((f[:, 2] if ir else 1-f[:, 2]) *
                      (f[:, 1] if ig else 1-f[:, 1]) *
                      (f[:, 0] if ib else 1-f[:, 0]))
        columns[:, k] = ((hi[:, 0] if ib else lo[:, 0]) * size +
                          (hi[:, 1] if ig else lo[:, 1])) * size + (hi[:, 2] if ir else lo[:, 2])
    basis = sparse.csr_matrix((weights.ravel(), columns.ravel(), np.arange(0, 8 * len(x) + 1, 8)),
                              shape=(len(x), size ** 3))
    basis.sum_duplicates()
    basis.eliminate_zeros()
    return basis


def _sample_split(source):
    flat = source.reshape(-1, 3)
    n = len(flat)
    # All temporaries stay bounded even for large source frames.
    candidates = np.linspace(0, n - 1, min(n, 4 * _MAX_SAMPLES), dtype=np.int64)
    first = np.full(16 ** 3, -1, np.int64)
    for start in range(0, n, 262144):
        bins = np.minimum((flat[start:start + 262144] * 16).astype(np.int32), 15)
        codes = bins[:, 0] + 16 * (bins[:, 1] + 16 * bins[:, 2])
        keys, indices = np.unique(codes, return_index=True)
        empty = first[keys] < 0
        first[keys[empty]] = start + indices[empty]
    representatives = first[first >= 0]
    candidates = np.setdiff1d(candidates, representatives, assume_unique=True)
    hashed = candidates.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
    hashed ^= hashed >> np.uint64(30)
    hashed *= np.uint64(0xBF58476D1CE4E5B9)
    train = candidates[(hashed & np.uint64(3)) != 0]
    valid = candidates[(hashed & np.uint64(3)) == 0]
    train = np.concatenate((representatives, train[np.linspace(0, len(train)-1,
                           min(len(train), _MAX_SAMPLES-len(representatives)), dtype=np.int64)]))
    if len(valid) > 80000:
        valid = valid[np.linspace(0, len(valid)-1, 80000, dtype=np.int64)]
    independent = bool(len(valid))
    if not independent:
        valid = train
    return train, valid, independent


def _apply_flat(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    size = grid.shape[0]
    lo, hi, f = _corners(image.reshape(-1, 3)[:, [2, 1, 0]], size)
    result = np.zeros((len(lo), 3), np.float32)
    for ib, ig, ir in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)):
        w = ((f[:, 2] if ir else 1-f[:, 2]) * (f[:, 1] if ig else 1-f[:, 1]) *
             (f[:, 0] if ib else 1-f[:, 0]))
        result += w[:, None].astype(np.float32) * grid[
            hi[:, 0] if ib else lo[:, 0], hi[:, 1] if ig else lo[:, 1],
            hi[:, 2] if ir else lo[:, 2]]
    return np.clip(result, 0.0, 1.0).reshape(image.shape)


def apply_lut(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    image = _check_image(image, "image")
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 4 or grid.shape[0] != grid.shape[1] or grid.shape[1] != grid.shape[2] or grid.shape[3] != 3:
        raise ValueError("grid must have shape (size, size, size, 3)")
    if grid.shape[0] < 2 or not np.isfinite(grid).all():
        raise ValueError("grid must be finite and have size >= 2")
    flat = image.reshape(-1, 3)
    result = np.empty_like(flat)
    for start in range(0, len(flat), 131072):
        result[start:start+131072] = _apply_flat(flat[start:start+131072], grid)
    return result.reshape(image.shape)


def fit_lut(source: np.ndarray, target: np.ndarray, size: int = 65):
    started = time.perf_counter()
    source, target = _check_image(source, "source"), _check_image(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shape")
    if not isinstance(size, (int, np.integer)) or int(size) not in (33, 65):
        raise ValueError("size must be 33 or 65")
    size = int(size)
    flat_n = source.shape[0] * source.shape[1]
    # Exact identity is a useful contract, not merely an optimization.
    if np.array_equal(source, target):
        axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
        b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
        grid = np.stack((r, g, b), axis=-1).astype(np.float32)
        val_count = min(flat_n, _MAX_SAMPLES)
        audit = {"sample_count": val_count, "validation_count": val_count, "rmse_255": 0.0,
                 "p95_error_255": 0.0, "max_error_255": 0.0, "size": size,
                 "fit_seconds": time.perf_counter() - started, "algorithm": "identity",
                 "notes": "RGB encoded trilinear LUT; unseen colours use conservative identity prior."}
        return grid, audit
    train, valid, independent = _sample_split(source)
    train_x = source.reshape(-1, 3)[train]
    train_y = target.reshape(-1, 3)[train]
    design = np.concatenate((train_x.astype(np.float64), np.ones((len(train_x), 1))), axis=1)
    # Minimum-change affine prior: identity is preferred in directions not
    # constrained by a thin/grey source gamut.  This is ridge, not pinv.
    prior = np.zeros((4, 3), np.float64)
    prior[:3] = np.eye(3)
    normal = design.T @ design
    scale = max(float(np.trace(normal)) / 4.0, 1.0)
    affine = np.linalg.solve(normal + np.eye(4) * (scale * 1e-6),
                             design.T @ train_y.astype(np.float64) + prior * (scale * 1e-6))
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    identity = np.stack((r, g, b), axis=-1).astype(np.float32)
    base = np.concatenate((identity.astype(np.float64), np.ones(identity.shape[:3] + (1,))), axis=-1) @ affine
    base = np.clip(base.reshape(-1, 3), 0.0, 1.0)
    basis = _basis(train_x, size)
    normal = (basis.T @ basis).tocsr()
    diagonal = normal.diagonal()
    strength = max(float(np.median(diagonal[diagonal > 0])), 1e-3)
    # Penalize curvature, not first derivatives: affine looks remain affine.
    second = sparse.diags([np.ones(size-2), -2*np.ones(size-2), np.ones(size-2)],
                          [0, 1, 2], shape=(size-2, size), format='csr')
    bend = second.T @ second
    curvature = sparse.kronsum(sparse.kronsum(bend, bend), bend, format='csr')
    system = normal + (2.0 * strength) * curvature + sparse.eye(size**3, format='csr') * (0.0002 * strength)
    rhs = basis.T @ (train_y.astype(np.float64) - basis @ base)
    preconditioner = sparse.diags(1.0 / system.diagonal(), format='csr')
    delta = np.empty_like(base)
    convergence = []
    for channel in range(3):
        delta[:, channel], info = cg(system, rhs[:, channel], M=preconditioner,
                                     rtol=3e-4, atol=1e-7, maxiter=1000)
        if info < 0 or not np.isfinite(delta[:, channel]).all():
            raise RuntimeError('LUT fit failed numerically')
        convergence.append(int(info))
    grid = np.clip(base + delta, 0.0, 1.0).reshape(size, size, size, 3).astype(np.float32)
    pred = _apply_flat(source.reshape(-1, 3)[valid], grid)
    err = np.linalg.norm(pred - target.reshape(-1, 3)[valid], axis=1) / np.sqrt(3.0) * 255.0
    audit = {"sample_count": int(len(train)), "validation_count": int(len(valid)),
             "rmse_255": float(np.sqrt(np.mean(err * err))), "p95_error_255": float(np.percentile(err, 95)),
             "max_error_255": float(np.max(err)), "size": size,
              "fit_seconds": time.perf_counter() - started, "algorithm": "curvature_regularized_trilinear_least_squares",
              "validation_independent": independent, "solver_converged": all(v == 0 for v in convergence),
              "solver_info": convergence,
              "notes": "Held-out pixels from this source image, not unseen-video validation. Unseen colours use a minimum-change affine prior; spatially varying mappings are averaged."}
    return grid, audit


def cube_bytes(grid: np.ndarray, metadata: Mapping[str, object] | None = None) -> bytes:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 4 or grid.shape[:3] != (grid.shape[0],) * 3 or grid.shape[3] != 3 or grid.shape[0] < 2:
        raise ValueError("grid must have shape (size, size, size, 3)")
    if not np.isfinite(grid).all():
        raise ValueError("grid contains non-finite values")
    size = grid.shape[0]
    lines = ["# Velum video-friendly 3D LUT", f"LUT_3D_SIZE {size}",
             "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    if metadata:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(metadata.get("title", ""))).strip("_")
        if safe:
            lines.insert(1, f"TITLE \"{safe[:120]}\"")
        lines.append("# metadata " + json.dumps(dict(metadata), ensure_ascii=True, allow_nan=False, default=str, separators=(",", ":")))
    grid = np.clip(grid, 0.0, 1.0)
    lines.extend("%.9f %.9f %.9f" % tuple(row) for row in grid.reshape(-1, 3))
    return ("\n".join(lines) + "\n").encode("ascii")
