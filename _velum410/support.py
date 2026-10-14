"""Closest chroma in a fixed-J section of the reference's joint UCS hull.

Keep already-supported 3.5 targets exactly. For an outside target, intersect
the triangulated hull with its J plane, then find the closest section segment.
This is one convex feasibility correction, not a reference density matcher.
"""
import numpy as np
from numba import njit, prange
from scipy.spatial import ConvexHull


@njit(cache=True, nogil=True, parallel=True)
def _section_project(features, planes, equalities, triangles, lower_j, upper_j,
                     cuts, offsets, candidates, section_planes):
    out = features.copy()
    changed = np.zeros(len(features), np.bool_)
    missing = np.zeros(len(features), np.bool_)
    eps = np.finfo(np.float64).eps
    for row in prange(len(features)):
        j, a, b = features[row]
        bucket = np.searchsorted(cuts, j, side='right')
        first, last = offsets[bucket], offsets[bucket + 1]
        inside = True
        for slot in range(first if section_planes else 0, last if section_planes else len(planes)):
            face = candidates[slot] if section_planes else slot
            n0, n1, n2, offset = planes[face]
            tolerance = 64 * eps * (1. + abs(n0*j) + abs(n1*a) + abs(n2*b) + abs(offset))
            if n0*j + n1*a + n2*b + offset > tolerance:
                inside = False
                break
        if inside:
            for eq in equalities:
                value = eq[0]*j + eq[1]*a + eq[2]*b
                if abs(value) > 64*eps*(1. + abs(j) + abs(a) + abs(b)):
                    inside = False
                    break
        if inside:
            continue
        best = np.inf
        points = np.empty((3, 2), np.float64)
        for slot in range(first, last):
            face = candidates[slot]
            # The original edge tests would all fail outside this interval.
            # Keep face order and arithmetic for every intersecting triangle.
            if j < lower_j[face] or j > upper_j[face]:
                continue
            tri = triangles[face]
            count = 0
            for edge in range(3):
                v, u = tri[edge], tri[(edge + 1) % 3]
                if v[0] == j:
                    points[count, 0], points[count, 1] = v[1], v[2]
                    count += 1
                elif (v[0] < j < u[0]) or (u[0] < j < v[0]):
                    t = (j - v[0]) / (u[0] - v[0])
                    points[count, 0] = v[1] + t*(u[1] - v[1])
                    points[count, 1] = v[2] + t*(u[2] - v[2])
                    count += 1
            for k in range(count):
                va, vb = points[k]
                ua, ub = points[(k + 1) % count]
                da, db = ua-va, ub-vb
                length = da*da + db*db
                t = min(1., max(0., ((a-va)*da + (b-vb)*db)/length)) if length else 0.
                ca, cb = va + t*da, vb + t*db
                distance = (a-ca)**2 + (b-cb)**2
                if distance < best:
                    best = distance
                    out[row, 1], out[row, 2] = ca, cb
        missing[row] = not np.isfinite(best)
        changed[row] = True
    return out, changed, missing


def section_project(features, planes, equalities, triangles, lower_j, upper_j):
    section_planes = len(equalities) == 0 and len(planes) == len(triangles) and len(triangles) > 64
    if section_planes:
        section_planes = not (np.any(features[:, 0] < lower_j.min()) or np.any(features[:, 0] > upper_j.max()))
    if section_planes:
        # A fixed-J convex section is bounded by faces intersecting that section.
        # Closed interval buckets retain boundary faces and original tie order.
        cuts = np.linspace(lower_j.min(), upper_j.max(), 129)[1:-1]
        starts = np.searchsorted(cuts, lower_j, side='right')
        ends = np.searchsorted(cuts, upper_j, side='right')
        groups = [np.flatnonzero((starts <= bucket) & (ends >= bucket)) for bucket in range(len(cuts) + 1)]
        offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
        candidates = np.concatenate(groups)
    else:
        cuts = np.empty(0, np.float64)
        offsets = np.array([0, len(triangles)], np.int64)
        candidates = np.arange(len(triangles))
    return _section_project(features, planes, equalities, triangles, lower_j, upper_j,
                            cuts, offsets, candidates, section_planes)


class JointSupport:
    def __init__(self, reference, weights):
        reference = np.asarray(reference, np.float64)
        weights = np.asarray(weights, np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference):
            raise ValueError('Expected nonempty reference Jprime/a/b samples.')
        if weights.shape != (len(reference),) or np.any(weights <= 0):
            raise ValueError('Reference atom weights must be positive.')
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(weights)):
            raise ValueError('Reference samples and weights must be finite.')
        self.center = np.average(reference, axis=0, weights=weights)
        self.scale = max(float(np.ptp(reference, axis=0).max()), np.finfo(float).tiny)
        if np.all(reference == reference[0]):
            self.scale = 1.
        ref = (reference - self.center) / self.scale
        padded = ref if len(ref) >= 3 else np.vstack((ref, np.zeros((3 - len(ref), 3))))
        _, singular, axes = np.linalg.svd(padded, full_matrices=False)
        threshold = np.finfo(float).eps * max(padded.shape) * singular[0]
        self.rank = int(np.count_nonzero(singular > threshold)) if singular[0] else 0
        basis = axes[:self.rank]
        intrinsic = ref @ basis.T
        if self.rank >= 2:
            hull = ConvexHull(intrinsic)
            normal = hull.equations[:, :-1] @ basis
            self.planes = np.ascontiguousarray(np.column_stack((normal, hull.equations[:, -1])))
            if self.rank == 3:
                self.triangles = np.ascontiguousarray(ref[hull.simplices])
            else:
                # A fan fills the rank-2 polygon. Its intersection is the
                # complete feasible segment, including a horizontal polygon.
                self.triangles = np.ascontiguousarray(np.concatenate((
                    np.zeros((len(hull.simplices), 1, 3)), ref[hull.simplices]), axis=1))
        elif self.rank == 1:
            self.planes = np.ascontiguousarray(np.array([
                np.r_[basis[0], -intrinsic[:, 0].max()],
                np.r_[-basis[0], intrinsic[:, 0].min()]]))
            lo, hi = np.argmin(intrinsic[:, 0]), np.argmax(intrinsic[:, 0])
            self.triangles = np.ascontiguousarray(ref[[lo, hi, hi]][None])
        else:
            self.planes = np.empty((0, 4), np.float64)
            self.triangles = np.ascontiguousarray(np.repeat(ref[:1], 3, axis=0)[None])
        self.equalities = np.ascontiguousarray(axes[self.rank:])
        self.lower_j = np.ascontiguousarray(self.triangles[:, :, 0].min(axis=1))
        self.upper_j = np.ascontiguousarray(self.triangles[:, :, 0].max(axis=1))
        j = reference[:, 0]
        self.low_j, self.high_j = float(j.min()), float(j.max())
        self.low = np.average(reference[j == self.low_j], axis=0, weights=weights[j == self.low_j])
        self.high = np.average(reference[j == self.high_j], axis=0, weights=weights[j == self.high_j])

    def correct(self, target):
        target = np.asarray(target, np.float64)
        if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
            raise ValueError('Expected finite N x 3 targets.')
        if np.any(target[:, 0] < self.low_j) or np.any(target[:, 0] > self.high_j):
            raise ValueError('Target Jprime must lie inside reference range.')
        normalized = (target - self.center) / self.scale
        projected, changed, missing = section_project(np.ascontiguousarray(normalized),
            self.planes, self.equalities, self.triangles, self.lower_j, self.upper_j)
        if np.any(missing):
            raise ArithmeticError('Could not intersect reference support at target J.')
        result = target.copy()
        result[changed, 1:] = (projected * self.scale + self.center)[changed, 1:]
        return result, changed
