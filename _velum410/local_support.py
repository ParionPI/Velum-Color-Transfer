"""Local reference colour graph: do not fill distant hue gaps with a hull.

The graph is mutual (d+1)-nearest neighbours in joint CAM16-UCS. Its local
spacing supplies a sampling-resolution acceptance neighbourhood. Outside
that neighbourhood, intersect real short edges with the fixed-J plane.
No spatial masks, colour names, saturation boosts or random assignments.
"""
import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange, get_num_threads


@njit(cache=True, nogil=True, parallel=True)
def _intersect_sweep(targets, reference, edges, start_order, end_order, lower, upper, chunks):
    """Independent sorted-J sweeps; exact edge ties, no dense distance matrix."""
    order = np.argsort(targets[:, 0])
    out = targets.copy()
    found = np.zeros(len(targets), np.bool_)
    comparisons = 0
    for chunk in prange(chunks):
        active = np.empty(len(edges), np.int64)
        position = np.full(len(edges), -1, np.int64)
        active_count = start = end = 0
        for index in range(len(order) * chunk // chunks, len(order) * (chunk + 1) // chunks):
            row = order[index]
            j, a, b = targets[row]
            while start < len(edges) and lower[start_order[start]] <= j:
                edge = start_order[start]
                position[edge] = active_count
                active[active_count] = edge
                active_count += 1
                start += 1
            while end < len(edges) and upper[end_order[end]] < j:
                edge = end_order[end]
                p = position[edge]
                if p >= 0:
                    active_count -= 1
                    last = active[active_count]
                    active[p] = last
                    position[last] = p
                    position[edge] = -1
                end += 1
            best = np.inf
            best_edge = len(edges)
            comparisons += active_count
            for slot in range(active_count):
                edge = active[slot]
                v, u = reference[edges[edge, 0]], reference[edges[edge, 1]]
                if u[0] != v[0]:
                    t = (j - v[0]) / (u[0] - v[0])
                else:
                    da, db = u[1]-v[1], u[2]-v[2]
                    length = da*da + db*db
                    t = min(1., max(0., ((a-v[1])*da + (b-v[2])*db)/length)) if length else 0.
                ca = v[1] + t*(u[1]-v[1])
                cb = v[2] + t*(u[2]-v[2])
                distance = (a-ca)**2 + (b-cb)**2
                if distance < best or (distance == best and edge < best_edge):
                    best, best_edge = distance, edge
                    out[row, 1], out[row, 2] = ca, cb
                    found[row] = True
    return out, found, comparisons


def intersect_sweep(targets, reference, edges, start_order, end_order, lower, upper):
    # Keep parallel active/position scratch below 64 MiB (or one serial sweep).
    chunks = min(get_num_threads(), max(1, len(targets) // 2048),
                 max(1, (64 * 1024 * 1024) // (16 * max(len(edges), 1))))
    return _intersect_sweep(targets, reference, edges, start_order, end_order, lower, upper, chunks)


class LocalReferenceSupport:
    def __init__(self, reference, workers=1):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer.')
        self.workers = workers
        self.reference = np.asarray(reference, np.float64)
        if (self.reference.ndim != 2 or self.reference.shape[1] != 3
                or not len(self.reference) or not np.all(np.isfinite(self.reference))):
            raise ValueError('Expected finite nonempty N x 3 reference.')
        self.tree = cKDTree(self.reference)
        n, dimension = self.reference.shape
        self.k = min(dimension + 1, n - 1)
        self.radius = np.zeros(n)
        if self.k:
            distances, neighbours = self.tree.query(self.reference, k=self.k + 1, workers=self.workers)
            neighbours, distances = neighbours[:, 1:], distances[:, 1:]
            ids = np.arange(n)
            source = np.repeat(ids, self.k)
            destination = neighbours.ravel()
            # The same reverse-edge membership check, without sorting global
            # integer edge codes; k is unchanged and output edge order is kept.
            mutual_matrix = np.empty((n, self.k), bool)
            for column in range(self.k):
                mutual_matrix[:, column] = np.any(neighbours[neighbours[:, column]] == ids[:, None], axis=1)
            mutual = mutual_matrix.ravel()
            np.maximum.at(self.radius, source[mutual], distances.ravel()[mutual])
            unique = mutual & (source < destination)
            graph = np.c_[source[unique], destination[unique]]
        else:
            graph = np.empty((0, 2), np.int64)
        self.graph_edges = len(graph)
        # Include isolated real samples at their exact J, without bridging gaps.
        vertices = np.arange(n)
        self.edges = np.ascontiguousarray(np.vstack((graph, np.c_[vertices, vertices])), dtype=np.int64)
        j0 = self.reference[self.edges[:, 0], 0]
        j1 = self.reference[self.edges[:, 1], 0]
        self.lower, self.upper = np.minimum(j0, j1), np.maximum(j0, j1)
        self.start_order = np.argsort(self.lower, kind='stable')
        self.end_order = np.argsort(self.upper, kind='stable')

    def correct(self, target) -> tuple[np.ndarray, np.ndarray, dict]:
        target = np.asarray(target, np.float64)
        if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
            raise ValueError('Expected finite N x 3 targets.')
        output = target.copy()
        changed = np.zeros(len(target), bool)
        if not len(target):
            return output, changed, dict(graph_edges=self.graph_edges, unsupported=0, fallback=0)
        k = min(self.k + 1, len(self.reference))
        distances, ids = self.tree.query(target, k=k, workers=self.workers)
        distances, ids = distances.reshape(len(target), k), ids.reshape(len(target), k)
        tolerance = 128*np.finfo(float).eps*(1 + np.max(np.abs(self.reference)))
        accepted = np.any(distances <= self.radius[ids] + tolerance, axis=1)
        rows = np.flatnonzero(~accepted)
        found = np.zeros(len(rows), bool)
        comparisons = 0
        if len(rows):
            corrected, found, comparisons = intersect_sweep(np.ascontiguousarray(target[rows]),
                self.reference, self.edges, self.start_order, self.end_order, self.lower, self.upper)
            selected = rows[found]
            output[selected, 1:] = corrected[found, 1:]
            changed[selected] = np.any(output[selected, 1:] != target[selected, 1:], axis=1)
        audit = dict(method='mutual-neighbour local support, nearest fixed-J graph intersection',
            neighbours=self.k, graph_edges=self.graph_edges, unsupported=int(len(rows)),
            fallback=int(np.count_nonzero(~found)), active_edge_comparisons=int(comparisons),
            median_reference_radius=float(np.median(self.radius)),
            acceptance_scope='local graph sampling scale, not exact density',
            fallback_policy='keep accepted 3.51 target when local graph has no J intersection')
        return output, changed, audit
