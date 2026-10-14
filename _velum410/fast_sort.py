"""Stable float64 radix argsort; exact ordering, not value quantization.

IEEE bits are transformed to monotone uint64 keys. Signed zeros share one
key and all NaNs share the last key, matching NumPy stable sorting semantics.
11-bit digits keep per-worker histograms small; small arrays retain NumPy.
"""
import numpy as np
from numba import njit, prange, get_num_threads

DIGIT_BITS = 11
BUCKETS = 1 << DIGIT_BITS


@njit(cache=True, nogil=True, parallel=True)
def radix_argsort(values, chunks):
    n = len(values)
    source_bits = values.view(np.uint64)
    keys = np.empty(n, np.uint64)
    order = np.empty(n, np.int64)
    sign = np.uint64(1) << np.uint64(63)
    last = np.uint64(0xffffffffffffffff)
    for i in prange(n):
        value = values[i]
        bits = source_bits[i]
        if np.isnan(value):
            keys[i] = last
        elif value == 0.:
            keys[i] = sign
        elif bits & sign:
            keys[i] = ~bits
        else:
            keys[i] = bits ^ sign
        order[i] = i
    ordered = True
    for i in range(1, n):
        if keys[i] < keys[i-1]:
            ordered = False
            break
    if ordered:
        return order
    next_keys = np.empty(n, np.uint64)
    next_order = np.empty(n, np.int64)
    counts = np.empty((chunks, BUCKETS), np.int64)
    offsets = np.empty_like(counts)
    mask = np.uint64(BUCKETS - 1)
    for shift in range(0, 64, DIGIT_BITS):
        for chunk in prange(chunks):
            counts[chunk, :] = 0
            start, stop = chunk*n//chunks, (chunk+1)*n//chunks
            for i in range(start, stop):
                digit = int((keys[i] >> np.uint64(shift)) & mask)
                counts[chunk, digit] += 1
        running = 0
        for digit in range(BUCKETS):
            for chunk in range(chunks):
                offsets[chunk, digit] = running
                running += counts[chunk, digit]
        for chunk in prange(chunks):
            start, stop = chunk*n//chunks, (chunk+1)*n//chunks
            for i in range(start, stop):
                key = keys[i]
                digit = int((key >> np.uint64(shift)) & mask)
                position = offsets[chunk, digit]
                next_keys[position] = key
                next_order[position] = order[i]
                offsets[chunk, digit] = position + 1
        keys, next_keys = next_keys, keys
        order, next_order = next_order, order
    return order


def stable_argsort(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 1:
        raise ValueError('Stable radix argsort expects a one-dimensional array.')
    # Pure scheduling crossover: at least one histogram bucket per input
    # element in each worker. It never changes a value or sorting criterion.
    workers = get_num_threads()
    if len(values) < BUCKETS * workers:
        return np.argsort(values, kind='stable')
    return radix_argsort(np.ascontiguousarray(values), workers)
