"""Weighted sliced transport for complete 2-D chroma residual distributions.

Each rotation exactly matches both weighted 1-D projection marginals in target
space. It creates only continuous TARGET coordinates; final RGB values are still
selected from the exact reference palette by the monotone solver.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .fast_sort import stable_argsort


@njit(cache=True, nogil=True)
def _integer_midpoints(order, weights, total):
    result = np.empty(len(order), np.float64)
    cumulative = np.int64(0)
    for i in range(len(order)):
        weight = weights[order[i]]
        cumulative += weight
        result[i] = (cumulative - weight / 2.) / total
    return result


def weighted_quantile_match(values,weights,reference_values,reference_weights):
    source_order=stable_argsort(values);reference_order=stable_argsort(reference_values)
    if weights.dtype == np.int64 and reference_weights.dtype == np.int64:
        source_u=_integer_midpoints(source_order,weights,np.sum(weights))
        reference_u=_integer_midpoints(reference_order,reference_weights,np.sum(reference_weights))
    else:
        source_u=(np.cumsum(weights[source_order])-weights[source_order]/2)/np.sum(weights)
        reference_u=(np.cumsum(reference_weights[reference_order])-reference_weights[reference_order]/2)/np.sum(reference_weights)
    mapped=np.interp(source_u,reference_u,reference_values[reference_order])
    result=np.empty_like(values,dtype=np.float64);result[source_order]=mapped
    return result


def weighted_sliced_transport(source,reference,source_weights,reference_weights,rotations=16):
    if rotations<1:raise ValueError('rotations must be positive.')
    transformed=np.asarray(source,dtype=np.float64).copy();reference=np.asarray(reference,dtype=np.float64)
    # CAM16-UCS a'/b' axes are perceptually defined axes, so the deterministic
    # quadrature starts there and covers [0,pi/2) with orthogonal pairs.
    angles=np.arange(rotations,dtype=np.float64)*np.pi/(2*rotations)
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle);first=np.array([c,s]);second=np.array([-s,c])
        p=weighted_quantile_match(transformed@first,source_weights,reference@first,reference_weights)
        q=weighted_quantile_match(transformed@second,source_weights,reference@second,reference_weights)
        transformed=p[:,None]*first+q[:,None]*second
    return transformed
