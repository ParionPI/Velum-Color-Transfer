"""Velum 4.10: isolated-runtime integration and serial AUTO scoring."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Isolated worker sets cache policy before importing native libraries.
from .color import RGBEncoding, ViewingConditions
from .quantile import QuantileResult
from .deband import DebandOptions


@dataclass(frozen=True)
class Performance:
    mode: str = 'fast'
    workers: int = 8
    backend: str = 'auto'

    def __post_init__(self):
        if self.mode != 'fast' or self.workers < 1:
            raise ValueError('3.53 exposes only the fast path.')
        if self.backend not in {'auto', 'cpu', 'gpu'}:
            raise ValueError('backend must be auto, cpu or gpu')


def transfer(source, reference, source_encoding=None, reference_encoding=None,
              viewing=ViewingConditions(), performance=Performance(), deband=DebandOptions()):
    if not isinstance(performance, Performance):
        raise TypeError('performance must be a Performance instance.')
    from . import quantile
    from . import gpu_mapping
    from numba import get_num_threads, set_num_threads
    previous = get_num_threads()
    set_num_threads(min(previous, performance.workers))
    try:
        with gpu_mapping.session(performance.backend) as backend_info:
            result = quantile.transfer(source, reference, source_encoding, reference_encoding, viewing, deband)
    finally:
        set_num_threads(previous)
    stats = result.stats
    seconds = stats.get('seconds', {})
    stats['performance'] = dict(mode=performance.mode, workers=performance.workers, backend=dict(backend_info),
        stages={key: dict(seconds=float(value)) for key, value in seconds.items()})
    stats['optimizer'] = dict(residual_reconstruction=stats['reference_support'],
        fixed_J_realisation=dict(method='3.53 fixed-J baseline realization before optional legacy mapping',
            scope='baseline only, not final output when deband is enabled',
            seconds=seconds['projection_scatter'],
            compressed_pixel_fraction=stats['compressed_pixel_fraction'],
            weighted_mean_alpha=stats['weighted_mean_alpha'],
            j_inversions=stats['ladder_inversions_float']))
    return result


__version__ = '4.10'
__all__ = ['transfer', 'QuantileResult', 'Performance', 'DebandOptions', 'RGBEncoding', 'ViewingConditions']
