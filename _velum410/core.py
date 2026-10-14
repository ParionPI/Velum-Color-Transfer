"""Velum 3.3: Velum 3.1 algorithm packaged with the six-panel benchmark."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .color import RGBEncoding, ViewingConditions
from .appearance import Appearance
from .solver import SearchBudget, solve, midpoint_group_map
from .relative import build_targets


def pack(rgb):
    x=np.asarray(rgb).reshape(-1,3)
    codes=x[:,0].astype(np.uint32)
    codes <<= 16
    codes |= x[:,1].astype(np.uint32) << 8
    codes |= x[:,2].astype(np.uint32)
    return codes


def unpack(codes):
    return np.column_stack([(codes>>16)&255,(codes>>8)&255,codes&255]).astype(np.uint8)


def palette(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.size==0:
        raise ValueError('Expected nonempty H,W,3 RGB array.')
    if rgb.dtype==np.uint8:
        codes,inv,counts=np.unique(pack(rgb),return_inverse=True,return_counts=True)
        return unpack(codes),inv,counts,codes.astype(np.int64)
    if rgb.dtype==np.float64:
        if not np.all(np.isfinite(rgb)) or np.any(rgb<0) or np.any(rgb>1):
            raise ValueError('Scientific linear RGB must be finite in [0,1].')
        colors,inv,counts=np.unique(rgb.reshape(-1,3),axis=0,return_inverse=True,return_counts=True)
        return colors,inv.reshape(-1),counts,np.full(len(colors),-1,np.int64)
    raise ValueError('Expected uint8 nonlinear RGB or float64 linear RGB.')


def palette_counts(rgb):
    """Reference colours and weights without the unused per-pixel inverse map."""
    if not isinstance(rgb,np.ndarray) or rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.size==0:
        raise ValueError('Expected nonempty H,W,3 RGB array.')
    if rgb.dtype==np.uint8:
        codes,counts=np.unique(pack(rgb),return_counts=True)
        return unpack(codes),counts
    colors,_,counts,_=palette(rgb)
    return colors,counts


@dataclass
class TransferResult:
    rgb: np.ndarray
    stats: dict


def transfer(source,reference,source_encoding=None,reference_encoding=None,
             viewing=ViewingConditions(),budget=SearchBudget()):
    if reference.dtype!=np.uint8:
        raise ValueError('Reference palette is currently 8-bit RGB.')
    if source_encoding is None:
        source_encoding=RGBEncoding.linear_srgb() if source.dtype==np.float64 else RGBEncoding.from_icc(None)
    reference_encoding=reference_encoding or RGBEncoding.from_icc(None)
    started=perf_counter()
    scolors,si,scounts,scodes=palette(source)
    rcolors,_,rcounts,rcodes=palette(reference)
    t1=perf_counter()
    cam=Appearance(viewing)
    sj,sc=cam.describe(scolors,source_encoding)
    rj,rc=cam.describe(rcolors,reference_encoding)
    so=np.argsort(sj,kind='stable');ro=np.argsort(rj,kind='stable')
    t2=perf_counter()
    quantile_map=midpoint_group_map(scounts[so],rcounts[ro])
    target,reference_features,relative_info=build_targets(sj[so],sc[so],rj[ro],rc[ro],scounts[so],rcounts[ro],quantile_map,budget.distribution_rotations)
    mapping,optimizer=solve(target,reference_features,scounts[so],rcounts[ro],scodes[so],rcodes[ro],quantile_map,budget)
    t3=perf_counter()
    chosen_codes=np.empty(len(scolors),np.uint32)
    chosen_codes[so]=rcodes[ro[mapping]].astype(np.uint32)
    output=unpack(chosen_codes[si]).reshape(source.shape)
    t4=perf_counter()
    stats={'version':'3.3','source_pixels':int(scounts.sum()),'reference_pixels':int(rcounts.sum()),
        'source_unique_colors':len(scolors),'reference_unique_colors':len(rcolors),
        'output_unique_colors':len(np.unique(chosen_codes)),
        'source_encoding':source_encoding.label,'reference_encoding':reference_encoding.label,
        'viewing':{'white':'D65','surround':'Average','LA':viewing.adapting_luminance,'Yb':viewing.background_luminance},
        'relative_model':relative_info,'optimizer':optimizer,'seconds':{'histograms':t1-started,'appearance_and_sort':t2-t1,
            'search':t3-t2,'scatter':t4-t3,'core_total':t4-started}}
    return TransferResult(output,stats)
