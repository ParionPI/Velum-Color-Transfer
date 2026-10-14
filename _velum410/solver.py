"""Monotone projection onto actual reference colors in a reference-relative space.

Primary distance is CAM16-UCS lightness plus the source-informed chromatic
subspace, without an empirical lightness/chroma trade-off coefficient.
DP is exact on the sparse candidate graph for the primary additive objective;
TV tie resolution and palette coverage are not claimed globally optimal.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from numba import njit


@dataclass(frozen=True)
class SearchBudget:
    neighbors:int=16
    distribution_rotations:int=16
    def __post_init__(self):
        if not 1<=self.neighbors<=1024 or not 1<=self.distribution_rotations<=256:
            raise ValueError('Invalid search/distribution compute budget.')


def midpoint_group_map(source_counts,reference_counts):
    n,m=int(source_counts.sum()),int(reference_counts.sum())
    if 2*n*m>np.iinfo(np.int64).max:raise OverflowError('Unsafe integer rank arithmetic.')
    centers=2*np.cumsum(source_counts)-source_counts
    return np.searchsorted(np.cumsum(reference_counts),(centers*m)//(2*n),side='right').astype(np.int32)


def objective(mapping,target,reference,counts,reference_counts,source_codes,reference_codes):
    w=counts/counts.sum()
    residual=reference[mapping]-target
    parts=np.sum(w[:,None]*residual*residual,axis=0)
    energy=float(np.sum(parts))
    histogram=np.zeros(len(reference),np.int64)
    np.add.at(histogram,mapping,counts)
    n,m=int(counts.sum()),int(reference_counts.sum())
    tv_num=n*m-int(np.minimum(histogram*m,reference_counts*n).sum())
    changed=int(counts[source_codes!=reference_codes[mapping]].sum())
    return (energy,tv_num,changed),histogram,parts


@njit(cache=True)
def _better(cost,overlap,changed,bestcost,bestoverlap,bestchanged):
    return cost<bestcost or (cost==bestcost and (overlap>bestoverlap or (overlap==bestoverlap and changed<bestchanged)))


@njit(cache=True)
def squared_distance(a,b):
    value=0.0
    for d in range(len(a)):
        delta=a[d]-b[d];value+=delta*delta
    return value


def minimax_cdf_certificate(source_counts,reference_counts):
    """Exact ordered-boundary minimax discrepancy and a witnessing mapping.

    This certificate is diagnostic. The production corridor is one maximum
    indivisible source atom: a universal data-derived granularity allowance.
    All arithmetic comparisons are exact integers.
    """
    n,m=int(source_counts.sum()),int(reference_counts.sum())
    source_boundaries=np.r_[0,np.cumsum(source_counts,dtype=np.int64)]
    reference_boundaries=np.cumsum(reference_counts,dtype=np.int64)
    positions=np.searchsorted(source_boundaries*m,reference_boundaries*n)
    left=np.maximum(positions-1,0);right=np.minimum(positions,len(source_boundaries)-1)
    dl=np.abs(source_boundaries[left]*m-reference_boundaries*n)
    dr=np.abs(source_boundaries[right]*m-reference_boundaries*n)
    chosen=np.where(dl<=dr,left,right)
    if np.any(np.diff(chosen)<0):raise AssertionError('Nearest cumulative boundaries must be monotone.')
    discrepancy=int(np.max(np.minimum(dl,dr)))
    labels=np.searchsorted(chosen,np.arange(len(source_counts)),side='right').astype(np.int32)
    return discrepancy,labels


@njit(cache=True)
def sparse_dp_cdf(candidates,target,reference,scounts,rcounts,scodes,rcodes,cdf_bound_numerator):
    """Sparse monotone DP with exact integer CDF-corridor transitions.

    cdf_bound_numerator compares |source_mass*M-reference_mass*N|. A transition
    finalizes every skipped reference boundary at the mass before this group.
    """
    u,k=candidates.shape;n=scounts.sum();m=rcounts.sum()
    ref_scaled=np.cumsum(rcounts)*n
    parent=np.full((u,k),-1,np.int16)
    costs=np.full(k,np.inf);overlaps=np.zeros(k,np.int64)
    changed=np.zeros(k,np.int64);masses=np.zeros(k,np.int64)
    source_before=0
    for i in range(u):
        nc=np.full(k,np.inf);no=np.zeros(k,np.int64);nh=np.zeros(k,np.int64);nm=np.zeros(k,np.int64)
        boundary_scaled=source_before*m
        low_value=boundary_scaled-cdf_bound_numerator
        high_value=boundary_scaled+cdf_bound_numerator
        low_label=np.searchsorted(ref_scaled,low_value)
        high_boundary=np.searchsorted(ref_scaled,high_value,side='right')-1
        cursor=0
        while i>0 and cursor<k and candidates[i-1,cursor]<low_label:cursor+=1
        prefix=-1
        for a in range(k):
            j=candidates[i,a]
            local=(scounts[i]/n)*squared_distance(target[i],reference[j])
            change=scounts[i] if scodes[i]!=rcodes[j] else 0
            if i==0:
                # Labels below j receive zero output mass and are finalized now.
                if j==0 or ref_scaled[j-1]<=cdf_bound_numerator:
                    nc[a]=local;no[a]=min(scounts[i]*m,rcounts[j]*n);nh[a]=change;nm[a]=scounts[i]
                continue
            while cursor<k and candidates[i-1,cursor]<j:
                if prefix<0 or _better(costs[cursor],overlaps[cursor],changed[cursor],costs[prefix],overlaps[prefix],changed[prefix]):prefix=cursor
                cursor+=1
            # A strict label increase finalizes boundaries previous_label..j-1.
            if j-1<=high_boundary and prefix>=0 and np.isfinite(costs[prefix]):
                nc[a]=costs[prefix]+local;no[a]=overlaps[prefix]+min(scounts[i]*m,rcounts[j]*n)
                nh[a]=changed[prefix]+change;nm[a]=scounts[i];parent[i,a]=prefix
            # Staying on j finalizes no reference boundary and is always legal.
            b=cursor
            while b<k and candidates[i-1,b]==j:
                if np.isfinite(costs[b]):
                    gain=min(scounts[i]*m,max(rcounts[j]*n-masses[b]*m,0))
                    co,ov,ch=costs[b]+local,overlaps[b]+gain,changed[b]+change
                    if _better(co,ov,ch,nc[a],no[a],nh[a]):
                        nc[a],no[a],nh[a],nm[a],parent[i,a]=co,ov,ch,masses[b]+scounts[i],b
                b+=1
        costs,overlaps,changed,masses=nc,no,nh,nm
        source_before+=scounts[i]
    best=-1
    total_scaled=n*m
    for a in range(k):
        j=candidates[u-1,a]
        # Boundaries j..last receive all output mass and are finalized at N.
        legal=ref_scaled[j]>=total_scaled-cdf_bound_numerator
        if legal and np.isfinite(costs[a]) and (best<0 or _better(costs[a],overlaps[a],changed[a],costs[best],overlaps[best],changed[best])):best=a
    if best<0:raise ValueError('CDF corridor and candidate graph have no feasible path.')
    mapping=np.empty(u,np.int32)
    for i in range(u-1,-1,-1):
        mapping[i]=candidates[i,best];best=parent[i,best]
    return mapping


def build_candidates(target,reference,rcounts,baseline,neighbors,enforce_range):
    k=min(neighbors,len(reference));u=len(target)
    tree=cKDTree(reference)
    nearest=np.asarray(tree.query(target,k=k,workers=-1)[1]).reshape(u,k).astype(np.int32)
    heavy=np.argsort(-rcounts,kind='stable')[:k].astype(np.int32)
    offsets=np.arange(k,dtype=np.int32)-k//2
    # Clip label INDICES to valid array bounds, never image values.
    tone=np.clip(baseline[:,None]+offsets[None,:],0,len(reference)-1).astype(np.int32)
    pieces=[nearest,tone,np.broadcast_to(heavy,(u,k)),baseline[:,None]]
    strata=min(int(np.ceil(np.sqrt(k))),len(reference))
    edges=np.linspace(0,len(reference),strata+1,dtype=np.int64)
    for left,right in zip(edges[:-1],edges[1:]):
        kk=min(max(1,k//strata),int(right-left));local=cKDTree(reference[left:right])
        idx=np.asarray(local.query(target,k=kk,workers=-1)[1]).reshape(u,kk)
        pieces.append((idx+left).astype(np.int32))
    candidates=np.sort(np.concatenate(pieces,axis=1),axis=1).astype(np.int32)
    if enforce_range:
        candidates[0,:]=0;candidates[-1,:]=len(reference)-1
    return candidates


def solve(target,reference,counts,rcounts,scodes,rcodes,quantile_map,budget=SearchBudget()):
    n,m=int(counts.sum()),int(rcounts.sum())
    minimax_numerator,baseline=minimax_cdf_certificate(counts,rcounts)
    # One maximum source atom is the universal CDF granularity allowance when
    # exact reference-range endpoints are also mandatory. Half an atom covers
    # nearest boundaries, but can be infeasible when a tiny endpoint color must
    # receive a whole first/last source group. No fitted visual tolerance.
    atom_bound_pixels=int(np.max(counts))
    corridor_numerator=atom_bound_pixels*m
    enforce_range=len(counts)>1
    if enforce_range:baseline[0]=0;baseline[-1]=len(reference)-1
    best=baseline.copy();best_score,hist,_=objective(best,target,reference,counts,rcounts,scodes,rcodes)
    initial=best_score
    baseline_cdf_numerator=int(np.max(np.abs(np.cumsum(hist,dtype=np.int64)*m-np.cumsum(rcounts,dtype=np.int64)*n)))
    if baseline_cdf_numerator>corridor_numerator:
        raise AssertionError('Minimax witness exceeds the atomic CDF corridor.')
    trace=[{'stage':'minimax-CDF witness','energy':best_score[0],
            'CDF_error':baseline_cdf_numerator/(n*m)}]
    columns=0
    if best_score!=(0.0,0,0):
        candidates=build_candidates(target,reference,rcounts,baseline,budget.neighbors,enforce_range)
        columns=candidates.shape[1]
        candidate=sparse_dp_cdf(candidates,target,reference,counts,rcounts,scodes,rcodes,corridor_numerator)
        score,_,_=objective(candidate,target,reference,counts,rcounts,scodes,rcodes)
        if score<best_score:best,best_score=candidate.copy(),score
        trace.append({'stage':'CDF-corridor monotone DP','energy':score[0]})
    if np.any(np.diff(best)<0):raise AssertionError('Internal monotonicity failure.')
    if enforce_range and (best[0]!=0 or best[-1]!=len(reference)-1):raise AssertionError('Range endpoint failure.')
    score,histogram,parts=objective(best,target,reference,counts,rcounts,scodes,rcodes)
    cdf_numerator=int(np.max(np.abs(np.cumsum(histogram,dtype=np.int64)*m-np.cumsum(rcounts,dtype=np.int64)*n)))
    if cdf_numerator>corridor_numerator:raise AssertionError('Output escaped the exact CDF corridor.')
    return best,{'method':'reference-relative chroma + atomic-CDF-corridor monotone palette projection',
        'global_optimum_claimed':False,'primary_exact_on_candidate_graph_with_CDF_corridor':True,
        'TV_global_optimum_claimed':False,
        'budget':{'neighbors':budget.neighbors},
        'candidate_columns':columns,'reference_range_endpoints_enforced':enforce_range,
        'degenerate_single_source_color':len(counts)==1,
        'CDF_corridor_definition':'one maximum source RGB-group mass; universal with exact range endpoints',
        'CDF_corridor':corridor_numerator/(n*m),'CDF_corridor_output_pixel_equivalent':atom_bound_pixels,
        'exact_minimax_CDF':minimax_numerator/(n*m),
        'exact_minimax_CDF_output_pixel_equivalent':minimax_numerator/m,
        'achieved_CDF':cdf_numerator/(n*m),'achieved_CDF_output_pixel_equivalent':cdf_numerator/m,
        'baseline_energy':initial[0],'final_energy':score[0],
        'lightness_component':float(parts[0]),'relative_chroma_component':float(np.sum(parts[1:])),
        'total_variation':score[1]/(n*m),'changed_code_pixels':score[2],'trace':trace}
