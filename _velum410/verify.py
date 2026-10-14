"""Full-resolution saved-file checks, including reference range and tone distribution."""
from __future__ import annotations
import numpy as np
from numba import njit
from .core import palette,pack,unpack
from .appearance import Appearance
from .solver import objective,midpoint_group_map,minimax_cdf_certificate
from .relative import build_targets
from .color import ViewingConditions


@njit(cache=True)
def tone_wasserstein2(j,output_counts,reference_counts):
    n=output_counts.sum();m=reference_counts.sum()
    i=0;k=0;left_o=0;left_r=0;total=0.0
    while True:
        while left_o==0 and i<len(j):
            left_o=output_counts[i]*m
            if left_o==0:i+=1
        while left_r==0 and k<len(j):
            left_r=reference_counts[k]*n
            if left_r==0:k+=1
        if i>=len(j) or k>=len(j):break
        mass=min(left_o,left_r)
        delta=j[i]-j[k];total+=mass*(delta*delta)
        left_o-=mass;left_r-=mass
        if left_o==0:i+=1
        if left_r==0:k+=1
    return np.sqrt(total/(n*m))


def distribution_metrics(rj,rc,output_counts,reference_counts):
    n,m=int(output_counts.sum()),int(reference_counts.sum())
    order=np.argsort(rj,kind='stable')
    j=rj[order];oc=output_counts[order];rr=reference_counts[order]
    delta=oc*m-rr*n
    cumulative=np.cumsum(delta)
    ends=np.r_[np.flatnonzero(np.diff(j)!=0),len(j)-1]
    ks=float(np.max(np.abs(cumulative[ends])))/(n*m)
    w1=float(np.sum(np.abs(cumulative[:-1])*np.diff(j)))/(n*m)
    w2=float(tone_wasserstein2(j,oc,rr))
    ow=output_counts/n;rw=reference_counts/m
    om=np.sum(ow[:,None]*rc,axis=0);rm=np.sum(rw[:,None]*rc,axis=0)
    ocov=((rc-om).T*ow)@(rc-om);rcov=((rc-rm).T*rw)@(rc-rm)
    chromatic_w1=[]
    for axis in range(2):
        ao=np.argsort(rc[:,axis],kind='stable')
        acdf=np.cumsum(output_counts[ao]*m-reference_counts[ao]*n)
        chromatic_w1.append(float(np.sum(np.abs(acdf[:-1])*np.diff(rc[ao,axis])))/(n*m))
    used=output_counts>0
    return {'chroma_axis_W1_a':chromatic_w1[0],'chroma_axis_W1_b':chromatic_w1[1],
        'reference_J_range':[float(np.min(rj)),float(np.max(rj))],
        'output_J_range':[float(np.min(rj[used])),float(np.max(rj[used]))],
        'J_distribution_KS':ks,'J_distribution_W1':w1,'J_distribution_W2':w2,
        'actual_chroma_mean':om.tolist(),'reference_chroma_mean':rm.tolist(),
        'actual_chroma_covariance':ocov.tolist(),'reference_chroma_covariance':rcov.tolist(),
        'actual_chroma_mean_error_norm':float(np.linalg.norm(om-rm)),
        'actual_chroma_covariance_error_norm':float(np.linalg.norm(ocov-rcov))}


def verify(source,reference,output,source_encoding,reference_encoding,viewing=ViewingConditions()):
    if output.shape!=source.shape or output.dtype!=np.uint8:return {'passed':False,'error':'Wrong output shape/dtype'}
    scolors,si,scounts,scodes=palette(source)
    rcolors,_,rcounts,rcodes=palette(reference)
    output_codes,output_counts=np.unique(pack(output),return_counts=True)
    outside=~np.isin(output_codes,rcodes,assume_unique=True)
    if np.any(outside):return {'passed':False,'outside_palette_pixels':int(output_counts[outside].sum())}
    flat=pack(output)
    lo=np.full(len(scolors),np.iinfo(np.uint32).max,np.uint32);hi=np.zeros(len(scolors),np.uint32)
    np.minimum.at(lo,si,flat);np.maximum.at(hi,si,flat)
    splits=int(np.count_nonzero(lo!=hi))
    report={'outside_palette_pixels':0,'split_source_color_groups':splits}
    if splits:report['passed']=False;return report
    cam=Appearance(viewing)
    sj,sc=cam.describe(scolors,source_encoding);rj,rc=cam.describe(rcolors,reference_encoding)
    mapping=np.searchsorted(rcodes,lo);oj=rj[mapping]
    so=np.argsort(sj,kind='stable');ro=np.argsort(rj,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(sj[so])!=0)+1]
    low=np.minimum.reduceat(oj[so],starts);high=np.maximum.reduceat(oj[so],starts)
    differences=np.maximum.accumulate(high)[:-1]-low[1:]
    inversions=int(np.count_nonzero(differences>0))
    rank=np.empty(len(ro),np.int32);rank[ro]=np.arange(len(ro),dtype=np.int32)
    q=midpoint_group_map(scounts[so],rcounts[ro])
    target,ref_features,model=build_targets(sj[so],sc[so],rj[ro],rc[ro],scounts[so],rcounts[ro],q)
    score,histogram_ordered,parts=objective(rank[mapping[so]],target,ref_features,scounts[so],rcounts[ro],scodes[so],rcodes[ro])
    hist=np.empty_like(histogram_ordered);hist[ro]=histogram_ordered
    n,m=int(scounts.sum()),int(rcounts.sum())
    dist=distribution_metrics(rj,rc,hist,rcounts)
    ordered_delta=np.cumsum(hist[ro],dtype=np.int64)*m-np.cumsum(rcounts[ro],dtype=np.int64)*n
    cdf_numerator=int(np.max(np.abs(ordered_delta)))
    cdf_bound_numerator=int(np.max(scounts))*m
    minimax_numerator,_=minimax_cdf_certificate(scounts[so],rcounts[ro])
    cdf_ok=cdf_numerator<=cdf_bound_numerator
    range_feasible=len(scolors)>1
    range_exact=dist['reference_J_range']==dist['output_J_range']
    range_ok=range_exact if range_feasible else np.min(oj)==np.max(oj)
    count_delta=hist*m-rcounts*n
    largest=np.argsort(-scounts,kind='stable')[:5]
    report.update({'passed':bool(inversions==0 and range_ok and cdf_ok),
        'cam16_inversion_groups':inversions,
        'cam16_max_inversion_J':max(0.0,float(differences.max())) if len(differences) else 0.0,
        'full_reference_range_feasible':range_feasible,'full_reference_J_range_exact':range_exact,
        'J_CDF_corridor_passed':bool(cdf_ok),
        'J_CDF_max_error':cdf_numerator/(n*m),
        'J_CDF_atomic_bound':cdf_bound_numerator/(n*m),
        'J_CDF_atomic_bound_pixel_equivalent':int(np.max(scounts)),
        'J_CDF_exact_minimax':minimax_numerator/(n*m),
        'J_CDF_exact_minimax_pixel_equivalent':minimax_numerator/m,
        'max_achievable_range_satisfied':bool(range_ok),
        'exact_reference_histogram':bool(np.all(count_delta==0)),
        'max_color_count_error_pixels':float(np.max(np.abs(count_delta)))/m,
        'total_variation':score[1]/(n*m),'output_unique_colors':len(output_codes),
        'reference_unique_colors':len(rcolors),'source_unique_colors':len(scolors),
        'source_unique_J':len(starts),'output_unique_J':len(np.unique(oj)),
        'projection_energy':score[0],'lightness_component':float(parts[0]),
        'relative_chroma_component':float(np.sum(parts[1:])),
        'largest_source_groups':[{'source_rgb':scolors[i].tolist(),'pixels':int(scounts[i]),'output_rgb':unpack(lo[i:i+1])[0].tolist()} for i in largest],
        'distribution':dist})
    if reference.shape==output.shape:
        diff=output.astype(np.int16)-reference.astype(np.int16)
        report['reference_reconstruction']={'exact_pixel_fraction':float(np.all(diff==0,axis=2).mean()),
            'rgb_mae_8bit':float(np.abs(diff).mean()),'rgb_max_error_8bit':int(np.abs(diff).max()),'exact':bool(np.all(diff==0))}
    if source.dtype==np.uint8 and source_encoding.icc==reference_encoding.icc and np.array_equal(source,reference):
        report['self_mapping_exact']=bool(np.array_equal(output,source));report['passed']&=report['self_mapping_exact']
    return report
