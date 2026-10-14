"""Reference-relative chroma transport conditioned on lightness rank.

No fitting to a desired output image. Inputs are ONLY source and reference.
The Gaussian Monge map is a 2x2 data-derived PSD map, not a tuned hue/saturation
coefficient. Rank-deficient source directions provide no chromatic evidence.
"""
from __future__ import annotations
import numpy as np
from .distribution import weighted_sliced_transport


def rank_coordinates(counts):
    w=counts/np.sum(counts)
    u=(np.cumsum(counts)-counts/2)/np.sum(counts)
    centered=u-0.5
    deviation=np.sqrt(np.sum(w*centered*centered))
    v=centered/deviation if deviation>0 else np.zeros_like(centered)
    return w,u,v


def conditional_statistics(chroma,counts):
    w,u,v=rank_coordinates(counts)
    # Anchored mean is exact for a constant field, avoiding false residuals.
    mean=chroma[0]+np.sum(w[:,None]*(chroma-chroma[0]),axis=0)
    centered=chroma-mean
    vv=float(np.sum(w*v*v))
    slope=np.sum(w[:,None]*v[:,None]*centered,axis=0)/vv if vv>0 else np.zeros(2)
    residual=centered-v[:,None]*slope
    covariance=(residual.T*w)@residual
    # Rank must be measured against the PRE-residualization arithmetic scale.
    # Otherwise a perfectly linear field's ~1e-30 cancellation covariance can
    # appear full-rank relative to itself and be amplified into real colors.
    joint=np.column_stack([v,centered])
    joint_covariance=(joint.T*w)@joint
    joint_norm=np.sqrt(max(0.0,float(np.linalg.eigvalsh(joint_covariance)[-1])))
    singular_roundoff=np.finfo(np.float64).eps*max(joint.shape)*joint_norm
    covariance_floor=singular_roundoff**2
    values,vectors,positive=psd_spectral(covariance)
    keep=positive&(values>covariance_floor)
    covariance=(vectors*np.where(keep,values,0.0))@vectors.T
    return {'mean':mean,'slope':slope,'covariance':covariance,'residual':residual,'v':v,'u':u,'w':w,
            'residual_covariance_roundoff_floor':covariance_floor}


def psd_spectral(matrix):
    matrix=(matrix+matrix.T)/2
    values,vectors=np.linalg.eigh(matrix)
    tolerance=np.finfo(np.float64).eps*matrix.shape[0]*max(float(np.max(np.abs(values))),0.0)
    if np.min(values)<-tolerance:
        raise ValueError('Covariance is not positive semidefinite beyond roundoff.')
    positive=values>tolerance
    return values,vectors,positive


def psd_power(matrix,power):
    values,vectors,positive=psd_spectral(matrix)
    scaled=np.zeros_like(values)
    scaled[positive]=values[positive]**power
    return (vectors*scaled)@vectors.T


def monge_map(source_covariance,reference_covariance):
    root=psd_power(source_covariance,0.5)
    inverse_root=psd_power(source_covariance,-0.5)
    middle=psd_power(root@reference_covariance@root,0.5)
    a=inverse_root@middle@inverse_root
    a=(a+a.T)/2
    achieved=a@source_covariance@a.T
    _,vectors,positive=psd_spectral(achieved)
    projector=(vectors*positive.astype(np.float64))@vectors.T
    return a,projector,int(np.count_nonzero(positive))


def lightness_ucs(j):
    return 1.7*j/(1+0.007*j)


def build_targets(source_j,source_c,reference_j,reference_c,source_counts,reference_counts,quantile_map,distribution_rotations=16):
    s=conditional_statistics(source_c,source_counts)
    r=conditional_statistics(reference_c,reference_counts)
    a,projector,informed_rank=monge_map(s['covariance'],r['covariance'])
    if informed_rank==2:
        # Transfer the COMPLETE weighted residual distribution instead of replacing
        # it by a Gaussian with only matching covariance. This preserves multimodal
        # reference chroma without color names, masks or semantic correspondence.
        relative=weighted_sliced_transport(s['residual'],r['residual'],source_counts,reference_counts,distribution_rotations)
    else:
        # A rank-deficient source contains no evidence in its null chromatic
        # direction. The PSD Monge map transports only the informed subspace;
        # this is a single linear-algebra rule, not grayscale-image detection.
        relative=s['residual']@a.T
    target_chroma=r['mean']+s['v'][:,None]*r['slope']+relative
    target_j=reference_j[quantile_map]
    target=np.column_stack([lightness_ucs(target_j),target_chroma@projector])
    reference=np.column_stack([lightness_ucs(reference_j),reference_c@projector])
    mean=np.sum(s['w'][:,None]*target_chroma,axis=0)
    center=target_chroma-mean
    cov=(center.T*s['w'])@center
    ref_mean=np.sum(r['w'][:,None]*reference_c,axis=0)
    ref_center=reference_c-ref_mean
    ref_cov=(ref_center.T*r['w'])@ref_center
    info={'definition':'reference mean + reference rank trend + weighted sliced-transported source rank residual',
        'source_chroma_mean':s['mean'].tolist(),'reference_chroma_mean':r['mean'].tolist(),
        'source_rank_trend':s['slope'].tolist(),'reference_rank_trend':r['slope'].tolist(),
        'source_residual_covariance':s['covariance'].tolist(),'reference_residual_covariance':r['covariance'].tolist(),
        'source_residual_roundoff_floor':s['residual_covariance_roundoff_floor'],
        'reference_residual_roundoff_floor':r['residual_covariance_roundoff_floor'],
        'distribution_method':'weighted sliced transport of conditional chroma residuals',
        'distribution_rotations':distribution_rotations,'distribution_projections':2*distribution_rotations,
        'informed_chroma_rank':informed_rank,'informed_chroma_projector':projector.tolist(),
        'target_chroma_mean':mean.tolist(),'target_chroma_covariance':cov.tolist(),
        'reference_chroma_covariance':ref_cov.tolist(),
        'target_mean_error_norm':float(np.linalg.norm(mean-ref_mean)),
        'target_covariance_error_norm':float(np.linalg.norm(cov-ref_cov)),
        'moment_note':'Sliced projection marginals target the full residual distribution; finite projections and final palette projection leave reported residual errors.'}
    return target,reference,info
