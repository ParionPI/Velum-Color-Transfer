"""CAM16 J and neutral-axis-relative CAM16-UCS chromatic coordinates.
No image-dependent thresholds. Neutral detection uses a float64 roundoff bound.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .color import CAM16, M16, D65, ViewingConditions


@njit(cache=True)
def _neutral_scale(j, aw, exponent, k):
    out = np.empty(len(j), np.float64)
    for i in range(len(j)):
        target = aw * (j[i] / 100.0) ** (1.0 / exponent)
        if target == 0.0:
            out[i] = 0.0
            continue
        lo, hi = 0.0, 1.0
        while True:
            v = (800*k[0]*hi/(27.13+k[0]*hi)
                 +400*k[1]*hi/(27.13+k[1]*hi)+20*k[2]*hi/(27.13+k[2]*hi))
            if v >= target:
                break
            hi *= 2.0
        while True:
            mid = (lo+hi)/2.0
            if mid == lo or mid == hi:
                break
            v = (800*k[0]*mid/(27.13+k[0]*mid)
                 +400*k[1]*mid/(27.13+k[1]*mid)+20*k[2]*mid/(27.13+k[2]*mid))
            if v < target:
                lo = mid
            else:
                hi = mid
        out[i] = ((lo+hi)/2.0) ** (1.0/0.42)
    return out


class Appearance(CAM16):
    def __init__(self, viewing=ViewingConditions()):
        super().__init__(viewing)
        self.n = viewing.background_luminance/100.0
        self.nbb = 0.725*self.n**(-0.2)
        white_adapted = (M16 @ (D65*100))*self.adaptation
        self.neutral_k = (self.fl*white_adapted/100)**0.42

    def raw(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        cone = (xyz[:,0,None]*M16[:,0]+xyz[:,1,None]*M16[:,1]+xyz[:,2,None]*M16[:,2])
        adapted = cone*self.adaptation
        p = (self.fl*np.abs(adapted)/100)**0.42
        r = np.sign(adapted)*400*p/(27.13+p)
        a_response = 2*r[:,0]+r[:,1]+r[:,2]/20
        if np.any(a_response < 0) or not np.all(np.isfinite(a_response)):
            raise ValueError('Outside real-valued CAM16 J domain; no clamp.')
        j = 100*(a_response/self.aw)**self.exponent
        a = r[:,0]-12*r[:,1]/11+r[:,2]/11
        b = (r[:,0]+r[:,1]-2*r[:,2])/9
        h = np.arctan2(b,a)
        et = (np.cos(h+2)+3.8)/4
        t = (50000/13)*self.nbb*et*np.hypot(a,b)/(r[:,0]+r[:,1]+1.05*r[:,2]+0.305)
        if np.any(t < 0):
            raise ValueError('Outside CAM16 colorfulness domain.')
        chroma = t**0.9*(1.64-0.29**self.n)**0.73*np.sqrt(j/100)
        m = chroma*self.fl**0.25
        mp = np.log1p(0.0228*m)/0.0228
        ab = np.column_stack((mp*np.cos(h),mp*np.sin(h)))
        return j, ab

    def neutral_ab(self, j):
        scale = _neutral_scale(np.asarray(j, np.float64), self.aw, self.exponent, self.neutral_k)
        _, ab = self.raw(scale[:,None]*D65*100)
        return ab

    def describe_xyz(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        j, ab = self.raw(xyz)
        relative = ab-self.neutral_ab(j)
        neutral_xyz = xyz[:,1,None]*D65
        # gamma_n = n*u/(1-n*u), u=float64 unit roundoff. n=16 bounds the
        # short matrix/white-axis arithmetic chain; NOT a saturation threshold.
        u = np.finfo(np.float64).eps/2
        gamma = 16*u/(1-16*u)
        bound = gamma*(np.abs(xyz)+np.abs(neutral_xyz))
        numerical_neutral = np.all(np.abs(xyz-neutral_xyz) <= bound, axis=1)
        relative[numerical_neutral] = 0.0
        if not np.all(np.isfinite(relative)):
            raise ValueError('Nonfinite chromatic coordinates.')
        return j, relative

    def describe(self, rgb, encoding, block_size=65536):
        j = np.empty(len(rgb), np.float64)
        c = np.empty((len(rgb),2), np.float64)
        for start in range(0,len(rgb),block_size):
            stop=start+block_size
            j[start:stop], c[start:stop] = self.describe_xyz(encoding.xyz(rgb[start:stop]))
        return j,c
