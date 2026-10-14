"""AUTO scoring without a JIT dispatcher; original float32 ufunc ordering."""
import numpy as np

SCORER_BACKEND = 'numpy_no_jit'


def make_scratch(size):
    return tuple(np.empty(int(size),dtype=np.float32) for _ in range(3))


def _vector(value, shape, name, writable=False):
    if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=shape or value.ndim!=1:
        raise ValueError(f'{name} must be a matching float32 vector')
    if not value.flags.c_contiguous or (writable and not value.flags.writeable):
        raise ValueError(f'{name} must be contiguous' + (' and writable' if writable else ''))
    return value


def score_amplitudes(h0a,h0b,h2a,h2b,h3a,h3b,weight,a2,a3,out,scratch=None):
    """Write weighted amplitudes; caller retains the original NumPy sum.

    No Numba import/decorator/cache loading is used by this module. Scratch is
    owned by one curve evaluator, not shared across requests or threads.
    """
    if not isinstance(out,np.ndarray) or out.ndim!=1:
        raise ValueError('out must be a vector')
    shape=out.shape
    _vector(out,shape,'out',True)
    fields=(h0a,h0b,h2a,h2b,h3a,h3b,weight)
    for index,value in enumerate(fields): _vector(value,shape,f'coefficient[{index}]')
    if scratch is None: scratch=make_scratch(out.size)
    if len(scratch)!=3: raise ValueError('Three scratch vectors required')
    a,b,temp=(_vector(value,shape,f'scratch[{i}]',True) for i,value in enumerate(scratch))
    writes=(out,a,b,temp)
    for i,value in enumerate(writes):
        if any(np.shares_memory(value,other) for other in fields+writes[:i]):
            raise ValueError('Scoring output/scratch must not alias inputs or each other')
    a2,a3=np.float32(a2),np.float32(a3)
    np.multiply(h2a,a2,out=a)
    np.add(a,h0a,out=a)
    np.multiply(h3a,a3,out=temp)
    np.add(a,temp,out=a)
    np.multiply(h2b,a2,out=b)
    np.add(b,h0b,out=b)
    np.multiply(h3b,a3,out=temp)
    np.add(b,temp,out=b)
    np.multiply(a,a,out=out)
    np.multiply(b,b,out=temp)
    np.add(out,temp,out=out)
    np.sqrt(out,out=out)
    np.multiply(out,weight,out=out)
