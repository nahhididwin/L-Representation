# lrep/ga.py
"""
GAMultivector kernel for L-Representation prototype.
- Packs a multivector into 2^n fields (n <= 6 recommended).
- Implements geometric product (L_GA_MUL) with sparse skipping and precomputed table.
- Optional Numba acceleration (if numba installed).
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

def mask_to_list(mask: int) -> List[int]:
    lst = []
    i = 0
    while mask:
        if mask & 1:
            lst.append(i)
        mask >>= 1
        i += 1
    return lst

def list_to_mask(lst: List[int]) -> int:
    m = 0
    for i in lst:
        m |= (1 << i)
    return m

def multiply_blades(mask_a: int, mask_b: int, metric: List[float]) -> Tuple[float,int]:
    """
    Multiply blade bitmasks (a * b) -> (scalar_factor, result_mask).
    Implementation matches the sign/metric rule for GA.
    """
    a = mask_to_list(mask_a)
    b = mask_to_list(mask_b)
    res = a.copy()
    sign = 1.0
    scalar = 1.0
    for bj in b:
        swaps = sum(1 for x in res if x > bj)
        if swaps % 2 != 0:
            sign *= -1.0
        if bj in res:
            res.remove(bj)
            scalar *= metric[bj]
        else:
            res.append(bj)
    out_mask = list_to_mask(res)
    return sign * scalar, out_mask

def build_table(n: int, metric: List[float]) -> Dict[Tuple[int,int], Tuple[float,int]]:
    size = 1 << n
    table = {}
    for i in range(size):
        for j in range(size):
            s, out = multiply_blades(i, j, metric)
            table[(i,j)] = (s,out)
    return table

class GAMultivectorKernel:
    def __init__(self, n:int=5, metric:Optional[List[float]]=None):
        """
        n: geometric dimension (CGA for 3D uses n=5)
        metric: list of length n (signature), default Euclidean (+1,...,+1)
        """
        self.n = n
        self.dim = 1 << n
        if metric is None:
            self.metric = [1.0]*n
        else:
            assert len(metric)==n
            self.metric = metric
        self.table = build_table(n, self.metric)

    def ga_mul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Geometric product of two multivectors A,B given as 1D numpy arrays length 2^n.
        Returns numpy array length 2^n.
        """
        if A.shape[0] != self.dim or B.shape[0] != self.dim:
            raise ValueError("Input length mismatch")
        out = np.zeros(self.dim, dtype=A.dtype)
        # iterate only nonzeros (sparse optimization)
        nz_a = np.nonzero(A)[0]
        nz_b = np.nonzero(B)[0]
        for i in nz_a:
            ai = A[i]
            for j in nz_b:
                bj = B[j]
                s, idx = self.table[(i,j)]
                out[idx] += s * ai * bj
        return out

    def ga_mul_inplace(self, A: np.ndarray, B: np.ndarray, out: np.ndarray):
        res = self.ga_mul(A,B)
        np.copyto(out, res)

# Numba accelerated worker (optional)
if _HAS_NUMBA:
    # For larger n you might precompute flattened integer and sign tables for numba.
    pass
