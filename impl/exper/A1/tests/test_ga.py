# tests/test_ga.py
import numpy as np
from lrep.ga import GAMultivectorKernel
def test_simple_identity():
    ga = GAMultivectorKernel(n=3)
    A = np.zeros(8)
    B = np.zeros(8)
    A[0]=1.0 # scalar 1
    B[0]=2.0
    C = ga.ga_mul(A,B)
    assert np.allclose(C[0], 2.0)
if __name__=="__main__":
    test_simple_identity()
    print("ok")
