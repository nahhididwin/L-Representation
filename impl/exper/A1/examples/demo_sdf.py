# examples/demo_sdf.py
"""
Demo pipeline:
- Build GAMultivectorKernel (n=5)
- Create two multivectors A,B, compute product
- Build tiny op tree: approx(sin, domain) applied to scalar field, multiply with B via VM
- Print results and show simple numeric checks
"""
import numpy as np
from lrep.ga import GAMultivectorKernel
from lrep.vm import LVMCompiler, Param, Approx, GAMul
import math

def main():
    n=5
    ga = GAMultivectorKernel(n=n)
    # Build A,B with few blades non-zero
    A = np.zeros(1<<n)
    B = np.zeros(1<<n)
    A[1<<0] = 1.0
    A[1<<1] = 0.5
    A[1<<2] = -0.25
    B[0] = 2.0
    B[1<<1] = 3.0
    B[1<<3] = 0.7
    print("A:", A)
    print("B:", B)
    C = ga.ga_mul(A,B)
    print("A*B:", C)
    # build VM pipeline
    def provider_p(x):
        out = A.copy()
        out[0] = float(x[0])
        return out
    P = Param("p", provider_p)
    Bparam = Param("B", lambda x: B.copy())
    approx_node = Approx(func=np.sin, domain=(-1.0,1.0), child=P, deg=30)
    root = GAMul(approx_node, Bparam)
    compiler = LVMCompiler(ga)
    prog = compiler.compile(root)
    x = np.array([0.5])
    res = compiler.run(prog, x)
    print("VM result:", res)
    # quick numeric sanity
    scalar_before = A[0]
    scalar_after = math.sin(0.5)
    print("example scalar transformation:", scalar_before, "->", scalar_after)

if __name__=="__main__":
    main()
