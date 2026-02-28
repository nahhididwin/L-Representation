# bench/bench_ga.py
import time
import numpy as np
from lrep.ga import GAMultivectorKernel
def bench(n=5, repetitions=100):
    ga = GAMultivectorKernel(n=n)
    dim = 1<<n
    rng = np.random.default_rng(42)
    A = rng.standard_normal(dim)
    B = rng.standard_normal(dim)
    # warmup
    ga.ga_mul(A,B)
    t0 = time.perf_counter()
    for _ in range(repetitions):
        ga.ga_mul(A,B)
    t1 = time.perf_counter()
    dt = (t1-t0)/repetitions
    print(f"GA multiply n={n} dim={dim} avg {dt*1e3:.4f} ms")
if __name__=="__main__":
    bench()
