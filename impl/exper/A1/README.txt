lrep/
  __init__.py
  ga.py           # GAMultivector kernel (n up to 5 default), sparse-optimized
  approx.py       # Chebyshev approximator + Horner evaluator
  vm.py           # Tiny L-VM (compile operator trees -> primitive sequence) + runtime
  autodiff.py     # Dual-number forward + small reverse-mode tape
examples/
  demo_sdf.py     # small demo: GA motor, approx, VM pipeline + prints/plots
bench/
  bench_ga.py     # microbenchmark GA kernel
tests/
  test_ga.py


## Quickstart
```bash
python -m pip install -r requirements.txt   # numpy, optional numba
python examples/demo_sdf.py
python bench/bench_ga.py
python tests/test_ga.py
