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
python examples/demo_sdf.py ; OR : python -m examples.demo_sdf
python bench/bench_ga.py ; OR : python -m bench.bench_ga
python tests/test_ga.py
