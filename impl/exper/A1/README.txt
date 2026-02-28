# File structure and cmd :


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
python tests/test_ga.py ; OR : python -m tests.test_ga


# result :


C:\Users\Admin\code.place\L-Rep\L-Rep-2>python -m examples.demo_sdf
A: [ 0.    1.    0.5   0.   -0.25  0.    0.    0.    0.    0.    0.    0.
  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
  0.    0.    0.    0.    0.    0.    0.    0.  ]
B: [2.  0.  3.  0.  0.  0.  0.  0.  0.7 0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
A*B: [ 1.5    2.     1.     3.    -0.5    0.     0.75   0.     0.     0.7
  0.35   0.    -0.175  0.     0.     0.     0.     0.     0.     0.
  0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.   ]
VM result: [ 2.45885108  2.          2.43827662  3.         -0.5         0.
  0.75        0.          0.33559788  0.7         0.35        0.
 -0.175       0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.        ]
example scalar transformation: 0.0 -> 0.479425538604203

C:\Users\Admin\code.place\L-Rep\L-Rep-2>python -m bench.bench_ga.py
C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe: Error while finding module specification for 'bench.bench_ga.py' (ModuleNotFoundError: __path__ attribute not found on 'bench.bench_ga' while trying to find 'bench.bench_ga.py'). Try using 'bench.bench_ga' instead of 'bench.bench_ga.py' as the module name.

C:\Users\Admin\code.place\L-Rep\L-Rep-2>python -m bench.bench_ga
GA multiply n=5 dim=32 avg 0.4895 ms

C:\Users\Admin\code.place\L-Rep\L-Rep-2>python -m test.test_ga
C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe: No module named test.test_ga

C:\Users\Admin\code.place\L-Rep\L-Rep-2>python -m tests.test_ga
ok
