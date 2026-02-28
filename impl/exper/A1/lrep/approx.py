# lrep/approx.py
from __future__ import annotations
import numpy as np
from numpy.polynomial import Chebyshev
from typing import Callable, Tuple

class ChebApproximator:
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], domain: Tuple[float,float], deg:int=30, samples:int=1024):
        self.func = func
        self.domain = domain
        self.deg = deg
        xs = np.linspace(domain[0], domain[1], samples)
        ys = func(xs)
        # fit Chebyshev polynomial on domain
        self.cheb = Chebyshev.fit(xs, ys, deg, domain=domain)
        # coefficients in Chebyshev basis
        self.coefs = self.cheb.coef  # can be used for Horner-like eval in Chebyshev basis
    def eval(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.cheb(x)
    def derivative(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.cheb.deriv()(x)
