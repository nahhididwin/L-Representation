# lrep/autodiff.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class Dual:
    val: np.ndarray
    dot: np.ndarray

def dual_add(a: Dual, b: Dual) -> Dual:
    return Dual(a.val + b.val, a.dot + b.dot)

def dual_mul(a: Dual, b: Dual) -> Dual:
    val = a.val * b.val
    dot = a.dot * b.val + a.val * b.dot
    return Dual(val, dot)

# Simple reverse tape
class Tape:
    def __init__(self):
        self.records = []  # tuple(opname, inputs, aux)
    def record(self, opname, inputs, aux=None):
        self.records.append((opname, inputs, aux))
    def backprop(self, out_adj: np.ndarray):
        # naive scalar-per-field adj accumulation
        adj: Dict[str, np.ndarray] = {}
        cur = out_adj.copy()
        for opname, inputs, aux in reversed(self.records):
            if opname=='add':
                a,b = inputs
                adj[a] = adj.get(a, np.zeros_like(cur)) + cur
                adj[b] = adj.get(b, np.zeros_like(cur)) + cur
            elif opname=='mul':
                a,b = inputs
                aval,bval = aux
                adj[a] = adj.get(a, np.zeros_like(cur)) + cur * bval
                adj[b] = adj.get(b, np.zeros_like(cur)) + cur * aval
            else:
                raise NotImplementedError(opname)
        return adj
