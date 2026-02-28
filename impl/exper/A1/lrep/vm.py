# lrep/vm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Dict
import numpy as np
from .ga import GAMultivectorKernel
from .approx import ChebApproximator

# Operator nodes
class Op: pass

@dataclass
class Param(Op):
    name: str
    provider: Callable[[np.ndarray], np.ndarray]  # x -> multivector (np.ndarray)

@dataclass
class FieldAdd(Op):
    left: Op
    right: Op

@dataclass
class GAMul(Op):
    left: Op
    right: Op

@dataclass
class Approx(Op):
    func: Callable[[np.ndarray], np.ndarray]
    domain: Tuple[float,float]
    child: Op
    deg:int = 30

@dataclass
class MinOp(Op):
    left: Op
    right: Op

@dataclass
class VMProgram:
    ops: List[Tuple[str, Any]]  # (opname, args)

class LVMCompiler:
    def __init__(self, ga_kernel: GAMultivectorKernel):
        self.ga = ga_kernel

    def compile(self, root: Op) -> VMProgram:
        ops=[]
        def emit(name, *args):
            ops.append((name,args))
            return len(ops)-1
        def walk(node):
            if isinstance(node, Param):
                return ('param', node.name, node.provider)
            elif isinstance(node, FieldAdd):
                a = walk(node.left); b = walk(node.right)
                idx = emit('add', a, b)
                return ('tmp', idx)
            elif isinstance(node, GAMul):
                a = walk(node.left); b = walk(node.right)
                idx = emit('ga_mul', a, b)
                return ('tmp', idx)
            elif isinstance(node, Approx):
                child = walk(node.child)
                approx = ChebApproximator(node.func, node.domain, deg=node.deg)
                idx = emit('approx', child, approx)
                return ('tmp', idx)
            elif isinstance(node, MinOp):
                a = walk(node.left); b = walk(node.right)
                idx = emit('min', a, b)
                return ('tmp', idx)
            else:
                raise NotImplementedError(type(node))
        walk(root)
        return VMProgram(ops)

    def run(self, program: VMProgram, x: np.ndarray, param_cache:Dict[str,np.ndarray]=None) -> np.ndarray:
        if param_cache is None:
            param_cache={}
        cache={}
        def resolve(op):
            if op is None: return None
            kind = op[0]
            if kind=='param':
                _,name,provider = op
                if name in param_cache:
                    return param_cache[name]
                val = provider(x)
                param_cache[name]=val
                return val
            if kind=='tmp':
                idx = op[1]
                if idx in cache: return cache[idx]
                opname,args = program.ops[idx]
                if opname=='add':
                    a = resolve(args[0]); b = resolve(args[1])
                    out = a + b
                elif opname=='ga_mul':
                    a = resolve(args[0]); b = resolve(args[1])
                    out = self.ga.ga_mul(a,b)
                elif opname=='approx':
                    child_op, approx = args
                    child = resolve(child_op)
                    # apply approx elementwise to scalar field 0 (common pattern)
                    out = child.copy()
                    out[0] = approx.eval(out[0])
                elif opname=='min':
                    a = resolve(args[0]); b = resolve(args[1])
                    out = np.minimum(a,b)
                else:
                    raise NotImplementedError(opname)
                cache[idx]=out
                return out
            raise RuntimeError("bad operand")
        if not program.ops:
            return np.zeros(self.ga.dim)
        last = ('tmp', len(program.ops)-1)
        return resolve(last)
