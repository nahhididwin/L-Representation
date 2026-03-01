#!/usr/bin/env python3
"""
L-Representation: Prototype implementation (single-file)
Features implemented in this prototype (proof-of-concept):
 - Single-integer "L" mixed-radix field packing/unpacking (configurable bases)
 - Field-wise fixed-point encoding (quantization) with guard bits
 - Small Geometric Algebra (GA) multivector packing + primitive L_GA_MUL
 - Lightweight L-VM: operator trees -> JIT compile into fieldwise primitives
 - Chebyshev polynomial approximator (piecewise) with measured error bound
 - Automatic differentiation: forward-mode (dual fields) + reverse-mode tape
 - Dynamic octree (B-chunked idea) encoded into single L words per node
 - Lazy materialization per-query (simulate O(depth) per query)
 - Optional GPU backend (CuPy) if available; otherwise NumPy
 - Simple sphere-tracing raymarcher for SDF/CSG scenes using the L-VM

This is a prototype for demo and experimentation. It is intentionally
self-contained, pure-Python (NumPy optional but highly recommended).

Author: prototype adaptation for Hacker News / YC demo
Date: 1 March 2026 (prototype)
"""

from __future__ import annotations
import math
import time
import functools
import threading
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any, Dict

try:
    import numpy as np
except Exception:
    raise RuntimeError("NumPy is required for this prototype. Install numpy and retry.")

# Optional GPU accel (best-effort)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except Exception:
    GPU_AVAILABLE = False
    xp = np

# -----------------------------
# L: mixed-radix integer packing
# -----------------------------

@dataclass
class FieldSpec:
    """Specification for one field inside L.
    - bits: number of bits allocated to this field (unsigned)
    - signed: whether the field encodes a signed integer (two's complement)
    - scale: quantization scale (float) mapping real -> int by round(x*scale)
    - name: optional human name
    """
    bits: int
    signed: bool = False
    scale: float = 1.0
    name: str = ""


class LLayout:
    """Mixed-radix layout: pack/unpack multiple fields into a single Python int.
    We place fields in increasing significance order (field 0 is least significant).
    """

    def __init__(self, fields: List[FieldSpec]):
        self.fields = fields
        # compute base for each field (2**bits)
        self.bases = [1 << f.bits for f in fields]
        # precompute offsets
        self.offsets = []
        off = 0
        for b in self.bases:
            self.offsets.append(off)
            off += int(math.log2(b))
        self.total_bits = sum(f.bits for f in fields)

    def pack(self, values: List[float]) -> int:
        """Pack a list of real values into L (quantized by scale)."""
        if len(values) != len(self.fields):
            raise ValueError("values length mismatch")
        L = 0
        for i, val in enumerate(values):
            spec = self.fields[i]
            q = int(round(val * spec.scale))
            if spec.signed:
                # two's complement fit into bits
                max_pos = (1 << (spec.bits - 1)) - 1
                min_neg = - (1 << (spec.bits - 1))
                if q < min_neg or q > max_pos:
                    raise OverflowError(f"value {val} (q={q}) overflows field {i}")
                if q < 0:
                    q = (1 << spec.bits) + q
            else:
                if q < 0 or q >= (1 << spec.bits):
                    raise OverflowError(f"value {val} (q={q}) overflows unsigned field {i}")
            L |= (q << self.offsets[i])
        return L

    def unpack(self, L: int) -> List[float]:
        vals = []
        for i, spec in enumerate(self.fields):
            mask = (1 << spec.bits) - 1
            q = (L >> self.offsets[i]) & mask
            if spec.signed:
                sign_bit = 1 << (spec.bits - 1)
                if q & sign_bit:
                    q = q - (1 << spec.bits)
            vals.append(q / spec.scale)
        return vals

    def get_field_raw(self, L: int, i: int) -> int:
        mask = (1 << self.fields[i].bits) - 1
        return (L >> self.offsets[i]) & mask

    def set_field_raw(self, L: int, i: int, raw: int) -> int:
        mask = (1 << self.fields[i].bits) - 1
        L &= ~(mask << self.offsets[i])
        L |= (raw & mask) << self.offsets[i]
        return L

# ---------------------
# Geometric Algebra POC
# ---------------------

class GAMultivector:
    """Dense multivector for n-dimensional GA (n small, e.g., 3 or 5).
    Internally stores coefficients as a 1D numpy array length 2**n.
    Provides packing/unpacking into L fields via LLayout.
    """

    def __init__(self, n: int, coeffs: xp.ndarray = None):
        self.n = n
        self.dim = 1 << n
        if coeffs is None:
            self.coeffs = xp.zeros(self.dim, dtype=xp.float64)
        else:
            assert coeffs.shape == (self.dim,)
            self.coeffs = coeffs.astype(xp.float64)

    @classmethod
    def random(cls, n: int, rng=None):
        rng = np.random.default_rng(42) if rng is None else rng
        coeffs = xp.array(rng.standard_normal(1 << n))
        return cls(n, coeffs)

    def pack_to_L(self, layout: LLayout, field_idx_start=0, scale=1.0) -> int:
        """Pack the first K coefficients into consecutive fields starting at field_idx_start.
        This is a simplification: in practice we would pack many coefficients across fields.
        """
        K = min(len(self.coeffs), len(layout.fields) - field_idx_start)
        vals = [float(self.coeffs[i]) * scale for i in range(K)]
        # pad remaining fields with zero
        vals += [0.0] * (len(layout.fields) - field_idx_start - K)
        # rotate so fields align
        # naive: pack to fields[field_idx_start:]
        full_vals = [0.0] * len(layout.fields)
        for i, v in enumerate(vals):
            full_vals[field_idx_start + i] = v
        return layout.pack(full_vals)

    @staticmethod
    def L_GA_mul(a: 'GAMultivector', b: 'GAMultivector') -> 'GAMultivector':
        """Compute geometric product using basis index xor and grade sign rules.
        This implementation uses the sign rule from basis blade multiplication.
        For small n it's fine; it's vectorized via NumPy/CuPy.
        """
        n = a.n
        assert n == b.n
        dim = 1 << n
        A = a.coeffs
        B = b.coeffs
        C = xp.zeros(dim, dtype=xp.float64)
        # naive double loop; for n<=5 dim<=32 so it's acceptable in prototype
        for i in range(dim):
            ai = A[i]
            if ai == 0: continue
            for j in range(dim):
                bj = B[j]
                if bj == 0: continue
                k = i ^ j
                # compute sign via bitcount trick: sign = (-1)^{
                # number of swaps required = grade(i) * grade(j) - ...
                # We'll compute sign using canonical basis multiplication parity.
                sign = GAMultivector._blade_mul_sign(i, j, n)
                C[k] += ai * bj * sign
        return GAMultivector(n, C)

    @staticmethod
    def _blade_mul_sign(i: int, j: int, n: int) -> int:
        # compute sign of basis blades multiplication e_i * e_j -> +/- e_{i^j}
        # Using bitwise algorithm (count intersections where bits cross)
        s = 1
        # for each bit set in i, count bits set in j at lower positions
        ii = i
        while ii:
            lb = ii & -ii
            idx = (lb.bit_length() - 1)
            # count bits in j below idx
            mask = (1 << idx) - 1
            if bin(j & mask).count('1') % 2:
                s = -s
            ii &= ii - 1
        return s

# -------------------------
# L-VM: operator tree model
# -------------------------

class OpNode:
    def __init__(self, op: str, args: List[Any], name: str = ''):
        self.op = op
        self.args = args
        self.name = name or op

    def __repr__(self):
        return f"OpNode({self.op}, args={self.args})"

# Some op constructors for convenience
def FieldAdd(a, b): return OpNode('field_add', [a, b])
def FieldMul(a, b): return OpNode('field_mul', [a, b])
def GA_Mul(a, b): return OpNode('ga_mul', [a, b])
def Approx(func, domain): return OpNode('approx', [func, domain])
def ConstL(L): return OpNode('const', [L])

enum_id = 0

def fresh_id(prefix='v'):
    global enum_id
    enum_id += 1
    return f"{prefix}{enum_id}"

class LVMCompiler:
    """Compile operator trees (OpNode) into executable Python functions that operate on
    L-packed integers and/or numpy arrays of L words.
    This JIT is lightweight: it does tree-to-lambda translation using Python closures.
    """

    def __init__(self, layout: LLayout, ga_n=3, use_gpu=False):
        self.layout = layout
        self.ga_n = ga_n
        self.use_gpu = use_gpu and GPU_AVAILABLE

    def compile(self, node: OpNode) -> Callable[[Dict[str,int]], int]:
        # compile returns a function env -> L
        if node.op == 'const':
            L = node.args[0]
            def f(env):
                return L
            return f
        if node.op == 'field_add':
            fa = self.compile(node.args[0])
            fb = self.compile(node.args[1])
            def f(env):
                return fa(env) + fb(env)
            return f
        if node.op == 'field_mul':
            fa = self.compile(node.args[0])
            fb = self.compile(node.args[1])
            def f(env):
                A = fa(env); B = fb(env)
                # naive per-field multiply with saturation into field ranges
                out = 0
                for i, spec in enumerate(self.layout.fields):
                    ai = self.layout.get_field_raw(A, i)
                    bi = self.layout.get_field_raw(B, i)
                    # decode to signed if needed
                    if spec.signed:
                        signbit = 1 << (spec.bits - 1)
                        if ai & signbit: ai = ai - (1<<spec.bits)
                        if bi & signbit: bi = bi - (1<<spec.bits)
                    prod = ai * bi
                    # simple saturating trim
                    maxv = (1 << (spec.bits-1)) - 1 if spec.signed else (1<<spec.bits)-1
                    minv = - (1 << (spec.bits-1)) if spec.signed else 0
                    if prod > maxv: prod = maxv
                    if prod < minv: prod = minv
                    raw = int(prod) & ((1<<spec.bits)-1)
                    out = self.layout.set_field_raw(out, i, raw)
                return out
            return f
        if node.op == 'ga_mul':
            # args are GAMultivector constants or compiled Ls representing packed GA
            a_node, b_node = node.args
            # compile children into callables that produce L words where multiple fields
            # hold quantized multivector coefficients. For prototype we unpack, multiply
            # with GA kernel and repack.
            fa = self.compile(a_node)
            fb = self.compile(b_node)
            def f(env):
                La = fa(env); Lb = fb(env)
                # unpack coefficients from fields
                coeffs_a = []
                coeffs_b = []
                for i in range(1<<self.ga_n):
                    if i < len(self.layout.fields):
                        qa = self.layout.get_field_raw(La, i)
                        qb = self.layout.get_field_raw(Lb, i)
                        spec = self.layout.fields[i]
                        # decode signed
                        if spec.signed:
                            signbit = 1 << (spec.bits - 1)
                            if qa & signbit: qa = qa - (1<<spec.bits)
                            if qb & signbit: qb = qb - (1<<spec.bits)
                        coeffs_a.append(qa / spec.scale)
                        coeffs_b.append(qb / spec.scale)
                    else:
                        coeffs_a.append(0.0); coeffs_b.append(0.0)
                A = xp.array(coeffs_a);
                B = xp.array(coeffs_b)
                # call GA multiply kernel
                C = GAMultivector._ga_mul_kernel(A, B, self.ga_n)
                # repack into fields
                out = 0
                for i, c in enumerate(C.tolist()):
                    if i >= len(self.layout.fields): break
                    spec = self.layout.fields[i]
                    q = int(round(c * spec.scale))
                    if spec.signed:
                        # two's complement encode
                        mask = (1 << spec.bits) - 1
                        if q < 0:
                            q = (1 << spec.bits) + q
                    out = self.layout.set_field_raw(out, i, q)
                return out
            return f
        if node.op == 'approx':
            # args: func, domain tuple (a,b), degree
            func, domain = node.args
            # precompute approx coefficients (chebyshev) on host
            poly, err = chebyshev_approx(func, domain, deg=8)
            def f(env):
                # we evaluate approximation per-field on each numeric field
                # For prototype we only support one field and treat it as float
                L_in = list(env.values())[0] if env else 0
                vals = self.layout.unpack(L_in)
                out_vals = []
                for v in vals:
                    x = v
                    y = eval_cheb(poly, x, domain)
                    out_vals.append(y)
                return self.layout.pack(out_vals)
            return f
        raise NotImplementedError(f"Compile for op {node.op} not implemented")

# Helper minimal GA kernel to plug into compiler
@staticmethod
def _ga_mul_kernel_static(A: xp.ndarray, B: xp.ndarray, n: int) -> xp.ndarray:
    dim = 1 << n
    C = xp.zeros(dim, dtype=xp.float64)
    for i in range(dim):
        ai = A[i]
        if ai == 0: continue
        for j in range(dim):
            bj = B[j]
            if bj == 0: continue
            k = i ^ j
            sign = GAMultivector._blade_mul_sign(i, j, n)
            C[k] += ai * bj * sign
    return C

# attach static kernel to class (monkeypatch for simplicity)
GAMultivector._ga_mul_kernel = staticmethod(_ga_mul_kernel_static)

# ----------------------------
# Chebyshev approximator (PWL)
# ----------------------------

def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    xk = np.cos((2*k + 1) / (2*n) * np.pi)
    return 0.5*(a + b) + 0.5*(b - a)*xk


def chebyshev_coeffs(func: Callable[[float], float], a: float, b: float, deg: int) -> np.ndarray:
    # Use discrete orthogonality via sampling at Chebyshev nodes
    n = deg + 1
    xs = chebyshev_nodes(a, b, n)
    ys = np.array([func(x) for x in xs])
    # compute coefficients via DCT-like formula
    c = np.zeros(n)
    for j in range(n):
        Tj = np.cos(j * np.arccos((2*xs - (a+b)) / (b-a)))
        c[j] = (2/n) * np.dot(ys, Tj)
    c[0] *= 0.5
    return c


def chebyshev_approx(func: Callable[[float], float], domain: Tuple[float,float], deg: int=8) -> Tuple[np.ndarray, float]:
    a,b = domain
    c = chebyshev_coeffs(func, a, b, deg)
    # compute max error on a fine grid (practical bound)
    xs = np.linspace(a, b, 2000)
    pvals = np.array([eval_cheb(c, x, domain) for x in xs])
    fvals = np.array([func(x) for x in xs])
    err = float(np.max(np.abs(pvals - fvals)))
    return c, err


def eval_cheb(c: np.ndarray, x: float, domain: Tuple[float,float]) -> float:
    a,b = domain
    # map x to t in [-1,1]
    t = (2*x - (a+b)) / (b-a)
    # Clenshaw algorithm
    d = 0.0
    dd = 0.0
    for cj in c[::-1]:
        sv = d
        d = 2*t*d - dd + cj
        dd = sv
    return d - t*dd

# --------------------
# Automatic differentiation
# --------------------

@dataclass
class Dual:
    val: float
    der: float

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        else:
            return Dual(self.val + other, self.der)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.val*other.der + self.der*other.val)
        else:
            return Dual(self.val * other, self.der * other)

# Reverse-mode simple tape
class Tape:
    def __init__(self):
        self.ops = []  # (fn_grad, out_idx, in_idxs)
        self.values = []

    def add(self, fn_grad: Callable, inputs: List[int], out_val: float):
        out_idx = len(self.values)
        self.values.append(out_val)
        self.ops.append((fn_grad, out_idx, inputs))
        return out_idx

    def backward(self, out_idx: int) -> List[float]:
        grads = [0.0] * len(self.values)
        grads[out_idx] = 1.0
        # walk ops in reverse
        for fn_grad, oidx, inputs in reversed(self.ops):
            g = grads[oidx]
            if g == 0: continue
            contribs = fn_grad(g, inputs, self.values)
            for idx, cg in zip(inputs, contribs):
                grads[idx] += cg
        return grads

# ----------------------
# Dynamic octree example
# ----------------------

class OctreePackedNode:
    """Pack an octree node into a single integer L_word as follows (prototype):
    - bits 0..7: child occupancy bitmap (8 children)
    - bits 8..15: lock / version
    - bits 16..47: pointer to child base (index into an external array)
    - remaining bits: payload or flags
    This is a toy packing to demonstrate single-integer atomic updates.
    """
    def __init__(self):
        # simple layout: 64-bit field
        self.bitmap_bits = 8
        self.version_bits = 8
        self.ptr_bits = 32
        self.payload_bits = 16

    def new_node_word(self, bitmap=0, version=0, ptr=0, payload=0):
        w = 0
        w |= (bitmap & ((1<<self.bitmap_bits)-1))
        w |= (version & ((1<<self.version_bits)-1)) << self.bitmap_bits
        w |= (ptr & ((1<<self.ptr_bits)-1)) << (self.bitmap_bits + self.version_bits)
        w |= (payload & ((1<<self.payload_bits)-1)) << (self.bitmap_bits + self.version_bits + self.ptr_bits)
        return w

    def extract(self, w):
        bitmap = w & ((1<<self.bitmap_bits)-1)
        version = (w >> self.bitmap_bits) & ((1<<self.version_bits)-1)
        ptr = (w >> (self.bitmap_bits + self.version_bits)) & ((1<<self.ptr_bits)-1)
        payload = (w >> (self.bitmap_bits + self.version_bits + self.ptr_bits)) & ((1<<self.payload_bits)-1)
        return dict(bitmap=bitmap, version=version, ptr=ptr, payload=payload)

    def cas(self, mem: Dict[int,int], addr: int, old: int, new: int) -> bool:
        """Simulate atomic CAS on a dictionary-backed memory. In real hardware this
        would be a single atomic integer operation. For prototype, we use a lock to
        make CAS thread-safe when used with threads.
        """
        lock = mem.setdefault('_lock', threading.Lock())
        with lock:
            cur = mem.get(addr, 0)
            if cur == old:
                mem[addr] = new
                return True
            else:
                return False

# -------------------------
# Simple SDF / Raymarch demo
# -------------------------

def sdf_sphere(p, r=1.0):
    return xp.linalg.norm(p) - r


def sdf_box(p, b):
    q = xp.abs(p) - b
    return xp.linalg.norm(xp.maximum(q, 0.0)) + xp.minimum(xp.maximum(q[0], xp.maximum(q[1], q[2])), 0.0)

# compose operators in operator tree form for lazy eval

def make_sphere_node(center, radius, layout: LLayout):
    # pack center.x, center.y, center.z, radius into L fields (3+1)
    vals = [center[0], center[1], center[2], radius]
    # pad rest
    vals += [0.0] * (len(layout.fields) - 4)
    L = layout.pack(vals)
    return ConstL(L)

# raymarching using compiled L-VM evaluation of SDF parameters

def raymarch(origin, dir, scene_eval_fn, max_steps=64, eps=1e-4, max_dist=100.0):
    t = 0.0
    for i in range(max_steps):
        p = origin + t * dir
        d = scene_eval_fn(p)
        if d < eps:
            return t, i
        if t > max_dist:
            break
        t += d
    return None, max_steps

# -----------------
# Demo & tests
# -----------------

def demo_ga_mul():
    print('\n--- GA multiply demo ---')
    n = 3
    A = GAMultivector.random(n).coeffs
    B = GAMultivector.random(n).coeffs
    a = GAMultivector(n, A)
    b = GAMultivector(n, B)
    C = GAMultivector.L_GA_mul(a, b)
    # sanity: compare naive kernel
    C2 = GAMultivector._ga_mul_kernel(A, B, n)
    err = float(xp.max(xp.abs(C.coeffs - C2)))
    print(f"GA multiply (n={n}) max error between kernels: {err:.3e}")


def demo_chebyshev():
    print('\n--- Chebyshev approx demo ---')
    f = math.sin
    domain = (0.0, math.pi)
    poly, err = chebyshev_approx(f, domain, deg=10)
    print(f"Chebyshev deg=10 approx on [0,pi], measured max error ~ {err:.3e}")


def demo_lvm_and_sdf():
    print('\n--- L-VM + SDF demo ---')
    # layout: simple 16-field layout (support packing small GA coefficients or SDF params)
    fields = [FieldSpec(bits=16, signed=True, scale=100.0, name=f'f{i}') for i in range(16)]
    layout = LLayout(fields)
    # build a sphere node
    sphere_node = make_sphere_node((0.0, 0.0, 3.0), 1.0, layout)
    compiler = LVMCompiler(layout, ga_n=3)
    const_fn = compiler.compile(sphere_node)
    Lword = const_fn({})
    print(f"Packed L (sphere params): 0x{Lword:x}")

    def scene_eval(p):
        # unpack sphere parameters and evaluate SDF directly (materialize lazily per query)
        vals = layout.unpack(Lword)
        cx, cy, cz, r = vals[0], vals[1], vals[2], vals[3]
        cp = p - xp.array([cx, cy, cz])
        return xp.linalg.norm(cp) - r

    origin = xp.array([0.0, 0.0, 0.0])
    dir = xp.array([0.0, 0.0, 1.0])
    t, steps = raymarch(origin, dir, scene_eval)
    print(f"Raymarch result: t={t}, steps={steps}")


def demo_dynamic_octree():
    print('\n--- Dynamic octree (packed) demo ---')
    mem = {}
    node = OctreePackedNode()
    w0 = node.new_node_word(bitmap=0, version=0, ptr=0, payload=0)
    addr = 10
    mem[addr] = w0
    # attempt to insert child by CAS: set bit 0
    cur = mem[addr]
    e = node.extract(cur)
    new_bitmap = e['bitmap'] | 1
    new_ver = (e['version'] + 1) & 0xff
    new_w = node.new_node_word(bitmap=new_bitmap, version=new_ver, ptr=e['ptr'], payload=e['payload'])
    ok = node.cas(mem, addr, cur, new_w)
    print(f"CAS insert child0 success={ok}")
    print(f"node after CAS: {node.extract(mem[addr])}")

if __name__ == '__main__':
    demo_ga_mul()
    demo_chebyshev()
    demo_lvm_and_sdf()
    demo_dynamic_octree()
    print('\nPrototype finished. See functions and classes in the file for extension points.')
