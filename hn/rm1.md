# Avoiding Overflow Bugs in Fixed-Point Arithmetic with Formal Verification

We built L-Representation (L-Rep), a framework for fixed-point arithmetic with machine-checked guarantees.

It targets hardware, safety-critical and DSP workloads where overflow and precision bugs are difficult to reason about.


**The core idea is a bit-allocation scheme that:**

- packs structured algebra computations into a single integer word
- provides tight, formally verified error bounds for every operation

We formalized the system in Lean 4 (no sorry proofs), and evaluated it on FPGA and GPU workloads.

**Preliminary results show:**

- higher efficiency per DSP vs standard approaches
- no overflow ("bleed") across millions of adversarial tests
- competitive throughput vs float32, with formal guarantees


**Paper** (Proof + Experiment) **(PDF)** : https://github.com/nahhididwin/L-Representation/blob/main/main/lrep_final.pdf

**Supplementary** : https://github.com/nahhididwin/L-Representation/blob/main/main/LRep_Supplementary_Corrigenda.tex

**If you're skimming:**

- Proofs (Lean 4): see middle sections  
- Implementation + experiments: see later sections  

This is early-stage work — I’d really appreciate feedback from people!

