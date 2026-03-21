# Avoiding Overflow Bugs in Fixed-Point Arithmetic with Formal Verification

We built L-Representation (L-Rep), a framework for fixed-point arithmetic with machine-checked guarantees.

It targets hardware, safety-critical and DSP workloads where overflow and precision bugs are difficult to reason about.


The core idea is a bit-allocation scheme that:

- packs structured algebra computations into a single integer word
- provides tight, formally verified error bounds for every operation

We formalized the system in Lean 4 (no sorry proofs), and evaluated it.

Preliminary results show:

- higher efficiency per DSP vs standard approaches
- no overflow ("bleed") across millions of adversarial tests
- competitive throughput vs float32, with formal guarantees


View **Paper** (Proof + Experiment) **(PDF)** : https://github.com/nahhididwin/L-Representation/blob/main/main/lrep_final.pdf

**If you're skimming:**

- Proofs (Lean 4): see middle sections  
- Implementation + experiments: see later sections  

**Debug:**

If your paper visibility is faulty, please check the **.tex** file: https://github.com/nahhididwin/L-Representation/blob/main/main/lrep_final.tex



**If you would like to read more** (supplementary version): https://github.com/nahhididwin/L-Representation/blob/main/main/LRep_Supplementary_Corrigenda.tex

**All versions and repositories are here:** https://github.com/nahhididwin/L-Representation

This is early-stage work — I’d really appreciate feedback from people!


# More info :

License : https://github.com/nahhididwin/L-Representation?tab=License-1-ov-file

Repositories Public Date (https://github.com/nahhididwin/L-Representation) : 27/02/2026 (DD/MM/YYYY)

WARNING : At the time of 28/02/2026 (DD/MM/YYYY), this work is an early-stage theoretical exploration developed independently by a student researcher. Due to practical constraints, the current version focuses on conceptual formulation and preliminary validation. The author welcomes feedback, critique, and collaboration from the community.

(You are reading version v4.0-beta)
