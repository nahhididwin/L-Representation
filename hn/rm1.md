# Avoiding Overflow Bugs in Fixed-Point Arithmetic with Formal Verification

(A Formally Verified Framework for Constraint-Aware Fixed-Point Arithmetic in Structured Convolution Algebras, with Applications to Hardware)

We built L-Representation (L-Rep), A Formally Verified Framework for Constraint-Aware Fixed-Point Arithmetic in Structured Convolution Algebras, with Applications to Hardware

It targets hardware/DSP/safety-critical workloads/Geometric Algebra/... where overflow and precision bugs are hard to reason about.

The core idea is a bit-allocation scheme that:
- packs structured algebra computations into a single integer word
- provides tight, formally verified error bounds for every operation

We formalized the system in Lean 4 (no sorry proofs), and tested it.

Preliminary results show:
- higher efficiency per DSP vs standard approaches
- no overflow ("bleed") across millions of adversarial tests
- competitive throughput vs float32, with formal guarantees



# Index of PDF and More :

**Index of PDF :**

Abstract : Pages 1, 2

Introduction and Compare : Pages 3, 4, 5

Experiment, Algorithm, Compare (Codes, Lean 4,...) : Pages 12 -> 31

References : Pages 31, 32

Proof (Math) : Pages 6 -> 11

**Read Now :**

Watch **Paper** (Proof + Experiment) **(PDF)** : https://github.com/nahhididwin/L-Representation/blob/main/main/lrep_final.pdf

**If you would like to read more!** (supplementary version): https://github.com/nahhididwin/L-Representation/blob/main/main/LRep_Supplementary_Corrigenda.tex

If your paper visibility is faulty, please check the **.tex** file: https://github.com/nahhididwin/L-Representation/blob/main/main/lrep_final.tex

**All versions and repositories are here:** https://github.com/nahhididwin/L-Representation

This is early-stage work — I’d really appreciate feedback from people!


# More info :

License : https://github.com/nahhididwin/L-Representation?tab=License-1-ov-file

Repositories Public Date (https://github.com/nahhididwin/L-Representation) : 27/02/2026 (DD/MM/YYYY)

WARNING : At the time of 28/02/2026 (DD/MM/YYYY), this work is an early-stage theoretical exploration developed independently by a student researcher. Due to practical constraints, the current version focuses on conceptual formulation and preliminary validation. The author welcomes feedback, critique, and collaboration from the community.

(You are reading version v4.0-beta)
