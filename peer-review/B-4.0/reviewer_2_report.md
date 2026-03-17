# Reviewer 2 Report: Experiments & Practical Impact Specialist

## Summary

This paper introduces L-Representation, a packed-integer arithmetic system for geometric algebra with formally verified error bounds, targeting safety-critical applications and resource-constrained hardware. The work claims to provide provably tight error bounds through constrained-input analysis, machine-verified Lean 4 proofs, quantization-aware training for Clifford neural networks, and FPGA validation. Experimental validation includes FPGA synthesis results showing 1.7×-2.3× speedups over float32 (2.8×-3.6× with sparsity), RTL simulation, and preliminary safety-critical system demonstrations. The paper positions L-Rep as a formal-correctness-first approach complementary to higher-throughput accelerators.

---

## Soundness: 3/5

### Experimental Design

**Strengths**:
- FPGA synthesis on Xilinx Artix-7 provides concrete hardware validation
- Multiple algebra configurations tested: G(3,0,1) motors, G(4,1,0) CGA, G(3,0,0) 3D
- RTL simulation with Verilator adds confidence in functional correctness
- Comparison with float32 baseline establishes performance context

**Critical Weaknesses**:
1. **No QAT experimental results**: Section 7 proposes quantization-aware training with detailed methodology but provides **zero experimental validation**—no accuracy curves, no bit-width ablations, no trained network results. This is a major gap for a paper claiming QAT as a key contribution.

2. **Incomplete hardware evaluation**: Section 9 reports only:
   - LUT counts and maximum frequency
   - No power consumption measurements
   - No resource utilization breakdown (DSPs, BRAMs, registers)
   - No throughput measurements (operations/second)
   - No latency analysis
   - No comparison with synthesized float32 GA on same FPGA

3. **Missing baseline comparisons**: No direct comparison with:
   - CliffordALU5 or CliffoSor on equivalent hardware
   - Optimized fixed-point GA implementations
   - Posit arithmetic for GA
   - GPU implementations

4. **Synthetic benchmarks only**: Section 9.2 uses "random unit motors" and "random CGA transformations"—no real-world datasets (robot trajectories, sensor data, actual neural network workloads)

5. **Statistical rigor lacking**: No error bars, confidence intervals, or significance tests for reported speedups. How many runs? What is the variance?

### Dataset and Experimental Setup

**Major Issues**:

1. **No real-world datasets**: All experiments use synthetic random data
   - Robot motion planning: Should use actual robot trajectory datasets
   - QAT: Should use real Clifford neural network training tasks
   - Safety-critical: Should use certified test suites

2. **Gauss-Seidel experiment (Sec 8.2)**: Claims L-Rep suitable for iterative solvers but provides:
   - No convergence analysis
   - No comparison with float32 convergence behavior
   - No iteration count vs. accuracy curves
   - No discussion of when fixed-point iteration fails

3. **Sparse L-Rep (Sec 8.3)**: Claims 2-8× memory savings but provides:
   - No experimental validation of sparsity patterns
   - No performance measurements for sparse operations
   - No comparison with sparse matrix libraries

4. **Sample size**: Unclear how many test cases were used for validation—"random motors" is vague

### Results and Claims

**Unsupported or Weakly Supported Claims**:

1. **"1.7×-2.3× speedup"**: Based on what metric? Clock cycles? Wall-clock time? Throughput (ops/sec)? The paper conflates different performance metrics.

2. **"2.8×-3.6× with sparse subalgebras"**: No experimental data provided—appears to be theoretical projection only.

3. **"IEC 61508 SIL 3/4 compliance"**: No evidence that any L-Rep implementation has undergone actual safety certification. This is a prospective claim, not a validated result.

4. **"Quantization-aware training enables deployment"**: No evidence—no trained networks, no deployment measurements, no accuracy comparisons.

5. **Table 6 (FPGA results)**: Reports LUTs and frequency but:
   - Missing: Power, energy per operation, throughput
   - No comparison with float32 synthesized on same FPGA
   - No validation that reported frequency is achievable under real workloads (vs. post-synthesis estimates)

### Reproducibility

**Positive**:
- GitHub repository provided
- Detailed bit allocation specifications (Table 3)
- RTL implementation described

**Negative**:
- No code for QAT experiments
- No datasets or benchmarks publicly available
- FPGA synthesis scripts not mentioned
- Verilator testbenches not provided
- Lean 4 proofs incomplete (only excerpts)

---

## Presentation: 3/5

### Clarity Issues

1. **QAT section misleading**: Section 7 is detailed and well-written but presents **methodology without results**—this creates false impression of validated contribution

2. **Performance metrics inconsistent**: 
   - Sometimes "speedup" means clock cycles
   - Sometimes means throughput
   - Sometimes means memory bandwidth
   - Needs clear, consistent definitions

3. **Figure quality**:
   - Figures 1-4 are excellent
   - Figure 5 (RTL) is unreadable at paper size
   - **Missing figures**: No experimental result plots, no accuracy curves, no convergence graphs, no resource utilization charts

4. **Table 6 incomplete**: Should include power, energy, throughput, latency—not just LUTs and frequency

### Organization

**Strengths**:
- Clear section structure
- Good use of tables for specifications
- Appropriate use of examples

**Weaknesses**:
- **Section 7 (QAT) should be moved to "Future Work"** or experimental results should be added
- **Section 8 (Applications)** contains claims without validation—reads more like "potential applications" than "validated applications"
- **Section 9 (Validation)** is too brief for the scope of claims

---

## Contribution: 3/5

### Assessment by Contribution

1. **Formal theory (Theorems 1-2)**: **Strong contribution**—novel and rigorous, partially machine-verified

2. **Constrained-input analysis (Theorem 2)**: **Strong contribution**—genuinely novel exploitation of algebraic structure

3. **Machine verification (Lean 4)**: **Moderate contribution**—only 5/9 theorems verified; incomplete

4. **Quantization-aware training**: **Weak contribution**—methodology described but **zero experimental validation**. Cannot assess significance without results.

5. **Safety-critical validation**: **Weak contribution**—prospective compliance pathways described but no actual certification or comprehensive validation

6. **FPGA implementation**: **Moderate contribution**—demonstrates feasibility but evaluation is incomplete

### Overall Significance

**Current state**: The paper makes **one strong theoretical contribution** (constrained-input analysis) and **one moderate contribution** (partial formal verification), but **experimental validation is insufficient** to support the claimed practical impact.

**With complete validation**: If QAT experiments, comprehensive hardware evaluation, and real-world dataset validation were added, this would be a **strong contribution** suitable for top venues.

**As submitted**: The experimental gaps reduce this to a **moderate contribution** more suitable for a workshop or a theory-focused venue.

---

## Strengths

1. **Novel constrained-input analysis** (Theorem 2): Elegant exploitation of motor normalization for provable bit savings—this is the paper's strongest contribution

2. **Formal rigor**: Combination of mathematical proofs and Lean 4 verification demonstrates serious commitment to correctness

3. **Clear problem motivation**: Safety-critical applications (IEC 61508, DO-178C) represent genuine industrial need

4. **Honest positioning**: Authors clearly state L-Rep is not throughput-focused and is complementary to existing accelerators—this intellectual honesty is commendable

5. **Comprehensive bit allocation analysis** (Table 3): Detailed specifications for multiple GA configurations

6. **RTL implementation**: Demonstrates feasibility beyond theoretical proposal

7. **Addresses real gap**: Prior GA accelerators lack formal verification—this is a genuine problem for safety-critical systems

---

## Weaknesses

### Experimental Validation (Critical)

1. **QAT experiments completely missing**: Section 7 describes methodology in detail but provides **no experimental results whatsoever**. This is unacceptable for a paper claiming QAT as a key contribution. Required:
   - Training curves for at least 2-3 Clifford neural network architectures
   - Accuracy vs. bit-width trade-off analysis
   - Comparison with full-precision and post-training quantization
   - Validation on real tasks (PDE modeling, point cloud processing)
   - Ablation studies on bit allocation strategies

2. **Hardware evaluation superficial**: Table 6 provides minimal data:
   - **Missing**: Power consumption (critical for embedded systems)
   - **Missing**: Energy per operation
   - **Missing**: Throughput (operations/second)
   - **Missing**: Latency distribution
   - **Missing**: Resource utilization breakdown
   - **Missing**: Comparison with float32 on same FPGA
   - **Missing**: Scaling analysis (how do resources scale with dimension n?)

3. **No real-world datasets**: All validation uses synthetic random data:
   - Robot motion planning: Use actual robot trajectory datasets (e.g., KITTI, TUM RGB-D)
   - Inverse kinematics: Use real manipulator specifications and task sequences
   - QAT: Use real Clifford neural network training tasks
   - Safety-critical: Use certified test suites from standards

4. **No baseline comparisons**: Missing comparisons with:
   - CliffordALU5/CliffoSor (claimed 4-23× speedups)
   - Optimized fixed-point implementations
   - GPU implementations
   - Posit arithmetic
   - Interval arithmetic (MPFI, INTLAB)

5. **Gauss-Seidel validation absent**: Section 8.2 claims suitability for iterative solvers but provides:
   - No convergence experiments
   - No comparison with float32 convergence
   - No analysis of fixed-point error accumulation
   - No failure cases identified

6. **Sparse L-Rep validation absent**: Section 8.3 claims 2-8× savings but provides:
   - No experimental measurements
   - No real sparse GA workloads
   - No performance comparison with sparse libraries

### Statistical Rigor

7. **No error bars or confidence intervals**: All reported numbers are point estimates with no variance information

8. **No significance testing**: No statistical tests to determine if speedups are significant

9. **Unclear sample sizes**: How many random motors tested? How many trials per configuration?

10. **No outlier analysis**: What happens with adversarial inputs near bound limits?

### Experimental Design Issues

11. **Synthetic data bias**: Random uniform sampling may not reflect real-world distributions of GA elements in applications

12. **Cherry-picking risk**: Without comprehensive benchmarks, unclear if reported speedups are representative

13. **Incomplete ablation studies**: Should ablate:
    - Effect of bit allocation strategy
    - Impact of sparsity patterns
    - Sensitivity to input bounds Aᵢ
    - Effect of rounding modes

14. **No failure case analysis**: What inputs cause largest errors? When does L-Rep perform poorly?

### Reproducibility Issues

15. **Code availability unclear**: GitHub repository mentioned but:
    - No link to QAT implementation
    - No FPGA synthesis scripts
    - No Verilator testbenches
    - Incomplete Lean 4 proofs

16. **Insufficient implementation details**:
    - FPGA synthesis tool versions not specified
    - Optimization flags not mentioned
    - Testbench methodology not described

---

## Questions for Authors

1. **QAT experiments**: When will quantization-aware training experimental results be available? This is essential for publication.

2. **Power measurements**: Can you provide power consumption data for FPGA implementations? This is critical for embedded systems evaluation.

3. **Real datasets**: Can you validate on real robot trajectory datasets, actual Clifford neural network tasks, and certified safety test suites?

4. **Direct hardware comparison**: Can you synthesize CliffordALU5/CliffoSor and L-Rep on the same FPGA and compare resources, power, and throughput directly?

5. **Statistical validation**: How many test cases were used? What are confidence intervals on reported speedups?

6. **Failure cases**: What inputs produce the worst-case errors? Can you characterize when L-Rep performs poorly?

7. **Convergence analysis**: For Gauss-Seidel (Sec 8.2), under what conditions does fixed-point iteration converge? Diverge?

8. **Sparse validation**: Can you provide experimental validation of sparse L-Rep claims with real sparse GA workloads?

9. **Bit allocation in practice**: How should practitioners determine σᵢ* for new problems? Can you provide a worked example or algorithm?

10. **Safety certification**: Have any L-Rep implementations undergone actual IEC 61508 or DO-178C certification, or is Section 8.1 entirely prospective?

---

## Suggestions

### Essential for Publication

1. **Conduct and report QAT experiments**: This is non-negotiable for claiming QAT as a contribution:
   - Train at least 2-3 Clifford neural network architectures
   - Report accuracy vs. bit-width curves
   - Compare with full-precision and PTQ baselines
   - Use real tasks (PDE modeling, point cloud processing)

2. **Comprehensive hardware evaluation**:
   - Measure power consumption
   - Report throughput (ops/sec) and latency
   - Provide detailed resource breakdown
   - Compare with float32 on same FPGA

3. **Real-world dataset validation**:
   - Robot trajectories (KITTI, TUM RGB-D)
   - Real Clifford neural network tasks
   - Certified safety test suites

4. **Statistical rigor**:
   - Report error bars and confidence intervals
   - Specify sample sizes
   - Conduct significance tests

### Highly Recommended

5. **Direct accelerator comparison**: Synthesize and compare with CliffordALU5/CliffoSor on same hardware

6. **Convergence analysis**: For iterative algorithms, provide convergence guarantees and failure case analysis

7. **Ablation studies**:
   - Bit allocation strategies
   - Sparsity patterns
   - Rounding modes
   - Input bound sensitivity

8. **Failure case analysis**: Characterize worst-case inputs and error distributions

9. **Reproducibility package**:
   - Complete Lean 4 proofs
   - FPGA synthesis scripts
   - Verilator testbenches
   - QAT training code
   - Datasets and benchmarks

### Recommended

10. **Extended baseline comparisons**: Include posit arithmetic, interval arithmetic, GPU implementations

11. **Scaling analysis**: How do resources/performance scale with dimension n and bit budget B?

12. **Application case studies**: Provide end-to-end demonstrations on real safety-critical systems

---

## Rating: 5/10 (Borderline Reject)

### Decision: Borderline Reject

**Justification**: While the paper presents **interesting theoretical ideas** (especially constrained-input analysis), the **experimental validation is critically insufficient** for a venue expecting rigorous empirical evaluation. The most significant issues:

1. **QAT contribution unvalidated**: An entire section (Sec 7) describes methodology with **zero experimental results**—this is unacceptable for claiming QAT as a contribution

2. **Hardware evaluation incomplete**: Missing power, throughput, latency, and direct comparisons with competing approaches

3. **No real-world validation**: All experiments use synthetic random data

4. **Statistical rigor absent**: No error bars, significance tests, or variance analysis

**This reads more like a technical report or workshop paper than a complete research contribution ready for a top venue.**

### Path to Acceptance

The paper could be accepted if the authors:
1. **Conduct QAT experiments** and report comprehensive results
2. **Complete hardware evaluation** with power, throughput, and direct comparisons
3. **Validate on real datasets** from target applications
4. **Add statistical rigor** to all experimental claims

The theoretical contributions (Theorems 1-2) are solid, but **experimental validation must match theoretical rigor** for publication at a top venue.

### Venue Suitability

**Current state**:
- ❌ NeurIPS/ICML: Insufficient ML experiments (no QAT results)
- ❌ ISCA/MICRO: Incomplete hardware evaluation
- ❌ PLDI/POPL: Insufficient focus on verification methodology
- ✅ Workshop: Appropriate for preliminary ideas
- ✅ ArXiv preprint: Suitable for early dissemination

**With complete validation**:
- ✅ FPGA/FPL: With comprehensive hardware evaluation
- ✅ CAV/FM: With focus on formal verification aspects
- ✅ NeurIPS (Hardware track): With complete QAT experiments
- ✅ Domain journals: TCAD, TACO, TVLSI (with full evaluation)

---

## Confidence: 5/5 (Very High)

I am very confident in this assessment based on:
- Extensive experience evaluating hardware accelerator papers
- Strong background in neural network quantization and QAT
- Clear identification of missing experimental validation
- Comparison with standards for experimental rigor in top venues

The experimental gaps are objective and unambiguous:
- Section 7 has no results
- Section 9 is missing standard hardware metrics
- No real datasets used
- No statistical analysis provided

These are not subjective judgments but factual observations about missing content.
