# Triton-to-KTIR Lowering: Thought Exercise on Nuances

This note captures a design discussion about Triton-to-KTIR lowering complexity.
The key observation is that the perceived complexity comes from starting with
GPU-shaped Triton input. This doc formalizes that insight: two structural
invariants, a clean pre-Triton/lowering boundary, and evidence that it holds
under stress.

---

## Core Hypothesis

**The Triton-to-KTIR lowering pass is a bounded, local translation.** Kernel-shape decisions (tile sizes, reduction topology) are absorbed by a pre-Triton stage; execution-level decisions (buffering, loop order, physical core mapping) are handled by the KTIR scheduler downstream. Two structural invariants — weight-stationary matmul, single-read-KV for attention — eliminate catastrophic failure modes. Lowering is invariant to everything else.

```
┌─────────────────────────────────────────────────────────┐
│  Pre-Triton Stage                                       │
│  (hyperparameter selection — autotunable)               │
│                                                         │
│  Constraints: HW capacity bounds, model arch, workload  │
│  Decides: tile sizes, reduction topology                │
│           (world count × world size, not physical cores)│
│  Tiles operands to fit scratchpad (M, N, K, KV)        │
│  Enforces: structural invariants                        │
└────────────────────────┬────────────────────────────────┘
                         │ Fully-shaped Triton kernel
                         │ (invariants satisfied, tiles chosen)
┌────────────────────────▼────────────────────────────────┐
│  Lowering (Triton → KTIR)                               │
│  (bounded, local, hardware-unaware)                     │
│                                                         │
│  tl.dot → matrix engine    tl.reduce → inter-core      │
│  tl.load/store → DMA       desc.gather → indirect tile │
│  loops → KTIR primitives                                │
└────────────────────────┬────────────────────────────────┘
                         │ KTIR module
                         │ (high-level: simple loops, no buffering)
┌────────────────────────▼────────────────────────────────┐
│  KTIR Scheduler (→ Schedule IR → deeptools)             │
│  (hardware-aware, topology-aware)                       │
│                                                         │
│  Decides: buffer depth, loop order, physical core       │
│           assignment (ring-local world mapping),        │
│           XRF-level tiling, DMA staging                 │
└─────────────────────────────────────────────────────────┘
```

Pre-Triton decisions reduce to hyperparameter selection — the same class of decisions GPU autotuners solve, amenable to search-based optimization rather than manual per-model engineering. The KTIR scheduler handles execution-level decisions that require hardware topology knowledge (ring adjacency, scratchpad banking, functional unit scheduling).

### Structural Invariants

| Invariant | Why | Always satisfiable? |
|-----------|-----|---------------------|
| Weight-stationary matmul (intra-core K-tiling) | Eliminates repeated weight DMA. Inter-core N-distribution is default (from layout); inter-core K-distribution with sum reduction when activation locality requires it (fused MLP down_proj) | Yes: layout [64, K, N/64] gives contiguous N-distribution; hardware sum-reduction available for K-distribution |
| KV read exactly once for attention (`Q_tiles × CORES_PER_Q_TILE ≤ 32`) | Violating costs 50-75% redundant bandwidth per extra pass — catastrophic for bandwidth-sensitive attention | Yes: `BLOCK_Q ≥ CHUNK_SIZE/32` always fits in 1MB scratchpad for practical chunk sizes |

Cross-core reduction (`tl.reduce`) is needed by attention kernels (online-softmax-combine for both prefill and decode) and by matmul when inter-core K-distribution is used (e.g., fused MLP down_proj where activation is already K-distributed from the prior op).

### Action Items

| Action | Type | Status |
|--------|------|--------|
| Extend `tl.reduce` with cross-program axis | **Build** (blocking) | Required for correctness — no single-kernel model without it (see FAQ Q1, Q3). |
| Pre-Triton stage enforces invariants + selects tile sizes and reduction topology | **Build** | Absorbs all HW/workload variation (FLOPS, BW, scratchpad, model arch, workload). |
| DMA pipeline bubble at pass boundaries (2–6%) | **Accept** for v1 | Latency-dominated, <10% all generations (Appendix A). Options for v2: post-lowering optimization, or pre-Triton async DMA hints. |

### Dos and Don'ts

- **Do**: Put tile sizes and reduction topology (world count × world size, BLOCK_Q/CORES_PER_Q_TILE co-design) in the pre-Triton stage — these define kernel shape
- **Do**: Use tensor descriptors with `gather`/`scatter` for indirect access (paged KV, MoE routing) — maps 1:1 to KTIR's `construct_indirect_access_tile` without de-linearization
- **Do**: Enforce structural invariants unconditionally
- **Do**: Leave buffer depth, loop order, physical core-to-world mapping, and XRF-level tiling to the KTIR scheduler — it knows ring adjacency and scratchpad banking
- **Don't**: Use pointer arithmetic for indirect patterns — requires de-linearization during lowering and cannot represent cases where the indirect index is multiplicatively mixed with other dimensions
- **Don't**: Make the lowering pass aware of hardware parameters, model architecture, or workload
- **Don't**: Block on async DMA overlap optimization for v1

### What Lowering Does

Bounded, deterministic translations — no per-kernel decisions, but real compiler work:

- `tl.dot` → matrix engine op
- `tl.load/store` → explicit DMA between off-chip and scratchpad (Spyre has no hardware cache — lowering bridges Triton's implicit memory model to explicit DMA with stick-alignment and capacity validation)
- `desc.gather/scatter` → `ktdp.construct_indirect_access_tile` + `ktdp.load/store` (index tensor maps to indirect dimension's auxiliary memory view; direct dimensions from descriptor shape/strides — 1:1 structural mapping, no de-linearization)
- `tl.reduce` (cross-program) → inter-core reduction (topology, synchronization, exchange buffers; software protocol for online-softmax-combine)
- Loop structure → KTIR loop primitives (tile iteration with load/compute/store sequencing)
- Type conversions (BF16 ↔ FP32 for accumulation)

DMA/compute overlap and double-buffer slot assignment are **not** expressed in KTIR or decided by lowering. The backend's dataflow execution model — concurrent DMA and compute programs coordinated through blocking handshakes — makes intra-pass overlap structural: the DMA engine leads compute by one tile via double-buffered scratchpad allocation. Lowering produces the representation the scheduler decomposes into these concurrent programs.

Overlap *across* pass boundaries (outer M-loop) is not automatic — the pipeline drains and restarts, producing a bubble at each boundary (quantified in Appendix A). Eliminating this would require scheduler-level outer-loop pipelining, orthogonal to lowering. Accepted for v1.

The work is complex to *build* (compiler engineering) but predictable to *use* (same transformations for every kernel).

---

## Rationale

### Matmul: Weight-Stationary

Hand-written Triton kernels already expose `BLOCK_M`, `BLOCK_K`, `BLOCK_N` as tunable hyperparameters — the autotuner sweeps them per GPU generation. For Spyre, you're adding constraints to the same parameter space:

- **GPU**: tile grid should saturate the SMs without excessive wave quantization.
- **Spyre**: `ceil(M/BLOCK_M) * ceil(N/BLOCK_N)` must be a multiple of 32 (or close), and each tile must fit in LX scratchpad.

The kernel *structure* doesn't change — you're constraining the parameter space, not restructuring the algorithm. If the tile grid exceeds 32, the kernel expresses residual iteration explicitly. Hand-written and Inductor-generated kernels fit the same model.

**Intra-core K-tiling:** Weights held stationary in scratchpad, activations stream through. K is iterated in tiles within each core, partial sums accumulate in registers. Weights load once and are reused across all M iterations — critical for decode where A is tiny but B is 32MB.

**Inter-core N-distribution (default):** Weight layout on device is [64, K, N/64] (innermost to outermost). N/64 is outermost → each core gets a contiguous N-partition with full K. Independent output columns, no cross-core reduction.

**Inter-core K-distribution (fused ops):** When the prior op's N-split leaves activations K-distributed across cores (e.g., MLP down_proj), each core computes a partial matmul on its local K-slice and cross-core sum reduction combines results. Motivated by activation locality — avoids expensive all-to-all gather.

---

### Spyre's Persistent Execution Model

On GPU, persistent kernels are opt-in (most kernels launch ephemeral thread blocks). On Spyre, **32 cores are always resident** — every kernel is persistent by construction. Two consequences:

1. **Cross-program `tl.reduce` is natural on Spyre.** Co-residency is *guaranteed* — all 32 programs are alive simultaneously. On GPU, this requires a special cooperative kernel launch mode. On Spyre, it's free.

2. **The programming model is uniform across all kernels**: assign work to 32 cores, loop over assigned tiles, reduce across cores when needed. The scheduling stage picks the assignment; lowering translates.

---

### Attention: Prefill (Chunked Flash Attention — 1K Chunk × 1M Sequence)

Prefill uses **chunked prefill**: query sequence broken into chunks (CHUNK_SIZE=1024), each attending to all KV tokens up to its position.

**GPU**: 2D grid over (batch×heads, ceil(CHUNK_SIZE / BLOCK_Q)). For long KV, FlashDecoding splits KV across thread blocks with a cross-block reduction.

**Spyre**: CHUNK_SIZE=1024 with BLOCK_Q=32 → 32 Q-tiles, 1 per core, no residual loop. KV dimension: up to ~15,625 iterations (BLOCK_KV=64).

**The core structure** (common to all BLOCK_Q choices):

```
core_id = ktdp.get_compute_tile_id()
q_group = core_id // CORES_PER_Q_TILE
kv_split = core_id % CORES_PER_Q_TILE

kv_start = kv_split * (num_kv_tiles // CORES_PER_Q_TILE)
kv_end = kv_start + (num_kv_tiles // CORES_PER_Q_TILE)

Q_block = load(Q[q_group * BLOCK_Q, :])          // BLOCK_Q x head_dim, stays resident
m_i = -inf, l_i = 0, acc = zeros(BLOCK_Q, head_dim)
for kv_tile in range(kv_start, kv_end):
    K_block = load(K[kv_tile, :])                 // BLOCK_KV x head_dim (double-buffered)
    V_block = load(V[kv_tile, :])                 // BLOCK_KV x head_dim (double-buffered)
    S = Q_block @ K_block^T                       // BLOCK_Q x BLOCK_KV
    m_new = max(m_i, rowmax(S))
    P = exp(S - m_new)
    l_new = l_i * exp(m_i - m_new) + rowsum(P)
    acc = acc * (l_i * exp(m_i - m_new) / l_new) + (P / l_new) @ V_block
    m_i, l_i = m_new, l_new

// Cross-program reduction (only when CORES_PER_Q_TILE > 1)
final_acc = tl.reduce((m_i, l_i, acc), axis='programs', group=q_group, op='online_softmax_combine')
store(final_acc)
```

**Structural invariant: KV is read exactly once per chunk.**

> **`Q_tiles × CORES_PER_Q_TILE ≤ 32` is a hard constraint, not a scheduling option.**

A residual Q-loop re-streams the entire KV cache per extra pass (50-75% redundant bandwidth — see FAQ Q6). Always satisfiable: `BLOCK_Q ≥ CHUNK_SIZE / 32` guarantees it, and the required BLOCK_Q (32–256) fits in scratchpad (2MB per core; analysis conservatively uses 1MB).

**BLOCK_Q / CORES_PER_Q_TILE co-design** (within the invariant):

| BLOCK_Q | Q-tiles | CORES_PER_Q_TILE | KV iters/core | Scratchpad | Cross-core? |
|---------|---------|------------------|---------------|------------|-------------|
| 32 | 32 | 1 | 15,625 | 96KB | No |
| 64 | 16 | 2 | 7,812 | 128KB | Yes |
| 128 | 8 | 4 | 3,906 | 192KB | Yes |
| 256 | 4 | 8 | 1,953 | 320KB | Yes |

All rows satisfy the invariant (single round, KV read once). Moving down: fewer iterations, larger matmuls (better matrix engine utilization), better compute-to-memory ratio, but cross-core reduction required. BLOCK_Q=32 with CORES_PER_Q_TILE>1 violates the invariant (32 × 2 = 64 > 32 cores → residual Q-loop). The choice of row is a hardware-dependent scheduling decision — the invariant itself is not.

**Scratchpad breakdown** (BLOCK_Q=128, BLOCK_KV=64, head_dim=128):

| Phase | Live buffers | Size |
|-------|-------------|------|
| KV loop | Q=32KB + K×2=32KB + V×2=32KB + S=32KB (128×64, FP32) + acc=64KB (128×128, FP32) | ~192KB |
| Reduction | own acc (64KB) + own m/l (1KB) + partner acc (64KB) + partner m/l (1KB) | ~130KB |

At reduction: Q, K/V buffers, and S are dead (freed 128KB). Partner's 65KB loads into freed space. Feasible.

**Verdict**: The invariant eliminates catastrophic multi-pass bandwidth waste. BLOCK_Q and CORES_PER_Q_TILE selection is a pre-Triton decision; lowering is unaffected.

---

### Attention: Paged Decode (Single-Token Generation, up to 1M Sequence)

During autoregressive decode, the query is a single token (seq_len_q=1) attending over up to 1M KV tokens stored in non-contiguous pages.

**GPU (PagedAttention)**: split KV pages across thread blocks, each produces a partial (m, l, acc), then a cross-block reduction combines them.

**Spyre**: The query is one vector (1 × head_dim) — no Q-tiling. The only parallelism axis is the KV pages. Three challenges specific to decode:

1. **Complex reduction**. Unlike matmul's simple sum, partial softmax results must be combined via log-sum-exp rescaling (online-softmax-combine) — stateful, not associative in the naive sense.

2. **Non-contiguous DMA (gather pattern)**. KV cache pages are not sequential in memory — `page_table` provides indirection. Tensor descriptors with `desc.gather(page_indices, offset)` express this structurally; lowering maps directly to `ktdp.construct_indirect_access_tile`.

3. **Bandwidth-bound**. The per-page compute is a vector-matrix multiply (1 × head_dim) @ (head_dim × PAGE_SIZE). The compute-to-memory ratio is poor. DMA strategy matters more than tile decomposition.

**Resolution**: Since all 32 programs are co-resident, `tl.reduce` with a cross-program axis resolves #1:

```
core_id = ktdp.get_compute_tile_id()
kv_desc = tl.make_tensor_descriptor(kv_ptr, shape=[TOTAL_PAGES, PAGE_SIZE, head_dim], ...)

// All 32 cores process their assigned KV pages
m_i = -inf, l_i = 0, acc = zeros(1, head_dim)
page_indices = load(page_table[core_id * pages_per_core : (core_id+1) * pages_per_core])
for page_local in range(pages_per_core):
    K_page = kv_desc.gather(page_indices[page_local], 0)  // PAGE_SIZE x head_dim (non-contiguous)
    V_page = kv_desc.gather(page_indices[page_local], 0)
    S = Q @ K_page^T                                       // 1 x PAGE_SIZE
    // online softmax update of (m_i, l_i, acc) ...

// Cross-program reduction
final_acc = tl.reduce((m_i, l_i, acc), axis='programs', op='online_softmax_combine')
if core_id == 0:
    store(final_acc)
```

**Scratchpad at 1M sequence** (PAGE_SIZE=64 tokens → 15,625 pages, ~488 per core):

| Phase | Live buffers | Size |
|-------|-------------|------|
| Page loop | Q (256B) + K×2 double-buf (32KB) + V×2 double-buf (32KB) + S (256B) + acc (512B) | ~65KB |
| Reduction | own acc (512B) + partner acc (512B) + m/l scalars | ~1KB |

Scratchpad is relaxed (Q is 1 vector). Total KV streamed per core: ~15MB. Bandwidth-bound regardless of sequence length.

**Verdict**: `tl.reduce` handles cross-core reduction. Non-contiguous DMA is expressed via tensor descriptor `gather` (maps 1:1 to KTIR indirect access tiles). Bandwidth-boundedness is a pre-Triton concern, not lowering complexity.

---

## FAQ

**Q1: Is `tl.reduce(axis='programs')` a real Triton op?**

No — pseudocode for a Spyre-specific extension in the IBM fork. The actual lowering target is `ktdp.reduce`.

**Q2: Sum and online-softmax-combine have very different costs — how can they share `tl.reduce`?**

They share the programming model, not the execution path. Sum uses the hardware reduction bus (fast path). Online-softmax-combine runs in software on SFP units: exp(), rescaling, weighted accumulation. `tl.reduce` dispatches to the appropriate backend based on `op=`. The cost asymmetry is real but hidden behind the same API.

**Q3: What does "single-kernel model" mean concretely?**

32 programs launch, read from off-chip, compute, reduce on-chip, write output — one off-chip round-trip, no intermediate writes. Without cross-program reduce, partial results must be written to off-chip memory between kernels (multi-kernel fallback), adding a full read-write round-trip per reduction stage.

**Q4: Does `tl.dot` consume B in 3D stick layout, or does lowering reshape to logical 2D?**

Lowering reshapes. `tl.dot` operates on logical 2D tiles (M×K @ K×N). The stick layout (num_sticks, N, 64) is a storage format — the matrix engine consumes data in stick-aligned slices, but the Triton-level abstraction is 2D. The stick↔logical mapping is handled during DMA load (address generation), not in the compute path.

**Q5: What happens when weights don't fit in scratchpad?**

Pre-Triton tiles all operand dimensions (M, N, K) to fit scratchpad. For Llama 70B FFN (8192×28672): N is tiled into blocks (e.g., BLOCK_N=4096), the kernel gets an outer N-loop. The weight-stationary invariant holds per-tile — weights load to scratchpad for each N-block, reuse across M iterations, then the next N-block loads.

**Q6: Why uniform KV assignment with causal masking? Where does 50-75% come from?**

Static uniform assignment is the v1 baseline. Dynamic assignment (more cores to later Q-tiles with longer KV ranges) is a valid pre-Triton optimization — it doesn't affect lowering.

The 50-75% figure: if the invariant is violated and a core runs a residual Q-loop of depth D, it re-streams KV D times. For D=2: the second pass re-reads all KV from the first pass plus its extension — ~50% redundant for uniform-length ranges, up to 75% with causal length skew.

---

## Appendix A: DMA Bubble Derivation

With weight-stationary matmul, weight B stays in scratchpad. The residual M-loop iterates over activation blocks. Shown here for inter-core K-distribution (the bubble applies equally to inter-core N-distribution — same pass-boundary structure, just without the reduction step):

```
// B_local already in scratchpad (stationary)
for m_block in range(batch / M_BLOCK):        // <-- pass boundary
    A_block = load(A[m_block, local_K])       // M_BLOCK × local_K
    C_partial = A_block × B_local             // M_BLOCK × N
    C_final = tl.reduce(C_partial, op='sum')  // cross-core sum (K-distribution only)
    store(C_final)
```

Within a pass, DMA is pipelined with compute — bulk data movement is overlapped. The overhead comes from the **pipeline bubble at pass boundaries**: compute cannot start the new pass until the minimum activation quanta arrives from off-chip, and the previous pass's tail write drains unmasked.

**Bubble model:** minimum quanta = 2 sticks/core × 32 cores = 8 KB. Both ramp-up (read) and ramp-down (write) are exposed:

```
bubble = 2 × (8 KB / off_chip_BW + DRAM_latency)
```

For M_BLOCK=32, N=4096, local_K=128 (BF16): FLOPs per pass = 2 × 32 × 128 × 4096 = 33.6 MFLOP.

| | AIU 1.0 | AIU 1.5 (single-chip) | AIU 1.5 (dual-chip) |
|--|---------|----------------------|---------------------|
| Peak (16-bit) | 98 TFLOPS (3.06 T/core) | 393 TFLOPS (12.3 T/core) | 393 TFLOPS (12.3 T/core) |
| Off-chip BW | 170 GBps | 1 TBps | 4.8 TBps |
| DRAM latency | 80 ns | 80 ns | 80 ns |
| Bubble | 2×(47+80) = 254 ns | 2×(8+80) = 176 ns | 2×(1.7+80) = 163 ns |
| Compute/pass | 11.0 µs | 2.7 µs | 2.7 µs |
| **Overhead** | **2.3%** | **6.4%** | **6.0%** |

The bubble is latency-dominated — higher off-chip BW barely helps (6.4% vs 6.0% for 1.5). Overhead grows on 1.5 because compute scales 4× while fixed DRAM latency is invariant. Still <10% in all cases.

Accept for v1. Options for v2 if profiling shows this matters:
- **(a)** Post-lowering KTIR optimization recovers overlap
- **(b)** Pre-Triton stage emits async DMA hints
- **(c)** Accept (likely acceptable indefinitely)

---

## Appendix B: Attention Bandwidth — What's Avoidable vs. Fundamental

**DMA overlap at Q-pass boundaries (avoidable, ~5% if invariant is violated)**:

If the invariant were violated — e.g., full 1M prefill with BLOCK_Q=256, forcing 122 Q-passes — each boundary re-reads ~192KB of KV redundantly. Over 122 boundaries against 500MB total KV: ~5% redundant bandwidth. Negligible compared to the fundamental cost below.

**Redundant KV bandwidth (fundamental, unavoidable with finite scratchpad)**:

Dense causal attention requires each query at position p to read KV[0:p]. Total KV bytes read across all queries ∝ O(seq_len²). With finite scratchpad, KV must be re-streamed for each Q-tile group — there is no way to "cache" 1M tokens of KV on-chip across Q-tiles.

Chunked prefill prevents the *extra* redundancy from a residual Q-loop within a chunk (which would multiply the O(n²) cost by the loop depth). It cannot prevent the baseline O(n²) cost — across chunks, the same KV regions are re-read by successive chunks. This is the algorithm, not an inefficiency.

Reducing the fundamental cost requires algorithmic changes orthogonal to lowering:
- Sliding-window attention (bound KV length per query)
- Sparse attention (attend to subset of KV)
- Linear attention (avoid O(n²) entirely)
- KV compression/quantization (reduce bytes per token)
