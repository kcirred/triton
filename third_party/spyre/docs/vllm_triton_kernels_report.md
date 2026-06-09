# vLLM Triton Kernels: Pointer Arithmetic vs. Descriptor-Based Access

**Date:** 2026-03-31 | **Purpose:** Assess Spyre/KTDP portability of vLLM's Triton kernels

---

## Background

Triton kernels access memory in two ways:

1. **Pointer arithmetic** (legacy): `tl.load(ptr + offsets, mask=...)` — the programmer computes addresses manually. The compiler sees raw arithmetic and cannot easily recover the tile structure.

2. **Block pointers / descriptors**: `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` — the programmer declares a structured tile descriptor. The compiler can directly map this to DMA engines (Spyre), TMA (Hopper), or structured memory access units.

**For Spyre**, block pointers are strongly preferred because the KTDP dialect's `construct_access_tile` / `ktdp.load` / `ktdp.store` ops align directly with `make_block_ptr` semantics. Recovering tile structure from pointer arithmetic IR would require complex, fragile analysis passes.

---

## vLLM Triton Kernel Inventory (~50 files, ~80+ kernels)

### Category 1: Core Inference — Attention (HIGH PRIORITY)

| File | Kernel(s) | Access Pattern | Conversion Difficulty |
|------|-----------|---------------|----------------------|
| `v1/attention/ops/prefix_prefill.py` | Paged attention prefill | **Pointer arith** — paged KV cache uses indirect indexing via block tables | **HARD** — scatter/gather for page table lookup |
| `v1/attention/ops/chunked_prefill_paged_decode.py` | Chunked prefill + decode | **Pointer arith** — paged KV with block table indirection | **HARD** — same page-table indirection issue |
| `v1/attention/ops/common.py` | Attention utility kernels | **Pointer arith** | Medium |
| `v1/attention/ops/triton_reshape_and_cache_flash.py` | KV cache reshape/store | **Pointer arith** — copies into paged cache slots | **HARD** — indirect slot indexing |
| `v1/attention/backends/flashinfer.py` | FlashInfer attention helpers | **Pointer arith** | Medium |
| `v1/attention/backends/mla/rocm_aiter_mla.py` | Multi-head Latent Attention | **Pointer arith** | Medium-Hard |

### Category 2: Core Inference — MoE (HIGH PRIORITY)

| File | Kernel(s) | Access Pattern | Conversion Difficulty |
|------|-----------|---------------|----------------------|
| `layers/fused_moe/fused_moe.py` | `fused_moe_kernel`, `write_zeros_to_output` | **Pointer arith** — expert routing with indirect indexing (token-to-expert mapping) | **HARD** — gather pattern for expert selection |
| `layers/fused_moe/batched_deep_gemm_moe.py` | Batched MoE GEMM helpers | **Pointer arith** | Medium-Hard |
| `layers/fused_moe/deep_gemm_utils.py` | Deep GEMM utilities | **Pointer arith** | Medium |
| `layers/fused_moe/router/base_router.py` | Top-k routing kernel | **Pointer arith** | Medium |

### Category 3: FLA (Flash Linear Attention) — Already Partially Converted

| File | Kernel(s) | Access Pattern | Conversion Difficulty |
|------|-----------|---------------|----------------------|
| `fla/ops/chunk_delta_h.py` | `chunk_gated_delta_rule_fwd_kernel_h_*` | **Block pointers** (extensive `make_block_ptr`) | Already done |
| `fla/ops/chunk_o.py` | `chunk_fwd_kernel_o` | **Block pointers** (5 `make_block_ptr` calls) | Already done |
| `fla/ops/chunk_scaled_dot_kkt.py` | `chunk_scaled_dot_kkt_fwd_kernel` | **Block pointers** + some scalar pointer loads | Mostly done |
| `fla/ops/cumsum.py` | `chunk_local_cumsum_*` | **Block pointers** | Already done |
| `fla/ops/kda.py` | 7 kernels (layernorm, attention, gating) | **Mixed** — most use `make_block_ptr`, one uses raw arithmetic | Easy to finish |
| `fla/ops/solve_tril.py` | 3 triangular solve kernels | **Block pointers** | Already done |
| `fla/ops/wy_fast.py` | `recompute_w_u_fwd_kernel` | **Block pointers** | Already done |
| `fla/ops/l2norm.py` | 3 L2 norm kernels | **Mixed** — `l2norm_fwd_kernel` uses block ptrs, others use pointer arith | Easy |
| `fla/ops/layernorm_guard.py` | `layer_norm_fwd_kernel` | **Pointer arith** with block ptr for some loads | Medium |
| `fla/ops/fused_recurrent.py` (via `op.py`) | Recurrent delta rule kernels | **Pointer arith** — sequential state updates | **HARD** — recurrent/sequential pattern |
| `fla/ops/fused_sigmoid_gating.py` (via `op.py`) | Sigmoid gating kernel | **Pointer arith** | Medium |

### Category 4: Mamba / SSM Kernels

| File | Kernel(s) | Access Pattern | Conversion Difficulty |
|------|-----------|---------------|----------------------|
| `mamba/ops/ssd_chunk_state.py` | SSD chunk state kernels | **Pointer arith** | Medium |
| `mamba/ops/ssd_state_passing.py` | State passing kernel | **Pointer arith** | Medium |
| `mamba/ops/ssd_bmm.py` | Batched matmul for SSD | **Pointer arith** | Easy-Medium |
| `mamba/gdn_linear_attn.py` | GDN linear attention | **Pointer arith** | Medium |

### Category 5: Quantization

| File | Kernel(s) | Access Pattern | Conversion Difficulty |
|------|-----------|---------------|----------------------|
| `quantization/utils/fp8_utils.py` | FP8 scaling/dequant kernels | **Pointer arith** | Easy-Medium |
| `quantization/utils/int8_utils.py` | INT8 dequant kernels | **Pointer arith** | Easy-Medium |
| `quantization/awq_triton.py` | AWQ dequantization | **Pointer arith** — bit manipulation + gather | **HARD** |

### Category 6: Sampling & Utility (LOWER PRIORITY)

| File | Kernel(s) | Access Pattern | Notes |
|------|-----------|---------------|-------|
| `v1/worker/gpu/sample/*.py` | min_p, logprob, bad_words, prompt_logprob | **Pointer arith** | Sampling — less critical for Spyre |
| `v1/worker/gpu/block_table.py` | Block table management | **Pointer arith** | Infrastructure |
| `v1/worker/gpu/input_batch.py` | Batch preparation | **Pointer arith** | Infrastructure |
| `layers/activation.py` | SiLU/GELU activation | **Pointer arith** | Easy |
| `layers/rotary_embedding/mrope.py` | Rotary embedding | **Pointer arith** | Medium |
| `layers/batch_invariant.py` | Batch-invariant matmul | **Pointer arith** | Medium |

---

## Key Patterns That Block Conversion

1. **Paged KV cache (attention kernels):** KV entries are stored in non-contiguous pages. The kernel does `kv_ptr + block_table[page_idx] * page_stride` — this is a gather/indirect access that `make_block_ptr` cannot express.

2. **Expert routing (MoE):** Tokens are dispatched to experts via `expert_ids[token_idx]`, creating a gather pattern.

3. **Recurrent state updates (FLA/Mamba):** Sequential dependencies where the output of step `t` feeds into step `t+1`. These use pointer arithmetic to walk through time steps with data-dependent control flow.

4. **Causal masking:** Triangular masks in attention cannot be expressed as rectangular block pointer boundary checks. Workaround: load the full rectangle via block pointer, apply mask post-load.

---

## Conversion Strategy for Spyre

### Phase 1: Low-Hanging Fruit (Easy)
- **Activation kernels** (`activation.py`): Simple elementwise — direct `make_block_ptr` replacement
- **L2 norm variants** (`l2norm.py`): Two of three already use block ptrs, convert the third
- **Quantization dequant** (`fp8_utils.py`, `int8_utils.py`): Regular tile patterns
- **FLA remaining** (`kda.py` layernorm1): One kernel still uses raw pointers

### Phase 2: Medium Effort — Dense GEMM Paths
- **Batch-invariant matmul** (`batch_invariant.py`): Standard tiled GEMM pattern
- **Mamba SSD BMM** (`ssd_bmm.py`): Batched matmul, mostly regular
- **Rotary embedding** (`mrope.py`): Regular per-element access

### Phase 3: Indirect Access — Now Supported via `ktdp.construct_indirect_access_tile`

The KTDP dialect now includes `ktdp.construct_indirect_access_tile` (per [RFC 0682](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md#5-ktdpconstruct_indirect_access_tile)), which supports gather/scatter-style indexing via auxiliary index tensors. The ConvertTTIRToKTDP pass converts `tt.descriptor_gather` and `tt.descriptor_scatter` to this op.

This unlocks:
- **Paged attention kernels**: The page table lookup (`kv_ptr + block_table[page_idx] * page_stride`) maps directly to indirect access tiles where the page table is an `index_view` operand. Triton's `tt.descriptor_gather` lowering produces the correct KTIR.
- **Fused MoE expert routing**: Token-to-expert dispatch via `sorted_token_ids` / `expert_ids` can be expressed as indirect access tiles with the routing tensor as an index view.

**RFC constraint:** The indirect access tile requires that indirect subscripts appear as standalone index values — they cannot be combined multiplicatively with other variables (RFC 0682, Section C.5). Complex linearized address expressions like `page_table[b, tkv/64] * stride + offset` must be de-linearized into structured multidimensional subscripts before representation.

Remaining hard cases:
- **Recurrent kernels** (FLA recurrent, Mamba scan): These are inherently sequential and may need a different execution model on Spyre — indirect access tiles don't help here.
- **Kernels using raw pointer arithmetic with indirect patterns** (not `tt.descriptor_gather`): These would need upstream conversion to use Triton's descriptor gather/scatter ops before the KTDP pass can handle them.

### Recommended Approach for Spyre
1. **Phase 1**: Target dense regular-access kernels (GEMM, activations, norms, FLA chunk ops) using `make_block_ptr` → `ktdp.construct_access_tile`
2. **Phase 2**: Target paged attention and MoE kernels by converting them to use `tt.descriptor_gather`/`tt.descriptor_scatter` → `ktdp.construct_indirect_access_tile`
3. **Phase 3**: For recurrent/scan kernels, evaluate whether Spyre's sequential execution model can support them or if alternative implementations are needed

---

## Summary Statistics (63 kernels cataloged)

| Category | # Kernels | Block Ptrs | Pointer Arith | Mixed | Spyre Priority |
|----------|:---------:|:----------:|:-------------:|:-----:|----------------|
| Attention (v1) | 15 | 0 | 12 | 0 | HIGH |
| Fused MoE | 8 | 0 | 8 | 0 | HIGH |
| FLA/KDA | 22 | 13 | 7 | 2 | MEDIUM (most done) |
| Lightning Attn | 5 | 3 | 1 | 1 | MEDIUM |
| Mamba/SSM | 10 | 0 | 9 | 0 | MEDIUM |
| Quantization | 2 | 0 | 2 | 0 | LOW-MEDIUM |
| Activation | 1 | 0 | 1 | 0 | LOW |
| **Total** | **63** | **16** | **40** | **3** | |

### Conversion Difficulty Breakdown (of 40 pointer-arithmetic kernels)

| Difficulty | Count | Examples |
|-----------|:-----:|---------|
| **Easy** | 8 | `write_zeros_to_output`, `_swiglustep_and_mul`, `merge_attn_states`, L2 norm variants |
| **Easy-Medium** | 3 | Layer norm variants (`layernorm_guard`, `layernorm_gated`) |
| **Medium** | 11 | Prefill attention GEMM paths, `_bmm_chunk_fwd`, `awq_dequantize`, reshape_and_cache |
| **Medium-Hard** | 5 | `fused_moe_kernel` GEMM path, unified attention, `awq_gemm` |
| **Hard** | 13 | Paged decode attention, MoE with GPTQ, all recurrent/scan kernels, pack_bitmatrix |

**Bottom line:** ~63% of vLLM's Triton kernels use pointer arithmetic exclusively. The FLA and Lightning Attention subsystems are the notable exceptions, having already adopted block pointers extensively. The highest-priority kernels (attention, MoE) are also the hardest to convert due to indirect/gather access patterns — these will likely require memory layout changes rather than pure kernel rewrites.
