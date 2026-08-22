# NVFP4 Volta Implementation Plan (v100-skinny -> llama.cpp)

Port `v100-skinny` QPN execution architecture (https://github.com/dnv2003/v100-skinny) from vLLM to llama.cpp.
Target: 4x V100 (SM70, cc=700) decode parity vs RTX 5090 Blackwell on Qwen3.8-27B NVFP4 mixed FP4/FP8.

## 1. What v100-skinny does

- Volta has no FP4/FP8 Tensor Core. QPN keeps weights compressed in HBM, translates fragment-by-fragment to FP16 register layout for Volta `m8n8k4` HMMA (`ggml/src/ggml-cuda/mma.cuh:VOLTA_MMA_AVAILABLE`), no upfront dequant.
- QPN2 (NVFP4): M=1 679.5 GB/s (77% of 879 ceiling), M=8 619.8 GB/s (71%). QPN8 (FP8): M=1-4 ~719 GB/s (82%). lm_head 842.9 GB/s (96%).
- M=8 trick: Volta tile = 8 rows. Qwen3.8 MTP k=7 verify = 1 target + 7 draft = 8 rows -> one HMMA tile. 1.38x tokens/round (5.89 vs 4.27 NInfer) cancels 1.35x latency (26.9ms vs 19.9ms) -> 1.02x parity (219.1 vs 214.7 tok/s).
- v1.0: FP8 regions converted to NVFP4. v1.1: mixed stays mixed FP4->QPN2, FP8->QPN8, act=FP16, KV=FP16. FP8 KV hit slow scalar attention -> forced FP16 KV.
- Packed format: `mlp.down_proj.weight U8 [5120,8704]` + `weight_scale F8_E4M3 [5120,1088]` where 1088*16=17408=8704*2 (2x 4b/byte, scale per 16). Runtime `skinny_codes/skinny_scales` + `_qpn_prepack` (fork_patches/marlin.py) unpacks nibbles (`qs & 0xF`, `qs>>4`) and interleaves to fragment-order tiles. Not a checkpoint tensor.

## 2. What llama.cpp does today

- Types exist: `ggml/include/ggml.h:429` `GGML_TYPE_MXFP4=39`, `430` `GGML_TYPE_NVFP4=40`; `ggml/src/ggml-common.h:214` `QK_MXFP4=32`, `221` `QK_NVFP4=64`, `222` `QK_NVFP4_SUB=16`, `214-227` `block_mxfp4{ e, qs[16]}`, `block_nvfp4{ d[4], qs[32]}`.
- Ref quant: `ggml/src/ggml-quants.c:350` `quantize_row_mxfp4_ref`, `384` `quantize_row_nvfp4_ref`.
- CUDA gated to Blackwell only: `ggml/src/ggml-cuda/common.cuh:360` `blackwell_mma_available(cc)` (cc>=1200), `ggml/src/ggml-cuda/mmq.cu:131` `use_native_fp4 = blackwell_mma_available(cc) && (MXFP4||NVFP4)`. On Volta (cc=70) falls back to `quantize_mmq_q8_1_cuda` + `block_q8_1_mmq` via `mmq-config-ampere.cuh`/`pascal.cuh` (`mmq.cuh:244` `ggml_cuda_mmq_get_config`). Blackwell tiles `mmq.cuh:51` `block_fp4_mmq`, `mmq-load-tiles.cuh:1586-1753` never selected on Volta.

## 3. Mapping vLLM -> llama.cpp

| vLLM | llama.cpp | Notes |
|---|---|---|
| `fork_patches/marlin.py` `skinny_codes/skinny_scales` + `_qpn_prepack` | `src/llama-model-loader.cpp` + `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | GGUF already packed; prepack becomes tile swizzle in CUDA loader |
| QPN2 NVFP4 M=1/M=8 | `ggml/src/ggml-cuda/mmq.cu` + `mmvq.cu/mmvf.cu` + `mmq-load-tiles.cuh:1681` | M=1 -> mmvq, M=8 -> small-batch mmq verify |
| QPN8 FP8 M=1-4/M=8 | New `GGML_TYPE_FP8` or reuse MXFP4 wrapper | llama has no FP8 weight type today |
| lm_head native 4-bit | `mmq.cuh:51` Q4 path | Keep, already 96% BW |
| 1cat-vLLM FlashAttention SM70 / CUDA Graphs / MTP k=7 | `ggml/src/ggml-cuda/fattn*.cu` + `src/llama-graph.cpp` + `src/llama-kv-cache.cpp` | Must use `--cache-type-k/v f16`, not FP8 |

## 4. Port - opt-in

**Flag is opt-in (default OFF).** No behavior change unless user enables.

- Env: `GGML_CUDA_VOLTA_QPN=1` to enable, `0` or unset = off (default).
- CLI: `--volta-qpn` (enable QPN2/QPN8 Volta path), no flag = off. Plumbed via `common/common.cpp` -> `ggml_cuda_set_volta_qpn(bool)` -> `ggml/src/ggml-cuda/ggml-cuda.cu:5441` / `common.cuh:360` area, same pattern as `GGML_CUDA_FORCE_*`.
- When off: current generic dequant/dp4a path unchanged. When on and `cc==700` and `src0->type==MXFP4||NVFP4` (and FP8 when added): route to new Volta path.

## 5. Implementation steps

1. **Config:** Add `ggml/src/ggml-cuda/mmq-config-volta.cuh` (or extend ampere) with `I/J/K_vram` tuned for M=1 vs M=8, `GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4` (`mmq.cuh:121`). Return it from `mmq.cuh:244` `ggml_cuda_mmq_get_config` when `volta_qpn_enabled && cc==700`.
2. **Tile loaders:** Add Volta loaders in `mmq-load-tiles.cuh` - unpack `qs & 0xF`/`qs>>4` via `kvalues_mxfp4` (`vecdotq.cuh:321`), `UE4M3->FP32` via `ggml_cuda_ue4m3_to_fp32` (`mmq-load-tiles.cuh:1715`), `__half` convert, `ldmatrix` fragment order for `m8n8k4`. Keep `block_mxfp4/nvfp4` in HBM.
3. **Dispatch:** Extend `mmq.cu:131` to `use_qpn = volta_qpn_enabled && cc==700 && (MXFP4||NVFP4)`; select `quantize_mmq_q8_1_cuda` for activations (FP16) and new weight tile. Add `GGML_TYPE_FP8` for QPN8 if block-128 scales needed.
4. **KV:** Document/test `--cache-type-k f16 --cache-type-v f16`; FP8 KV remains slow scalar path.
5. **Tests:** `tests-volta/decode_tps.py` + `tps_suite.sh` at M=1, M=8 (k=7 verify), M=4, long-ctx 65K sweep (k=3 vs k=7 vs off), `q4_K_M` regression with flag off.

## 6. Scope / non-goals

- Decode BW-bound parity only; prefill stays ~4x behind 5090/NInfer (compute-bound).
- 4x V100 TP4, ~A$600 accelerator HW, 300W/card, not power/density win.
- Whole-model scope (TP, attention, graphs, sampling) not GEMM-only; prefill not cloned.

## 7. Repro

- Repo: https://github.com/dnv2003/v100-skinny
- Checkpoint: `RadixArk/Qwen3.8-27B-NVFP4` (packed, no conversion script; prepack is in-memory at load)
- Llama side: `llama-server -m qwen3.8-nvfp4.gguf --volta-qpn --cache-type-k f16 --cache-type-v f16` (once implemented)
