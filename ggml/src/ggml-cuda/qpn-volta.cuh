#pragma once

#include "common.cuh"

// Volta (SM70) QPN path for NVFP4, ported from v100-skinny (github.com/dnv2003/v100-skinny).
// Weights stay 4-bit in HBM and are decoded straight into the FP16 register operands of
// mma.sync.m8n8k4. Opt-in: GGML_CUDA_VOLTA_QPN=1 / --volta-qpn.

void ggml_cuda_qpn_free_cache_range(const void * base, size_t size);

bool ggml_cuda_should_use_qpn_volta(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc, cudaStream_t stream);

void ggml_cuda_mul_mat_qpn_volta(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
