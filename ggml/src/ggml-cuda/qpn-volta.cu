#include "qpn-volta.cuh"
#include "convert.cuh"

#include <mutex>
#include <unordered_map>
#include <chrono>

// Port of the QPN2 NVFP4 path from v100-skinny (github.com/dnv2003/v100-skinny,
// kernels/skinny_kernels.cu). Volta has no FP4 tensor core, so the 4-bit codes are
// decoded by bit manipulation straight into the FP16 register operands of
// mma.sync.m8n8k4. No shared memory in the main loop; the only barrier is the
// cross-warp K reduce in the epilogue.
//
// Weights are pre-permuted once into fragment order so the decoder output register
// pair IS the B fragment. Layout: codes uint2[tile N/32][group K/16][lane 32],
// scales u8 with the same index. Lane owns one N column, so one warp reads 256 B
// per group.

#define QPN_TILE_N  32
#define QPN_WARPS    4
#define QPN_GROUP_K 16

// lane -> column inside the 32-wide N tile (skinny_kernels.cu:1165)
#define QPN_LANE_QP(lane)  (((lane) >> 2) & 3)
#define QPN_LANE_R(lane)   (((lane) & 3) + (((lane) & 16) ? 4 : 0))
#define QPN_LANE_COL(lane) (QPN_LANE_QP(lane)*8 + QPN_LANE_R(lane))

// mirrors VLLM_SKINNY_QPN2=0: fall back to the fixed-4-warp kernel for M <= 8
static bool ggml_cuda_qpn2_disabled() {
    const char * env = getenv("GGML_CUDA_VOLTA_QPN2");
    return env && env[0] == '0';
}

static __global__ void qpn_prepack_nvfp4(
        const block_nvfp4 * __restrict__ src, uint8_t * __restrict__ codes, uint8_t * __restrict__ scales,
        const int N, const int K) {
    const int64_t idx = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;

    const int     G     = K / QPN_GROUP_K;
    const int64_t total = (int64_t)(N / QPN_TILE_N) * G * 32;
    if (idx >= total) {
        return;
    }

    const int lane = idx & 31;
    const int g    = (int) ((idx >> 5) % G);
    const int tile = (int) ((idx >> 5) / G);
    const int n    = tile*QPN_TILE_N + QPN_LANE_COL(lane);

    const block_nvfp4 * blk = src + (int64_t) n * (K/QK_NVFP4) + g/4;
    const uint8_t     * qs  = blk->qs + (g & 3)*(QK_NVFP4_SUB/2);

    // value kk of this 16-wide group: kk<8 low nibble of qs[kk], else high nibble of qs[kk-8]
    uint8_t w[QPN_GROUP_K];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        w[kk    ] = qs[kk] & 0x0F;
        w[kk + 8] = qs[kk] >>   4;
    }

    // nibble j = k(2j), nibble j+4 = k(2j+1) within each 8-value half
    uint8_t out[8];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int b = 8*h;
        out[4*h + 0] = (uint8_t) ((w[b + 2] << 4) | w[b + 0]);
        out[4*h + 1] = (uint8_t) ((w[b + 6] << 4) | w[b + 4]);
        out[4*h + 2] = (uint8_t) ((w[b + 3] << 4) | w[b + 1]);
        out[4*h + 3] = (uint8_t) ((w[b + 7] << 4) | w[b + 5]);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        codes[idx*8 + i] = out[i];
    }
    scales[idx] = blk->d[g & 3];
}


#define QPN_MMA(C, A0, A1, B0, B1)                                       \
    asm volatile(                                                        \
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                 \
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                    \
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),     \
          "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                              \
        : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// out[i] = half2(nibble i, nibble i+4), each nibble placed as FP16 sign + exp/mant
// exact ue4m3 -> broadcast half2 (normals and subnormals); no NaN handling,
// matching ggml_cuda_ue4m3_to_fp32's zero-for-NaN applied by callers
static __device__ __forceinline__ half2 qpn_scale_h2(uint8_t b) {
    const unsigned short hb =
        (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));
    return __halves2half2(hs, hs);
}

// decoder output carries a 2^-14 exponent bias; group scales stay raw here
// (bits*scale <= 0.2 never overflows half) and the bias is recovered once per
// output row in the epilogues (QPN_UNBIAS below)
static __device__ __forceinline__ void qpn_decode8(unsigned q, half2 sc2, half2 out[4]) {
    constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
    unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
    unsigned v1 = ((q <<  8) & S) | ((q << 5) & EM);
    unsigned v2 = ((q <<  4) & S) | ((q << 1) & EM);
    unsigned v3 = ( q        & S) | ((q >> 3) & EM);
    out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2);
    out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2);
    out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2);
    out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2);
}

template <int MT>
static __global__ void qpn_nvfp4_mma(
        const uint8_t * __restrict__ codes, const uint8_t * __restrict__ scales,
        const half * __restrict__ x, float * __restrict__ dst,
        const int N, const int K, const int M) {
#ifdef VOLTA_MMA_AVAILABLE
    __shared__ float cs[QPN_WARPS][MT*256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int qp   = QPN_LANE_QP(lane);
    const int r    = QPN_LANE_R(lane);

    const int G  = K >> 4;
    const int Gq = G / QPN_WARPS;
    const int g0 = warp*Gq;

    const uint2   * cb = reinterpret_cast<const uint2 *>(codes) + (size_t) tile*G*32 + lane;
    const uint8_t * sb = scales + (size_t) tile*G*32 + lane;

    float c[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[t][i] = 0.0f;
        }
    }

    for (int g = g0; g < g0 + Gq; ++g) {
        const uint2 q2 = __ldcs(cb + (size_t) g*32);
        const half2 sc2 = qpn_scale_h2(__ldg(sb + (size_t) g*32));

        half2 b[8];
        qpn_decode8(q2.x, sc2, b + 0);
        qpn_decode8(q2.y, sc2, b + 4);
        const unsigned * B = reinterpret_cast<const unsigned *>(b);

#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int ar = t*8 + r;

            uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
            if (ar < M) {
                const half * xrow = x + (size_t) ar*K;
                a01 = *reinterpret_cast<const uint4 *>(xrow + g*QPN_GROUP_K);
                a23 = *reinterpret_cast<const uint4 *>(xrow + g*QPN_GROUP_K + 8);
            }
            const unsigned * A0 = reinterpret_cast<const unsigned *>(&a01);
            const unsigned * A1 = reinterpret_cast<const unsigned *>(&a23);

            QPN_MMA(c[t], A0[0], A0[1], B[0], B[1]);
            QPN_MMA(c[t], A0[2], A0[3], B[2], B[3]);
            QPN_MMA(c[t], A1[0], A1[1], B[4], B[5]);
            QPN_MMA(c[t], A1[2], A1[3], B[6], B[7]);
        }
    }

    // recover the decoder's 2^-14 bias once per accumulator
#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[t][i] *= 16384.0f;
        }
    }

    // C fragment map (skinny_kernels.cu:1207)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(t*8 + row)*32 + qp*8 + col] = c[t][i];
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < MT*256; e += blockDim.x) {
        const int row = e >> 5;
        if (row >= M) {
            continue;
        }
        const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
        dst[(size_t) row*N + (size_t) tile*QPN_TILE_N + (e & 31)] = v;
    }
#else
    GGML_UNUSED_VARS(codes, scales, x, dst, N, K, M);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}


// QPN2 (skinny_kernels.cu:1394): same QP-N architecture and prepacked layout, with
// SPLITK warps per CTA splitting K on one N=32 tile and NACC independent accumulators
// across the four k-slice mma ops. M <= 8 only; M 9..16 stays on qpn_nvfp4_mma<2>.
template <int SPLITK, int NACC>
static __global__ void qpn_nvfp4_qpn2(
        const uint8_t * __restrict__ codes, const uint8_t * __restrict__ scales,
        const half * __restrict__ x, float * __restrict__ dst,
        const int N, const int K, const int M) {
#ifdef VOLTA_MMA_AVAILABLE
    __shared__ float cs[SPLITK][256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int qp   = QPN_LANE_QP(lane);
    const int r    = QPN_LANE_R(lane);

    const int G  = K >> 4;
    const int Gq = G / SPLITK;
    const int g0 = warp*Gq;

    const uint2   * cb = reinterpret_cast<const uint2 *>(codes) + (size_t) tile*G*32 + lane;
    const uint8_t * sb = scales + (size_t) tile*G*32 + lane;

    float c[NACC][8];
#pragma unroll
    for (int a = 0; a < NACC; ++a) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[a][i] = 0.0f;
        }
    }

#pragma unroll 4
    for (int g = g0; g < g0 + Gq; ++g) {
        const uint2 q2 = __ldcs(cb + (size_t) g*32);
        const half2 sc2 = qpn_scale_h2(__ldg(sb + (size_t) g*32));

        half2 b[8];
        qpn_decode8(q2.x, sc2, b + 0);
        qpn_decode8(q2.y, sc2, b + 4);
        const unsigned * B = reinterpret_cast<const unsigned *>(b);

        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (r < M) {
            const half * xrow = x + (size_t) r*K;
            a01 = *reinterpret_cast<const uint4 *>(xrow + g*QPN_GROUP_K);
            a23 = *reinterpret_cast<const uint4 *>(xrow + g*QPN_GROUP_K + 8);
        }
        const unsigned * A0 = reinterpret_cast<const unsigned *>(&a01);
        const unsigned * A1 = reinterpret_cast<const unsigned *>(&a23);

        QPN_MMA(c[0],        A0[0], A0[1], B[0], B[1]);
        QPN_MMA(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
        QPN_MMA(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
        QPN_MMA(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
    }

#pragma unroll
    for (int a = 1; a < NACC; ++a) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[0][i] += c[a][i];
        }
    }

    // recover the decoder's 2^-14 bias once per accumulator
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        c[0][i] *= 16384.0f;
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row*32 + qp*8 + col] = c[0][i];
    }
    __syncthreads();

    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        const int row = e >> 5;
        if (row >= M) {
            continue;
        }
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SPLITK; ++w) {
            v += cs[w][e];
        }
        dst[(size_t) row*N + (size_t) tile*QPN_TILE_N + (e & 31)] = v;
    }
#else
    GGML_UNUSED_VARS(codes, scales, x, dst, N, K, M);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

// llama block_nvfp4 packs byte i of each 8-byte group as (code i+8 << 4) | code i,
// the reference packed format is sequential pairs. Rebuild decoder input words so
// nibble j = code j.
static __device__ __forceinline__ unsigned qpn_pack_nib(unsigned x) {
    const unsigned p = (x | (x >> 4)) & 0x00FF00FFu;
    return __byte_perm(p, 0, 0x5420);
}

static __device__ __forceinline__ uint2 qpn_codes_canon(unsigned a, unsigned b) {
    const unsigned m = 0x0F0F0F0Fu;
    return make_uint2(qpn_pack_nib(a & m)          | (qpn_pack_nib(b & m) << 16),
                      qpn_pack_nib((a >> 4) & m)   | (qpn_pack_nib((b >> 4) & m) << 16));
}

// Faithful port of skinny_nvfp4_simt (skinny_kernels.cu, gemm_simt entry): one
// warp per output row, 8 warps per block, R rows per warp. x is staged to smem in
// KC chunks with an XOR bank swizzle; fp16 accumulation window per 16-code segment
// flushed to fp32. Not on the default route (qpn2 owns M 1..8); kept for the
// no-prepack seam, mirroring marlin.py's gemm_simt fallback. Reads block_nvfp4
// directly - no prepack copy on this path.
template <int M, int KC, int R>
static __global__ void qpn_nvfp4_simt(
        const block_nvfp4 * __restrict__ src, const half * __restrict__ x,
        float * __restrict__ dst, const int N, const int K) {
#ifdef VOLTA_MMA_AVAILABLE
    extern __shared__ char smem_raw[];
    half2 * xs = reinterpret_cast<half2 *>(smem_raw);   // [M][KC/2] swizzled
    constexpr int P2 = KC / 2;

    // XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the
    // simt read pattern (lane-groups sharing a bank base differ in p>>5)
    const auto swz = [](int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); };

    // stage 8 contiguous activation halves as four (k, k+4) half2 pairs, matching
    // the interleaved output of qpn_decode8
    const auto stage_pairs = [&](half2 * dst2, int base_pair, const uint4 & v) {
        const unsigned * r = reinterpret_cast<const unsigned *>(&v);
        const unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                               __byte_perm(r[0], r[2], 0x7632),
                               __byte_perm(r[1], r[3], 0x5410),
                               __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst2[swz(base_pair + j)] = *reinterpret_cast<const half2 *>(o + j);
        }
    };

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = (blockIdx.x * 8 + warp) * R;

    const block_nvfp4 * brow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        brow[r] = src + (size_t)(n + r) * (K / QK_NVFP4);
    }

    float accf[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            accf[r][m] = 0.0f;
        }
    }

    // one 16-code segment == one scale group: load codes and scale, decode to
    // half2, hfma2 against the staged x, flush the window to fp32. s is the
    // chunk-relative segment index, g the global group index.
    const auto seg_body = [&](int s, int g) {
        uint2 q2[R];
        half2 sc2[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_nvfp4 * blk = brow[r] + (g >> 2);
            const uint8_t * q8 = blk->qs + (g & 3)*(QK_NVFP4_SUB/2);
            // qs sits at a 4B-aligned 36 B stride, so one u32 pair instead of u2
            q2[r] = qpn_codes_canon(*reinterpret_cast<const uint32_t *>(q8),
                                    *reinterpret_cast<const uint32_t *>(q8 + 4));
            sc2[r] = qpn_scale_h2(blk->d[g & 3]);
        }

        // fp16 accumulation window is one 16-code segment (8 products per half2
        // lane); flushed to fp32 so activation outliers cannot overflow half range
        half2 acch[R][M];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acch[r][m] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            half2 w4[R][4];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                qpn_decode8(w == 0 ? q2[r].x : q2[r].y, sc2[r], w4[r]);
            }
#pragma unroll
            for (int pi = 0; pi < 4; ++pi) {
                const int psw = swz(s*8 + w*4 + pi);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const half2 xv = xs[m*P2 + psw];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float2 f = __half22float2(acch[r][m]);
                accf[r][m] += f.x + f.y;
            }
        }
    };

    int k0 = 0;
    for (; k0 + KC <= K; k0 += KC) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < M*(KC/8); idx += blockDim.x) {
            const int m = idx / (KC/8), j4 = idx % (KC/8);
            const uint4 v = *reinterpret_cast<const uint4 *>(x + (size_t) m*K + k0 + j4*8);
            stage_pairs(xs + m*P2, j4*4, v);
        }
        __syncthreads();
        const int gb = k0 >> 4;
#pragma unroll
        for (int i = 0; i < KC/512; ++i) {
            const int s = lane + 32*i;
            seg_body(s, gb + s);
        }
    }

    // tail chunk: K % KC remainder (any multiple of 128). Same layout and
    // swizzle, runtime segment bound with idle-lane guard.
    const int tail = K - k0;
    if (tail > 0) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < M*(tail/8); idx += blockDim.x) {
            const int m = idx / (tail/8), j4 = idx % (tail/8);
            const uint4 v = *reinterpret_cast<const uint4 *>(x + (size_t) m*K + k0 + j4*8);
            stage_pairs(xs + m*P2, j4*4, v);
        }
        __syncthreads();
        const int gb = k0 >> 4;
        const int nseg = tail >> 4;
        for (int s = lane; s < nseg; s += 32) {
            seg_body(s, gb + s);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = accf[r][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0) {
                dst[(size_t) m*N + n + r] = v * 16384.0f;
            }
        }
    }
#else
    GGML_UNUSED_VARS(src, x, dst, N, K);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

// marlin.py:_QPN2_TABLE + _qpn2_cfg, ported as written.
static bool qpn2_cfg(int K, int N, int & splitk, int & nacc) {
    // TEMP splitk sweep knob: GGML_CUDA_VOLTA_QPN_SPLITK=16[,2]
    if (const char * e = getenv("GGML_CUDA_VOLTA_QPN_SPLITK")) {
        const int s = atoi(e);
        if ((K / QPN_GROUP_K) % s == 0) {
            splitk = s;
            nacc   = s >= 16 ? 2 : 1;
            return true;
        }
    }
    static const struct { int k, n, splitk, nacc; } table[] = {
        { 1536,  5120, 16, 2}, { 4352,  5120, 16, 2}, { 5120,  8704,  8, 2},
        { 5120,  4096, 16, 2}, { 5120,  2048, 32, 2}, { 5120, 62080,  8, 1},
        { 5120,  3584, 16, 2},
    };
    const int G = K / QPN_GROUP_K;
    for (const auto & e : table) {
        if (e.k == K && e.n == N && G % e.splitk == 0) {
            splitk = e.splitk;
            nacc   = e.nacc;
            return true;
        }
    }
    // smallest split that puts ~640+ warps in flight (80 SMs x 8)
    for (int s : {8, 16, 32}) {
        if (G % s == 0 && (N/QPN_TILE_N)*s >= 640) {
            splitk = s;
            nacc   = s >= 16 ? 2 : 1;
            return true;
        }
    }
    for (int s : {16, 8}) {
        if (G % s == 0) {
            splitk = s;
            nacc   = s >= 16 ? 2 : 1;
            return true;
        }
    }
    return false;
}

// Prepacked copy of one weight tensor. Same total size as the source blocks
// (9 bytes per 16 weights either way) but a different order, so it is held
// alongside the original - the original still serves mmq for M > 16.
struct qpn_weights {
    uint8_t * codes  = nullptr;
    uint8_t * scales = nullptr;
    int       device = -1;
};

static std::mutex                                    g_qpn_mutex;
static std::unordered_map<const void *, qpn_weights> g_qpn_cache;

static const qpn_weights * qpn_get_weights(const ggml_tensor * src0, int device, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_qpn_mutex);

    qpn_weights w;

    auto it = g_qpn_cache.find(src0->data);
    if (it != g_qpn_cache.end()) {
        // device < 0 marks a remembered allocation failure: do not retry, the
        // caller keeps that tensor on the stock paths
        return it->second.device < 0 ? nullptr : &it->second;
    }

    // allocating during graph capture is illegal; report no-prepack without
    // caching so a later eager call can still build it
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    // keep a VRAM reserve for activations/workspace: greedy prepack starves
    // every other allocation on the device (GGML_CUDA_VOLTA_QPN_RESERVE_MB)
    static const size_t reserve_mb = [] {
        const char * e = getenv("GGML_CUDA_VOLTA_QPN_RESERVE_MB");
        return e ? (size_t) atoi(e) : (size_t) 1536;
    } ();
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t need = (size_t)(src0->ne[0] * src0->ne[1] / 2)
        + (size_t)(src0->ne[0] * src0->ne[1] / QPN_GROUP_K);
    if (free_b < need + reserve_mb * 1024 * 1024) {
        w.device = -1;
        g_qpn_cache.emplace(src0->data, w);
        return nullptr;
    }

    const int     K   = src0->ne[0];
    const int     N   = src0->ne[1];
    const int64_t nw  = (int64_t) N * K;

    w.device = device;
    if (cudaMalloc(&w.codes, nw/2) != cudaSuccess) {
        cudaGetLastError();
        w.device = -1;
        g_qpn_cache.emplace(src0->data, w);
        return nullptr;
    }
    if (cudaMalloc(&w.scales, nw/QPN_GROUP_K) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(w.codes);
        w.device = -1;
        g_qpn_cache.emplace(src0->data, w);
        return nullptr;
    }

    const int64_t total = nw / QPN_GROUP_K;
    const int     block = 256;
    qpn_prepack_nvfp4<<<(total + block - 1)/block, block, 0, stream>>>(
        (const block_nvfp4 *) src0->data, w.codes, w.scales, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (getenv("GGML_CUDA_VOLTA_QPN_TRACE")) {
        int dev = -1;
        cudaGetDevice(&dev);
        fprintf(stderr, "QPN-PREPACK: K=%d N=%d dev=%d done\n", K, N, dev);
    }

    return &g_qpn_cache.emplace(src0->data, w).first->second;
}

void ggml_cuda_qpn_free_cache_range(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(g_qpn_mutex);
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    for (auto it = g_qpn_cache.begin(); it != g_qpn_cache.end(); ) {
        const char * p = (const char *) it->first;
        if (p >= lo && p < hi) {
            cudaFree(it->second.codes);
            cudaFree(it->second.scales);
            it = g_qpn_cache.erase(it);
        } else {
            ++it;
        }
    }
}

bool ggml_cuda_should_use_qpn_volta(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc, cudaStream_t stream) {
    if (!volta_qpn_available(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // 2D only, contiguous, and the tile/group geometry must divide exactly
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[0] % QK_NVFP4 != 0 || src0->ne[1] % QPN_TILE_N != 0) {
        return false;
    }
    // K must split evenly across the 4 warps
    if ((src0->ne[0] / QPN_GROUP_K) % QPN_WARPS != 0) {
        return false;
    }
    // size floor: on tiny projections (hybrid GDN layers have many N=32-class
    // GEMVs) mmvq's leaner launch wins; qpn2 pays off once weight streaming
    // dominates. The reference model has no such tiny shapes.
    if ((int64_t) src0->ne[0] * src0->ne[1] < (int64_t) 14 * 1024 * 1024) {
        return false;
    }
    const int M = (int) src1->ne[1];
    // prepack must exist: the fragment kernels cannot run from the GGUF
    // blocks, and when the prepack allocation failed (VRAM budget) those
    // tensors stay on the stock paths entirely. The direct-read simt seam is
    // NOT used here - its per-group nibble canonicalization loses to mmvq.
    const qpn_weights * w = qpn_get_weights(src0, ggml_cuda_get_device(), stream);
    if (w == nullptr) {
        return false;
    }
    return M <= 16;
}

// TEMP kernel timing helper, see ggml_cuda_mul_mat_qpn_volta
static void dispatch_qpn_kernels(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const half * src1_f16, cudaStream_t stream);

void ggml_cuda_mul_mat_qpn_volta(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = src0->ne[0];
    const int N = src0->ne[1];
    const int M = src1->ne[1];

    {   // TEMP route trace, GGML_CUDA_VOLTA_QPN_TRACE=1 to enable
        static const bool trace = getenv("GGML_CUDA_VOLTA_QPN_TRACE") != nullptr;
        if (trace) {
            int tk = 0, tn = 0;
            const bool cfg_ok = M <= 8 && qpn2_cfg(K, N, tk, tn);
            fprintf(stderr, "QPN-ROUTE: K=%d N=%d M=%d qpn2=%d splitk=%d nacc=%d\n", K, N, M, (int) cfg_ok, tk, tn);
        }
    }


    // TEMP kernel timing, GGML_CUDA_VOLTA_QPN_TIMING=1: sample every 37th call,
    // sync-bracketed, aggregated per shape and flushed at exit
    static const bool timing = getenv("GGML_CUDA_VOLTA_QPN_TIMING") != nullptr;
    struct qpn_time_acc {
        std::unordered_map<uint64_t, std::pair<double, long>> per_shape;
        ~qpn_time_acc() {
            for (const auto & e : per_shape) {
                const int k = (int) (e.first >> 40), n = (int) ((e.first >> 20) & 0xFFFFF), m = (int) (e.first & 0xFFFFF);
                fprintf(stderr, "QPN-TIME: K=%d N=%d M=%d calls=%ld us_per_call=%.2f\n",
                        k, n, m, e.second.second, e.second.first / e.second.second);
            }
        }
    };
    static qpn_time_acc qpn_times;
    static long qpn_timing_n = 0;
    const uint64_t shape_key = ((uint64_t) K << 40) | ((uint64_t) N << 20) | (uint64_t) M;
    const bool sample = timing && (qpn_timing_n++ % 37) == 0;

    cudaStream_t stream = ctx.stream();

    float * dst_d = (float *) dst->data;

    // kernels consume FP16 activations, same as the reference
    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(), (size_t) M * K);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);

    if (sample) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t0 = std::chrono::steady_clock::now();
        to_fp16(src1->data, src1_f16.get(), (int64_t) M * K, stream);
        dispatch_qpn_kernels(ctx, src0, src1, dst, src1_f16.get(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t1 = std::chrono::steady_clock::now();
        auto & acc = qpn_times.per_shape[shape_key];
        acc.first += std::chrono::duration<double, std::micro>(t1 - t0).count();
        acc.second++;
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    to_fp16(src1->data, src1_f16.get(), (int64_t) M * K, stream);
    CUDA_CHECK(cudaGetLastError());

    dispatch_qpn_kernels(ctx, src0, src1, dst, src1_f16.get(), stream);
    CUDA_CHECK(cudaGetLastError());
}

static void dispatch_qpn_kernels(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const half * src1_f16, cudaStream_t stream) {
    const int K = src0->ne[0];
    const int N = src0->ne[1];
    const int M = src1->ne[1];
    float * dst_d = (float *) dst->data;

    // Dispatch mirrors marlin.py::_skinny_linear: qpn2 owns M 1..8 ("qpn2 owns
    // M 1..8, msweep: 550-800 GB/s at M=1"); the fixed-4-warp kernel takes over
    // when qpn2 is disabled or has no split config, M 9..16 rides the MT=2 tile.
    const qpn_weights * w = qpn_get_weights(src0, ctx.device, stream);

    if (w == nullptr) {
        // prepack did not fit: serve the decode band straight from the GGUF
        // blocks, mirroring marlin.py's gemm_simt fallback when no prepack exists
        GGML_ASSERT(M <= 3);
        constexpr int KC = 1024;
        const bool two_rows = (K <= 2048) && (N % 16 == 0);
        const dim3 grid2(two_rows ? N/16 : N/8, 1, 1);
        const dim3 block2(256, 1, 1);
        const size_t smem = (size_t) M*(KC/2)*sizeof(half2);
        auto launch = [&](auto fn) {
            fn<<<grid2, block2, smem, stream>>>((const block_nvfp4 *) src0->data, src1_f16, dst_d, N, K);
        };
#define QPN_SIMT_CASE(MM, RR)                                                                 \
        if (M == (MM) && two_rows == ((RR) == 2)) {                                           \
            launch(qpn_nvfp4_simt<MM, KC, RR>);                                               \
        } else
        QPN_SIMT_CASE(1, 1)
        QPN_SIMT_CASE(1, 2)
        QPN_SIMT_CASE(2, 1)
        QPN_SIMT_CASE(2, 2)
        QPN_SIMT_CASE(3, 1)
        QPN_SIMT_CASE(3, 2)
#undef QPN_SIMT_CASE
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid(N / QPN_TILE_N, 1, 1);

    int splitk = 0, nacc = 0;
    if (M <= 8 && !ggml_cuda_qpn2_disabled() && qpn2_cfg(K, N, splitk, nacc)) {
        const dim3 block(splitk*32, 1, 1);
        const uint8_t * c = w->codes;
        const uint8_t * sc = w->scales;
        const half    * a = src1_f16;
#define QPN2_CASE(SK, NA)                                                                          \
        if (splitk == (SK) && nacc == (NA)) {                                                      \
            qpn_nvfp4_qpn2<SK, NA><<<grid, block, 0, stream>>>(c, sc, a, dst_d, N, K, M);          \
        } else
        QPN2_CASE( 8, 1)
        QPN2_CASE( 8, 2)
        QPN2_CASE(16, 1)
        QPN2_CASE(16, 2)
        QPN2_CASE(32, 1)
        QPN2_CASE(32, 2)
        {
            GGML_ABORT("qpn2: unhandled splitk=%d nacc=%d", splitk, nacc);
        }
#undef QPN2_CASE
    } else {
        const dim3 block(QPN_WARPS*32, 1, 1);
        if (M <= 8) {
            qpn_nvfp4_mma<1><<<grid, block, 0, stream>>>(w->codes, w->scales, src1_f16, dst_d, N, K, M);
        } else {
            qpn_nvfp4_mma<2><<<grid, block, 0, stream>>>(w->codes, w->scales, src1_f16, dst_d, N, K, M);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
