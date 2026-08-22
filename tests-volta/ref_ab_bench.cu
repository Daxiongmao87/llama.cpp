// A/B microbench: verbatim v100-skinny qpn2 kernel vs the llama.cpp port,
// same shapes, same launch geometry, CUDA-event timed, no framework overhead.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define DEV_INLINE __device__ __forceinline__

#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

DEV_INLINE void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2p);
}

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));
  return __halves2half2(hs, hs);
}

// ---- verbatim v100-skinny skinny_nvfp4_qpn2 ----
template <int SPLITK, int NACC>
__global__ void ref_qpn2(const uint8_t *__restrict__ bcodes,
                         const uint8_t *__restrict__ bscales,
                         const half *__restrict__ x, half *__restrict__ y,
                         int N, int K, int M, float gscale) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint2 *cb = reinterpret_cast<const uint2 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = bscales + (size_t)tile * G * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;
#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg((const unsigned char *)(sb + (size_t)g * 32))), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);
    dequant8_tm(q2.y, sc2, b + 4);
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = x + (size_t)r * K;
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }
#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// ---- verbatim llama.cpp port qpn_nvfp4_qpn2 ----
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
static __device__ __forceinline__ unsigned qpn_f2pack(float a, float b) {
  const __half2 h = __float22half2_rn(make_float2(a, b));
  return *reinterpret_cast<const unsigned *>(&h);
}
static __device__ __forceinline__ void qpn_x_cvt8(const float * xp, uint4 & lo, uint4 & hi) {
  const float4 f0 = *reinterpret_cast<const float4 *>(xp);
  const float4 f1 = *reinterpret_cast<const float4 *>(xp + 4);
  const float4 f2 = *reinterpret_cast<const float4 *>(xp + 8);
  const float4 f3 = *reinterpret_cast<const float4 *>(xp + 12);
  lo = make_uint4(qpn_f2pack(f0.x, f0.y), qpn_f2pack(f0.z, f0.w),
                  qpn_f2pack(f1.x, f1.y), qpn_f2pack(f1.z, f1.w));
  hi = make_uint4(qpn_f2pack(f2.x, f2.y), qpn_f2pack(f2.z, f2.w),
                  qpn_f2pack(f3.x, f3.y), qpn_f2pack(f3.z, f3.w));
}
template <int SPLITK, int NACC>
__global__ void our_qpn2(const uint8_t *__restrict__ codes,
                         const uint8_t *__restrict__ scales,
                         const float *__restrict__ x, float *__restrict__ dst,
                         const int N, const int K, const int M) {
  __shared__ float cs[SPLITK][256];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp   = (lane >> 2) & 3;
  const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G  = K >> 4;
  const int Gq = G / SPLITK;
  const int g0 = warp*Gq;
  const uint2   * cb = reinterpret_cast<const uint2 *>(codes) + (size_t) tile*G*32 + lane;
  const uint8_t * sb = scales + (size_t) tile*G*32*sizeof(__half) + lane*sizeof(__half);
  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) c[a][i] = 0.0f;
#pragma unroll 4
  for (int g = g0; g < g0 + Gq; ++g) {
    const uint2 q2 = __ldcs(cb + (size_t) g*32);
    const half2 sc2 = __half2half2(__ldg(reinterpret_cast<const __half *>(sb) + (size_t) g*32));
    half2 b[8];
    qpn_decode8(q2.x, sc2, b + 0);
    qpn_decode8(q2.y, sc2, b + 4);
    const unsigned * B = reinterpret_cast<const unsigned *>(b);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      qpn_x_cvt8(x + (size_t) r*K + g*16, a01, a23);
    }
    const unsigned * A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned * A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0],        A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) c[0][i] += c[a][i];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row*32 + qp*8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < SPLITK; ++w) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) dst[(size_t) row*N + (size_t) tile*32 + col] = v;
  }
}

static __device__ __forceinline__ half2 qpn_scale_h2(uint8_t b) {
  const unsigned short hb =
      (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));
  return __halves2half2(hs, hs);
}

template <int SPLITK, int NACC>
__global__ void our_qpn2_h(const uint8_t *__restrict__ codes,
                           const uint8_t *__restrict__ scales,
                           const half *__restrict__ x, float *__restrict__ dst,
                           const int N, const int K, const int M) {
  __shared__ float cs[SPLITK][256];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp   = (lane >> 2) & 3;
  const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G  = K >> 4;
  const int Gq = G / SPLITK;
  const int g0 = warp*Gq;
  const uint2   * cb = reinterpret_cast<const uint2 *>(codes) + (size_t) tile*G*32 + lane;
  const uint8_t * sb = scales + (size_t) tile*G*32 + lane;
  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) c[a][i] = 0.0f;
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
      a01 = *reinterpret_cast<const uint4 *>(xrow + g*16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g*16 + 8);
    }
    const unsigned * A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned * A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0],        A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) c[0][i] += c[a][i];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row*32 + qp*8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < SPLITK; ++w) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) dst[(size_t) row*N + (size_t) tile*32 + col] = v;
  }
}

struct shapespec { int n, k, m, splitk, nacc; };

template <typename F>
double time_kernel(F launch, int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  launch();  // warmup / prepack-independent
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.0 / iters;   // us
}

int main() {
  const int iters = 300;
  const std::vector<shapespec> shapes = {
      {8704, 5120, 1, 8, 2}, {13824, 5120, 1, 8, 2}, {62080, 5120, 1, 8, 1},
      {4096, 5120, 1, 8, 2}, {2048, 5120, 1, 8, 2},  {8704, 5120, 4, 8, 2},
      {8704, 5120, 8, 8, 2},
  };
  printf("%8s %3s %11s %11s %11s %9s %9s %9s\n", "N", "M", "ref_us", "ourF_us", "ourH_us", "ref_GBps", "ourF", "ourH");
  for (const shapespec & s : shapes) {
    const int tiles = s.n / 32, G = s.k / 16;
    std::vector<uint8_t> codes((size_t) tiles*G*32*8, 0x24);
    std::vector<uint8_t> scales_u8((size_t) tiles*G*32, 0x30);
    std::vector<__half> scales_h((size_t) tiles*G*32, __float2half(0.125f*16384.f));
    std::vector<__half> xh((size_t) s.m*s.k, __float2half(0.125f));
    std::vector<float> xf((size_t) s.m*s.k, 0.125f);
    std::vector<half> yr((size_t) s.m*s.n);
    std::vector<float> yo((size_t) s.m*s.n);

    uint8_t * dc, * ds_u8; __half * ds_h, * dxh; float * dxf, * dyo; half * dyr;
    cudaMalloc(&dc, codes.size());            cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&ds_u8, scales_u8.size());     cudaMemcpy(ds_u8, scales_u8.data(), scales_u8.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&ds_h, scales_h.size()*2);     cudaMemcpy(ds_h, scales_h.data(), scales_h.size()*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dxh, xh.size()*2);            cudaMemcpy(dxh, xh.data(), xh.size()*2, cudaMemcpyHostToDevice);
    cudaMalloc(&dxf, xf.size()*4);            cudaMemcpy(dxf, xf.data(), xf.size()*4, cudaMemcpyHostToDevice);
    cudaMalloc(&dyr, yr.size()*2);
    cudaMalloc(&dyo, yo.size()*4);

    const dim3 grid(tiles), block(s.splitk*32);
    double us_ref = time_kernel([&] {
        if (s.splitk == 8 && s.nacc == 2)
            ref_qpn2<8,2><<<grid, block>>>(dc, ds_u8, dxh, dyr, s.n, s.k, s.m, 1.0f);
        else
            ref_qpn2<8,1><<<grid, block>>>(dc, ds_u8, dxh, dyr, s.n, s.k, s.m, 1.0f);
    }, iters);
    double us_our = time_kernel([&] {
        if (s.splitk == 8 && s.nacc == 2)
            our_qpn2<8,2><<<grid, block>>>((const uint8_t *)dc, (const uint8_t *)ds_h, (const float *)dxf, dyo, s.n, s.k, s.m);
        else
            our_qpn2<8,1><<<grid, block>>>((const uint8_t *)dc, (const uint8_t *)ds_h, (const float *)dxf, dyo, s.n, s.k, s.m);
    }, iters);
    double us_h = time_kernel([&] {
        if (s.splitk == 8 && s.nacc == 2)
            our_qpn2_h<8,2><<<grid, block>>>((const uint8_t *)dc, (const uint8_t *)ds_h, dxh, dyo, s.n, s.k, s.m);
        else
            our_qpn2_h<8,1><<<grid, block>>>((const uint8_t *)dc, (const uint8_t *)ds_h, dxh, dyo, s.n, s.k, s.m);
    }, iters);

    const double mb = (double) s.k * s.n * 0.5625 / 1e6;
    printf("%8d %3d %11.2f %11.2f %11.2f %9.0f %9.0f %9.0f\n",
           s.n, s.m, us_ref, us_our, us_h,
           mb/1e3/(us_ref*1e-6), mb/1e3/(us_our*1e-6), mb/1e3/(us_h*1e-6));

    cudaFree(dc); cudaFree(ds_u8); cudaFree(ds_h); cudaFree(dxh);
    cudaFree(dxf); cudaFree(dyr); cudaFree(dyo);
  }
  return 0;
}
