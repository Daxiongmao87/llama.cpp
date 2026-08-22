// Synthetic per-shape NVFP4 GEMV bandwidth bench for the Volta QPN path.
// Times ggml_mul_mat(NVFP4 [K,N], F32 [K,M]) with CUDA events so 27B shapes can
// be measured without loading a model. QPN on/off via GGML_CUDA_VOLTA_QPN env.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct shape { int n, k, m; };

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 300;
    const std::vector<shape> shapes = {
        // qwen3.8-27B geometries (marlin.py _QPN2_TABLE)
        {8704, 5120, 1}, {4096, 5120, 1}, {2048, 5120, 1},
        {3584, 5120, 1}, {5120, 4352, 1}, {62080, 5120, 1},
        {13824, 5120, 1}, {5120, 1536, 1}, {2560, 9216, 1},
        // M variants on the down-proj-like shape (MTP verify band)
        {8704, 5120, 2}, {8704, 5120, 4}, {8704, 5120, 8},
        {8704, 5120, 10}, {8704, 5120, 16},
    };

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "CUDA", 4) == 0) {
            dev = d;
            break;
        }
    }
    if (!dev) { fprintf(stderr, "no CUDA backend\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    printf("%8s %6s %2s %11s %9s\n", "N", "K", "M", "us/call", "GB/s");
    for (const shape & s : shapes) {
        ggml_init_params ip = { 512u*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, s.k, s.n);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.m);
        ggml_tensor * y = ggml_mul_mat(ctx, a, b);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

        // benign fill: e4m3 scale bytes ~0x30 region, code nibbles small values
        std::vector<uint8_t> zdata(ggml_nbytes(a), 0x33);
        ggml_backend_tensor_set(a, zdata.data(), 0, ggml_nbytes(a));
        std::vector<float> xdata(s.k*s.m, 0.125f);
        ggml_backend_tensor_set(b, xdata.data(), 0, ggml_nbytes(b));

        ggml_backend_graph_compute(backend, gf);   // warmup + prepack

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(backend, gf);
        }
        const auto t1 = std::chrono::steady_clock::now();
        ggml_backend_synchronize(backend);

        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        const double mb = (double) s.k * s.n * 0.5625 / 1e6;
        printf("%8d %6d %2d %11.2f %9.0f\n", s.n, s.k, s.m, us, mb/1e3/(us*1e-6));

        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
