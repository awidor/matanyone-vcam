// MatAnyone — CUDA kernel launchers (uses embedded cubin + CUDA driver API)
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

#include "matanyone_cubin.h"

// Module handle and kernel function handles
static CUmodule  g_cubin_module  = nullptr;
static CUfunction g_kernel_bgra_to_rgb   = nullptr;
static CUfunction g_kernel_alpha_composite = nullptr;
static CUfunction g_kernel_alpha_output   = nullptr;
static CUfunction g_kernel_copy_bgra      = nullptr;

// ============================================================
// Initialize cubin module — call once during filter creation
// ============================================================
bool init_cuda_kernels() {
    if (g_cubin_module) return true;  // already loaded

    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "[MatAnyone] cudaSetDevice failed\n");
        return false;
    }
    cudaFree(0);
    cuInit(0);

    CUresult err = cuModuleLoadData(&g_cubin_module, g_cubin_data);
    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "[MatAnyone] cuModuleLoadData failed: %s\n", msg ? msg : "unknown");
        return false;
    }

    auto get_func = [](CUfunction& func, const char* name) -> bool {
        CUresult e = cuModuleGetFunction(&func, g_cubin_module, name);
        if (e != CUDA_SUCCESS) {
            fprintf(stderr, "[MatAnyone] cuModuleGetFunction(%s) failed\n", name);
            return false;
        }
        return true;
    };

    bool ok = true;
    ok = ok && get_func(g_kernel_bgra_to_rgb,   "_Z23bgra_to_rgb_nchw_kernelPKhiiiPfii");
    ok = ok && get_func(g_kernel_alpha_composite, "_Z22alpha_composite_kernelPKfS0_fffPhii");
    ok = ok && get_func(g_kernel_alpha_output,   "_Z19alpha_output_kernelPKfPhii");
    ok = ok && get_func(g_kernel_copy_bgra,      "_Z16copy_bgra_kernelPKhiPhiii");
    return ok;
}

// ============================================================
// Kernel launchers
// ============================================================
void launch_bgra_to_rgb_nchw(const uint8_t* bgra, int src_w, int src_h, int src_pitch,
                             float* rgb_nchw, int dst_w, int dst_h,
                             cudaStream_t stream)
{
    if (!g_kernel_bgra_to_rgb) return;
    void* args[] = { &bgra, &src_w, &src_h, &src_pitch, &rgb_nchw, &dst_w, &dst_h };
    int grid_x = (dst_w + 31) / 32;
    int grid_y = (dst_h + 15) / 16;
    cuLaunchKernel(g_kernel_bgra_to_rgb, grid_x, grid_y, 1, 32, 16, 1, 0, stream, args, nullptr);
}

void launch_alpha_composite(const float* src_rgb, const float* alpha,
                            float bg_r, float bg_g, float bg_b,
                            uint8_t* out_bgra, int h, int w,
                            cudaStream_t stream)
{
    if (!g_kernel_alpha_composite) return;
    void* args[] = { &src_rgb, &alpha, &bg_r, &bg_g, &bg_b, &out_bgra, &h, &w };
    int grid_x = (w + 31) / 32;
    int grid_y = (h + 15) / 16;
    cuLaunchKernel(g_kernel_alpha_composite, grid_x, grid_y, 1, 32, 16, 1, 0, stream, args, nullptr);
}

void launch_alpha_output(const float* alpha, uint8_t* out_alpha, int h, int w, cudaStream_t stream)
{
    if (!g_kernel_alpha_output) return;
    void* args[] = { &alpha, &out_alpha, &h, &w };
    int grid_x = (w + 31) / 32;
    int grid_y = (h + 15) / 16;
    cuLaunchKernel(g_kernel_alpha_output, grid_x, grid_y, 1, 32, 16, 1, 0, stream, args, nullptr);
}

void launch_copy_bgra(const uint8_t* src, int src_pitch, uint8_t* dst, int dst_pitch,
                      int h, int w, cudaStream_t stream)
{
    if (!g_kernel_copy_bgra) return;
    void* args[] = { &src, &src_pitch, &dst, &dst_pitch, &h, &w };
    int grid_x = (w + 31) / 32;
    int grid_y = (h + 15) / 16;
    cuLaunchKernel(g_kernel_copy_bgra, grid_x, grid_y, 1, 32, 16, 1, 0, stream, args, nullptr);
}
