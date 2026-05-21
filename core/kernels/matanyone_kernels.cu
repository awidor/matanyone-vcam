// MatAnyone — CUDA kernels for standalone core pipeline
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cuda_runtime.h>
#include <cstdint>

__device__ inline float sample_bilinear(const uint8_t* base, int pitch, int w, int h, float sx, float sy, int channel) {
    sx = fmaxf(0.0f, fminf(sx, w - 1.001f));
    sy = fmaxf(0.0f, fminf(sy, h - 1.001f));
    const int sx0 = static_cast<int>(sx);
    const int sy0 = static_cast<int>(sy);
    const int sx1 = min(sx0 + 1, w - 1);
    const int sy1 = min(sy0 + 1, h - 1);
    const float fx = sx - sx0;
    const float fy = sy - sy0;

    auto load = [&](int px, int py) -> float {
        const uint8_t* p = base + static_cast<size_t>(py) * pitch + static_cast<size_t>(px) * 3;
        return static_cast<float>(p[channel]) / 255.0f;
    };

    return (1.0f - fx) * (1.0f - fy) * load(sx0, sy0) +
           fx * (1.0f - fy) * load(sx1, sy0) +
           (1.0f - fx) * fy * load(sx0, sy1) +
           fx * fy * load(sx1, sy1);
}

__global__ void bgr_to_rgb_nchw_kernel(
    const uint8_t* __restrict__ bgr, int src_w, int src_h, int src_pitch,
    float* __restrict__ rgb_nchw, int dst_w, int dst_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const float sx = (static_cast<float>(x) + 0.5f) * src_w / dst_w - 0.5f;
    const float sy = (static_cast<float>(y) + 0.5f) * src_h / dst_h - 0.5f;

    const size_t idx = static_cast<size_t>(y) * dst_w + x;
    rgb_nchw[0 * dst_h * dst_w + idx] = sample_bilinear(bgr, src_pitch, src_w, src_h, sx, sy, 2);
    rgb_nchw[1 * dst_h * dst_w + idx] = sample_bilinear(bgr, src_pitch, src_w, src_h, sx, sy, 1);
    rgb_nchw[2 * dst_h * dst_w + idx] = sample_bilinear(bgr, src_pitch, src_w, src_h, sx, sy, 0);
}

__global__ void bgra_to_rgb_nchw_kernel(
    const uint8_t* __restrict__ bgra, int src_w, int src_h, int src_pitch,
    float* __restrict__ rgb_nchw, int dst_w, int dst_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float sx = (static_cast<float>(x) + 0.5f) * src_w / dst_w - 0.5f;
    float sy = (static_cast<float>(y) + 0.5f) * src_h / dst_h - 0.5f;
    sx = fmaxf(0.0f, fminf(sx, src_w - 1.001f));
    sy = fmaxf(0.0f, fminf(sy, src_h - 1.001f));

    const int sx0 = static_cast<int>(sx);
    const int sy0 = static_cast<int>(sy);
    const int sx1 = min(sx0 + 1, src_w - 1);
    const int sy1 = min(sy0 + 1, src_h - 1);
    const float fx = sx - sx0;
    const float fy = sy - sy0;

    auto load = [&](int px, int py) -> uint4 {
        const uint8_t* p = bgra + static_cast<size_t>(py) * src_pitch + static_cast<size_t>(px) * 4;
        return make_uint4(p[0], p[1], p[2], p[3]);
    };

    const uint4 c00 = load(sx0, sy0);
    const uint4 c10 = load(sx1, sy0);
    const uint4 c01 = load(sx0, sy1);
    const uint4 c11 = load(sx1, sy1);

    const float b = (1.0f - fx) * (1.0f - fy) * c00.x + fx * (1.0f - fy) * c10.x +
                    (1.0f - fx) * fy * c01.x + fx * fy * c11.x;
    const float g = (1.0f - fx) * (1.0f - fy) * c00.y + fx * (1.0f - fy) * c10.y +
                    (1.0f - fx) * fy * c01.y + fx * fy * c11.y;
    const float r = (1.0f - fx) * (1.0f - fy) * c00.z + fx * (1.0f - fy) * c10.z +
                    (1.0f - fx) * fy * c01.z + fx * fy * c11.z;

    const size_t idx = static_cast<size_t>(y) * dst_w + x;
    rgb_nchw[0 * dst_h * dst_w + idx] = r / 255.0f;
    rgb_nchw[1 * dst_h * dst_w + idx] = g / 255.0f;
    rgb_nchw[2 * dst_h * dst_w + idx] = b / 255.0f;
}

__global__ void alpha_composite_kernel(
    const float* __restrict__ src_rgb,
    const float* __restrict__ alpha,
    float bg_r, float bg_g, float bg_b,
    uint8_t* __restrict__ out_bgra,
    int h, int w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const size_t idx = static_cast<size_t>(y) * w + x;
    float a = alpha[idx];
    a = fminf(1.0f, fmaxf(0.0f, a));

    const float r = src_rgb[0 * h * w + idx];
    const float g = src_rgb[1 * h * w + idx];
    const float b = src_rgb[2 * h * w + idx];

    const float or_ = a * r + (1.0f - a) * bg_r;
    const float og_ = a * g + (1.0f - a) * bg_g;
    const float ob_ = a * b + (1.0f - a) * bg_b;

    out_bgra[idx * 4 + 0] = static_cast<uint8_t>(fminf(255.0f, ob_ * 255.0f));
    out_bgra[idx * 4 + 1] = static_cast<uint8_t>(fminf(255.0f, og_ * 255.0f));
    out_bgra[idx * 4 + 2] = static_cast<uint8_t>(fminf(255.0f, or_ * 255.0f));
    out_bgra[idx * 4 + 3] = 255;
}

__global__ void bgra_to_bgr_kernel(
    const uint8_t* __restrict__ bgra, int src_pitch,
    uint8_t* __restrict__ bgr, int dst_pitch,
    int h, int w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const size_t src_off = static_cast<size_t>(y) * src_pitch + static_cast<size_t>(x) * 4;
    const size_t dst_off = static_cast<size_t>(y) * dst_pitch + static_cast<size_t>(x) * 3;
    bgr[dst_off + 0] = bgra[src_off + 0];
    bgr[dst_off + 1] = bgra[src_off + 1];
    bgr[dst_off + 2] = bgra[src_off + 2];
}

void launch_bgr_to_rgb_nchw(const uint8_t* bgr, int src_w, int src_h, int src_pitch,
                            float* rgb_nchw, int dst_w, int dst_h, cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((dst_w + 31) / 32, (dst_h + 15) / 16);
    bgr_to_rgb_nchw_kernel<<<grid, block, 0, stream>>>(bgr, src_w, src_h, src_pitch, rgb_nchw, dst_w, dst_h);
}

void launch_bgra_to_rgb_nchw(const uint8_t* bgra, int src_w, int src_h, int src_pitch,
                             float* rgb_nchw, int dst_w, int dst_h, cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((dst_w + 31) / 32, (dst_h + 15) / 16);
    bgra_to_rgb_nchw_kernel<<<grid, block, 0, stream>>>(bgra, src_w, src_h, src_pitch, rgb_nchw, dst_w, dst_h);
}

void launch_alpha_composite(const float* src_rgb, const float* alpha,
                            float bg_r, float bg_g, float bg_b,
                            uint8_t* out_bgra, int h, int w, cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + 31) / 32, (h + 15) / 16);
    alpha_composite_kernel<<<grid, block, 0, stream>>>(src_rgb, alpha, bg_r, bg_g, bg_b, out_bgra, h, w);
}

void launch_bgra_to_bgr(const uint8_t* bgra, int src_pitch,
                        uint8_t* bgr, int dst_pitch,
                        int h, int w, cudaStream_t stream)
{
    dim3 block(32, 16);
    dim3 grid((w + 31) / 32, (h + 15) / 16);
    bgra_to_bgr_kernel<<<grid, block, 0, stream>>>(bgra, src_pitch, bgr, dst_pitch, h, w);
}
