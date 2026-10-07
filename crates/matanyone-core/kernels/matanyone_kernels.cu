// MatAnyone pre/post-processing kernels.
// Regenerate matanyone_cubin.h with scripts/build_kernels.ps1 after editing.
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>

// Packed BGR8 (any size) -> planar RGB float in [0, 1] at the model size.
// Bilinear with half-pixel centres (cv2.INTER_LINEAR); exact copy when sizes match.
extern "C" __global__ void bgr_to_rgb_nchw(
    const uint8_t* __restrict__ bgr, int src_w, int src_h, int src_pitch,
    float* __restrict__ rgb, int dst_w, int dst_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const size_t plane = static_cast<size_t>(dst_w) * dst_h;
    const size_t idx = static_cast<size_t>(y) * dst_w + x;
    constexpr float kInv255 = 1.0f / 255.0f;

    if (src_w == dst_w && src_h == dst_h) {
        const uint8_t* p = bgr + static_cast<size_t>(y) * src_pitch + static_cast<size_t>(x) * 3;
        rgb[idx] = p[2] * kInv255;
        rgb[plane + idx] = p[1] * kInv255;
        rgb[2 * plane + idx] = p[0] * kInv255;
        return;
    }

    const float sx = fminf(fmaxf((x + 0.5f) * src_w / dst_w - 0.5f, 0.0f), src_w - 1.0f);
    const float sy = fminf(fmaxf((y + 0.5f) * src_h / dst_h - 0.5f, 0.0f), src_h - 1.0f);
    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);
    const float fx = sx - x0;
    const float fy = sy - y0;
    const uint8_t* r0 = bgr + static_cast<size_t>(y0) * src_pitch;
    const uint8_t* r1 = bgr + static_cast<size_t>(y1) * src_pitch;

    for (int c = 0; c < 3; ++c) {
        const float top = r0[x0 * 3 + c] + fx * (r0[x1 * 3 + c] - r0[x0 * 3 + c]);
        const float bottom = r1[x0 * 3 + c] + fx * (r1[x1 * 3 + c] - r1[x0 * 3 + c]);
        rgb[(2 - c) * plane + idx] = (top + fy * (bottom - top)) * kInv255;
    }
}

// Planar RGB float + alpha -> packed BGR8 composited over a solid colour.
extern "C" __global__ void composite_bgr(
    const float* __restrict__ rgb, const float* __restrict__ alpha,
    float bg_r, float bg_g, float bg_b,
    uint8_t* __restrict__ bgr, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const size_t plane = static_cast<size_t>(w) * h;
    const size_t idx = static_cast<size_t>(y) * w + x;
    const float a = fminf(1.0f, fmaxf(0.0f, alpha[idx]));
    const float r = bg_r + a * (rgb[idx] - bg_r);
    const float g = bg_g + a * (rgb[plane + idx] - bg_g);
    const float b = bg_b + a * (rgb[2 * plane + idx] - bg_b);

    uint8_t* out = bgr + idx * 3;
    out[0] = static_cast<uint8_t>(fminf(255.0f, b * 255.0f + 0.5f));
    out[1] = static_cast<uint8_t>(fminf(255.0f, g * 255.0f + 0.5f));
    out[2] = static_cast<uint8_t>(fminf(255.0f, r * 255.0f + 0.5f));
}

// dst += src; the object memory is a streaming sum (MemoryManager.add_memory).
extern "C" __global__ void accumulate_f32(float* __restrict__ dst, const float* __restrict__ src, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
