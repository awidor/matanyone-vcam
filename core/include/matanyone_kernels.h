#pragma once

#include <cuda_runtime.h>
#include <cstdint>

bool init_cuda_kernels();

void launch_bgra_to_rgb_nchw(const uint8_t* bgra, int src_w, int src_h, int src_pitch,
                             float* rgb_nchw, int dst_w, int dst_h, cudaStream_t stream);

void launch_alpha_composite(const float* src_rgb, const float* alpha,
                            float bg_r, float bg_g, float bg_b,
                            uint8_t* out_bgra, int h, int w, cudaStream_t stream);

void launch_alpha_output(const float* alpha, uint8_t* out_alpha, int h, int w, cudaStream_t stream);

void launch_copy_bgra(const uint8_t* src, int src_pitch, uint8_t* dst, int dst_pitch,
                      int h, int w, cudaStream_t stream);
