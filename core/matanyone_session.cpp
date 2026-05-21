#include "matanyone_core_api.h"

#include "matanyone_kernels.h"
#include "matanyone_runner.h"
#include "trt_common.h"

#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int MODEL_W = 1280;
constexpr int MODEL_H = 720;

thread_local std::string g_last_error;

void set_error(const std::string& message) {
    g_last_error = message;
}

std::vector<float> resize_mask_nearest(const float* src, int src_w, int src_h, int dst_w, int dst_h) {
    std::vector<float> dst(static_cast<size_t>(dst_w) * dst_h);
    for (int y = 0; y < dst_h; ++y) {
        const int sy = std::min(src_h - 1, y * src_h / dst_h);
        for (int x = 0; x < dst_w; ++x) {
            const int sx = std::min(src_w - 1, x * src_w / dst_w);
            dst[static_cast<size_t>(y) * dst_w + x] = src[static_cast<size_t>(sy) * src_w + sx];
        }
    }
    return dst;
}

std::vector<float> load_mask_png(const char* path, int dst_w, int dst_h) {
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 0);
    if (!data) {
        throw std::runtime_error(std::string("failed to load mask image: ") + path);
    }

    std::vector<float> alpha(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            const unsigned char* px = data + idx * static_cast<size_t>(channels);
            float a = 0.0f;
            if (channels >= 4) {
                a = px[3] > 0 ? static_cast<float>(px[3]) / 255.0f
                              : static_cast<float>(std::max({px[0], px[1], px[2]})) / 255.0f;
            } else if (channels == 1) {
                a = static_cast<float>(px[0]) / 255.0f;
            } else {
                a = static_cast<float>(std::max({px[0], px[1], px[2]})) / 255.0f;
            }
            alpha[idx] = a;
        }
    }
    stbi_image_free(data);

    if (w == dst_w && h == dst_h) {
        return alpha;
    }
    return resize_mask_nearest(alpha.data(), w, h, dst_w, dst_h);
}

struct MatAnyoneSessionImpl {
    std::unique_ptr<MatAnyoneRunner> runner;
    float* d_rgb_nchw = nullptr;
    float* d_alpha = nullptr;
    uint8_t* d_bgra_in = nullptr;
    int d_bgra_in_capacity = 0;
    uint8_t* d_bgra_out = nullptr;
    bool initialized = false;
    uint64_t frame_index = 0;
};

void bgra_from_bgr_host(const uint8_t* bgr, int w, int h, std::vector<uint8_t>& bgra) {
    bgra.resize(static_cast<size_t>(w) * h * 4);
    for (int i = 0; i < w * h; ++i) {
        bgra[i * 4 + 0] = bgr[i * 3 + 0];
        bgra[i * 4 + 1] = bgr[i * 3 + 1];
        bgra[i * 4 + 2] = bgr[i * 3 + 2];
        bgra[i * 4 + 3] = 255;
    }
}

void bgr_from_bgra_host(const uint8_t* bgra, int w, int h, uint8_t* bgr) {
    for (int i = 0; i < w * h; ++i) {
        bgr[i * 3 + 0] = bgra[i * 4 + 0];
        bgr[i * 3 + 1] = bgra[i * 4 + 1];
        bgr[i * 3 + 2] = bgra[i * 4 + 2];
    }
}

void upload_bgr_to_rgb(MatAnyoneSessionImpl& session, const uint8_t* bgr, int src_w, int src_h, cudaStream_t stream) {
    std::vector<uint8_t> bgra_host;
    bgra_from_bgr_host(bgr, src_w, src_h, bgra_host);
    const size_t bytes = bgra_host.size();
    if (session.d_bgra_in_capacity < static_cast<int>(bytes)) {
        if (session.d_bgra_in) {
            cudaFree(session.d_bgra_in);
        }
        checkCuda(cudaMalloc(&session.d_bgra_in, bytes), "cudaMalloc d_bgra_in");
        session.d_bgra_in_capacity = static_cast<int>(bytes);
    }
    checkCuda(cudaMemcpyAsync(session.d_bgra_in, bgra_host.data(), bytes, cudaMemcpyHostToDevice, stream),
              "upload bgra");
    launch_bgra_to_rgb_nchw(session.d_bgra_in, src_w, src_h, src_w * 4, session.d_rgb_nchw, MODEL_W, MODEL_H, stream);
}

void readback_bgr(MatAnyoneSessionImpl& session, uint8_t* bgr_out, int dst_w, int dst_h, cudaStream_t stream) {
    std::vector<uint8_t> host_bgra(static_cast<size_t>(MODEL_W) * MODEL_H * 4);
    checkCuda(cudaMemcpyAsync(host_bgra.data(), session.d_bgra_out, host_bgra.size(), cudaMemcpyDeviceToHost, stream),
              "readback bgra");
    checkCuda(cudaStreamSynchronize(stream), "readback sync");

    if (dst_w == MODEL_W && dst_h == MODEL_H) {
        bgr_from_bgra_host(host_bgra.data(), MODEL_W, MODEL_H, bgr_out);
        return;
    }

    std::vector<uint8_t> model_bgr(static_cast<size_t>(MODEL_W) * MODEL_H * 3);
    bgr_from_bgra_host(host_bgra.data(), MODEL_W, MODEL_H, model_bgr.data());
    for (int y = 0; y < dst_h; ++y) {
        const int sy = std::min(MODEL_H - 1, y * MODEL_H / dst_h);
        for (int x = 0; x < dst_w; ++x) {
            const int sx = std::min(MODEL_W - 1, x * MODEL_W / dst_w);
            const size_t src_idx = (static_cast<size_t>(sy) * MODEL_W + sx) * 3;
            const size_t dst_idx = (static_cast<size_t>(y) * dst_w + x) * 3;
            bgr_out[dst_idx + 0] = model_bgr[src_idx + 0];
            bgr_out[dst_idx + 1] = model_bgr[src_idx + 1];
            bgr_out[dst_idx + 2] = model_bgr[src_idx + 2];
        }
    }
}

int initialize_session(MatAnyoneSessionImpl& session, const uint8_t* bgr, int src_w, int src_h,
                       const float* mask_hw, int mask_w, int mask_h) {
    try {
        cudaStream_t stream = session.runner->stream();
        upload_bgr_to_rgb(session, bgr, src_w, src_h, stream);

        std::vector<float> mask_model;
        if (mask_w == MODEL_W && mask_h == MODEL_H) {
            mask_model.assign(mask_hw, mask_hw + static_cast<size_t>(MODEL_W) * MODEL_H);
        } else {
            mask_model = resize_mask_nearest(mask_hw, mask_w, mask_h, MODEL_W, MODEL_H);
        }

        float* d_init_alpha = nullptr;
        checkCuda(cudaMalloc(&d_init_alpha, static_cast<size_t>(MODEL_W) * MODEL_H * sizeof(float)), "cudaMalloc init alpha");
        checkCuda(cudaMemcpyAsync(d_init_alpha, mask_model.data(), static_cast<size_t>(MODEL_W) * MODEL_H * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "upload init alpha");
        session.runner->InitializeDevice(session.d_rgb_nchw, d_init_alpha);
        session.runner->CopyAlphaToDevice(session.d_alpha);
        cudaFree(d_init_alpha);

        session.initialized = true;
        session.frame_index = 1;
        checkCuda(cudaStreamSynchronize(stream), "init sync");
        return 0;
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

} // namespace

extern "C" {

static MatAnyoneSessionImpl* as_impl(MatAnyoneSession* session) {
    return reinterpret_cast<MatAnyoneSessionImpl*>(session);
}

MatAnyoneSession* matanyone_create(const char* engine_dir) {
    try {
        g_last_error.clear();
        if (!init_cuda_kernels()) {
            set_error("failed to initialize CUDA kernels");
            return nullptr;
        }

        auto* session = new MatAnyoneSessionImpl();
        session->runner = std::make_unique<MatAnyoneRunner>(engine_dir ? engine_dir : "engines/faithful");

        const size_t rgb_bytes = 3ULL * MODEL_H * MODEL_W * sizeof(float);
        const size_t alpha_bytes = 1ULL * MODEL_H * MODEL_W * sizeof(float);
        const size_t bgra_bytes = static_cast<size_t>(MODEL_W) * MODEL_H * 4;
        checkCuda(cudaMalloc(&session->d_rgb_nchw, rgb_bytes), "cudaMalloc d_rgb_nchw");
        checkCuda(cudaMalloc(&session->d_alpha, alpha_bytes), "cudaMalloc d_alpha");
        checkCuda(cudaMalloc(&session->d_bgra_out, bgra_bytes), "cudaMalloc d_bgra_out");
        return reinterpret_cast<MatAnyoneSession*>(session);
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return nullptr;
    }
}

void matanyone_destroy(MatAnyoneSession* session) {
    auto* impl = as_impl(session);
    if (!impl) {
        return;
    }
    if (impl->d_rgb_nchw) cudaFree(impl->d_rgb_nchw);
    if (impl->d_alpha) cudaFree(impl->d_alpha);
    if (impl->d_bgra_in) cudaFree(impl->d_bgra_in);
    if (impl->d_bgra_out) cudaFree(impl->d_bgra_out);
    delete impl;
}

const char* matanyone_last_error(void) {
    return g_last_error.c_str();
}

int matanyone_internal_width(void) {
    return MODEL_W;
}

int matanyone_internal_height(void) {
    return MODEL_H;
}

int matanyone_reset(MatAnyoneSession* session) {
    auto* impl = as_impl(session);
    if (!impl) {
        set_error("null session");
        return -1;
    }
    impl->initialized = false;
    impl->frame_index = 0;
    return 0;
}

int matanyone_init_from_mask_file(MatAnyoneSession* session, const uint8_t* bgr, int src_w, int src_h, const char* mask_png_path) {
    auto* impl = as_impl(session);
    if (!impl || !impl->runner) {
        set_error("invalid session");
        return -1;
    }
    if (!bgr || src_w <= 0 || src_h <= 0 || !mask_png_path || !mask_png_path[0]) {
        set_error("invalid init arguments");
        return -1;
    }
    try {
        const auto mask = load_mask_png(mask_png_path, MODEL_W, MODEL_H);
        return initialize_session(*impl, bgr, src_w, src_h, mask.data(), MODEL_W, MODEL_H);
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

int matanyone_init_from_mask(MatAnyoneSession* session, const uint8_t* bgr, int src_w, int src_h,
                             const float* mask_hw, int mask_w, int mask_h) {
    auto* impl = as_impl(session);
    if (!impl || !impl->runner) {
        set_error("invalid session");
        return -1;
    }
    if (!bgr || src_w <= 0 || src_h <= 0 || !mask_hw || mask_w <= 0 || mask_h <= 0) {
        set_error("invalid init arguments");
        return -1;
    }
    return initialize_session(*impl, bgr, src_w, src_h, mask_hw, mask_w, mask_h);
}

int matanyone_process_bgr(MatAnyoneSession* session, const uint8_t* bgr_in, int src_w, int src_h,
                          uint8_t* bgr_out, int dst_w, int dst_h,
                          float bg_r, float bg_g, float bg_b, int memory_update) {
    auto* impl = as_impl(session);
    if (!impl || !impl->runner) {
        set_error("invalid session");
        return -1;
    }
    if (!impl->initialized) {
        set_error("session not initialized");
        return -1;
    }
    if (!bgr_in || !bgr_out || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        set_error("invalid process arguments");
        return -1;
    }

    try {
        cudaStream_t stream = impl->runner->stream();
        upload_bgr_to_rgb(*impl, bgr_in, src_w, src_h, stream);

        impl->runner->ProcessFrameRgbDevice(impl->d_rgb_nchw, memory_update != 0);
        impl->runner->CopyAlphaToDevice(impl->d_alpha);
        ++impl->frame_index;

        launch_alpha_composite(impl->d_rgb_nchw, impl->d_alpha, bg_r, bg_g, bg_b,
                               impl->d_bgra_out, MODEL_H, MODEL_W, stream);
        readback_bgr(*impl, bgr_out, dst_w, dst_h, stream);
        return 0;
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return -1;
    }
}

} // extern "C"
