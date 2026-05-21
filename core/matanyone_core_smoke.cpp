#include "matanyone_core_api.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<uint8_t> read_bgr_file(const fs::path& path, int& w, int& h) {
    int channels = 0;
    unsigned char* data = stbi_load(path.string().c_str(), &w, &h, &channels, 3);
    if (!data) {
        throw std::runtime_error("failed to load image: " + path.string());
    }
    std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3);
    for (int i = 0; i < w * h; ++i) {
        bgr[i * 3 + 0] = data[i * 3 + 2];
        bgr[i * 3 + 1] = data[i * 3 + 1];
        bgr[i * 3 + 2] = data[i * 3 + 0];
    }
    stbi_image_free(data);
    return bgr;
}

static std::vector<uint8_t> make_synthetic_bgr(int w, int h, int frame) {
    std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            bgr[idx + 0] = static_cast<uint8_t>((x + frame * 3) % 256);
            bgr[idx + 1] = static_cast<uint8_t>((y + frame * 2) % 256);
            bgr[idx + 2] = static_cast<uint8_t>((x + y + frame) % 256);
        }
    }
    return bgr;
}

static std::vector<float> make_center_mask(int w, int h) {
    std::vector<float> mask(static_cast<size_t>(w) * h, 0.0f);
    const float cx = w * 0.5f;
    const float cy = h * 0.5f;
    const float rx = w * 0.22f;
    const float ry = h * 0.35f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = (x - cx) / rx;
            const float dy = (y - cy) / ry;
            if (dx * dx + dy * dy <= 1.0f) {
                mask[static_cast<size_t>(y) * w + x] = 1.0f;
            }
        }
    }
    return mask;
}

static int run_synthetic(MatAnyoneSession* session, const fs::path& output_path, int max_frames) {
    const int out_w = matanyone_internal_width();
    const int out_h = matanyone_internal_height();
    auto first = make_synthetic_bgr(out_w, out_h, 0);
    auto mask = make_center_mask(out_w, out_h);
    if (matanyone_init_from_mask(session, first.data(), out_w, out_h, mask.data(), out_w, out_h) != 0) {
        std::cerr << "init failed: " << matanyone_last_error() << "\n";
        return 1;
    }

    std::vector<uint8_t> out_bgr(static_cast<size_t>(out_w) * out_h * 3);
    const float bg_r = 0.0f;
    const float bg_g = 180.0f / 255.0f;
    const float bg_b = 80.0f / 255.0f;

    for (int i = 1; i < max_frames; ++i) {
        auto bgr = make_synthetic_bgr(out_w, out_h, i);
        const int memory_update = (i % 5) == 0 ? 1 : 0;
        if (matanyone_process_bgr(session, bgr.data(), out_w, out_h, out_bgr.data(), out_w, out_h,
                                  bg_r, bg_g, bg_b, memory_update) != 0) {
            std::cerr << "process failed frame " << i << ": " << matanyone_last_error() << "\n";
            return 1;
        }
    }

    fs::create_directories(output_path.parent_path());
    std::vector<uint8_t> rgb(static_cast<size_t>(out_w) * out_h * 3);
    for (int i = 0; i < out_w * out_h; ++i) {
        rgb[i * 3 + 0] = out_bgr[i * 3 + 2];
        rgb[i * 3 + 1] = out_bgr[i * 3 + 1];
        rgb[i * 3 + 2] = out_bgr[i * 3 + 0];
    }
    if (!stbi_write_png(output_path.string().c_str(), out_w, out_h, 3, rgb.data(), out_w * 3)) {
        std::cerr << "failed to write " << output_path << "\n";
        return 1;
    }
    std::cout << "wrote=" << output_path.string() << " mode=synthetic frames=" << max_frames << "\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        std::string engine_dir = "engines/faithful";
        fs::path input_dir = "raw_sample/frames";
        fs::path mask_path = "raw_sample/mask.png";
        fs::path output_path = "output/core_smoke.png";
        int max_frames = 5;
        bool synthetic = false;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--synthetic") == 0) {
                synthetic = true;
            } else if (engine_dir == "engines/faithful" && argv[i][0] != '-') {
                engine_dir = argv[i];
            }
        }
        if (argc > 2 && !synthetic) input_dir = argv[2];
        if (argc > 3 && !synthetic) mask_path = argv[3];
        if (argc > 4 && !synthetic) output_path = argv[4];
        if (argc > 5 && !synthetic) max_frames = std::stoi(argv[5]);

        MatAnyoneSession* session = matanyone_create(engine_dir.c_str());
        if (!session) {
            std::cerr << "create failed: " << matanyone_last_error() << "\n";
            return 1;
        }

        int rc = 0;
        if (synthetic || !fs::exists(input_dir)) {
            rc = run_synthetic(session, output_path, max_frames);
        } else {
            const int out_w = matanyone_internal_width();
            const int out_h = matanyone_internal_height();
            int in_w = 0;
            int in_h = 0;
            fs::path first_frame = input_dir / "00000.png";
            if (!fs::exists(first_frame)) {
                first_frame = input_dir / "00000.jpg";
            }
            auto first_bgr = read_bgr_file(first_frame, in_w, in_h);
            if (matanyone_init_from_mask_file(session, first_bgr.data(), in_w, in_h, mask_path.string().c_str()) != 0) {
                std::cerr << "init failed: " << matanyone_last_error() << "\n";
                matanyone_destroy(session);
                return 1;
            }

            std::vector<uint8_t> out_bgr(static_cast<size_t>(out_w) * out_h * 3);
            const float bg_r = 0.0f;
            const float bg_g = 180.0f / 255.0f;
            const float bg_b = 80.0f / 255.0f;

            for (int i = 0; i < max_frames; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "%05d.png", i);
                fs::path frame_path = input_dir / name;
                if (!fs::exists(frame_path)) {
                    std::snprintf(name, sizeof(name), "%05d.jpg", i);
                    frame_path = input_dir / name;
                }
                if (!fs::exists(frame_path)) {
                    break;
                }
                auto bgr = read_bgr_file(frame_path, in_w, in_h);
                const int memory_update = (i > 0 && (i % 5) == 0) ? 1 : 0;
                if (matanyone_process_bgr(session, bgr.data(), in_w, in_h, out_bgr.data(), out_w, out_h,
                                          bg_r, bg_g, bg_b, memory_update) != 0) {
                    std::cerr << "process failed frame " << i << ": " << matanyone_last_error() << "\n";
                    matanyone_destroy(session);
                    return 1;
                }
            }

            fs::create_directories(output_path.parent_path());
            std::vector<uint8_t> rgb(static_cast<size_t>(out_w) * out_h * 3);
            for (int i = 0; i < out_w * out_h; ++i) {
                rgb[i * 3 + 0] = out_bgr[i * 3 + 2];
                rgb[i * 3 + 1] = out_bgr[i * 3 + 1];
                rgb[i * 3 + 2] = out_bgr[i * 3 + 0];
            }
            if (!stbi_write_png(output_path.string().c_str(), out_w, out_h, 3, rgb.data(), out_w * 3)) {
                std::cerr << "failed to write " << output_path << "\n";
                rc = 1;
            } else {
                std::cout << "wrote=" << output_path.string() << "\n";
            }
        }

        matanyone_destroy(session);
        return rc;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
