#include "matanyone_runner.h"
#include "trt_common.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

std::vector<float> readFloats(const fs::path& path, size_t count) {
    std::vector<float> data(count);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        throw std::runtime_error("failed to read " + path.string());
    }
    return data;
}

void writeFloats(const fs::path& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

int main(int argc, char** argv) {
    try {
        std::string engineDir = "engines/faithful";
        fs::path sampleDir = "raw_sample";
        fs::path outDir = "output/raw_alpha";
        if (argc > 1) {
            sampleDir = argv[1];
        }
        if (argc > 2) {
            engineDir = argv[2];
        }
        fs::create_directories(outDir);

        const size_t imageCount = 3ULL * 720ULL * 1280ULL;
        const size_t alphaCount = 720ULL * 1280ULL;
        auto mask = readFloats(sampleDir / "mask.f32", alphaCount);
        auto first = readFloats(sampleDir / "frames" / "00000.rgbf32", imageCount);

        MatAnyoneRunner runner(engineDir);
        runner.Initialize(first.data(), mask.data());

        std::vector<double> timings;
        std::vector<float> alpha(alphaCount);
        runner.CopyAlphaToHost(alpha.data());
        writeFloats(outDir / "00000.alphaf32", alpha);

        int frameIndex = 1;
        for (;; ++frameIndex) {
            fs::path framePath = sampleDir / "frames" / (std::string(5 - std::to_string(frameIndex).length(), '0') +
                                                          std::to_string(frameIndex) + ".rgbf32");
            if (!fs::exists(framePath)) {
                break;
            }
            auto frame = readFloats(framePath, imageCount);
            const auto start = std::chrono::high_resolution_clock::now();
            runner.ProcessFrameRgb(frame.data(), frameIndex % 5 == 0);
            checkCuda(cudaStreamSynchronize(runner.stream()), "frame sync");
            const auto end = std::chrono::high_resolution_clock::now();
            timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            runner.CopyAlphaToHost(alpha.data());
            writeFloats(outDir / (std::string(5 - std::to_string(frameIndex).length(), '0') + std::to_string(frameIndex) +
                                  ".alphaf32"),
                        alpha);
        }

        std::cout << "frames=" << frameIndex << "\n";
        if (!timings.empty()) {
            std::cout << "mean_ms=" << mean(timings) << "\n";
            std::cout << "p99_ms=" << percentile(timings, 99.0) << "\n";
        }
        std::cout << "wrote=" << outDir.string() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
