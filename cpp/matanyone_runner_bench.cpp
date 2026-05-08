#include "matanyone_runner.h"
#include "trt_common.h"

#include <chrono>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::string engineDir = "engines/faithful";
        int warmup = 30;
        int iters = 100;
        if (argc > 1) {
            engineDir = argv[1];
        }

        MatAnyoneRunner runner(engineDir);
        for (int i = 0; i < warmup; ++i) {
            runner.ProcessFrame(i % 5 == 0);
        }
        checkCuda(cudaStreamSynchronize(runner.stream()), "warmup sync");

        std::vector<double> timings;
        timings.reserve(static_cast<size_t>(iters));
        for (int i = 0; i < iters; ++i) {
            const bool memoryUpdate = i % 5 == 0;
            const auto start = std::chrono::high_resolution_clock::now();
            runner.ProcessFrame(memoryUpdate);
            checkCuda(cudaStreamSynchronize(runner.stream()), "process sync");
            const auto end = std::chrono::high_resolution_clock::now();
            timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::cout << "mixed_mean_ms=" << mean(timings) << "\n";
        std::cout << "mixed_p99_ms=" << percentile(timings, 99.0) << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
