#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

using DevicePtr = std::unique_ptr<void, CudaDeleter>;

void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open " + path);
    }
    const auto size = file.tellg();
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(data.data(), size);
    return data;
}

size_t elementSize(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT64:
        return 8;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

int64_t volume(nvinfer1::Dims dims) {
    int64_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        total *= dims.d[i];
    }
    return total;
}

class Engine {
public:
    Engine(nvinfer1::IRuntime& runtime, const std::string& path) : path_(path) {
        const auto data = readFile(path);
        engine_.reset(runtime.deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize " + path);
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create context for " + path);
        }

        const int n = engine_->getNbIOTensors();
        for (int i = 0; i < n; ++i) {
            const char* name = engine_->getIOTensorName(i);
            names_.emplace_back(name);
            const auto dims = context_->getTensorShape(name);
            const auto bytes = static_cast<size_t>(volume(dims)) * elementSize(engine_->getTensorDataType(name));
            void* ptr = nullptr;
            checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
            buffers_.emplace(name, DevicePtr(ptr));
            if (!context_->setTensorAddress(name, ptr)) {
                throw std::runtime_error("setTensorAddress failed for " + std::string(name));
            }
        }
    }

    void enqueue(cudaStream_t stream) {
        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("enqueueV3 failed for " + path_);
        }
    }

private:
    struct RuntimeDeleter {
        template <typename T>
        void operator()(T* ptr) const {
            delete ptr;
        }
    };

    std::string path_;
    std::unique_ptr<nvinfer1::ICudaEngine, RuntimeDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, RuntimeDeleter> context_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, DevicePtr> buffers_;
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(idx);
    const auto hi = std::min(lo + 1, values.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string engineDir = "engines/faithful";
        int warmup = 30;
        int iters = 100;
        if (argc > 1) {
            engineDir = argv[1];
        }

        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("createInferRuntime failed");
        }

        Engine encode(*runtime, engineDir + "/encode_image_fp16.engine");
        Engine read(*runtime, engineDir + "/read_memory_fp16.engine");
        Engine segment(*runtime, engineDir + "/segment_fp16.engine");
        Engine encodeMask(*runtime, engineDir + "/encode_mask_fp16.engine");
        Engine encodeMaskShallow(*runtime, engineDir + "/encode_mask_shallow_fp16.engine");

        cudaStream_t stream{};
        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        auto runNormal = [&]() {
            encode.enqueue(stream);
            read.enqueue(stream);
            segment.enqueue(stream);
            encodeMaskShallow.enqueue(stream);
        };
        auto runUpdate = [&]() {
            encode.enqueue(stream);
            read.enqueue(stream);
            segment.enqueue(stream);
            encodeMask.enqueue(stream);
        };

        for (int i = 0; i < warmup; ++i) {
            runNormal();
        }
        checkCuda(cudaStreamSynchronize(stream), "warmup sync");

        std::vector<double> normalTimes;
        normalTimes.reserve(static_cast<size_t>(iters));
        for (int i = 0; i < iters; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            runNormal();
            checkCuda(cudaStreamSynchronize(stream), "normal sync");
            const auto end = std::chrono::high_resolution_clock::now();
            normalTimes.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        for (int i = 0; i < warmup; ++i) {
            runUpdate();
        }
        checkCuda(cudaStreamSynchronize(stream), "update warmup sync");

        std::vector<double> updateTimes;
        updateTimes.reserve(static_cast<size_t>(iters));
        for (int i = 0; i < iters; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            runUpdate();
            checkCuda(cudaStreamSynchronize(stream), "update sync");
            const auto end = std::chrono::high_resolution_clock::now();
            updateTimes.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        std::cout << "normal_mean_ms=" << mean(normalTimes) << "\n";
        std::cout << "normal_p99_ms=" << percentile(normalTimes, 99.0) << "\n";
        std::cout << "memory_update_mean_ms=" << mean(updateTimes) << "\n";
        std::cout << "memory_update_p99_ms=" << percentile(updateTimes, 99.0) << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
