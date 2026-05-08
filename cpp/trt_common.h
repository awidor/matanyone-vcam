#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

void checkCuda(cudaError_t status, const char* what);
std::vector<char> readFile(const std::string& path);
size_t trtElementSize(nvinfer1::DataType dtype);
int64_t trtVolume(nvinfer1::Dims dims);
double percentile(std::vector<double> values, double p);
double mean(const std::vector<double>& values);

struct CudaDeleter {
    void operator()(void* ptr) const;
};

using DevicePtr = std::unique_ptr<void, CudaDeleter>;

struct TensorBuffer {
    DevicePtr ptr;
    size_t bytes = 0;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
};

struct TrtDestroy {
    template <typename T>
    void operator()(T* ptr) const {
        delete ptr;
    }
};

class TrtEngine {
public:
    TrtEngine(nvinfer1::IRuntime& runtime, const std::string& path);

    const std::string& path() const;
    bool hasTensor(const std::string& name) const;
    nvinfer1::TensorIOMode tensorMode(const std::string& name) const;
    nvinfer1::Dims tensorShape(const std::string& name) const;
    nvinfer1::DataType tensorType(const std::string& name) const;
    std::vector<std::string> tensorNames() const;
    void bind(const std::string& name, void* ptr);
    void enqueue(cudaStream_t stream);

private:
    std::string path_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy> context_;
    std::vector<std::string> names_;
};
