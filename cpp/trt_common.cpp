#include "trt_common.h"

#include <iostream>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << "\n";
    }
}

void CudaDeleter::operator()(void* ptr) const {
    if (ptr) {
        cudaFree(ptr);
    }
}

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

size_t trtElementSize(nvinfer1::DataType dtype) {
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

int64_t trtVolume(nvinfer1::Dims dims) {
    int64_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        total *= dims.d[i];
    }
    return total;
}

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

TrtEngine::TrtEngine(nvinfer1::IRuntime& runtime, const std::string& path) : path_(path) {
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
        names_.emplace_back(engine_->getIOTensorName(i));
    }
}

const std::string& TrtEngine::path() const {
    return path_;
}

bool TrtEngine::hasTensor(const std::string& name) const {
    return std::find(names_.begin(), names_.end(), name) != names_.end();
}

nvinfer1::TensorIOMode TrtEngine::tensorMode(const std::string& name) const {
    return engine_->getTensorIOMode(name.c_str());
}

nvinfer1::Dims TrtEngine::tensorShape(const std::string& name) const {
    return context_->getTensorShape(name.c_str());
}

nvinfer1::DataType TrtEngine::tensorType(const std::string& name) const {
    return engine_->getTensorDataType(name.c_str());
}

std::vector<std::string> TrtEngine::tensorNames() const {
    return names_;
}

void TrtEngine::bind(const std::string& name, void* ptr) {
    if (!hasTensor(name)) {
        return;
    }
    if (!context_->setTensorAddress(name.c_str(), ptr)) {
        throw std::runtime_error("setTensorAddress failed for " + name + " in " + path_);
    }
}

void TrtEngine::enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed for " + path_);
    }
}
