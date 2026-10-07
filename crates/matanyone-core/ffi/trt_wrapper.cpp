#include "trt_wrapper.h"

#include <NvInfer.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

Logger g_logger;

struct TrtDestroy {
    template <typename T>
    void operator()(T* ptr) const {
        delete ptr;
    }
};

std::vector<char> readFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open ") + path);
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
        return 0;
    }
}

int64_t volume(nvinfer1::Dims dims) {
    int64_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        total *= dims.d[i];
    }
    return total;
}

struct TrtRuntimeImpl {
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy> runtime;
};

struct TrtEngineImpl {
    std::string path;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy> context;
    std::vector<std::string> names;
};

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

struct TrtBenchEngineImpl {
    std::unique_ptr<TrtEngineImpl> engine;
    std::vector<std::unique_ptr<void, CudaDeleter>> buffers;
};

} // namespace

extern "C" {

TrtRuntime* matanyone_trt_create_runtime(void) {
    try {
        auto* impl = new TrtRuntimeImpl();
        impl->runtime.reset(nvinfer1::createInferRuntime(g_logger));
        if (!impl->runtime) {
            delete impl;
            return nullptr;
        }
        return reinterpret_cast<TrtRuntime*>(impl);
    } catch (...) {
        return nullptr;
    }
}

void matanyone_trt_destroy_runtime(TrtRuntime* runtime) {
    delete reinterpret_cast<TrtRuntimeImpl*>(runtime);
}

TrtEngineHandle* matanyone_trt_load_engine(TrtRuntime* runtime, const char* path) {
    try {
        auto* rt = reinterpret_cast<TrtRuntimeImpl*>(runtime);
        if (!rt || !rt->runtime || !path) {
            return nullptr;
        }
        auto* impl = new TrtEngineImpl();
        impl->path = path;
        const auto data = readFile(path);
        impl->engine.reset(rt->runtime->deserializeCudaEngine(data.data(), data.size()));
        if (!impl->engine) {
            delete impl;
            return nullptr;
        }
        impl->context.reset(impl->engine->createExecutionContext());
        if (!impl->context) {
            delete impl;
            return nullptr;
        }
        const int n = impl->engine->getNbIOTensors();
        for (int i = 0; i < n; ++i) {
            impl->names.emplace_back(impl->engine->getIOTensorName(i));
        }
        return reinterpret_cast<TrtEngineHandle*>(impl);
    } catch (...) {
        return nullptr;
    }
}

void matanyone_trt_destroy_engine(TrtEngineHandle* engine) {
    delete reinterpret_cast<TrtEngineImpl*>(engine);
}

int matanyone_trt_has_tensor(TrtEngineHandle* engine, const char* name) {
    auto* impl = reinterpret_cast<TrtEngineImpl*>(engine);
    if (!impl || !name) {
        return 0;
    }
    return std::find(impl->names.begin(), impl->names.end(), name) != impl->names.end() ? 1 : 0;
}

int matanyone_trt_bind_tensor(TrtEngineHandle* engine, const char* name, void* ptr) {
    try {
        auto* impl = reinterpret_cast<TrtEngineImpl*>(engine);
        if (!impl || !name || !matanyone_trt_has_tensor(engine, name)) {
            return 0;
        }
        return impl->context->setTensorAddress(name, ptr) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int matanyone_trt_enqueue(TrtEngineHandle* engine, cudaStream_t stream) {
    try {
        auto* impl = reinterpret_cast<TrtEngineImpl*>(engine);
        if (!impl) {
            return 0;
        }
        return impl->context->enqueueV3(stream) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int64_t matanyone_trt_tensor_volume(TrtEngineHandle* engine, const char* name) {
    auto* impl = reinterpret_cast<TrtEngineImpl*>(engine);
    if (!impl || !name || !matanyone_trt_has_tensor(engine, name)) {
        return 0;
    }
    return volume(impl->context->getTensorShape(name));
}

int matanyone_trt_tensor_dtype(TrtEngineHandle* engine, const char* name) {
    auto* impl = reinterpret_cast<TrtEngineImpl*>(engine);
    if (!impl || !name || !matanyone_trt_has_tensor(engine, name)) {
        return -1;
    }
    return static_cast<int>(impl->engine->getTensorDataType(name));
}

TrtBenchEngine* matanyone_trt_load_bench_engine(TrtRuntime* runtime, const char* path) {
    try {
        auto* handle = matanyone_trt_load_engine(runtime, path);
        if (!handle) {
            return nullptr;
        }
        auto* bench = new TrtBenchEngineImpl();
        bench->engine.reset(reinterpret_cast<TrtEngineImpl*>(handle));
        auto* impl = bench->engine.get();
        for (const auto& name : impl->names) {
            const auto dims = impl->context->getTensorShape(name.c_str());
            const auto bytes = static_cast<size_t>(volume(dims)) * elementSize(impl->engine->getTensorDataType(name.c_str()));
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
                matanyone_trt_destroy_bench_engine(reinterpret_cast<TrtBenchEngine*>(bench));
                return nullptr;
            }
            bench->buffers.emplace_back(ptr);
            if (!impl->context->setTensorAddress(name.c_str(), ptr)) {
                matanyone_trt_destroy_bench_engine(reinterpret_cast<TrtBenchEngine*>(bench));
                return nullptr;
            }
        }
        return reinterpret_cast<TrtBenchEngine*>(bench);
    } catch (...) {
        return nullptr;
    }
}

void matanyone_trt_destroy_bench_engine(TrtBenchEngine* engine) {
    delete reinterpret_cast<TrtBenchEngineImpl*>(engine);
}

int matanyone_trt_bench_enqueue(TrtBenchEngine* engine, cudaStream_t stream) {
    try {
        auto* bench = reinterpret_cast<TrtBenchEngineImpl*>(engine);
        if (!bench || !bench->engine) {
            return 0;
        }
        return bench->engine->context->enqueueV3(stream) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

} // extern "C"
