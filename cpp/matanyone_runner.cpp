#include "matanyone_runner.h"

#include <iostream>

namespace {

nvinfer1::Dims dims(std::initializer_list<int64_t> values) {
    nvinfer1::Dims d{};
    d.nbDims = static_cast<int32_t>(values.size());
    int i = 0;
    for (const auto value : values) {
        d.d[i++] = value;
    }
    return d;
}

} // namespace

MatAnyoneRunner::MatAnyoneRunner(const std::string& engineDir) {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("createInferRuntime failed");
    }

    encode_ = std::make_unique<TrtEngine>(*runtime_, engineDir + "/encode_image_fp16.engine");
    read_ = std::make_unique<TrtEngine>(*runtime_, engineDir + "/read_memory_fp16.engine");
    segment_ = std::make_unique<TrtEngine>(*runtime_, engineDir + "/segment_fp16.engine");
    encodeMask_ = std::make_unique<TrtEngine>(*runtime_, engineDir + "/encode_mask_fp16.engine");
    encodeMaskShallow_ = std::make_unique<TrtEngine>(*runtime_, engineDir + "/encode_mask_shallow_fp16.engine");

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    allocateStaticBuffers();
    bindEngines();

    // Initialize first-frame memory with deterministic garbage-free values for benchmarking.
    checkCuda(cudaMemsetAsync(buffers_.at("memory_key").ptr.get(), 0, buffers_.at("memory_key").bytes, stream_), "memset memory_key");
    checkCuda(cudaMemsetAsync(buffers_.at("memory_shrinkage").ptr.get(), 0, buffers_.at("memory_shrinkage").bytes, stream_), "memset memory_shrinkage");
    checkCuda(cudaMemsetAsync(buffers_.at("memory_value").ptr.get(), 0, buffers_.at("memory_value").bytes, stream_), "memset memory_value");
    checkCuda(cudaMemsetAsync(buffers_.at("obj_memory").ptr.get(), 0, buffers_.at("obj_memory").bytes, stream_), "memset obj_memory");
    checkCuda(cudaStreamSynchronize(stream_), "init sync");
}

MatAnyoneRunner::~MatAnyoneRunner() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

cudaStream_t MatAnyoneRunner::stream() const {
    return stream_;
}

void MatAnyoneRunner::allocateBuffer(const std::string& name, nvinfer1::Dims d, nvinfer1::DataType dtype) {
    if (buffers_.find(name) != buffers_.end()) {
        return;
    }
    const auto bytes = static_cast<size_t>(trtVolume(d)) * trtElementSize(dtype);
    void* ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, bytes), ("cudaMalloc " + name).c_str());
    buffers_.emplace(name, TensorBuffer{DevicePtr(ptr), bytes, d, dtype});
}

void MatAnyoneRunner::allocateStaticBuffers() {
    const auto f32 = nvinfer1::DataType::kFLOAT;
    allocateBuffer("image", dims({1, 3, 720, 1280}), f32);
    allocateBuffer("last_mask", dims({1, 1, 720, 1280}), f32);
    allocateBuffer("alpha", dims({1, 1, 720, 1280}), f32);
    allocateBuffer("f16", dims({1, 1024, 45, 80}), f32);
    allocateBuffer("f8", dims({1, 512, 90, 160}), f32);
    allocateBuffer("f4", dims({1, 256, 180, 320}), f32);
    allocateBuffer("f2", dims({1, 64, 360, 640}), f32);
    allocateBuffer("f1", dims({1, 3, 720, 1280}), f32);
    allocateBuffer("pix_feat", dims({1, 256, 45, 80}), f32);
    allocateBuffer("last_pix_feat", dims({1, 256, 45, 80}), f32);
    allocateBuffer("key", dims({1, 64, 45, 80}), f32);
    allocateBuffer("shrinkage", dims({1, 1, 45, 80}), f32);
    allocateBuffer("selection", dims({1, 64, 45, 80}), f32);
    allocateBuffer("memory_key", dims({1, 64, 5, 45, 80}), f32);
    allocateBuffer("memory_shrinkage", dims({1, 1, 5, 45, 80}), f32);
    allocateBuffer("memory_value", dims({1, 1, 256, 5, 45, 80}), f32);
    allocateBuffer("last_msk_value", dims({1, 1, 256, 45, 80}), f32);
    allocateBuffer("sensory", dims({1, 1, 256, 45, 80}), f32);
    allocateBuffer("new_sensory", dims({1, 1, 256, 45, 80}), f32);
    allocateBuffer("obj_memory", dims({1, 1, 1, 16, 257}), f32);
    allocateBuffer("memory_readout", dims({1, 1, 256, 45, 80}), f32);
    allocateBuffer("mask_value", dims({1, 1, 256, 45, 80}), f32);
    allocateBuffer("object_summaries", dims({1, 1, 16, 257}), f32);
}

void MatAnyoneRunner::bindEngines() {
    encode_->bind("input_0", buffers_.at("image").ptr.get());
    encode_->bind("f16", buffers_.at("f16").ptr.get());
    encode_->bind("f8", buffers_.at("f8").ptr.get());
    encode_->bind("f4", buffers_.at("f4").ptr.get());
    encode_->bind("f2", buffers_.at("f2").ptr.get());
    encode_->bind("f1", buffers_.at("f1").ptr.get());
    encode_->bind("pix_feat", buffers_.at("pix_feat").ptr.get());
    encode_->bind("key", buffers_.at("key").ptr.get());
    encode_->bind("shrinkage", buffers_.at("shrinkage").ptr.get());
    encode_->bind("selection", buffers_.at("selection").ptr.get());

    read_->bind("input_0", buffers_.at("key").ptr.get());
    read_->bind("input_1", buffers_.at("selection").ptr.get());
    read_->bind("input_2", buffers_.at("memory_key").ptr.get());
    read_->bind("input_3", buffers_.at("memory_shrinkage").ptr.get());
    read_->bind("input_4", buffers_.at("memory_value").ptr.get());
    read_->bind("input_5", buffers_.at("last_pix_feat").ptr.get());
    read_->bind("input_6", buffers_.at("pix_feat").ptr.get());
    read_->bind("input_7", buffers_.at("last_mask").ptr.get());
    read_->bind("input_8", buffers_.at("last_msk_value").ptr.get());
    read_->bind("input_9", buffers_.at("sensory").ptr.get());
    read_->bind("input_10", buffers_.at("obj_memory").ptr.get());
    read_->bind("memory_readout", buffers_.at("memory_readout").ptr.get());

    segment_->bind("input_0", buffers_.at("f16").ptr.get());
    segment_->bind("input_1", buffers_.at("f8").ptr.get());
    segment_->bind("input_2", buffers_.at("f4").ptr.get());
    segment_->bind("input_3", buffers_.at("f2").ptr.get());
    segment_->bind("input_4", buffers_.at("f1").ptr.get());
    segment_->bind("input_5", buffers_.at("memory_readout").ptr.get());
    segment_->bind("input_6", buffers_.at("sensory").ptr.get());
    segment_->bind("new_sensory", buffers_.at("new_sensory").ptr.get());
    segment_->bind("alpha", buffers_.at("alpha").ptr.get());

    encodeMask_->bind("input_0", buffers_.at("image").ptr.get());
    encodeMask_->bind("input_1", buffers_.at("pix_feat").ptr.get());
    encodeMask_->bind("input_2", buffers_.at("new_sensory").ptr.get());
    encodeMask_->bind("input_3", buffers_.at("alpha").ptr.get());
    encodeMask_->bind("mask_value", buffers_.at("mask_value").ptr.get());
    encodeMask_->bind("new_sensory", buffers_.at("sensory").ptr.get());
    encodeMask_->bind("object_summaries", buffers_.at("object_summaries").ptr.get());

    encodeMaskShallow_->bind("input_0", buffers_.at("image").ptr.get());
    encodeMaskShallow_->bind("input_1", buffers_.at("pix_feat").ptr.get());
    encodeMaskShallow_->bind("input_3", buffers_.at("alpha").ptr.get());
    encodeMaskShallow_->bind("mask_value", buffers_.at("last_msk_value").ptr.get());
}

void MatAnyoneRunner::advanceMemoryBank(bool deepUpdate) {
    const size_t keyFrameBytes = 64ULL * 45ULL * 80ULL * sizeof(float);
    const size_t shrinkFrameBytes = 45ULL * 80ULL * sizeof(float);
    const size_t valueFrameBytes = 256ULL * 45ULL * 80ULL * sizeof(float);
    const int slot = memSlot_ < 5 ? memSlot_ : 4;

    auto* memoryKey = static_cast<std::uint8_t*>(buffers_.at("memory_key").ptr.get());
    auto* memoryShrink = static_cast<std::uint8_t*>(buffers_.at("memory_shrinkage").ptr.get());
    auto* memoryValue = static_cast<std::uint8_t*>(buffers_.at("memory_value").ptr.get());
    if (memSlot_ >= 5) {
        checkCuda(cudaMemcpyAsync(memoryKey + keyFrameBytes, memoryKey + 2 * keyFrameBytes, 3 * keyFrameBytes,
                                  cudaMemcpyDeviceToDevice, stream_),
                  "shift memory_key");
        checkCuda(cudaMemcpyAsync(memoryShrink + shrinkFrameBytes, memoryShrink + 2 * shrinkFrameBytes,
                                  3 * shrinkFrameBytes, cudaMemcpyDeviceToDevice, stream_),
                  "shift memory_shrinkage");
        checkCuda(cudaMemcpyAsync(memoryValue + valueFrameBytes, memoryValue + 2 * valueFrameBytes,
                                  3 * valueFrameBytes, cudaMemcpyDeviceToDevice, stream_),
                  "shift memory_value");
    }

    checkCuda(cudaMemcpyAsync(memoryKey + slot * keyFrameBytes, buffers_.at("key").ptr.get(), keyFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "copy memory_key slot");
    checkCuda(cudaMemcpyAsync(memoryShrink + slot * shrinkFrameBytes, buffers_.at("shrinkage").ptr.get(), shrinkFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "copy memory_shrinkage slot");
    checkCuda(cudaMemcpyAsync(memoryValue + slot * valueFrameBytes, buffers_.at("mask_value").ptr.get(), valueFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "copy memory_value slot");
    checkCuda(cudaMemcpyAsync(buffers_.at("last_msk_value").ptr.get(), buffers_.at("mask_value").ptr.get(), valueFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "copy last_msk_value");
    if (deepUpdate) {
        checkCuda(cudaMemcpyAsync(buffers_.at("obj_memory").ptr.get(), buffers_.at("object_summaries").ptr.get(),
                                  1ULL * 1ULL * 16ULL * 257ULL * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
                  "copy obj_memory");
        checkCuda(cudaMemcpyAsync(buffers_.at("last_pix_feat").ptr.get(), buffers_.at("pix_feat").ptr.get(), valueFrameBytes,
                                  cudaMemcpyDeviceToDevice, stream_),
                  "copy last_pix_feat");
        checkCuda(cudaMemcpyAsync(buffers_.at("last_mask").ptr.get(), buffers_.at("alpha").ptr.get(), 720ULL * 1280ULL * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream_),
                  "copy last_mask");
    }
    ++memSlot_;
}

void MatAnyoneRunner::seedMemoryBank() {
    const size_t keyFrameBytes = 64ULL * 45ULL * 80ULL * sizeof(float);
    const size_t shrinkFrameBytes = 45ULL * 80ULL * sizeof(float);
    const size_t valueFrameBytes = 256ULL * 45ULL * 80ULL * sizeof(float);
    auto* memoryKey = static_cast<std::uint8_t*>(buffers_.at("memory_key").ptr.get());
    auto* memoryShrink = static_cast<std::uint8_t*>(buffers_.at("memory_shrinkage").ptr.get());
    auto* memoryValue = static_cast<std::uint8_t*>(buffers_.at("memory_value").ptr.get());
    for (int slot = 0; slot < 5; ++slot) {
        checkCuda(cudaMemcpyAsync(memoryKey + slot * keyFrameBytes, buffers_.at("key").ptr.get(), keyFrameBytes,
                                  cudaMemcpyDeviceToDevice, stream_),
                  "seed memory_key");
        checkCuda(cudaMemcpyAsync(memoryShrink + slot * shrinkFrameBytes, buffers_.at("shrinkage").ptr.get(),
                                  shrinkFrameBytes, cudaMemcpyDeviceToDevice, stream_),
                  "seed memory_shrinkage");
        checkCuda(cudaMemcpyAsync(memoryValue + slot * valueFrameBytes, buffers_.at("mask_value").ptr.get(), valueFrameBytes,
                                  cudaMemcpyDeviceToDevice, stream_),
                  "seed memory_value");
    }
    checkCuda(cudaMemcpyAsync(buffers_.at("obj_memory").ptr.get(), buffers_.at("object_summaries").ptr.get(),
                              1ULL * 1ULL * 16ULL * 257ULL * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
              "seed obj_memory");
    checkCuda(cudaMemcpyAsync(buffers_.at("last_pix_feat").ptr.get(), buffers_.at("pix_feat").ptr.get(), valueFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "seed last_pix_feat");
    checkCuda(cudaMemcpyAsync(buffers_.at("last_msk_value").ptr.get(), buffers_.at("mask_value").ptr.get(), valueFrameBytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "seed last_msk_value");
    checkCuda(cudaMemcpyAsync(buffers_.at("last_mask").ptr.get(), buffers_.at("alpha").ptr.get(), 720ULL * 1280ULL * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream_),
              "seed last_mask");
    memSlot_ = 1;
}

void MatAnyoneRunner::Initialize(const float* rgbNchw, const float* alphaMaskNchw) {
    checkCuda(cudaMemcpyAsync(buffers_.at("image").ptr.get(), rgbNchw, 3ULL * 720ULL * 1280ULL * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "copy initial image");
    checkCuda(cudaMemcpyAsync(buffers_.at("alpha").ptr.get(), alphaMaskNchw, 720ULL * 1280ULL * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "copy initial alpha");
    encode_->enqueue(stream_);
    encodeMask_->enqueue(stream_);
    seedMemoryBank();
    checkCuda(cudaStreamSynchronize(stream_), "initialize sync");
}

void MatAnyoneRunner::ProcessFrameRgb(const float* rgbNchw, bool memoryUpdate) {
    checkCuda(cudaMemcpyAsync(buffers_.at("image").ptr.get(), rgbNchw, 3ULL * 720ULL * 1280ULL * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "copy frame image");
    ProcessFrame(memoryUpdate);
}

void MatAnyoneRunner::CopyAlphaToHost(float* alphaNchw) {
    checkCuda(cudaMemcpyAsync(alphaNchw, buffers_.at("alpha").ptr.get(), 720ULL * 1280ULL * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "copy alpha to host");
    checkCuda(cudaStreamSynchronize(stream_), "copy alpha sync");
}

void MatAnyoneRunner::ProcessFrame(bool memoryUpdate) {
    encode_->enqueue(stream_);
    read_->enqueue(stream_);
    segment_->enqueue(stream_);
    if (memoryUpdate) {
        encodeMask_->enqueue(stream_);
        advanceMemoryBank(true);
    } else {
        encodeMaskShallow_->enqueue(stream_);
        advanceMemoryBank(false);
    }
}
