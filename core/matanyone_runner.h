#pragma once

#include "trt_common.h"

#include <memory>
#include <string>
#include <unordered_map>

class MatAnyoneRunner {
public:
    explicit MatAnyoneRunner(const std::string& engineDir);
    ~MatAnyoneRunner();

    MatAnyoneRunner(const MatAnyoneRunner&) = delete;
    MatAnyoneRunner& operator=(const MatAnyoneRunner&) = delete;

    void ProcessFrame(bool memoryUpdate);
    void Initialize(const float* rgbNchw, const float* alphaMaskNchw);
    void InitializeDevice(const float* rgbNchwDevice, const float* alphaMaskNchwDevice);
    void ProcessFrameRgb(const float* rgbNchw, bool memoryUpdate);
    void ProcessFrameRgbDevice(const float* rgbNchwDevice, bool memoryUpdate);
    void CopyAlphaToHost(float* alphaNchw);
    void CopyAlphaToDevice(float* alphaDevice);
    cudaStream_t stream() const;

private:
    void allocateBuffer(const std::string& name, nvinfer1::Dims dims, nvinfer1::DataType dtype);
    void allocateStaticBuffers();
    void bindEngines();
    void advanceMemoryBank(bool deepUpdate);
    void seedMemoryBank();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy> runtime_;
    std::unique_ptr<TrtEngine> encode_;
    std::unique_ptr<TrtEngine> read_;
    std::unique_ptr<TrtEngine> pixelFusion_;
    std::unique_ptr<TrtEngine> segment_;
    std::unique_ptr<TrtEngine> encodeMask_;
    cudaStream_t stream_{};
    std::unordered_map<std::string, TensorBuffer> buffers_;
    int memSlot_ = 1;
};
