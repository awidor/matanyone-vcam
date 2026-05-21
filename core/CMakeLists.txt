cmake_minimum_required(VERSION 3.28)
project(matanyone_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "C:/Tools/TensorRT-10.16.1.11" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" CACHE PATH "CUDA toolkit root")

add_library(matanyone_core STATIC
    matanyone_runner.cpp
    trt_common.cpp
    matanyone_session.cpp
    stb_image_impl.cpp
    stb_image_write_impl.cpp
    kernels/matanyone_kernels.cpp
)

target_include_directories(matanyone_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${TENSORRT_ROOT}/include"
    "${CUDA_ROOT}/include"
)

target_link_directories(matanyone_core PUBLIC
    "${TENSORRT_ROOT}/lib"
    "${CUDA_ROOT}/lib/x64"
)

target_link_libraries(matanyone_core PUBLIC
    nvinfer_10
    cudart
    cuda
)

target_compile_definitions(matanyone_core PRIVATE NOMINMAX)

add_executable(matanyone_core_smoke matanyone_core_smoke.cpp)
target_link_libraries(matanyone_core_smoke PRIVATE matanyone_core)
