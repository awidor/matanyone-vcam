cmake_minimum_required(VERSION 3.28)
project(matanyone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "C:/Tools/TensorRT-10.16.1.11" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" CACHE PATH "CUDA toolkit root")

add_subdirectory(core)

add_executable(matanyone_trt_bench
    cpp/matanyone_trt_bench.cpp
    cpp/trt_common.cpp
)

add_executable(matanyone_runner_bench
    cpp/matanyone_runner_bench.cpp
    core/matanyone_runner.cpp
    core/trt_common.cpp
)

add_executable(matanyone_runner_raw
    cpp/matanyone_runner_raw.cpp
    core/matanyone_runner.cpp
    core/trt_common.cpp
)

foreach(target matanyone_trt_bench matanyone_runner_bench matanyone_runner_raw)
    target_include_directories(${target} PRIVATE
        "${TENSORRT_ROOT}/include"
        "${CUDA_ROOT}/include"
        core
    )
    target_link_directories(${target} PRIVATE
        "${TENSORRT_ROOT}/lib"
        "${CUDA_ROOT}/lib/x64"
    )
    target_link_libraries(${target} PRIVATE nvinfer_10 cudart)
    target_compile_definitions(${target} PRIVATE NOMINMAX)
endforeach()
