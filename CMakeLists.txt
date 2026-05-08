cmake_minimum_required(VERSION 3.28)
project(matanyone_obs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TENSORRT_ROOT "C:/Tools/TensorRT-10.16.1.11" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" CACHE PATH "CUDA toolkit root")

add_executable(matanyone_trt_bench
    cpp/matanyone_trt_bench.cpp
    cpp/trt_common.cpp
)

add_executable(matanyone_runner_bench
    cpp/matanyone_runner_bench.cpp
    cpp/matanyone_runner.cpp
    cpp/trt_common.cpp
)

add_executable(matanyone_runner_raw
    cpp/matanyone_runner_raw.cpp
    cpp/matanyone_runner.cpp
    cpp/trt_common.cpp
)

target_include_directories(matanyone_trt_bench PRIVATE
    "${TENSORRT_ROOT}/include"
    "${CUDA_ROOT}/include"
)

target_include_directories(matanyone_runner_bench PRIVATE
    "${TENSORRT_ROOT}/include"
    "${CUDA_ROOT}/include"
)

target_include_directories(matanyone_runner_raw PRIVATE
    "${TENSORRT_ROOT}/include"
    "${CUDA_ROOT}/include"
)

target_link_directories(matanyone_trt_bench PRIVATE
    "${TENSORRT_ROOT}/lib"
    "${CUDA_ROOT}/lib/x64"
)

target_link_directories(matanyone_runner_bench PRIVATE
    "${TENSORRT_ROOT}/lib"
    "${CUDA_ROOT}/lib/x64"
)

target_link_directories(matanyone_runner_raw PRIVATE
    "${TENSORRT_ROOT}/lib"
    "${CUDA_ROOT}/lib/x64"
)

target_link_libraries(matanyone_trt_bench PRIVATE
    nvinfer_10
    cudart
)

target_link_libraries(matanyone_runner_bench PRIVATE
    nvinfer_10
    cudart
)

target_link_libraries(matanyone_runner_raw PRIVATE
    nvinfer_10
    cudart
)

target_compile_definitions(matanyone_trt_bench PRIVATE
    NOMINMAX
)

target_compile_definitions(matanyone_runner_bench PRIVATE
    NOMINMAX
)

target_compile_definitions(matanyone_runner_raw PRIVATE
    NOMINMAX
)

option(BUILD_OBS_PLUGIN "Build OBS plugin scaffold" OFF)
set(OBS_ROOT "" CACHE PATH "OBS source/build or development package root")

if(BUILD_OBS_PLUGIN)
    find_package(OBS REQUIRED)
    add_library(matanyone-obs MODULE
        obs-plugin/matanyone_obs_plugin.cpp
        obs-plugin/matanyone-obs.rc
        cpp/matanyone_runner.cpp
        cpp/trt_common.cpp
    )
    target_include_directories(matanyone-obs PRIVATE
        "${TENSORRT_ROOT}/include"
        "${CUDA_ROOT}/include"
        cpp
    )
    target_link_directories(matanyone-obs PRIVATE
        "${TENSORRT_ROOT}/lib"
        "${CUDA_ROOT}/lib/x64"
    )
    target_link_libraries(matanyone-obs PRIVATE
        OBS::libobs
        nvinfer_10
        cudart
    )
    target_compile_definitions(matanyone-obs PRIVATE
        NOMINMAX
    )
endif()
