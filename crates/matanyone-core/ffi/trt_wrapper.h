#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef _WIN32
#define MATANYONE_TRT_API __declspec(dllexport)
#else
#define MATANYONE_TRT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtRuntime TrtRuntime;
typedef struct TrtEngineHandle TrtEngineHandle;

MATANYONE_TRT_API TrtRuntime* matanyone_trt_create_runtime(void);
MATANYONE_TRT_API void matanyone_trt_destroy_runtime(TrtRuntime* runtime);

MATANYONE_TRT_API TrtEngineHandle* matanyone_trt_load_engine(TrtRuntime* runtime, const char* path);
MATANYONE_TRT_API void matanyone_trt_destroy_engine(TrtEngineHandle* engine);

MATANYONE_TRT_API int matanyone_trt_has_tensor(TrtEngineHandle* engine, const char* name);
MATANYONE_TRT_API int matanyone_trt_bind_tensor(TrtEngineHandle* engine, const char* name, void* ptr);
MATANYONE_TRT_API int matanyone_trt_enqueue(TrtEngineHandle* engine, cudaStream_t stream);

MATANYONE_TRT_API int64_t matanyone_trt_tensor_volume(TrtEngineHandle* engine, const char* name);
MATANYONE_TRT_API int matanyone_trt_tensor_dtype(TrtEngineHandle* engine, const char* name);

typedef struct TrtBenchEngine TrtBenchEngine;

MATANYONE_TRT_API TrtBenchEngine* matanyone_trt_load_bench_engine(TrtRuntime* runtime, const char* path);
MATANYONE_TRT_API void matanyone_trt_destroy_bench_engine(TrtBenchEngine* engine);
MATANYONE_TRT_API int matanyone_trt_bench_enqueue(TrtBenchEngine* engine, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
