#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef MATANYONE_CORE_EXPORT
#    define MATANYONE_API __declspec(dllexport)
#  else
#    define MATANYONE_API
#  endif
#else
#  define MATANYONE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MatAnyoneSession MatAnyoneSession;

MATANYONE_API MatAnyoneSession* matanyone_create(const char* engine_dir);
MATANYONE_API void matanyone_destroy(MatAnyoneSession* session);
MATANYONE_API const char* matanyone_last_error(void);

MATANYONE_API int matanyone_internal_width(void);
MATANYONE_API int matanyone_internal_height(void);

MATANYONE_API int matanyone_init_from_mask_file(MatAnyoneSession* session,
                                                const uint8_t* bgr, int src_w, int src_h,
                                                const char* mask_png_path);

MATANYONE_API int matanyone_init_from_mask(MatAnyoneSession* session,
                                           const uint8_t* bgr, int src_w, int src_h,
                                           const float* mask_hw, int mask_w, int mask_h);

MATANYONE_API int matanyone_process_bgr(MatAnyoneSession* session,
                                        const uint8_t* bgr_in, int src_w, int src_h,
                                        uint8_t* bgr_out, int dst_w, int dst_h,
                                        float bg_r, float bg_g, float bg_b,
                                        int memory_update);

MATANYONE_API int matanyone_reset(MatAnyoneSession* session);

#ifdef __cplusplus
}
#endif
