#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("matanyone-obs", "en-US")

struct MatAnyoneFilter {
    obs_source_t* source = nullptr;
};

static const char* matanyone_filter_name(void*) {
    return "MatAnyone AI Matting";
}

static void* matanyone_filter_create(obs_data_t*, obs_source_t* source) {
    auto* filter = new MatAnyoneFilter();
    filter->source = source;
    return filter;
}

static void matanyone_filter_destroy(void* data) {
    delete static_cast<MatAnyoneFilter*>(data);
}

static void matanyone_filter_render(void* data, gs_effect_t*) {
    auto* filter = static_cast<MatAnyoneFilter*>(data);
    if (!obs_source_process_filter_begin(filter->source, GS_RGBA, OBS_NO_DIRECT_RENDERING)) {
        obs_source_skip_video_filter(filter->source);
        return;
    }

    // Passthrough scaffold. The C++ MatAnyoneRunner and DX11/CUDA interop will be wired here.
    obs_source_process_filter_end(filter->source, nullptr, 0, 0);
}

static obs_properties_t* matanyone_filter_properties(void*) {
    auto* props = obs_properties_create();
    obs_properties_add_path(props, "background_image", "Background Image", OBS_PATH_FILE,
                            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)", nullptr);
    obs_properties_add_color(props, "background_color", "Background Color");
    obs_properties_add_int_slider(props, "internal_height", "Internal Height", 360, 1080, 180);
    return props;
}

static void matanyone_filter_defaults(obs_data_t* settings) {
    obs_data_set_default_int(settings, "internal_height", 720);
    obs_data_set_default_int(settings, "background_color", 0x000000);
}

bool obs_module_load(void) {
    obs_source_info info = {};
    info.id = "matanyone_ai_matting_filter";
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_VIDEO;
    info.get_name = matanyone_filter_name;
    info.create = matanyone_filter_create;
    info.destroy = matanyone_filter_destroy;
    info.video_render = matanyone_filter_render;
    info.get_properties = matanyone_filter_properties;
    info.get_defaults = matanyone_filter_defaults;

    obs_register_source(&info);
    return true;
}
