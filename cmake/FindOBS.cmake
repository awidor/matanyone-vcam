find_path(OBS_INCLUDE_DIR
    NAMES obs-module.h
    PATHS
        "${OBS_ROOT}/include"
        "${OBS_ROOT}/libobs"
        "${OBS_ROOT}/UI/obs-frontend-api"
)

find_library(OBS_LIB
    NAMES obs libobs
    PATHS
        "${OBS_ROOT}/lib"
        "${OBS_ROOT}/build/libobs/Release"
        "${OBS_ROOT}/build/libobs"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OBS REQUIRED_VARS OBS_INCLUDE_DIR OBS_LIB)

if(OBS_FOUND)
    add_library(OBS::libobs UNKNOWN IMPORTED)
    set_target_properties(OBS::libobs PROPERTIES
        IMPORTED_LOCATION "${OBS_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${OBS_INCLUDE_DIR}"
    )
endif()
