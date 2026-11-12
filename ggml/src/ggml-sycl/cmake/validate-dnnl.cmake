# Validate that the selected oneDNN package really supports the requested
# SYCL GPU target. oneDNN 3.5-3.7 did not export DNNL_GPU_VENDOR in its CMake
# package, even when the vendor is present in its installed configuration header.
function(ggml_sycl_validate_dnnl target_vendor)
    if(NOT DEFINED DNNL_GPU_RUNTIME OR NOT DNNL_GPU_RUNTIME MATCHES "^(SYCL|DPCPP)$")
        message(FATAL_ERROR "oneDNN must use the SYCL GPU runtime; found '${DNNL_GPU_RUNTIME}' at ${DNNL_DIR}")
    endif()

    if(DEFINED DNNL_GPU_VENDOR)
        if(NOT "${DNNL_GPU_VENDOR}" STREQUAL "${target_vendor}")
            message(FATAL_ERROR "oneDNN GPU vendor '${DNNL_GPU_VENDOR}' does not match '${target_vendor}'")
        endif()
    endif()

    if(NOT target_vendor MATCHES "^(INTEL|NVIDIA|AMD)$")
        message(FATAL_ERROR "Cannot validate unsupported oneDNN GPU vendor '${target_vendor}'")
    endif()

    # Compile against the include directories exported by this exact DNNL
    # target, not another installation found independently on the system.
    # A static try-compile avoids needing runtime devices or linking a DLL.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES DNNL::dnnl)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    unset(GGML_DNNL_HEADER_MATCHES_TARGET CACHE)
    check_cxx_source_compiles("
        #include <oneapi/dnnl/dnnl_config.h>
        #if !defined(DNNL_GPU_RUNTIME) || !defined(DNNL_RUNTIME_SYCL) || DNNL_GPU_RUNTIME != DNNL_RUNTIME_SYCL
        #error oneDNN GPU runtime is not SYCL
        #endif
        #if !defined(DNNL_GPU_VENDOR) || !defined(DNNL_VENDOR_${target_vendor}) || DNNL_GPU_VENDOR != DNNL_VENDOR_${target_vendor}
        #error oneDNN GPU vendor does not match
        #endif
        int main() { return 0; }
    " GGML_DNNL_HEADER_MATCHES_TARGET)
    if(NOT GGML_DNNL_HEADER_MATCHES_TARGET)
        message(FATAL_ERROR
            "oneDNN at ${DNNL_DIR} has installed headers that do not confirm "
            "SYCL GPU support for ${target_vendor}")
    endif()
    message(STATUS "oneDNN: installed headers confirm SYCL GPU support for ${target_vendor}")
endfunction()
