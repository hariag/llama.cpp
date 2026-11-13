#pragma once

#include <sycl/sycl.hpp>

// work_group_static first appeared after the oneAPI 2025.0 headers.  The
// earlier local-memory extension provides the same per-work-group lifetime.
#if __has_include(<sycl/ext/oneapi/work_group_static.hpp>)
#include <sycl/ext/oneapi/work_group_static.hpp>
#define GGML_SYCL_LOCAL_STORAGE(name, size, item) \
    sycl::ext::oneapi::experimental::work_group_static<char[size]> name; \
    auto * name##_ptr = &name
#else
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <cstddef>
template <std::size_t N> struct alignas(16) ggml_sycl_local_storage {
    char bytes[N];
};
#define GGML_SYCL_LOCAL_STORAGE(name, size, item) \
    auto name = sycl::ext::oneapi::group_local_memory_for_overwrite< \
        ggml_sycl_local_storage<size>>((item).get_group()); \
    auto * name##_ptr = name.get()
#endif
