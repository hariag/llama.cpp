#pragma once

#include <type_traits>

// Some oneAPI releases do not declare newer architecture enum members. Use
// detection on the actual enum instead of compiler-version guesses: SDKs may
// backport device support independently of the compiler version.
#define GGML_SYCL_OPTIONAL_ARCH_MATCHER(function_name, member_name)                         \
    template <typename Arch, typename = void>                                               \
    struct function_name##_impl {                                                           \
        static constexpr bool matches(Arch) { return false; }                               \
    };                                                                                      \
    template <typename Arch>                                                                \
    struct function_name##_impl<Arch, std::void_t<decltype(Arch::member_name)>> {            \
        static constexpr bool matches(Arch arch) { return arch == Arch::member_name; }       \
    };                                                                                      \
    template <typename Arch>                                                                \
    constexpr bool function_name(Arch arch) { return function_name##_impl<Arch>::matches(arch); }

GGML_SYCL_OPTIONAL_ARCH_MATCHER(ggml_sycl_is_bmg_g31, intel_gpu_bmg_g31)
GGML_SYCL_OPTIONAL_ARCH_MATCHER(ggml_sycl_is_ptl_h, intel_gpu_ptl_h)
GGML_SYCL_OPTIONAL_ARCH_MATCHER(ggml_sycl_is_ptl_u, intel_gpu_ptl_u)
GGML_SYCL_OPTIONAL_ARCH_MATCHER(ggml_sycl_is_wcl, intel_gpu_wcl)

#undef GGML_SYCL_OPTIONAL_ARCH_MATCHER
