#include "../ggml/src/ggml-sycl/sycl-arch-compat.hpp"

// Model the combinations of architecture enum members seen in oneAPI SDKs.
// No integer IDs are assumed by the implementation.
enum class old_arch { other, intel_gpu_bmg_g21 };
enum class middle_arch { other, intel_gpu_ptl_h, intel_gpu_ptl_u };
enum class new_arch { other, intel_gpu_bmg_g31, intel_gpu_ptl_h, intel_gpu_ptl_u, intel_gpu_wcl };

static_assert(!ggml_sycl_is_bmg_g31(old_arch::other));
static_assert(!ggml_sycl_is_ptl_h(old_arch::other));
static_assert(!ggml_sycl_is_ptl_u(old_arch::other));
static_assert(!ggml_sycl_is_wcl(old_arch::other));
static_assert(!ggml_sycl_is_bmg_g31(old_arch::intel_gpu_bmg_g21));
static_assert(ggml_sycl_is_ptl_h(middle_arch::intel_gpu_ptl_h));
static_assert(ggml_sycl_is_ptl_u(middle_arch::intel_gpu_ptl_u));
static_assert(!ggml_sycl_is_bmg_g31(middle_arch::other));
static_assert(!ggml_sycl_is_wcl(middle_arch::other));
static_assert(ggml_sycl_is_bmg_g31(new_arch::intel_gpu_bmg_g31));
static_assert(ggml_sycl_is_ptl_h(new_arch::intel_gpu_ptl_h));
static_assert(ggml_sycl_is_ptl_u(new_arch::intel_gpu_ptl_u));
static_assert(ggml_sycl_is_wcl(new_arch::intel_gpu_wcl));
static_assert(!ggml_sycl_is_bmg_g31(new_arch::intel_gpu_ptl_h));
static_assert(!ggml_sycl_is_ptl_h(new_arch::intel_gpu_bmg_g31));
static_assert(!ggml_sycl_is_ptl_u(new_arch::intel_gpu_wcl));
static_assert(!ggml_sycl_is_wcl(new_arch::intel_gpu_ptl_u));

int main() { return 0; }
