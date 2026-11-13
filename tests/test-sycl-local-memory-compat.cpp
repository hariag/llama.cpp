#include "../ggml/src/ggml-sycl/sycl-local-memory-compat.hpp"

int main() {
    sycl::queue queue;
    int output[64] = {};
    {
        sycl::buffer<int, 1> result(output, sycl::range<1>{64});
        queue.submit([&](sycl::handler & cgh) {
            auto out = result.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{64}, sycl::range<1>{16}},
                             [=](sycl::nd_item<1> item) {
                GGML_SYCL_LOCAL_STORAGE(lsm, 256, item);
                auto * data = reinterpret_cast<int *>(lsm_ptr);
                data[item.get_local_linear_id()] = int(item.get_global_linear_id());
                item.barrier(sycl::access::fence_space::local_space);
                out[item.get_global_linear_id()] = data[(item.get_local_linear_id() + 1) % 16];
            });
        });
        queue.wait_and_throw();
    }
    for (int i = 0; i < 64; ++i) {
        if (output[i] != (i / 16) * 16 + (i + 1) % 16) return 1;
    }
}
