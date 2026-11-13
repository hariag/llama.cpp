#include "../ggml/src/ggml-sycl/sycl-event-compat.hpp"

#include <cassert>
#include <optional>

struct event { int id; };

int main() {
    const event first{42};
    const auto direct = ggml_sycl_last_event_dependencies<event>(first);
    assert(direct.size() == 1 && direct[0].id == 42);
    const auto present = ggml_sycl_last_event_dependencies<event>(std::optional<event>{first});
    assert(present.size() == 1 && present[0].id == 42);
    assert(ggml_sycl_last_event_dependencies<event>(std::optional<event>{}).empty());
    const event * ptr = &first;
    assert(ggml_sycl_last_event_dependencies<event>(ptr)[0].id == 42);
    ptr = nullptr;
    assert(ggml_sycl_last_event_dependencies<event>(ptr).empty());
}
