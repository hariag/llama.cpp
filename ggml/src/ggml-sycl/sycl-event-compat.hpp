#pragma once

#include <type_traits>
#include <utility>
#include <vector>

// oneAPI SDK versions return either an event directly or an optional event.
// An empty optional means that no prior event needs to be waited on.
template <typename Event, typename LastEvent>
static inline std::vector<Event> ggml_sycl_last_event_dependencies(LastEvent && last) {
    if constexpr (std::is_same_v<std::decay_t<LastEvent>, Event>) {
        return { std::forward<LastEvent>(last) };
    } else {
        return last ? std::vector<Event>{ *last } : std::vector<Event>{};
    }
}
