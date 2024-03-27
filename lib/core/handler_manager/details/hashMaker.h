#pragma once

namespace oklt::hm::detail {
template <typename T>
std::size_t makeHash(T v) {
    return std::hash<T>()(v);
}

template <typename T, typename... Args>
std::size_t makeHash(T first, Args... args) {
    std::size_t h = std::hash<T>()(first) ^ (makeHash(std::forward<Args>(args)...) << 1);
    return h;
}
}  // namespace oklt::hm::detail
