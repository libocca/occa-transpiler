#pragma once

#include <tuple>

namespace oklt {
template <typename x_Function>
struct function_traits;

// specialization for functions
template <typename x_Result, typename... x_Args>
struct function_traits<x_Result(x_Args...)> {
    using arguments = ::std::tuple<x_Args...>;
};

template <typename FuncType, std::size_t I>
struct func_param_type {
    using type =
        typename std::tuple_element_t<I - 1, typename function_traits<FuncType>::arguments>;
};

template <typename FuncType>
struct func_num_arguments {
    static constexpr size_t value =
        std::tuple_size_v<typename function_traits<FuncType>::arguments>;
};
}  // namespace oklt
