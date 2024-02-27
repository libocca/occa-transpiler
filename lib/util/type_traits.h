#pragma once

#include <string>
#include <tuple>
#include <type_traits>

namespace oklt {

template <typename...>
struct is_one_of {
    static constexpr bool value = false;
};

template <typename F, typename S, typename... T>
struct is_one_of<F, S, T...> {
    static constexpr bool value = std::is_same<F, S>::value || is_one_of<F, T...>::value;
};

template <typename F, typename S, typename... T>
inline constexpr bool is_one_of_v = is_one_of<F, S, T...>::value;

template <typename T>
struct is_string {
    static constexpr bool value =
        is_one_of_v<T, std::string, std::wstring, std::u16string, std::u32string>;
};

template <typename T>
inline constexpr bool is_string_v = is_string<T>::value;

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
    static constexpr size_t value = std::tuple_size_v<typename function_traits<FuncType>::arguments>;
};

}  // namespace oklt
