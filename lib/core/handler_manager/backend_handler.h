#pragma once

#include "core/handler_manager/attr_handler.h"

namespace oklt {

using BackendHandler = AttrHandler;

template <enum HandleType T>
struct HandlerKey<T, std::enable_if_t<T == HandleType::BACKEND>> : public HandleKeyBase {
    typedef BackendHandler HandlerType;

    template <typename... Ts>
    explicit HandlerKey(Ts&&... params)
        : HandleKeyBase(T) {
        using params_t = std::tuple<Ts&&...>;
        static_assert(
            std::is_same_v<std::decay_t<std::tuple_element_t<0, params_t>>, TargetBackend>);
        static_assert(std::is_same_v<std::decay_t<std::tuple_element_t<1, params_t>>, std::string>);

        auto p = std::forward_as_tuple(params...);
        backend = std::get<0>(p);
        attr = std::get<1>(p);
        if constexpr (std::tuple_size_v<params_t> == 3) {
            if constexpr (std::is_same_v<std::decay_t<std::tuple_element_t<2, params_t>>,
                                         clang::ASTNodeKind>) {
                kind = std::get<2>(p);
            }
        }
    }
};

template <typename F>
inline bool HandlerManager::registerBackendHandler(TargetBackend backend,
                                                   std::string attr,
                                                   F& func) {
    return _map().insert(HandlerKey<HandleType::BACKEND>(backend, attr), func);
};

}  // namespace oklt
