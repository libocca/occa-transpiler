#pragma once

#include "core/handler_manager/handler_manager.h"
#include "util/type_traits.h"

#include "util/string_utils.hpp"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>
#include <tl/expected.hpp>

#include <any>
#include <functional>

namespace oklt {

class SessionStage;

class ImplicitHandler : public NodeHandler {
    friend class HandlerMap;

   private:
    using HandlerType = std::function<HandleResult(SessionStage&, const clang::DynTypedNode&)>;
    const HandlerType handler;

    template <class F>
    HandlerType wrapHandler(F&);

   public:
    template <class F>
    explicit ImplicitHandler(F& func)
        : handler(wrapHandler(func)) {
        kind = clang::ASTNodeKind::getFromNodeKind<std::decay_t<func_param_type_t<F, 1>>>();
    };

    HandleResult handle(SessionStage& stage, const clang::DynTypedNode& node) override {
        return handler(stage, node);
    }
};

template <class F>
ImplicitHandler::HandlerType ImplicitHandler::wrapHandler(F& f) {
    constexpr size_t nargs = func_num_arguments<F>::value;
    static_assert(nargs == 2);
    static_assert(std::is_same_v<SessionStage, std::decay_t<func_param_type_t<F, 0>>>);
    static_assert(std::is_base_of_v<clang::Decl, std::decay_t<func_param_type_t<F, 1>>> ||
                  std::is_base_of_v<clang::Stmt, std::decay_t<func_param_type_t<F, 1>>>);

    return HandlerType{[&f](SessionStage& stage, const clang::DynTypedNode& node) -> HandleResult {
        using NodeT = std::decay_t<func_param_type_t<F, 1>>;
        const auto localNode = node.get<NodeT>();
        if (!localNode) {
            auto baseNodeTypeName = node.getNodeKind().asStringRef();
            auto handleNodeTypeName = clang::ASTNodeKind::getFromNodeKind<NodeT>().asStringRef();
            return tl::make_unexpected(Error{
                {},
                util::fmt(
                    "Failed to cast {} to {}", baseNodeTypeName.str(), handleNodeTypeName.str())
                    .value()});
        }

        return f(stage, *localNode);
    }};
}

template <enum HandleType T>
struct HandlerKey<T, std::enable_if_t<T == HandleType::IMPLICIT>> : public HandleKeyBase {
    typedef ImplicitHandler HandlerType;

    template <typename... Ts>
    explicit HandlerKey(Ts&&... params)
        : HandleKeyBase(T) {
        using params_t = std::tuple<Ts&&...>;
        static_assert(
            std::is_same_v<std::decay_t<std::tuple_element_t<0, params_t>>, TargetBackend>);

        auto p = std::forward_as_tuple(params...);
        backend = std::get<0>(p);
        if constexpr (std::tuple_size_v<params_t> == 2) {
            if constexpr (std::is_same_v<std::decay_t<std::tuple_element_t<1, params_t>>,
                                         clang::ASTNodeKind>) {
                kind = std::get<1>(p);
            }
        }
    };
};

template <typename F>
inline bool registerImplicitHandler(TargetBackend backend, F& func) {
    return HandlerManager::_map().insert(HandlerKey<HandleType::IMPLICIT>(backend), func);
}

}  // namespace oklt
