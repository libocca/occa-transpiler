#pragma once

#include "core/handler_manager/handler_manager.h"
#include "util/type_traits.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include "util/string_utils.hpp"
#include <tl/expected.hpp>

#include <any>
#include <functional>

namespace oklt {

class SessionStage;

class SemaHandler : public NodeHandler {
   private:
    using HandlerType =
        std::function<HandleResult(SessionStage&, const clang::DynTypedNode&, const clang::Attr*)>;
    const HandlerType preHandler;
    const HandlerType postHandler;

    template <class F>
    HandlerType wrapHandler(F&);

   public:
    template <class F>
    explicit SemaHandler(F& pre, F& post)
        : preHandler(wrapHandler(pre)),
          postHandler(wrapHandler(post)) {
        kind = clang::ASTNodeKind::getFromNodeKind<std::decay_t<func_param_type_t<F, 1>>>();
    };

    HandleResult pre(SessionStage& stage,
                     const clang::DynTypedNode& node,
                     const clang::Attr* attr) override {
        return preHandler(stage, node, attr);
    }

    HandleResult post(SessionStage& stage,
                      const clang::DynTypedNode& node,
                      const clang::Attr* attr) override {
        return postHandler(stage, node, attr);
    }
};

template <class F>
SemaHandler::HandlerType SemaHandler::wrapHandler(F& f) {
    constexpr size_t nargs = func_num_arguments<F>::value;
    static_assert(nargs == 3);
    static_assert(std::is_same_v<SessionStage, std::decay_t<func_param_type_t<F, 0>>>);
    static_assert(std::is_base_of_v<clang::Decl, std::decay_t<func_param_type_t<F, 1>>> ||
                  std::is_base_of_v<clang::Stmt, std::decay_t<func_param_type_t<F, 1>>>);
    static_assert(
        std::is_same_v<const clang::Attr,
                       std::remove_pointer_t<std::remove_reference_t<func_param_type_t<F, 2>>>>);

    return HandlerType{[&f](SessionStage& stage,
                            const clang::DynTypedNode& node,
                            const clang::Attr* attr) -> HandleResult {
        using NodeT = std::remove_reference_t<func_param_type_t<F, 1>>;
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

        if constexpr (!std::is_reference_v<func_param_type_t<F, 2>>) {
            return f(stage, *localNode, attr);
        } else {
            if (!attr) {
                auto NodeType = clang::ASTNodeKind::getFromNodeKind<NodeT>().asStringRef();
                return tl::make_unexpected(
                    Error{{}, util::fmt("nullptr attr for {}", NodeType.str()).value()});
            }
            return f(stage, *localNode, *attr);
        }
    }};
};

template <enum HandleType H>
struct HandlerKey<H, std::enable_if_t<H == HandleType::SEMA>> : public HandleKeyBase {
    typedef SemaHandler HandlerType;

    template <typename... Ts>
    explicit HandlerKey(Ts&&... params)
        : HandleKeyBase(H) {
        using params_t = std::tuple<Ts&&...>;
        static_assert(std::is_same_v<std::decay_t<std::tuple_element_t<0, params_t>>, std::string>);

        auto p = std::forward_as_tuple(params...);
        attr = std::get<0>(p);
        if constexpr (std::tuple_size_v<params_t> == 2) {
            if constexpr (std::is_same_v<std::decay_t<std::tuple_element_t<1, params_t>>,
                                         clang::ASTNodeKind>) {
                kind = std::get<1>(p);
            }
        }
    }
};

template <typename F>
inline bool registerSemaHandler(std::string attr, F& pre, F& post) {
    return HandlerManager::_map().insert(HandlerKey<HandleType::SEMA>(attr), pre, post);
}

}  // namespace oklt
