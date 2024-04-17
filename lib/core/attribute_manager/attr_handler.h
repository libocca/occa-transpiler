#pragma once

#include "core/attribute_manager/attribute_manager.h"
#include "util/type_traits.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <oklt/util/string_utils.h>
#include <tl/expected.hpp>

#include <any>
#include <functional>

namespace oklt {

class SessionStage;

class AttrHandler : public NodeHandler {
   private:
    using HandleType = std::function<HandleResult(SessionStage&,
                                                  const clang::DynTypedNode&,
                                                  const clang::Attr&,
                                                  const std::any*)>;
    const HandleType handler;

    template <class F>
    HandleType wrapHandler(F&);

   public:
    template <class F>
    explicit AttrHandler(F& func)
        : handler(wrapHandler(func)) {
        kind = clang::ASTNodeKind::getFromNodeKind<std::decay_t<func_param_type_t<F, 1>>>();
    };

    HandleResult handle(SessionStage& stage,
                        const clang::DynTypedNode& node,
                        const clang::Attr& attr,
                        const std::any* param) override {
        return handler(stage, node, attr, param);
    }
};

template <class F>
AttrHandler::HandleType AttrHandler::wrapHandler(F& f) {
    constexpr size_t nargs = func_num_arguments<F>::value;
    static_assert(nargs == 3 || nargs == 4);
    static_assert(std::is_same_v<SessionStage, std::decay_t<func_param_type_t<F, 0>>>);
    static_assert(std::is_base_of_v<clang::Decl, std::decay_t<func_param_type_t<F, 1>>> ||
                  std::is_base_of_v<clang::Stmt, std::decay_t<func_param_type_t<F, 1>>>);
    static_assert(std::is_same_v<clang::Attr, std::decay_t<func_param_type_t<F, 2>>>);

    return HandleType{[&f, nargs](SessionStage& stage,
                                  const clang::DynTypedNode& node,
                                  const clang::Attr& attr,
                                  const std::any* params) -> HandleResult {
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

        if constexpr (nargs == 3) {
            return f(stage, *localNode, attr);
        } else {
            if constexpr (std::is_same_v<func_param_type_t<F, 3>, const std::any*>) {
                return f(stage, *localNode, attr, params);
            } else {
                using ParamsType = std::remove_pointer_t<std::remove_cv_t<func_param_type_t<F, 3>>>;
                const ParamsType* p = params->type() == typeid(ParamsType)
                                          ? std::any_cast<ParamsType>(params)
                                          : nullptr;
                if (!p) {
                    auto tn = typeid(ParamsType).name();
                    return tl::make_unexpected(Error{
                        {},
                        util::fmt("Any cast fail: failed to cast to {}", typeid(ParamsType).name())
                            .value()});
                }
                return f(stage, *localNode, attr, p);
            }
        }
    }};
};

template <enum HandleType H>
struct HandlerKey<H,
                  std::enable_if_t<H == HandleType::COMMON || H == HandleType::SEMA_PRE ||
                                   H == HandleType::SEMA_POST>> : public HandleKeyBase {
    typedef AttrHandler HandlerType;

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
inline bool AttributeManager::registerCommonHandler(std::string attr, F& func) {
    return _handlers.insert(HandlerKey<HandleType::COMMON>(attr), func);
};

}  // namespace oklt
