#pragma once

#include <oklt/core/ast_processor_types.h>
#include <oklt/util/string_utils.h>

#include "core/attribute_manager/result.h"
#include "util/type_traits.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

#include <map>

namespace oklt {

class SessionStage;
class OklSemaCtx;

class AstProcessorManager {
   protected:
    AstProcessorManager() = default;
    ~AstProcessorManager() = default;

   public:
    using AttrKeyType = std::tuple<AstProcessorType, std::string>;
    using DefaultKeyType = std::tuple<AstProcessorType>;

    using DeclHandleType = std::function<
        HandleResult(SessionStage&, OklSemaCtx&, const clang::Decl&, const clang::Attr*)>;
    using StmtHandleType = std::function<
        HandleResult(SessionStage&, OklSemaCtx&, const clang::Stmt&, const clang::Attr*)>;

    struct DeclNodeHandle {
        // run in direction from parent to child
        DeclHandleType preAction;
        // run in direction from child to parent
        DeclHandleType postAction;
    };

    struct StmtNodeHandle {
        // run in direction from parent to child
        StmtHandleType preAction;
        // run in direction from child to parent
        StmtHandleType postAction;
    };

    static AstProcessorManager& instance();

    AstProcessorManager(const AstProcessorManager&) = delete;
    AstProcessorManager(AstProcessorManager&&) = delete;
    AstProcessorManager& operator=(const AstProcessorManager&) = delete;
    AstProcessorManager& operator=(AstProcessorManager&&) = delete;

    bool registerDefaultHandle(DefaultKeyType key, DeclNodeHandle handle);
    bool registerDefaultHandle(DefaultKeyType key, StmtNodeHandle handle);

    bool registerSpecificNodeHandle(AttrKeyType key, DeclNodeHandle handle);
    bool registerSpecificNodeHandle(AttrKeyType key, StmtNodeHandle handle);

    HandleResult runPreActionNodeHandle(AstProcessorType procType,
                                        SessionStage& stage,
                                        OklSemaCtx& sema,
                                        const clang::Decl& decl,
                                        const clang::Attr* attr);
    HandleResult runPostActionNodeHandle(AstProcessorType procType,
                                         SessionStage& stage,
                                         OklSemaCtx& sema,
                                         const clang::Decl& decl,
                                         const clang::Attr* attr);
    HandleResult runPreActionNodeHandle(AstProcessorType procType,
                                        SessionStage& stage,
                                        OklSemaCtx& sema,
                                        const clang::Stmt& stmt,
                                        const clang::Attr* attr);
    HandleResult runPostActionNodeHandle(AstProcessorType procType,
                                         SessionStage& stage,
                                         OklSemaCtx& sema,
                                         const clang::Stmt& stmt,
                                         const clang::Attr* attr);

   private:
    std::map<DefaultKeyType, DeclNodeHandle> _defaultDeclHandlers;
    std::map<DefaultKeyType, StmtNodeHandle> _defaultStmtHandlers;
    std::map<AttrKeyType, DeclNodeHandle> _declHandlers;
    std::map<AttrKeyType, StmtNodeHandle> _stmtHandlers;
};

// INFO: helper functions to register specific Decl/Stmt handler in Ast Processor Manager
//  so far 2 separate helpers for decl and stmt because no time to play with meta programming
//  USE only by existing examples other cases are not tested!!!
namespace detail {
constexpr size_t HANDLE_NUM_OF_ARGS = 4;
template <typename Handler, typename NodeType, typename HandleType>
HandleType makeSpecificSemaXXXHandle(Handler& handler) {
    using ExprType = typename std::remove_reference_t<typename func_param_type<Handler, 3>::type>;
    constexpr size_t n_arguments = func_num_arguments<Handler>::value;

    return HandleType{[&handler, n_arguments](SessionStage& stage,
                                              OklSemaCtx& sema,
                                              const NodeType& node,
                                              const clang::Attr* attr) -> HandleResult {
        static_assert(n_arguments == HANDLE_NUM_OF_ARGS, "Handler must have 4 arguments");
        if (!attr) {
            auto handleNodeTypeName = typeid(ExprType).name();
            return tl::make_unexpected(
                Error{{}, util::fmt("nullptr attr for {}", handleNodeTypeName).value()});
        }

        const auto localNode = clang::dyn_cast_or_null<ExprType>(&node);
        if (!localNode) {
            auto baseNodeTypeName = typeid(NodeType).name();
            auto handleNodeTypeName = typeid(ExprType).name();
            return tl::make_unexpected(
                Error{{},
                      util::fmt("Failed to cast {} to {}", baseNodeTypeName, handleNodeTypeName)
                          .value()});
        }
        return handler(stage, sema, *localNode, *attr);
    }};
};

// TODO: maybe remove dublication with makeSpecificSemaXXXHandle
template <typename Handler, typename NodeType, typename HandleType>
HandleType makeDefaultSemaXXXHandle(Handler& handler) {
    using ExprType = typename std::remove_reference_t<typename func_param_type<Handler, 3>::type>;
    constexpr size_t n_arguments = func_num_arguments<Handler>::value;

    return HandleType{[&handler, n_arguments](SessionStage& stage,
                                              OklSemaCtx& sema,
                                              const NodeType& node,
                                              const clang::Attr* attr) -> HandleResult {
        static_assert(n_arguments == HANDLE_NUM_OF_ARGS, "Handler must have 4 arguments");
        const auto localNode = clang::dyn_cast_or_null<ExprType>(&node);
        if (!localNode) {
            auto baseNodeTypeName = typeid(NodeType).name();
            auto handleNodeTypeName = typeid(ExprType).name();
            return tl::make_unexpected(
                Error{{},
                      util::fmt("Failed to cast {} to {}", baseNodeTypeName, handleNodeTypeName)
                          .value()});
        }
        return handler(stage, sema, *localNode, attr);
    }};
};

}  // namespace detail

template <typename Handler>
auto makeSpecificSemaHandle(Handler& handler) {
    using DeclOrStmt = typename std::remove_const_t<
        typename std::remove_reference_t<typename func_param_type<Handler, 3>::type>>;

    if constexpr (std::is_base_of_v<clang::Decl, DeclOrStmt>) {
        return detail::makeSpecificSemaXXXHandle<Handler,
                                                 clang::Decl,
                                                 AstProcessorManager::DeclHandleType>(handler);
    } else {
        return detail::makeSpecificSemaXXXHandle<Handler,
                                                 clang::Stmt,
                                                 AstProcessorManager::StmtHandleType>(handler);
    }
}

template <typename Handler>
auto makeDefaultSemaHandle(Handler& handler) {
    using DeclOrStmt = typename std::remove_const_t<
        typename std::remove_reference_t<typename func_param_type<Handler, 3>::type>>;

    if constexpr (std::is_base_of_v<clang::Decl, DeclOrStmt>) {
        return detail::makeDefaultSemaXXXHandle<Handler,
                                                clang::Decl,
                                                AstProcessorManager::DeclHandleType>(handler);
    } else {
        return detail::makeDefaultSemaXXXHandle<Handler,
                                                clang::Stmt,
                                                AstProcessorManager::StmtHandleType>(handler);
    }
}

}  // namespace oklt
