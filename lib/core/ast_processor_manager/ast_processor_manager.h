#pragma once

#include <oklt/core/ast_processor_types.h>
#include "util/type_traits.h"

#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

#include <map>

namespace oklt {

struct SessionStage;

class AstProcessorManager {
   protected:
    AstProcessorManager() = default;
    ~AstProcessorManager() = default;

   public:
    using KeyType = std::tuple<AstProcessorType, int>;
    using DeclHandleType = std::function<bool(const clang::Decl*, SessionStage&)>;
    using StmtHandleType = std::function<bool(const clang::Stmt*, SessionStage&)>;

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

    using DeclNodeHandlers = std::map<KeyType, DeclNodeHandle>;
    using StmtNodeHandlers = std::map<KeyType, StmtNodeHandle>;

    static AstProcessorManager& instance();

    AstProcessorManager(const AstProcessorManager&) = delete;
    AstProcessorManager(AstProcessorManager&&) = delete;
    AstProcessorManager& operator=(const AstProcessorManager&) = delete;
    AstProcessorManager& operator=(AstProcessorManager&&) = delete;

    bool registerGenericHandle(AstProcessorType procType, DeclNodeHandle handle);
    bool registerGenericHandle(AstProcessorType procType, StmtNodeHandle handle);

    bool registerSpecificNodeHandle(KeyType key, DeclNodeHandle handle);
    bool registerSpecificNodeHandle(KeyType key, StmtNodeHandle handle);

    bool runPreActionNodeHandle(AstProcessorType procType,
                                const clang::Decl* decl,
                                SessionStage& stage);
    bool runPostActionNodeHandle(AstProcessorType procType,
                                 const clang::Decl* decl,
                                 SessionStage& stage);
    bool runPreActionNodeHandle(AstProcessorType procType,
                                const clang::Stmt* stmt,
                                SessionStage& stage);
    bool runPostActionNodeHandle(AstProcessorType procType,
                                 const clang::Stmt* stmt,
                                 SessionStage& stage);

   private:
    std::map<AstProcessorType, DeclNodeHandle> _genericDeclHandle;
    std::map<AstProcessorType, StmtNodeHandle> _genericStmtHandle;
    std::map<KeyType, DeclNodeHandle> _declHandlers;
    std::map<KeyType, StmtNodeHandle> _stmtHandlers;
};

// INFO: helper functions to register specific Decl/Stmt handler in Ast Processor Manager
//  so far 2 separate helpers for decl and stmt because no time to play with meta programming
//  USE only by existing examples other cases are not tested!!!
template <typename DeclHandleType>
AstProcessorManager::DeclNodeHandle makeSpecificDeclHandle(DeclHandleType& preAction,
                                                           DeclHandleType& postAction) {
    using SpecificDeclType =
        typename std::remove_pointer_t<typename func_param_type<DeclHandleType, 1>::type>;

    return AstProcessorManager::DeclNodeHandle{
        .preAction =
            [&preAction](const clang::Decl* decl, SessionStage& stage) {
                return preAction(clang::dyn_cast_or_null<SpecificDeclType>(decl), stage);
            },
        .postAction =
            [&postAction](const clang::Decl* decl, SessionStage& stage) {
                return postAction(clang::dyn_cast_or_null<SpecificDeclType>(decl), stage);
            }};
}

template <typename StmtHandleType>
AstProcessorManager::StmtNodeHandle makeSpecificStmtHandle(StmtHandleType& preAction,
                                                           StmtHandleType& postAction) {
    using SpecificStmtType =
        std::remove_pointer_t<typename func_param_type<StmtHandleType, 1>::type>;

    return AstProcessorManager::StmtNodeHandle{
        .preAction =
            [&preAction](const clang::Stmt* stmt, SessionStage& stage) {
                return preAction(clang::dyn_cast_or_null<SpecificStmtType>(stmt), stage);
            },
        .postAction =
            [&postAction](const clang::Stmt* stmt, SessionStage& stage) {
                return postAction(clang::dyn_cast_or_null<SpecificStmtType>(stmt), stage);
            }};
}
}  // namespace oklt
