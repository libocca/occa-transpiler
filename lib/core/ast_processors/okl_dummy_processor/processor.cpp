#include <oklt/util/string_utils.h>
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

HandleResult runPreActionDecl(const Attr*, const Decl& decl, OklSemaCtx&, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif
    return {};
}

HandleResult runPostActionDecl(const Attr* attr,
                               const Decl& decl,
                               OklSemaCtx&,
                               SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif

    auto& am = stage.getAttrManager();
    if (!attr) {
        return am.handleDecl(decl, stage);
    }

    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    return am.handleAttr(*attr, decl, &params.value(), stage);
}

HandleResult runPreActionStmt(const Attr*, const Stmt& stmt, OklSemaCtx&, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    return {};
}

HandleResult runPostActionStmt(const Attr* attr,
                               const Stmt& stmt,
                               OklSemaCtx&,
                               SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    auto& am = stage.getAttrManager();
    if (!attr) {
        return am.handleStmt(stmt, stage);
    }

    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    return am.handleAttr(*attr, stmt, &params.value(), stage);
}

__attribute__((constructor)) void registerAstNodeHanlder() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    auto ok = mng.registerDefaultHandle(
        {AstProcessorType::OKL_NO_SEMA},
        DeclHandle{.preAction = runPreActionDecl, .postAction = runPostActionDecl});
    assert(ok);

    ok = mng.registerDefaultHandle(
        AstProcessorType::OKL_NO_SEMA,
        StmtHandle{.preAction = runPreActionStmt, .postAction = runPostActionStmt});
    assert(ok);
}
}  // namespace
