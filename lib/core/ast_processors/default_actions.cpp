#include "core/ast_processors/default_actions.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>
namespace {
using namespace oklt;
using namespace clang;

std::string_view getNodeName(const Decl& d) {
    return d.getDeclKindName();
}

std::string_view getNodeName(const Stmt& s) {
    return s.getStmtClassName();
}

template <typename NodeType>
HandleResult runDefaultPreActionImpl(const Attr* attr,
                                     const NodeType& node,
                                     OklSemaCtx&,
                                     SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << getNodeName(node) << '\n';
#endif
    return {};
}

template <typename NodeType>
HandleResult runDefaultPostActionImpl(const Attr* attr,
                                      const NodeType& node,
                                      OklSemaCtx&,
                                      SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << getNodeName(node) << '\n';
#endif
    auto& am = stage.getAttrManager();
    if (!attr) {
        return am.handleNode(node, stage);
    }

    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    return am.handleAttr(*attr, node, &params.value(), stage);
}
}  // namespace
namespace oklt {
HandleResult runDefaultPreActionDecl(const Attr* attr,
                                     const Decl& decl,
                                     OklSemaCtx& sema,
                                     SessionStage& stage) {
    return runDefaultPreActionImpl(attr, decl, sema, stage);
}

HandleResult runDefaultPreActionStmt(const Attr* attr,
                                     const Stmt& stmt,
                                     OklSemaCtx& sema,
                                     SessionStage& stage) {
    return runDefaultPreActionImpl(attr, stmt, sema, stage);
}

HandleResult runDefaultPostActionDecl(const Attr* attr,
                                      const Decl& decl,
                                      OklSemaCtx& sema,
                                      SessionStage& stage) {
    return runDefaultPostActionImpl(attr, decl, sema, stage);
}

HandleResult runDefaultPostActionStmt(const Attr* attr,
                                      const Stmt& stmt,
                                      OklSemaCtx& sema,
                                      SessionStage& stage) {
    return runDefaultPostActionImpl(attr, stmt, sema, stage);
}
}  // namespace oklt
