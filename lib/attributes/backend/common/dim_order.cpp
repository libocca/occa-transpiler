#include <core/attribute_manager/attributed_type_map.h>
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Expr.h>

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;
using DimOrder = std::vector<size_t>;

HandleResult handleDimOrderDeclAttribute(const clang::Attr& a,
                                         const clang::Decl& decl,
                                         SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle @dimOrder decl: "
                 << getSourceText(decl.getSourceRange(), s.getCompiler().getASTContext()) << "\n";
#endif
    removeAttribute(a, s);
    return {};
}

HandleResult handleDimOrderStmtAttribute(const clang::Attr& a,
                                         const clang::Stmt& stmt,
                                         SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "Called empty stmt [@dimOrder] handler\n";
#endif
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIMORDER_ATTR_NAME, makeSpecificAttrHandle(handleDimOrderDeclAttribute));

    ok = ok && oklt::AttributeManager::instance().registerCommonHandler(
                   DIMORDER_ATTR_NAME, makeSpecificAttrHandle(handleDimOrderStmtAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIMORDER_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
