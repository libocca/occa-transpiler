#include <core/attribute_manager/attributed_type_map.h>
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Expr.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;
using DimOrder = std::vector<size_t>;

HandleResult handleDimOrderDeclAttribute(SessionStage& s,
                                         const clang::Decl& decl,
                                         const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@dimOrder] decl: {}",
                 getSourceText(decl.getSourceRange(), decl.getASTContext()));
    removeAttribute(s, a);
    return {};
}

HandleResult handleDimOrderStmtAttribute(SessionStage& s,
                                         const clang::Stmt& stmt,
                                         const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty stmt [@dimOrder] handler");
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        {DIM_ORDER_ATTR_NAME, ASTNodeKind::getFromNodeKind<Decl>()},
        makeSpecificAttrHandle(handleDimOrderDeclAttribute));

    ok = ok && oklt::AttributeManager::instance().registerCommonHandler(
                   {DIM_ORDER_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
                   makeSpecificAttrHandle(handleDimOrderStmtAttribute));
    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute stmt handler", DIM_ORDER_ATTR_NAME);
    }
}
}  // namespace
