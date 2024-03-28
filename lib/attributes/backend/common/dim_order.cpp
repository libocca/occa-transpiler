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

HandleResult handleDimOrderDeclAttribute(const clang::Attr& a,
                                         const clang::Decl& decl,
                                         SessionStage& s) {
    SPDLOG_DEBUG("Handle [@dimOrder] decl: {}",
                 getSourceText(decl.getSourceRange(), decl.getASTContext()));
    removeAttribute(a, s);
    return {};
}

HandleResult handleDimOrderStmtAttribute(const clang::Attr& a,
                                         const clang::Stmt& stmt,
                                         SessionStage& s) {
    SPDLOG_DEBUG("Called empty stmt [@dimOrder] handler");
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIMORDER_ATTR_NAME, makeSpecificAttrHandle(handleDimOrderDeclAttribute));

    ok = ok && oklt::AttributeManager::instance().registerCommonHandler(
                   DIMORDER_ATTR_NAME, makeSpecificAttrHandle(handleDimOrderStmtAttribute));
    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute stmt handler", DIMORDER_ATTR_NAME);
    }
}
}  // namespace
