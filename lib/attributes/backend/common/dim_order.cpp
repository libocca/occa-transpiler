#include <core/attribute_manager/attributed_type_map.h>
#include "attributes/attribute_names.h"
#include "attributes/frontend/params/dim.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <numeric>

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;
using DimOrder = std::vector<size_t>;

HandleResult handleDimOrderDeclAttribute(const clang::Attr& a,
                                         const clang::Decl& decl,
                                         SessionStage& s) {
    llvm::outs() << "handle @dimOrder decl: "
                 << getSourceText(decl.getSourceRange(), s.getCompiler().getASTContext()) << "\n";

    s.getRewriter().RemoveText(getAttrFullSourceRange(a));
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIMORDER_ATTR_NAME, makeSpecificAttrHandle(handleDimOrderDeclAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIMORDER_ATTR_NAME << " attribute decl handler\n";
    }
}
}  // namespace
