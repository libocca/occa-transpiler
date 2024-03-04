#include "attributes/frontend/params/dim.h"
#include <core/attribute_manager/attributed_type_map.h>
#include "attributes/attribute_names.h"
#include "attributes/frontend/params/dim.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <numeric>

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;
using DimOrder = std::vector<size_t>;

HandleResult handleDimDeclAttribute(const clang::Attr& a,
                                    const clang::Decl& decl,
                                    const AttributedDim* params,
                                    SessionStage& s) {
    llvm::outs() << "handle @dim decl: "
                 << getSourceText(decl.getSourceRange(), s.getCompiler().getASTContext()) << "\n";

    // Just remove attribute
    return TranspilationBuilder(s.getCompiler().getSourceManager(), "dim", 1)
        .addReplacement(OKL_TRANSPILED_ATTR, getAttrFullSourceRange(a), "")
        .build();
}

// Given variable, returns dim order (0, 1, .. n) if no @dimOrder, or parses @dimOrder if present
tl::expected<DimOrder, Error> getDimOrder(const clang::DeclRefExpr* var,
                                          const AttributedDim* params,
                                          SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    DimOrder dimOrder(params->dim.size());
    // Default dimensions order - 0, 1, ... n
    std::iota(dimOrder.begin(), dimOrder.end(), 0);

    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto attrs = attrTypeMap.get(ctx, var->getType());
    for (const auto* attr : attrs) {
        auto name = attr->getNormalizedFullName();
        if (name == DIMORDER_ATTR_NAME) {
            auto dimOrderParams = stage.getAttrManager().parseAttr(*attr, stage);
            if (!dimOrderParams) {
                return tl::make_unexpected(dimOrderParams.error());
            }
            dimOrder = std::any_cast<AttributedDimOrder>(dimOrderParams.value()).idx;
            break;
        }
    }
    if (dimOrder.size() != params->dim.size()) {
        return tl::make_unexpected(Error{{}, "[@dimOrder] wrong number of arguments"});
    }

    // Verify if dimensions are correct
    auto n_dims = params->dim.size();
    for (const auto& dim : dimOrder) {
        if (dim >= n_dims) {
            return tl::make_unexpected(Error{
                {},
                util::fmt("[@dimOrder] Dimensions must be in range [0, {}]", n_dims - 1).value()});
        }
    }
    return dimOrder;
}

// Get a list of recovery expressions inside call to dim variable, and validate size
tl::expected<ExprVec, Error> validateDim(const RecoveryExpr& rec,
                                         const AttributedDim& params,
                                         SessionStage& s) {
    auto& ctx = s.getCompiler().getASTContext();
    auto n_dims = params.dim.size();

    auto missing_dim_err = util::fmt("Missing dimensions, expected {} argument(s)", n_dims).value();
    auto too_many_dim_err =
        util::fmt("Too many dimensions, expected {} argument(s)", n_dims).value();

    auto subExpr = rec.subExpressions();
    if (subExpr.size() <= 1) {
        return tl::make_unexpected(Error{{}, missing_dim_err});
    }

    ExprVec recoveryExpressions(subExpr.begin(), subExpr.end());
    auto n_args = recoveryExpressions.size() - 1;  // First element - dim variable

    if (n_args < n_dims) {
        return tl::make_unexpected(Error{{}, missing_dim_err});
    }
    if (n_args > n_dims) {
        return tl::make_unexpected(Error{{}, too_many_dim_err});
    }
    return recoveryExpressions;
}

// TODO: maybe recursion would look better?
std::string buildIndexCalculation(const ExprVec& dimVarArgs,
                                  const AttributedDim* params,
                                  const DimOrder& dimOrder,
                                  SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    int n_dims = params->dim.size();
    std::string indexCalculation;
    // Open brackets
    for (int dim = 0; dim < n_dims - 1; ++dim) {
        auto idx = dimOrder[dim];
        auto dimVarArgStr = getSourceText(*dimVarArgs[idx], ctx);
        indexCalculation += util::fmt("{} + ({} * (", dimVarArgStr, params->dim[idx]).value();
    }
    auto idx = dimOrder[n_dims - 1];
    indexCalculation += getSourceText(*dimVarArgs[idx], ctx);
    // Close brackets
    for (int i = 0; i < 2 * (n_dims - 1); ++i) {
        indexCalculation += ")";
    }
    return indexCalculation;
}

HandleResult handleDimStmtAttribute(const clang::Attr& a,
                                    const clang::RecoveryExpr& rec,
                                    const AttributedDim* params,
                                    SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle @dim stmt: "
                 << getSourceText(rec.getSourceRange(), stage.getCompiler().getASTContext())
                 << " with params: ";
    for (const auto& dim : params->dim) {
        llvm::outs() << dim << ", ";
    }
    llvm::outs() << "\n";
#endif

    auto& ctx = stage.getCompiler().getASTContext();

    auto recoveryExpressions = validateDim(rec, *params, stage);
    if (!recoveryExpressions) {
        return tl::make_unexpected(recoveryExpressions.error());
    }

    auto* dimVarExpr = recoveryExpressions.value()[0];
    auto* dimVarDeclExpr = dyn_cast_or_null<DeclRefExpr>(dimVarExpr);
    if (!dimVarDeclExpr) {
        return tl::make_unexpected(Error{{}, "Failed to cast [@dim] variable Expr to DeclRefExpr"});
    }

    auto dimOrder = getDimOrder(dimVarDeclExpr, params, stage);
    if (!dimOrder) {
        return tl::make_unexpected(dimOrder.error());
    }

    auto dimVarNameStr = getSourceText(*dimVarExpr, ctx);
    ExprVec dimVarArgs(recoveryExpressions.value().begin() + 1, recoveryExpressions.value().end());
    auto indexCalculation = buildIndexCalculation(dimVarArgs, params, dimOrder.value(), stage);

    return TranspilationBuilder(stage.getCompiler().getSourceManager(), "dim", 1)
        .addReplacement(OKL_DIM_ACCESS,
                        rec.getSourceRange(),
                        util::fmt("{}[{}]", dimVarNameStr, indexCalculation).value())
        .build();
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, makeSpecificAttrHandle(handleDimDeclAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute decl handler\n";
    }

    ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, makeSpecificAttrHandle(handleDimStmtAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
