#include "attributes/frontend/params/dim.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "core/handler_manager/attr_handler.h"
#include "core/transpiler_session/attributed_type_map.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;
using DimOrder = std::vector<size_t>;

HandleResult handleDimDeclAttribute(SessionStage& s,
                                    const clang::Decl& decl,
                                    const clang::Attr& a,
                                    const AttributedDim* params) {
    SPDLOG_DEBUG("Handle [@dim] decl: {}",
                 getSourceText(decl.getSourceRange(), decl.getASTContext()));
    s.getRewriter().RemoveText(getAttrFullSourceRange(a));
    return {};
}

// Given variable, returns dim order (0, 1, .. n) if no @dimOrder, or parses @dimOrder if present
tl::expected<DimOrder, Error> getDimOrder(SessionStage& stage,
                                          const clang::DeclRefExpr* var,
                                          const AttributedDim* params) {
    auto& ctx = stage.getCompiler().getASTContext();
    DimOrder dimOrder(params->dim.size());
    // Default dimensions order - 0, 1, ... n
    std::iota(dimOrder.begin(), dimOrder.end(), 0);

    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto attrs = attrTypeMap.get(ctx, var->getType());
    auto dimOrderAttr = std::find_if(attrs.begin(), attrs.end(), [](const auto* attr) {
        auto name = attr->getNormalizedFullName();
        return (name == DIM_ORDER_ATTR_NAME);
    });
    auto status = [&]() -> tl::expected<void, Error> {
        if (dimOrderAttr != attrs.end()) {
            auto dimOrderParams = stage.getAttrManager().parseAttr(stage, **dimOrderAttr);
            if (!dimOrderParams) {
                return tl::make_unexpected(dimOrderParams.error());
            }
            auto* attributedDimOrder = std::any_cast<AttributedDimOrder>(&(dimOrderParams.value()));
            if (!attributedDimOrder) {
                return tl::make_unexpected(
                    Error{{}, "Failed to cast @dimOrder parameters to AttributedDimOrder"});
            }
            dimOrder = attributedDimOrder->idx;
        }
        if (dimOrder.size() != params->dim.size()) {
            return tl::make_unexpected(Error{{}, "[@dimOrder] wrong number of arguments"});
        }

        // Verify if dimensions are correct
        auto nDims = params->dim.size();
        for (const auto& dim : dimOrder) {
            if (dim >= nDims) {
                return tl::make_unexpected(
                    Error{{},
                          util::fmt("[@dimOrder] Dimensions must be in range [0, {}]", nDims - 1)
                              .value()});
            }
        }
        return {};
    }();
    if (!status) {
        if (dimOrderAttr != attrs.end()) {
            status.error().ctx = (*dimOrderAttr)->getRange();
        }
        return tl::make_unexpected(std::move(status.error()));
    }
    return dimOrder;
}

// Get a list of recovery expressions inside call to dim variable, and validate size
tl::expected<ExprVec, Error> validateDim(SessionStage& s,
                                         const RecoveryExpr& rec,
                                         const AttributedDim& params) {
    auto& ctx = s.getCompiler().getASTContext();
    auto nDims = params.dim.size();

    auto missingDimErr = util::fmt("Missing dimensions, expected {} argument(s)", nDims).value();
    auto tooManyDimErr = util::fmt("Too many dimensions, expected {} argument(s)", nDims).value();

    auto subExpr = rec.subExpressions();
    if (subExpr.size() <= 1) {
        return tl::make_unexpected(Error{{}, missingDimErr});
    }

    ExprVec expressions(subExpr.begin(), subExpr.end());
    auto nArgs = expressions.size() - 1;  // First element - dim variable

    if (nArgs < nDims) {
        return tl::make_unexpected(Error{{}, missingDimErr});
    }
    if (nArgs > nDims) {
        return tl::make_unexpected(Error{{}, tooManyDimErr});
    }
    return expressions;
}

tl::expected<ExprVec, Error> validateDim(SessionStage& s,
                                         const CallExpr& expr,
                                         const AttributedDim& params) {
    auto& ctx = s.getCompiler().getASTContext();
    auto nDims = params.dim.size();

    auto missingDimErr = util::fmt("Missing dimensions, expected {} argument(s)", nDims).value();
    auto tooManyDimErr = util::fmt("Too many dimensions, expected {} argument(s)", nDims).value();

    auto args = expr.getArgs();
    if (expr.getNumArgs() == 0 || args == nullptr) {
        return tl::make_unexpected(Error{{}, missingDimErr});
    }

    ExprVec expressions;
    expressions.push_back(expr.getCallee());
    for (size_t i = 0; i < expr.getNumArgs(); ++i) {
        expressions.push_back(args[i]);
    }
    auto nArgs = expressions.size() - 1;  // First element - dim variable

    if (nArgs < nDims) {
        return tl::make_unexpected(Error{{}, missingDimErr});
    }
    if (nArgs > nDims) {
        return tl::make_unexpected(Error{{}, tooManyDimErr});
    }
    return expressions;
}

std::string buildIndexCalculation(SessionStage& stage,
                                  const ExprVec& dimVarArgs,
                                  const AttributedDim* params,
                                  const DimOrder& dimOrder) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto& rewriter = stage.getRewriter();
    int nDims = params->dim.size();
    std::string indexCalculation;
    // Open brackets
    for (int dim = 0; dim < nDims - 1; ++dim) {
        auto idx = dimOrder[dim];
        auto dimVarArgStr = getLatestSourceText(*dimVarArgs[idx], rewriter);
        indexCalculation += util::fmt("{} + ({} * (", dimVarArgStr, params->dim[idx]).value();
    }
    auto idx = dimOrder[nDims - 1];
    indexCalculation += getLatestSourceText(*dimVarArgs[idx], rewriter);
    // Close brackets
    for (int i = 0; i < 2 * (nDims - 1); ++i) {
        indexCalculation += ")";
    }
    return indexCalculation;
}

HandleResult handleDimStmtAttribute(SessionStage& stage,
                                    const clang::Stmt& stmt,
                                    const clang::Attr& a,
                                    const AttributedDim* params) {
    if (!isa<RecoveryExpr, CallExpr>(stmt)) {
        return {};
    }

    SPDLOG_DEBUG("Handle [@dim] stmt: {}",
                 getSourceText(stmt.getSourceRange(), stage.getCompiler().getASTContext()));

    auto& ctx = stage.getCompiler().getASTContext();

    // TODO: Error from validateDim points to @dim, but should point to variable
    auto expressions = [&]() -> tl::expected<ExprVec, Error> {
        // Dispatch statement
        if (isa<RecoveryExpr>(stmt)) {
            return validateDim(stage, *dyn_cast<RecoveryExpr>(&stmt), *params);
        }
        if (isa<CallExpr>(stmt)) {
            return validateDim(stage, *dyn_cast<CallExpr>(&stmt), *params);
        }
        return tl::make_unexpected(Error{{}, "Incorrect statement type for [@dim]"});
    }();
    if (!expressions) {
        return tl::make_unexpected(std::move(expressions.error()));
    }

    auto* dimVarExpr = expressions.value()[0];
    auto* dimVarDeclExpr = dyn_cast_or_null<DeclRefExpr>(dimVarExpr);
    if (!dimVarDeclExpr) {
        return tl::make_unexpected(Error{{}, "Failed to cast [@dim] variable Expr to DeclRefExpr"});
    }

    auto dimOrder = getDimOrder(stage, dimVarDeclExpr, params);
    if (!dimOrder) {
        return tl::make_unexpected(std::move(dimOrder.error()));
    }

    auto dimVarNameStr = getSourceText(*dimVarExpr, ctx);
    ExprVec dimVarArgs(expressions.value().begin() + 1, expressions.value().end());
    auto indexCalculation = buildIndexCalculation(stage, dimVarArgs, params, dimOrder.value());

    stage.getRewriter().ReplaceText(stmt.getSourceRange(),
                                    util::fmt("{}[{}]", dimVarNameStr, indexCalculation).value());
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerCommonHandler(DIM_ATTR_NAME, handleDimDeclAttribute);
    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute decl handler", DIM_ATTR_NAME);
    }

    ok = registerCommonHandler(DIM_ATTR_NAME, handleDimStmtAttribute);
    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute stmt handler", DIM_ATTR_NAME);
    }
}
}  // namespace
