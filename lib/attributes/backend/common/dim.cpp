#include "attributes/frontend/params/dim.h"
#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

namespace {
using namespace oklt;
using namespace clang;
using ExprVec = std::vector<const Expr*>;

HandleResult handleDimDeclAttrbute(const clang::Attr& a,
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

//
tl::expected<ExprVec, Error> validateDim(const RecoveryExpr* rec,
                                         const AttributedDim* params,
                                         SessionStage& s) {
    auto& ctx = s.getCompiler().getASTContext();
    auto n_dims = params->dim.size();

    auto missing_dim_err = util::fmt("Missing dimensions, expected {} argument(s)", n_dims).value();
    auto too_many_dim_err =
        util::fmt("Too many dimensions, expected {} argument(s)", n_dims).value();

    auto subExpr = rec->subExpressions();
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
                                  SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    int n_dims = params->dim.size();
    std::string indexCalculation;
    // Open brackets
    for (int dim = 0; dim < n_dims - 1; ++dim) {
        auto dimVarArgStr = getSourceText(*dimVarArgs[dim], ctx);
        indexCalculation += util::fmt("{} + ({} * (", dimVarArgStr, params->dim[dim]).value();
    }
    indexCalculation += getSourceText(*dimVarArgs[n_dims - 1], ctx);
    // Close brackets
    for (int i = 0; i < 2 * (n_dims - 1); ++i) {
        indexCalculation += ")";
    }
    return indexCalculation;
}

HandleResult handleDimStmtAttrbute(const clang::Attr& a,
                                   const clang::Stmt& stmt,
                                   const AttributedDim* params,
                                   SessionStage& s) {
    llvm::outs() << "handle @dim stmt: "
                 << getSourceText(stmt.getSourceRange(), s.getCompiler().getASTContext())
                 << " with params: ";
    for (const auto& dim : params->dim) {
        llvm::outs() << dim << ", ";
    }
    llvm::outs() << "\n";

    auto& ctx = s.getCompiler().getASTContext();
    auto n_dims = params->dim.size();

    auto rec = dyn_cast<RecoveryExpr>(&stmt);
    if (!rec) {
        return tl::make_unexpected(Error{{}, "Failed to cast Stmt to RecoveryExpr"});
    }
    auto recoveryExpressions = validateDim(rec, params, s);
    if (!recoveryExpressions) {
        return tl::make_unexpected(recoveryExpressions.error());
    }

    auto dimVarNameStr = getSourceText(*recoveryExpressions.value()[0], ctx);
    ExprVec dimVarArgs(recoveryExpressions.value().begin() + 1, recoveryExpressions.value().end());
    auto indexCalculation = buildIndexCalculation(dimVarArgs, params, s);

    return TranspilationBuilder(s.getCompiler().getSourceManager(), "dim", 1)
        .addReplacement(OKL_DIM_ACCESS,
                        stmt.getSourceRange(),
                        util::fmt("{}[{}]", dimVarNameStr, indexCalculation).value())
        .build();
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, makeSpecificAttrHandle(handleDimDeclAttrbute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute decl handler\n";
    }

    ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, makeSpecificAttrHandle(handleDimStmtAttrbute));
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
