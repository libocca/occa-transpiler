#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/ast_processor_manager/ast_processor_manager.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"

#include "core/transpiler_session/kernel_generator.h"
#include "core/transpiler_session/kernel_metadata_generator.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpilation_node.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/AST/Attr.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>

namespace {
using namespace oklt;
using namespace clang;

template <typename TraversalType, typename ExprType>
bool dispatchTraverseFunc(TraversalType& traversal, ExprType expr) {
    using PureType = std::remove_pointer_t<ExprType>;
    if constexpr (std::is_same_v<PureType, Stmt>) {
        auto* expr_ = [](auto* expr) {
            if (expr->getStmtClass() == clang::Stmt::AttributedStmtClass) {
                return cast<AttributedStmt>(expr)->getSubStmt();
            }
            return expr;
        }(expr);
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseStmt(expr_);
    } else if constexpr (std::is_same_v<PureType, Decl>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseDecl(expr);
    } else if constexpr (std::is_same_v<PureType, TranslationUnitDecl>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseTranslationUnitDecl(
            expr);
    }
    return false;
}

const Decl& tryGetAttrNode(const Decl& d) {
    return d;
}

const Stmt& tryGetAttrNode(const Stmt& s) {
    if (s.getStmtClass() != Stmt::AttributedStmtClass) {
        return s;
    }
    return *cast<AttributedStmt>(s).getSubStmt();
}

int getNodeType(const Decl& d) {
    return d.getKind();
}

int getNodeType(const Stmt& s) {
    return s.getStmtClass();
}

tl::expected<std::set<const Attr*>, Error> getNodeAttrs(const Decl& decl, SessionStage& stage) {
    return stage.getAttrManager().checkAttrs(decl, stage);
}

tl::expected<std::set<const Attr*>, Error> tryGetDeclRefExprAttrs(const clang::DeclRefExpr& expr,
                                                                  SessionStage& stage) {
    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto& ctx = stage.getCompiler().getASTContext();
    auto attrs = attrTypeMap.get(ctx, expr.getType());

    return std::set<const Attr*>(attrs.begin(), attrs.end());
}

tl::expected<std::set<const Attr*>, Error> tryGetRecoveryExprAttrs(const clang::RecoveryExpr& expr,
                                                                   SessionStage& stage) {
    auto subExpr = expr.subExpressions();
    if (subExpr.empty()) {
        return {};
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
    if (!declRefExpr) {
        return {};
    }

    return tryGetDeclRefExprAttrs(*declRefExpr, stage);
}

tl::expected<std::set<const Attr*>, Error> tryGetCallExprAttrs(const clang::CallExpr& expr,
                                                               SessionStage& stage) {
    // If no errors, call default postValidate.
    if (!expr.containsErrors()) {
        return {};
    }

    // Otherwise, we should try to call dim handler
    auto* args = expr.getArgs();
    if (expr.getNumArgs() == 0 || args == nullptr) {
        return {};
    }

    auto* callee = expr.getCallee();
    if (callee == nullptr) {
        return {};
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(callee);
    if (!declRefExpr) {
        return {};
    }

    return tryGetDeclRefExprAttrs(*declRefExpr, stage);
}

tl::expected<std::set<const Attr*>, Error> getNodeAttrs(const Stmt& stmt, SessionStage& stage) {
    switch (stmt.getStmtClass()) {
        case Stmt::RecoveryExprClass:
            return tryGetRecoveryExprAttrs(cast<RecoveryExpr>(stmt), stage);
        case Stmt::CallExprClass:
            return tryGetCallExprAttrs(cast<CallExpr>(stmt), stage);
        case Stmt::DeclRefExprClass:
            return tryGetDeclRefExprAttrs(cast<DeclRefExpr>(stmt), stage);
        default:
            return stage.getAttrManager().checkAttrs(stmt, stage);
    }

    return {};
}

template <typename TraversalType, typename NodeType>
HandleResult runFromRootToLeaves(TraversalType& traversal,
                                 AstProcessorManager& procMng,
                                 AstProcessorType procType,
                                 const std::set<const Attr*>& attrs,
                                 NodeType& node,
                                 OklSemaCtx& sema,
                                 SessionStage& stage) {
    if (attrs.empty()) {
        return procMng.runPreActionNodeHandle(procType, nullptr, node, sema, stage);
    }

    for (const auto* attr : attrs) {
        auto result = procMng.runPreActionNodeHandle(procType, attr, node, sema, stage);
        if (!result) {
            return result;
        }
    }

    return {};
}

template <typename TraversalType, typename NodeType>
HandleResult runFromLeavesToRoot(TraversalType& traversal,
                                 AstProcessorManager& procMng,
                                 AstProcessorType procType,
                                 const std::set<const Attr*>& attrs,
                                 NodeType& node,
                                 OklSemaCtx& sema,
                                 SessionStage& stage) {
    auto& transpilationAccumulator = stage.tryEmplaceUserCtx<TranspilationNodes>();
    auto* ki = sema.getParsingKernelInfo();
    auto* cl = sema.getLoopInfo();

    // non attributed node
    if (attrs.empty()) {
        auto result = procMng.runPostActionNodeHandle(procType, nullptr, node, sema, stage);
        if (!result) {
            return result;
        }
        if (stage.getAttrManager().hasImplicitHandler(stage.getBackend(), getNodeType(node))) {
            transpilationAccumulator.push_back(TranspilationNode{
                .ki = ki, .li = cl, .attr = nullptr, .node = DynTypedNode::create(node)});
        }
    }

    // attributed node
    for (const auto* attr : attrs) {
        auto result = procMng.runPostActionNodeHandle(procType, attr, node, sema, stage);
        if (!result) {
            return result;
        }
        transpilationAccumulator.push_back(TranspilationNode{
            .ki = ki, .li = cl, .attr = attr, .node = DynTypedNode::create(node)});
    }

    return {};
}

template <typename TraversalType, typename NodeType>
bool traverseNode(TraversalType& traversal,
                  NodeType* node,
                  AstProcessorManager& procMng,
                  SessionStage& stage) {
    if (node == nullptr) {
        return true;
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto procType = stage.getAstProccesorType();

    auto attrsResult = getNodeAttrs(*node, stage);
    if (!attrsResult) {
        stage.pushError(std::move(attrsResult.error()));
        return false;
    }

    const auto& attrNode = tryGetAttrNode(*node);
    auto result = runFromRootToLeaves(
        traversal, procMng, procType, attrsResult.value(), attrNode, sema, stage);
    if (!result) {
        stage.pushError(std::move(result.error()));
        return false;
    }

    // dispatch the next node
    if (!dispatchTraverseFunc(traversal, node)) {
        stage.pushError(Error{.ec = std::error_code(), .desc = "traverse is stopped"});
        return false;
    }

    result = runFromLeavesToRoot(
        traversal, procMng, procType, attrsResult.value(), attrNode, sema, stage);
    if (!result) {
        stage.pushError(std::move(result.error()));
        return false;
    }

    return true;
}



tl::expected<std::string, Error> generateLauncher(SessionStage& stage) {
    const auto& transpilationNodes = stage.tryEmplaceUserCtx<TranspilationNodes>();
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    for (const auto& t : transpilationNodes) {
        sema.setParsedKernelInfo(t.ki);
        sema.setLoopInfo(t.li);
        auto result = applyTranspilationToNode(t.attr, t.node, stage);
        if (!result) {
            return tl::make_unexpected(result.error());
        }
    }

    return stage.getRewriterResult();
}

}  // namespace
namespace oklt {

PreorderNlrTraversal::PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage)
    : _procMng(procMng),
      _stage(stage) {
    // create storage for lazy transpiled nodes
    _stage.tryEmplaceUserCtx<OklSemaCtx>();
    _stage.tryEmplaceUserCtx<TranspilationNodes>();
}

bool PreorderNlrTraversal::TraverseDecl(clang::Decl* decl) {
    return traverseNode(*this, decl, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseStmt(clang::Stmt* stmt) {
    return traverseNode(*this, stmt, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseTranslationUnitDecl(
    clang::TranslationUnitDecl* translationUnitDecl) {
    return traverseNode(*this, translationUnitDecl, _procMng, _stage);
}

tl::expected<std::string, Error> PreorderNlrTraversal::applyAstProcessor(
    clang::TranslationUnitDecl* translationUnitDecl) {
    // traverse AST and generate sema metadata
    if (!TraverseTranslationUnitDecl(translationUnitDecl)) {
        return tl::make_unexpected(Error{{}, "error during AST traversing"});
    }

    // 1. generate transpiled kernel
    auto transpiledKernelResult = generateTranspiledKernel(_stage);
    if (!transpiledKernelResult) {
        return transpiledKernelResult;
    }
    _stage.getSession().output.kernel.sourceCode = std::move(transpiledKernelResult.value());

    // 2. generate build json transpile
    auto kernelMetaData = generateKernelMetaData(_stage);
    if (!kernelMetaData) {
        return kernelMetaData;
    }
    _stage.getSession().output.kernel.metadataJson = std::move(kernelMetaData.value());


    if (!isHostCategory(_stage.getBackend())) {
        _stage.setLauncherMode();
        auto launcherResult = generateLauncher(_stage);
        if (!transpiledKernelResult) {
            return tl::make_unexpected(transpiledKernelResult.error());
        }
        _stage.getSession().output.launcher.sourceCode = std::move(launcherResult.value());

        // 3. generate launcher and metadata
    }

    return std::move(transpiledKernelResult.value());
}
}  // namespace oklt
