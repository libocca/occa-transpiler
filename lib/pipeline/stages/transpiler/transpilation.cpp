#include <oklt/util/format.h>

#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/session_stage.h"

#include "core/handler_manager/handler_manager.h"
#include "core/transpiler_session/attributed_type_map.h"

#include "core/transpiler_session/code_generator.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpilation_node.h"
#include "core/transpiler_session/transpiler_session.h"

#include "pipeline/core/stage_action.h"
#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"

#include <clang/AST/RecursiveASTVisitor.h>

#include <spdlog/spdlog.h>

namespace {

using namespace oklt;
using namespace clang;

struct PreorderNlrTraversal;

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
        return traversal.template RecursiveASTVisitor<PreorderNlrTraversal>::TraverseStmt(expr_);
    } else if constexpr (std::is_same_v<PureType, Decl>) {
        return traversal.template RecursiveASTVisitor<PreorderNlrTraversal>::TraverseDecl(expr);
    } else if constexpr (std::is_same_v<PureType, TranslationUnitDecl>) {
        return traversal
            .template RecursiveASTVisitor<PreorderNlrTraversal>::TraverseTranslationUnitDecl(expr);
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

tl::expected<std::set<const Attr*>, Error> getNodeAttrs(SessionStage& stage, const Decl& decl) {
    return stage.getAttrManager().checkAttrs(stage, DynTypedNode::create(decl));
}

tl::expected<std::set<const Attr*>, Error> tryGetDeclRefExprAttrs(SessionStage& stage,
                                                                  const clang::DeclRefExpr& expr) {
    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto& ctx = stage.getCompiler().getASTContext();
    auto attrs = attrTypeMap.get(ctx, expr.getType());
    auto res = std::set<const Attr*>(attrs.begin(), attrs.end());

    return res;
}

tl::expected<std::set<const Attr*>, Error> tryGetRecoveryExprAttrs(
    SessionStage& stage,
    const clang::RecoveryExpr& expr) {
    auto subExpr = expr.subExpressions();
    if (subExpr.empty()) {
        return {};
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
    if (!declRefExpr) {
        return {};
    }

    return tryGetDeclRefExprAttrs(stage, *declRefExpr);
}

tl::expected<std::set<const Attr*>, Error> tryGetCallExprAttrs(SessionStage& stage,
                                                               const clang::CallExpr& expr) {
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

    return tryGetDeclRefExprAttrs(stage, *declRefExpr);
}

tl::expected<std::set<const Attr*>, Error> getNodeAttrs(SessionStage& stage, const Stmt& stmt) {
    switch (stmt.getStmtClass()) {
        case Stmt::RecoveryExprClass:
            return tryGetRecoveryExprAttrs(stage, cast<RecoveryExpr>(stmt));
        case Stmt::CallExprClass:
            return tryGetCallExprAttrs(stage, cast<CallExpr>(stmt));
        case Stmt::DeclRefExprClass:
            return tryGetDeclRefExprAttrs(stage, cast<DeclRefExpr>(stmt));
        default:
            return stage.getAttrManager().checkAttrs(stage, DynTypedNode::create(stmt));
    }

    return {};
}

template <typename TraversalType>
HandleResult runFromRootToLeaves(TraversalType& traversal,
                                 SessionStage& stage,
                                 OklSemaCtx& sema,
                                 DynTypedNode& node,
                                 const std::set<const Attr*>& attrs) {
    auto& am = stage.getAttrManager();
    if (attrs.empty()) {
        return am.handleSemaPre(stage, node, nullptr);
    }

    for (const auto* attr : attrs) {
        auto result = am.handleSemaPre(stage, node, attr);
        if (!result) {
            if (!result.error().ctx.has_value() && attr) {
                result.error().ctx = attr->getRange();
            }
            return result;
        }
    }

    return {};
}

template <typename TraversalType>
HandleResult runFromLeavesToRoot(TraversalType& traversal,
                                 SessionStage& stage,
                                 OklSemaCtx& sema,
                                 DynTypedNode& node,
                                 const std::set<const Attr*>& attrs) {
    auto& am = stage.getAttrManager();
    auto& transpilationAccumulator = stage.tryEmplaceUserCtx<TranspilationNodes>();
    auto* ki = sema.getParsingKernelInfo();
    auto* cl = sema.getLoopInfo();

    // non attributed node
    if (attrs.empty()) {
        auto result = am.handleSemaPost(stage, node, nullptr);
        if (!result) {
            if (!result.error().ctx.has_value()) {
                result.error().ctx = node.getSourceRange();
            }
            return result;
        }
    }

    // attributed node
    for (const auto* attr : attrs) {
        auto result = am.handleSemaPost(stage, node, attr);
        if (!result) {
            if (!result.error().ctx.has_value() && attr) {
                result.error().ctx = attr->getRange();
            }
            return result;
        }
        transpilationAccumulator.push_back(TranspilationNode{.ki = ki, .li = cl, .attr = attr, .node = node});
    }

    if (stage.getAttrManager().hasImplicitHandler(stage.getBackend(), node.getNodeKind())) {
        transpilationAccumulator.push_back(TranspilationNode{
            .ki = ki, .li = cl, .attr = nullptr, .node = node});
    }

    return {};
}

template <typename NodeType>
bool skipNode(SessionStage& s, const NodeType& n) {
    const auto& sm = s.getCompiler().getSourceManager();
    const auto& loc = n.getEndLoc();

    if (sm.isInSystemHeader(loc)) {
        return true;
    }

    if (sm.isInSystemMacro(loc)) {
        return true;
    }

    return false;
}

template <typename TraversalType, typename NodeType>
bool traverseNode(TraversalType& traversal, SessionStage& stage, NodeType* node) {
    if (node == nullptr) {
        return true;
    }

    // node in non user header - skip traverse it
    // TODO add more robust verification
    if (skipNode(stage, *node)) {
        return true;
    }

    auto result = [&]() -> HandleResult {
        auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();

        auto attrsResult = getNodeAttrs(stage, *node);
        if (!attrsResult) {
            if (!attrsResult.error().ctx.has_value()) {
                attrsResult.error().ctx = node->getSourceRange();
            }
            return tl::make_unexpected(attrsResult.error());
        }

        auto attrNode = DynTypedNode::create(tryGetAttrNode(*node));
        auto result = runFromRootToLeaves(traversal, stage, sema, attrNode, attrsResult.value());
        if (!result) {
            return result;
        }

        // dispatch the next node
        if (!dispatchTraverseFunc(traversal, node)) {
            return tl::make_unexpected(Error{});
        }

        result = runFromLeavesToRoot(traversal, stage, sema, attrNode, attrsResult.value());
        if (!result) {
            return result;
        }
        return {};
    }();

    if (!result) {
        stage.pushError(result.error());
        return false;
    }

    return true;
};

class PreorderNlrTraversal : public clang::RecursiveASTVisitor<PreorderNlrTraversal> {
   public:
    PreorderNlrTraversal(SessionStage& stage)
        : _stage(stage),
          _tu(nullptr) {
        // create storage for lazy transpiled nodes
        _stage.tryEmplaceUserCtx<OklSemaCtx>();
        _stage.tryEmplaceUserCtx<TranspilationNodes>();
    }

    bool TraverseDecl(clang::Decl* decl) { return traverseNode(*this, _stage, decl); }

    bool TraverseStmt(clang::Stmt* stmt) { return traverseNode(*this, _stage, stmt); }

    bool TraverseTranslationUnitDecl(clang::TranslationUnitDecl* translationUnitDecl) {
        auto& sema = _stage.tryEmplaceUserCtx<OklSemaCtx>();
        sema.clear();

        auto& tnodes = _stage.tryEmplaceUserCtx<TranspilationNodes>();
        tnodes.clear();

        _tu = translationUnitDecl;
        return traverseNode(*this, _stage, translationUnitDecl);
    }

    tl::expected<std::pair<std::string, std::string>, Error> applyAstProcessor(
        clang::TranslationUnitDecl* translationUnitDecl) {
        // traverse AST and generate sema metadata if required
        if (!_tu || _tu != translationUnitDecl) {
            SPDLOG_INFO("Start AST traversal");
            if (!TraverseTranslationUnitDecl(translationUnitDecl)) {
                return tl::make_unexpected(Error{{}, "error during AST traversing"});
            }
        }

        // 0. Clear Kernel metadata
        auto& sema = _stage.tryEmplaceUserCtx<OklSemaCtx>();
        sema.getProgramMetaData().kernels.clear();

        // 1. generate transpiled code
        SPDLOG_INFO("Apply transpilation");
        auto transpiledResult = generateTranspiledCode(_stage);
        if (!transpiledResult) {
            return tl::make_unexpected(transpiledResult.error());
        }

        if (sema.getProgramMetaData().kernels.empty()) {
            return tl::make_unexpected(Error{{}, "Error: No [@kernel] functions found"});
        }

        // 2. generate build json
        SPDLOG_INFO("Build metadata json");
        auto transpiledMetaData = generateTranspiledCodeMetaData(_stage);
        if (!transpiledMetaData) {
            return tl::make_unexpected(transpiledMetaData.error());
        }

        return std::make_pair(std::move(transpiledResult.value()),
                              std::move(transpiledMetaData.value()));
    }

   private:
    SessionStage& _stage;
    clang::TranslationUnitDecl* _tu;
};

class TranspilationConsumer : public clang::ASTConsumer {
   public:
    TranspilationConsumer(SessionStage& stage)
        : _stage(stage) {}

    void HandleTranslationUnit(ASTContext& context) override {
        // get the root of parsed AST that contains main file and all headers
        TranslationUnitDecl* tu = context.getTranslationUnitDecl();

        auto traversal = std::make_unique<PreorderNlrTraversal>(_stage);

        auto& output = _stage.getSession().getOutput();
        // traverse AST and apply processor sema/backend handlers
        // retrieve final transpiled kernel code that fused all user includes
        {
            auto result = traversal->applyAstProcessor(tu);
            if (!result) {
                _stage.pushError(result.error());
                return;
            }

            // no errors and empty output could mean that the source is already transpiled
            // so use input as output and lets the next stage try to figure out
            if (result->first.empty()) {
                result->first = _stage.getSession().getStagedSource();
            }
            output.kernel.source = oklt::format(std::move(result->first));
            output.kernel.metadata = std::move(result->second);
        }

        // reuse traversed AST
        // retrieve launcher code and metadata if required
        if (isDeviceCategory(_stage.getBackend())) {
            _stage.setLauncherMode();

            auto result = traversal->applyAstProcessor(tu);
            if (!result) {
                return;
            }

            // no errors and empty output could mean that the source is already transpiled
            // so use input as output and lets the next stage try to figure out
            if (result->first.empty()) {
                result->first = _stage.getSession().getStagedSource();;
            }
            output.launcher.source = oklt::format(std::move(result->first));
            output.launcher.metadata = std::move(result->second);
        }
    }

    SessionStage& getSessionStage() { return _stage; }

   private:
    SessionStage& _stage;
};

class Transpilation : public StageAction {
   protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,
                                                   llvm::StringRef in_file) {
        if (!compiler.hasPreprocessor()) {
            SPDLOG_CRITICAL("no preporcessor: {}", __PRETTY_FUNCTION__);
        }

        auto& deps = _stage->tryEmplaceUserCtx<HeaderDepsInfo>();
        std::unique_ptr<PPCallbacks> callback =
            std::make_unique<InclusionDirectiveCallback>(deps, compiler.getSourceManager());
        // setup preprocessor hook to gather all user/system includes
        compiler.getPreprocessor().addPPCallbacks(std::move(callback));

        compiler.getDiagnostics().setClient(new DiagConsumer(*_stage));
        compiler.getDiagnostics().setShowColors(true);

        return std::make_unique<TranspilationConsumer>(*_stage);
    }
};

StagePluginRegistry::Add<Transpilation> transpilation(TRANSPILATION_STAGE, "");
}  // namespace
