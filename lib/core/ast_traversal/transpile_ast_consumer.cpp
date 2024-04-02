#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
    // get the root of parsed AST that contains main file and all headers
    TranslationUnitDecl* tu = context.getTranslationUnitDecl();

    auto t = std::make_unique<PreorderNlrTraversal>(AstProcessorManager::instance(), _stage);

    // traverse AST and apply processor sema/backend handlers
    // retrieve final transpiled kernel code that fused all user includes
    {
        auto result = t->applyAstProcessor(tu);
        if (!result) {
            return;
        }

        // no errors and empty output could mean that the source is already transpiled
        // so use input as output and lets the next stage try to figure out
        if (result->first.empty()) {
            result->first = _stage.getSession().input.sourceCode;
        }
        _stage.getSession().output.kernel.sourceCode = std::move(result->first);
        _stage.getSession().output.kernel.metadataJson = std::move(result->second);
    }

    // reuse traversed AST
    // retrieve launcher code and metadata if required
    if (isDeviceCategory(_stage.getBackend())) {
        _stage.setLauncherMode();

        auto result = t->applyAstProcessor(tu);
        if (!result) {
            return;
        }

        // no errors and empty output could mean that the source is already transpiled
        // so use input as output and lets the next stage try to figure out
        if (result->first.empty()) {
            result->first = _stage.getSession().input.sourceCode;
        }
        _stage.getSession().output.launcher.sourceCode = std::move(result->first);
        _stage.getSession().output.launcher.metadataJson = std::move(result->second);
    }
}

SessionStage& TranspileASTConsumer::getSessionStage() {
    return _stage;
}

}  // namespace oklt
