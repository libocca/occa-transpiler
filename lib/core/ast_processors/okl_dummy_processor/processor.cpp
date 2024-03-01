#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_processors/default_actions.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerAstNodeHanlder() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    auto ok = mng.registerDefaultHandle(
        {AstProcessorType::OKL_NO_SEMA},
        DeclHandle{.preAction = runDefaultPreActionDecl, .postAction = runDefaultPostActionDecl});
    assert(ok);

    ok = mng.registerDefaultHandle(
        AstProcessorType::OKL_NO_SEMA,
        StmtHandle{.preAction = runDefaultPreActionStmt, .postAction = runDefaultPostActionStmt});
    assert(ok);
}

}  // namespace
