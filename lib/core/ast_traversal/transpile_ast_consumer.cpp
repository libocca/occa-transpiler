#include <oklt/core/ast_processor_manager/ast_processor_manager.h>
#include <oklt/core/ast_processors/okl_sema_processor/okl_sema_ctx.h>
#include <oklt/core/ast_traversal/preorder_traversal_nlr.h>
#include <oklt/core/ast_traversal/transpile_ast_consumer.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage) : _stage(stage) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();

  PreorderNlrTraversal traversal(AstProcessorManager::instance(), _stage);
  traversal.TraverseTranslationUnitDecl(tu);

#ifdef OKL_SEMA_DEBUG_LOG
  auto& md = _stage.tryEmplaceUserCtx<OklSemaCtx>().getProgramMetaData();
  for (const auto& k : md.kernels) {
    printf("parsed okl kernel\n name: %s\n num_of_args: %d\n instances: %d\n", k.name.c_str(),
           (uint32_t)k.args.size(), (uint32_t)k.instances.size());
    for (const auto& arg : k.args) {
      printf("parsed args \n name: %s\n is_const: %d\n is_ptr: %d\n is_custom: %d\n",
             arg.name.c_str(), arg.is_const, arg.is_ptr,
             arg.dtype.type == DatatypeCategory::CUSTOM);
    }
  }
#endif
}

SessionStage& TranspileASTConsumer::getSessionStage() {
  return _stage;
}

}  // namespace oklt
