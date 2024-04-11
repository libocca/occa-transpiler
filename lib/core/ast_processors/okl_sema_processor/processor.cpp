#include <oklt/util/string_utils.h>

#include "attributes/attribute_names.h"

#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"

#include <clang/AST/AST.h>
#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

__attribute__((constructor)) void registerOklSemaProcessor() {
    auto& mng = AstProcessorManager::instance();

    // sema handler for OKL kernel attribute
    auto ok = mng.registerHandle({AstProcessorType::OKL_WITH_SEMA,
                                  KERNEL_ATTR_NAME,
                                  ASTNodeKind::getFromNodeKind<FunctionDecl>()},
                                 {.preAction = makeSemaHandle(preValidateOklKernel),
                                  .postAction = makeSemaHandle(postValidateOklKernel)});
    assert(ok);

    // sema handler for OKL tile attribute
    ok = mng.registerHandle(
        {AstProcessorType::OKL_WITH_SEMA, TILE_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        {.preAction = makeSemaHandle(preValidateOklForLoop),
         .postAction = makeSemaHandle(postValidateOklForLoop)});
    assert(ok);

    // sema handler for OKL outer attribute
    ok = mng.registerHandle(
        {AstProcessorType::OKL_WITH_SEMA, OUTER_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        {.preAction = makeSemaHandle(preValidateOklForLoop),
         .postAction = makeSemaHandle(postValidateOklForLoop)});
    assert(ok);

    // sema handler for OKL inner attribute
    ok = mng.registerHandle(
        {AstProcessorType::OKL_WITH_SEMA, INNER_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        {.preAction = makeSemaHandle(preValidateOklForLoop),
         .postAction = makeSemaHandle(postValidateOklForLoop)});
    assert(ok);

    ok = mng.registerHandle(
        {AstProcessorType::OKL_WITH_SEMA, "", ASTNodeKind::getFromNodeKind<ForStmt>()},
        {.preAction = makeSemaHandle(preValidateOklForLoopWithoutAttribute),
         .postAction = makeSemaHandle(postValidateOklForLoopWithoutAttribute)});

    assert(ok);
}

}  // namespace
