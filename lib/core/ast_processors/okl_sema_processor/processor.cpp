#include <oklt/util/string_utils.h>

#include "attributes/attribute_names.h"

#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

__attribute__((constructor)) void registerOklSemaProcessor() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    // sema handler for OKL kernel attribute
    auto ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, KERNEL_ATTR_NAME},
        DeclHandle{.preAction = makeSpecificSemaHandle(preValidateOklKernel),
                   .postAction = makeSpecificSemaHandle(postValidateOklKernel)});
    assert(ok);

    // sema handler for OKL tile attribute
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, TILE_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);

    // sema handler for OKL outer attribute
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, OUTER_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);

    // sema handler for OKL inner attribute
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, INNER_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);
}
}  // namespace
