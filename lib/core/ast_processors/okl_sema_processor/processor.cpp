#include "attributes/attribute_names.h"

#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"
#include "core/attribute_manager/sema_handler.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

__attribute__((constructor)) void registerOklSemaProcessor() {
    auto& am = AttributeManager::instance();

    // sema handler for OKL kernel attribute
    auto ok = am.registerSemaHandler(KERNEL_ATTR_NAME, preValidateOklKernel, postValidateOklKernel);
    assert(ok);

    // sema handler for OKL tile attribute
    ok = am.registerSemaHandler(TILE_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL outer attribute
    ok = am.registerSemaHandler(OUTER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL inner attribute
    ok = am.registerSemaHandler(INNER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    ok = am.registerSemaHandler(
        "", preValidateOklForLoopWithoutAttribute, postValidateOklForLoopWithoutAttribute);

    assert(ok);
}

}  // namespace
