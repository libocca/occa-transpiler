#include "attributes/attribute_names.h"

#include "core/handler_manager/sema_handler.h"
#include "core/sema/handlers/function.h"
#include "core/sema/handlers/loop.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

__attribute__((constructor)) void registerOklSemaProcessor() {
    // sema handler for OKL kernel attribute
    auto ok = HandlerManager::registerSemaHandler(
        KERNEL_ATTR_NAME, preValidateOklKernel, postValidateOklKernel);
    assert(ok);

    // sema handler for OKL tile attribute
    ok = HandlerManager::registerSemaHandler(
        TILE_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL outer attribute
    ok = HandlerManager::registerSemaHandler(
        OUTER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL inner attribute
    ok = HandlerManager::registerSemaHandler(
        INNER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    ok = HandlerManager::registerSemaHandler(
        "", preValidateOklForLoopWithoutAttribute, postValidateOklForLoopWithoutAttribute);

    assert(ok);
}

}  // namespace
