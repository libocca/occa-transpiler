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
    auto ok =
        mng.registerSemaHandler(KERNEL_ATTR_NAME, preValidateOklKernel, postValidateOklKernel);
    assert(ok);

    // sema handler for OKL tile attribute
    ok = mng.registerSemaHandler(TILE_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL outer attribute
    ok = mng.registerSemaHandler(OUTER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    // sema handler for OKL inner attribute
    ok = mng.registerSemaHandler(INNER_ATTR_NAME, preValidateOklForLoop, postValidateOklForLoop);
    assert(ok);

    ok = mng.registerSemaHandler(
        "", preValidateOklForLoopWithoutAttribute, postValidateOklForLoopWithoutAttribute);

    assert(ok);
}

}  // namespace
