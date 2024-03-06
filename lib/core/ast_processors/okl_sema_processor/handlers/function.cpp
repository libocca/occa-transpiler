#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

HandleResult preValidateOklKernel(const Attr& attr,
                                  const FunctionDecl& fd,
                                  OklSemaCtx& sema,
                                  SessionStage& stage) {
    if (sema.isParsingOklKernel()) {
        // TODO nested okl kernel function
        //  make approptiate error code
        return tl::make_unexpected(
            Error{.ec = std::error_code(), .desc = "nested OKL kernels are illegal"});
    }

    if (!sema.startParsingOklKernel(fd)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(), .desc = "nested OKL kernels are illegal"});
    }

    return {};
}

HandleResult postValidateOklKernel(const Attr& attr,
                                   const FunctionDecl& fd,
                                   OklSemaCtx& sema,
                                   SessionStage& stage) {
    // stop parsing of current kernel info and reset internal state of sema
    sema.stopParsingKernelInfo();

    return {};
}
}  // namespace oklt
