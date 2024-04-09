#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

HandleResult preValidateOklKernel(SessionStage& stage,
                                  OklSemaCtx& sema,
                                  const clang::FunctionDecl& fd,
                                  const clang::Attr& attr) {
    if (sema.isParsingOklKernel()) {
        // TODO nested okl kernel function
        //  make appropriate error code
        return tl::make_unexpected(
            Error{.ec = std::error_code(), .desc = "nested OKL kernels are illegal"});
    }

    if (!sema.startParsingOklKernel(fd)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(), .desc = "nested OKL kernels are illegal"});
    }

    return {};
}

HandleResult postValidateOklKernel(SessionStage& stage,
                                   OklSemaCtx& sema,
                                   const clang::FunctionDecl& fd,
                                   const clang::Attr& attr) {
    // stop parsing of current kernel info and reset internal state of sema
    sema.stopParsingKernelInfo();

    return {};
}
}  // namespace oklt
