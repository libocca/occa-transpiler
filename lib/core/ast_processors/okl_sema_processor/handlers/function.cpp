#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/default_actions.h"
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

    auto* parsingKernInfo = sema.startParsingOklKernel(fd);
    assert(parsingKernInfo);

    return {};
}

HandleResult postValidateOklKernel(const Attr& attr,
                                   const FunctionDecl& fd,
                                   OklSemaCtx& sema,
                                   SessionStage& stage) {
    auto result = runDefaultPostActionDecl(&attr, fd, sema, stage);
    if (!result) {
        return result;
    }

    // stop parsing of current kernel info and reset internal state of sema
    sema.stopParsingKernelInfo();

    return result;
}

HandleResult preValidateOklKernelAttrArg(const Attr&,
                                         const ParmVarDecl& parm,
                                         OklSemaCtx& sema,
                                         SessionStage& stage) {
    if (!sema.isParsingOklKernel()) {
        return {};
    }

    // skip if param belongs to nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return {};
    }

    return {};
}

HandleResult postValidateOklKernelAttrArg(const Attr& attr,
                                          const ParmVarDecl& parm,
                                          OklSemaCtx& sema,
                                          SessionStage& stage) {
    auto result = runDefaultPostActionDecl(&attr, parm, sema, stage);
    if (!result) {
        return result;
    }

    // skip nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return {};
    }

    return result;
}

HandleResult preValidateOklKernelParam(const ParmVarDecl& parm,
                                       OklSemaCtx& sema,
                                       SessionStage& stage) {
    if (!sema.isParsingOklKernel()) {
        return {};
    }

    // skip if param belongs to nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return {};
    }

    return {};
}

HandleResult postValidateOklKernelParam(const ParmVarDecl& parm,
                                        OklSemaCtx& sema,
                                        SessionStage& stage) {
    auto result = runDefaultPostActionDecl(nullptr, parm, sema, stage);
    if (!result) {
        return result;
    }

    // skip nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return result;
    }

    return result;
}
}  // namespace oklt
