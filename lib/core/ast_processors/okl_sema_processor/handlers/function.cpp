#include <oklt/core/error.h>

#include "core/ast_processors/default_actions.h"
#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpilation_decoders.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

#define OKL_SEMA_DEBUG

namespace {
using namespace clang;
using namespace oklt;
}  // namespace

namespace oklt {
HandleResult preValidateOklKernel(const Attr& attr,
                                  const FunctionDecl& fd,
                                  OklSemaCtx& sema,
                                  SessionStage& stage) {
    if (sema.isParsingOklKernel()) {
        // TODO nested okl kernel function
        //  make appropriate error code
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

    // set transpiled kernel attribute
    auto kernelModifier = decodeKernelModifier(result.value());
    if (!kernelModifier) {
        return tl::make_unexpected(std::move(kernelModifier.error()));
    }
    sema.setKernelTranspiledAttrStr(kernelModifier.value());

    auto* ki = sema.getParsingKernelInfo();
    if (ki->kernInfo->childrens.size() > 1) {
        // TODO perform kernel split
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

    // fill parsed function parameter info
    // even it's attributed we care about regular properties: name, is_const, is_ptr, custom, etc
    sema.setKernelArgInfo(parm);

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

    auto paramModifier = decodeParamModifier(result.value());
    if (!paramModifier) {
        return tl::make_unexpected(std::move(paramModifier.error()));
    }

    sema.setTranspiledArgStr(parm, paramModifier.value());

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

    // fill parsed function parameter info
    // even it's attributed we care about regular properties: name, is_const, is_ptr, custom, etc
    sema.setKernelArgInfo(parm);

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

    sema.setTranspiledArgStr(parm);

    return result;
}
}  // namespace oklt
