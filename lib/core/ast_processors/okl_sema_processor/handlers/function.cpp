#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

#define OKL_SEMA_DEBUG

namespace {
using namespace clang;
using namespace oklt;
}  // namespace

namespace oklt {
bool preValidateOklKernelSema(const FunctionDecl& fd, SessionStage& stage, OklSemaCtx& sema) {
    if (sema.isParsingOklKernel()) {
        // TODO nested okl kernel function
        //  make approptiate error code
        stage.pushError(std::error_code(), "nested OKL kernels are illegal");
        return false;
    }

    auto* parsingKernInfo = sema.startParsingOklKernel(fd);
    assert(parsingKernInfo);

    return true;
}

bool postValidateOklKernelSema(const FunctionDecl& fd, SessionStage& stage, OklSemaCtx& sema) {
    // bypass possible nested functions/closures
    if (!sema.isParsingOklKernel()) {
        return true;
    }

    auto* ki = sema.getParsingKernelInfo();
    if (ki->kernInfo->instances.size() > 1) {
        // TODO perfrom kernel split
    }

    // stop parsing of current kernel info and reset internal state of sema
    sema.stopParsingKernelInfo();

    return true;
}

bool preValidateOklKernelParamSema(const ParmVarDecl& parm, SessionStage& stage, OklSemaCtx& sema) {
    if (!sema.isParsingOklKernel()) {
        return true;
    }

    // skip if param belongs to nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return true;
    }

    // fill parsed function parameter info
    // even it's attributed we care about regular propertities: name, is_const, is_ptr, custom, etc
    sema.setKernelArgInfo(parm);

    return true;
}

bool postValidateOklKernelParamSema(const ParmVarDecl& parm,
                                    SessionStage& stage,
                                    OklSemaCtx& sema) {
    // skip nested function/closure
    if (!sema.isDeclInLexicalTraversal(parm)) {
        return true;
    }

    // for attributed param decl backend handler is responsible to fill raw string representation of
    // func argument
    if (!parm.hasAttrs()) {
        sema.setKernelArgRawString(parm);
        return true;
    }

    return true;
}
}  // namespace oklt
