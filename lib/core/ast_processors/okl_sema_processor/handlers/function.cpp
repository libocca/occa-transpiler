#include <oklt/core/ast_processors/okl_sema_processor/okl_sema_ctx.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/metadata/program.h>
#include <oklt/core/transpiler_session/session_stage.h>

#include <clang/AST/AST.h>

#define OKL_SEMA_DEBUG

namespace {
using namespace clang;
using namespace oklt;

bool isOklKernel(const FunctionDecl* fd, SessionStage& stage) {
    // we intersting only in okl kernel function
    if (!fd->hasAttrs()) {
        return false;
    }

    const auto& attrs = fd->getAttrs();
    if (attrs.size() > 1) {
        // TODO multiple attributes are not supported by legacy OKL
        //  make approptiate error code
        stage.pushError(std::error_code(), "multiplte attributes for function decl");
        return false;
    }

    return attrs[0]->getNormalizedFullName() == KERNEL_ATTR_NAME;
}

bool applyBackendHandlers(const Decl* decl, SessionStage& stage) {
    if (!decl->hasAttrs()) {
        return AttributeManager::instance().handleDecl(decl, stage);
    }
    for (auto* attr : decl->getAttrs()) {
        if (!AttributeManager::instance().handleAttr(attr, decl, stage)) {
            return false;
        }
    }
    return true;
}
}  // namespace

namespace oklt {
bool prepareOklKernelFunction(const FunctionDecl* fd, SessionStage& stage) {
    // we intersting only in okl kernel function
    if (!isOklKernel(fd, stage)) {
        return true;
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
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

bool transpileOklKernelFunction(const FunctionDecl* fd, SessionStage& stage) {
    // validation is done on preAction so braverely get the first attribute and run traclpile handle
    auto cont = applyBackendHandlers(fd, stage);
    if (!cont) {
        return false;
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
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

bool prepareOklKernelParam(const ParmVarDecl* parm, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return true;
    }

    // skip if param belongs to nested function/closure
    if (!sema.isKernelParmVar(parm)) {
        return true;
    }

    // fill parsed function parameter info
    // even it's attributed we care about regular propertities: name, is_const, is_ptr, custom, etc
    sema.setKernelArgInfo(parm);

    return true;
}

bool transpileOklKernelParam(const ParmVarDecl* parm, SessionStage& stage) {
    auto cont = applyBackendHandlers(parm, stage);
    if (!cont) {
        return false;
    }

    // it is just regular function continue
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return true;
    }

    // skip nested function/closure
    if (!sema.isKernelParmVar(parm)) {
        return true;
    }

    // for attributed param decl backend handler is responsible to fill raw string representation of
    // func argument
    if (!parm->hasAttrs()) {
        sema.setKernelArgRawString(parm);
        return true;
    }

    return true;
}
}  // namespace oklt
