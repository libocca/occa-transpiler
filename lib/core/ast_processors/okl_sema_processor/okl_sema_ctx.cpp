#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/metadata/program.h"
#include "core/utils/type_converter.h"

namespace {
using namespace clang;
using namespace oklt;

DatatypeCategory toDatatypeCategory(const QualType& qt) {
    if (qt->isBuiltinType()) {
        return DatatypeCategory::BUILTIN;
    }
    return DatatypeCategory::CUSTOM;
}

}  // namespace
namespace oklt {
OklSemaCtx::ParsingKernelInfo* OklSemaCtx::startParsingOklKernel(const FunctionDecl* fd) {
    if (_parsingKernInfo.has_value()) {
        return nullptr;
    }

    auto* kiPtr = &_programMetaData.addKernelInfo(fd->getNameAsString(), fd->param_size(), 1);

    // link created slot with current parsing kernel context
    _parsingKernInfo = ParsingKernelInfo{.kernInfo = kiPtr,
                                         .argStrs = std::vector<std::string>(fd->param_size()),
                                         .kernFuncDecl = fd};

    return &_parsingKernInfo.value();
}

OklSemaCtx::ParsingKernelInfo* OklSemaCtx::getParsingKernelInfo() {
    return _parsingKernInfo ? &_parsingKernInfo.value() : nullptr;
}

void OklSemaCtx::stopParsingKernelInfo() {
    _parsingKernInfo.reset();
}

bool OklSemaCtx::isParsingOklKernel() const {
    return _parsingKernInfo.has_value();
}

bool OklSemaCtx::isKernelParmVar(const ParmVarDecl* parm) const {
    if (!(parm || _parsingKernInfo.has_value())) {
        return false;
    }

    return parm->getParentFunctionOrMethod() == _parsingKernInfo->kernFuncDecl;
}

void OklSemaCtx::setKernelArgInfo(const ParmVarDecl* parm) {
    assert(_parsingKernInfo.has_value());
    auto result = toOklArgInfo(*parm);
    if (!result) {
        llvm::errs() << "failed to convert parm var decl to okl data type\n";
        return;
    }

    auto* ki = _parsingKernInfo.value().kernInfo;
    ki->args[parm->getFunctionScopeIndex()] = std::move(result.value());
}

void OklSemaCtx::setKernelArgRawString(const ParmVarDecl* parm, std::string_view transpiledType) {
    assert(_parsingKernInfo.has_value());

    auto varType = [](const auto* p, auto transpiledType) {
        return !transpiledType.empty() ? std::string(transpiledType) : p->getType().getAsString();
    }(parm, transpiledType);

    auto& pki = _parsingKernInfo.value();
    pki.argStrs[parm->getFunctionScopeIndex()] = varType + " " + parm->getNameAsString();
}

void OklSemaCtx::setKernelTranspiledAttrStr(std::string attrStr) {
    assert(_parsingKernInfo.has_value());
    _parsingKernInfo.value().transpiledFuncAttrStr = std::move(attrStr);
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
    return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
    return _programMetaData;
}
}  // namespace oklt
