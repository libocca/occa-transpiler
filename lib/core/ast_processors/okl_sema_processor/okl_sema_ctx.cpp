#include <oklt/core/ast_processors/okl_sema_processor/okl_sema_ctx.h>
#include <oklt/core/kernel_info/kernel_info.h>

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
  // create new slot for kenrnel info
  _programMetaData.kernels.emplace_back(
    std::move(KernelInfo{.name = fd->getNameAsString(),
                         .args = std::vector<ArgumentInfo>(fd->param_size()),
                         .argRawStrings = std::vector<std::string>(fd->param_size())}));

  // link created slot with current parsing kernel context
  auto* ki = &_programMetaData.kernels.back();
  _parsingKernInfo = ParsingKernelInfo{.kernInfo = ki, .kernFuncDecl = fd};

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

  // templated arg is abstract
  if (parm->isTemplated()) {
    return;
  }

  auto paramQualType = parm->getType();
  auto typeSize = parm->getASTContext().getTypeSize(paramQualType);
  auto idx = parm->getFunctionScopeIndex();

  auto* ki = _parsingKernInfo.value().kernInfo;
  ki->args[idx].is_const = paramQualType.isConstQualified();

  ki->args[idx].dtype.name = paramQualType.getAsString();
  ki->args[idx].dtype.type = toDatatypeCategory(paramQualType);
  ki->args[idx].dtype.bytes = static_cast<int>(typeSize);

  ki->args[idx].name = parm->getNameAsString();
  ki->args[idx].is_ptr = paramQualType->isPointerType();
}

void OklSemaCtx::setKernelArgRawString(const ParmVarDecl* parm, std::string_view transpiledType) {
  assert(_parsingKernInfo.has_value());

  auto idx = parm->getFunctionScopeIndex();
  auto* ki = _parsingKernInfo.value().kernInfo;

  auto varType = [](const auto* p, auto transpiledType) {
    return !transpiledType.empty() ? std::string(transpiledType) : p->getType().getAsString();
  }(parm, transpiledType);

  ki->argRawStrings[idx] = varType + " " + parm->getNameAsString();
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
  return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
  return _programMetaData;
}
}  // namespace oklt
