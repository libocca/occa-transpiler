#include <clang/AST/Type.h>
#include <clang/Basic/LLVM.h>
#include <oklt/core/ast_traversal/transpile_types/param_info.h>

namespace oklt {
using namespace clang;

// INFO: simple mapping
DatatypeCategory makeDatatypeCategory(const QualType& qt) {
  if (qt->isBuiltinType()) {
    return DatatypeCategory::BUILTIN;
  }
  return DatatypeCategory::CUSTOM;
}

ParamInfoBase::~ParamInfoBase() = default;

OriginalParamInfo::OriginalParamInfo(clang::ParmVarDecl* param) : ParamInfoBase(), _param(param) {}

std::string OriginalParamInfo::toString() const {
  std::string result;
  auto& ctx = _param->getASTContext();
  llvm::raw_string_ostream ostream(result);
  _param->print(ostream, ctx.getPrintingPolicy());
  return result;
}

ArgumentInfo OriginalParamInfo::makeArgument() const {
  auto paramQualType = _param->getType();
  auto typeInfo = _param->getASTContext().getTypeInfo(paramQualType);
  return ArgumentInfo{.is_const = paramQualType.isConstQualified(),
                      .dtype =
                        DataType{
                          .name = paramQualType.getAsString(),
                          .type = makeDatatypeCategory(paramQualType),
                          .bytes = static_cast<int>(typeInfo.Width),
                        },
                      .name = _param->getNameAsString(),
                      .is_ptr = paramQualType->isPointerType()};
}

ConvertedParamInfo::ConvertedParamInfo(clang::ParmVarDecl* param, std::string&& converted)
    : _param(param), _converted(std::move(converted)) {}

std::string ConvertedParamInfo::toString() const {
  return _converted;
}

ArgumentInfo ConvertedParamInfo::makeArgument() const {
  auto paramQualType = _param->getType();
  auto typeInfo = _param->getASTContext().getTypeInfo(paramQualType);
  return ArgumentInfo{.is_const = paramQualType.isConstQualified(),
                      .dtype =
                        DataType{
                          .name = paramQualType.getAsString(),
                          .type = makeDatatypeCategory(paramQualType),
                          .bytes = static_cast<int>(typeInfo.Width),
                        },
                      .name = _param->getNameAsString(),
                      .is_ptr = paramQualType->isPointerType()};
}

ExtraParamInfo::ExtraParamInfo(std::string&& representation, ArgumentInfo&& info)
    : _represent(std::move(representation)), _info(std::move(info)) {}

std::string ExtraParamInfo::toString() const {
  return _represent;
}

ArgumentInfo ExtraParamInfo::makeArgument() const {
  return _info;
}

}  // namespace oklt
