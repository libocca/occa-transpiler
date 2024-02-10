#pragma once

#include <string>
#include <clang/AST/AST.h>
#include <oklt/core/kernel_info/kernel_info.h>
#include <memory>

namespace oklt {

struct ParamInfoBase {
  virtual ~ParamInfoBase() = 0;
  [[nodiscard]] virtual std::string toString() const = 0;
  [[nodiscard]] virtual ArgumentInfo makeArgument() const = 0;
};

struct OriginalParamInfo : public ParamInfoBase {
  explicit OriginalParamInfo(clang::ParmVarDecl *param);
  ~OriginalParamInfo() override = default;
  [[nodiscard]] std::string toString() const override;
  [[nodiscard]] ArgumentInfo makeArgument() const override;
 private:
    clang::ParmVarDecl *_param;
};

//INFO: need deeper investigation for makeArgument method
struct ConvertedParamInfo : public ParamInfoBase {
    ConvertedParamInfo(clang::ParmVarDecl *param, std::string &&converted);
    ~ConvertedParamInfo() override = default;
    [[nodiscard]] std::string toString() const override;
    [[nodiscard]] ArgumentInfo makeArgument() const override;
private:
  clang::ParmVarDecl *_param;
  std::string _converted;
};

//INFO: for adding sycl custom params at kernel function processing time
struct ExtraParamInfo: public ParamInfoBase {
  explicit ExtraParamInfo(std::string &&representation, ArgumentInfo &&info);
  ~ExtraParamInfo() override = default;
  [[nodiscard]] std::string toString() const override;
  [[nodiscard]] ArgumentInfo makeArgument() const override;
 private:
  std::string _represent;
  ArgumentInfo _info;
};

using ParamInfoPtr = std::shared_ptr<ParamInfoBase>;

}
