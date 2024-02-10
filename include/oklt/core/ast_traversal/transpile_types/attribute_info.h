#pragma once

#include <clang/AST/AST.h>
#include <oklt/core/kernel_info/kernel_info.h>
#include <memory>
#include <string>

namespace oklt {

struct AttributeInfoBase {
  virtual ~AttributeInfoBase() = 0;
  [[nodiscard]] virtual std::string toString() const = 0;
};

struct OriginAttributeInfo : public AttributeInfoBase {
  OriginAttributeInfo(clang::Attr* attr, clang::ASTContext& ctx);
  ~OriginAttributeInfo() override = default;
  [[nodiscard]] std::string toString() const override;

 private:
  clang::Attr* _attr;
  clang::ASTContext& _ctx;
};

struct ConvertedAttributeInfo : public AttributeInfoBase {
  explicit ConvertedAttributeInfo(std::string&& converted);
  ~ConvertedAttributeInfo() override = default;
  [[nodiscard]] std::string toString() const override;

 private:
  std::string _converted;
};

using AttributeInfoPtr = std::shared_ptr<AttributeInfoBase>;

}  // namespace oklt
