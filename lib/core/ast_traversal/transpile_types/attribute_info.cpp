#include <oklt/core/ast_traversal/transpile_types/attribute_info.h>
#include <clang/AST/Attr.h>

namespace oklt {


AttributeInfoBase::~AttributeInfoBase() = default;

OriginAttributeInfo::OriginAttributeInfo(clang::Attr *attr, clang::ASTContext &ctx)
    : _attr(attr)
      , _ctx(ctx)
{}

std::string OriginAttributeInfo::toString() const {
  std::string result;
  llvm::raw_string_ostream ostream(result);
  _attr->printPretty(ostream, _ctx.getPrintingPolicy());
  return result;
}

ConvertedAttributeInfo::ConvertedAttributeInfo(std::string &&converted)
    :_converted(std::move(converted))
{}
std::string ConvertedAttributeInfo::toString() const {
  return _converted;
}

//struct AttributeInfoBase {
//  virtual ~AttributeInfoBase() = 0;
//  [[nodiscard]] virtual std::string toString() const = 0;
//};

//struct OriginAttributeInfo : public AttributeInfoBase {
//  OriginAttributeInfo::OriginAttributeInfo(clang::Attr *attr, clang::ASTContext &ctx)
//  {}
//  ~OriginAttributeInfo() override = default;
//  [[nodiscard]]  std::string OriginAttributeInfo::toString() const override;
// private:
//  clang::Attr *_attr;
//  clang::ASTContext &_ctx;
//};

//struct ConvertedAttributeInfo : public AttributeInfoBase {
//  explicit ConvertedAttributeInfo(std::string &&converted);
//  ~ConvertedAttributeInfo() override = default;
//  [[nodiscard]]  std::string toString() const override;
// private:
//  std::string _converted;
//};


}
