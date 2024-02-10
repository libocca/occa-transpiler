#include <clang/AST/Attr.h>
#include <oklt/core/ast_traversal/transpile_types/attribute_info.h>

namespace oklt {

AttributeInfoBase::~AttributeInfoBase() = default;

OriginAttributeInfo::OriginAttributeInfo(clang::Attr* attr, clang::ASTContext& ctx)
    : _attr(attr), _ctx(ctx) {}

std::string OriginAttributeInfo::toString() const {
  std::string result;
  llvm::raw_string_ostream ostream(result);
  _attr->printPretty(ostream, _ctx.getPrintingPolicy());
  return result;
}

ConvertedAttributeInfo::ConvertedAttributeInfo(std::string&& converted)
    : _converted(std::move(converted)) {}
std::string ConvertedAttributeInfo::toString() const {
  return _converted;
}
}  // namespace oklt
