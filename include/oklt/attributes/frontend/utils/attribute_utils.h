#include <clang/AST/Attr.h>
#include <oklt/core/error.h>
#include <string>
#include <tl/expected.hpp>

namespace oklt {
// Given C++ suppress attribute returns it's parameter as string. Expected attribute to have only
// one attribute
tl::expected<std::string, Error> parseCppAttributeParameterStr(const clang::SuppressAttr* attr);

std::vector<std::string> splitCSV(const std::string& str);

// Returns OKL attribute parameters as vector of string
tl::expected<std::vector<std::string>, Error> parseOKLAttributeParamsStr(const clang::Attr* a);
}  // namespace oklt