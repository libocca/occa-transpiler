#include <clang/AST/Attr.h>
#include <oklt/core/error.h>
#include <string>
#include <tl/expected.hpp>

namespace oklt {
std::vector<std::string> splitCSV(const std::string& str);

// Returns OKL attribute parameters as vector of string
tl::expected<std::vector<std::string>, Error> parseOKLAttributeParamsStr(const clang::Attr* a);
}  // namespace oklt
