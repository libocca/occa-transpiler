#include "core/metadata/program.h"

#include <tl/expected.hpp>

namespace clang {
struct QualType;
struct VarDecl;
}  // namespace clang

namespace oklt {
tl::expected<DataType, std::error_code> toOklDataType(const clang::VarDecl&);
tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl&);
DatatypeCategory toOklDatatypeCategory(const clang::QualType&);
}  // namespace oklt
// namespace oklt
