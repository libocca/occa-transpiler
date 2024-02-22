#include <oklt/core/kernel_metadata.h>

#include <tl/expected.hpp>

namespace clang {
struct QualType;
struct VarDecl;
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;
struct LoopMetaData;

tl::expected<DataType, std::error_code> toOklDataType(const clang::VarDecl&);
tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl&);
DatatypeCategory toOklDatatypeCategory(const clang::QualType&);
}  // namespace oklt
