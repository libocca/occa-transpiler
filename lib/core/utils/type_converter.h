#pragma once

#include <oklt/core/kernel_metadata.h>
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Type.h>

#include <tl/expected.hpp>

namespace clang {
struct QualType;
struct FunctionDecl;
struct VarDecl;
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;

tl::expected<KernelInfo, std::error_code> toOklKernelInfo(const clang::FunctionDecl&,
                                                          const std::string& suffix = "");
tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl&);
tl::expected<StructFieldInfo, std::error_code> toOklStructFieldInfo(const clang::FieldDecl&);
DatatypeCategory toOklDatatypeCategory(const clang::QualType&);
tl::expected<DataType, std::error_code> toOklDataType(const clang::VarDecl& var);
tl::expected<DataType, std::error_code> toOklDataType(const clang::FieldDecl& var);

}  // namespace oklt
