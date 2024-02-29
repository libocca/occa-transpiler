#pragma once

#include <oklt/core/error.h>
#include <system_error>

// this should be in the global namespace to make make_error_code works for automatic convertions
enum class OkltTranspilerErrorCode {
    EMPTY_SOURCE_STRING = 1,
    NO_TOKENS_FROM_SOURCE = 10,
    OKL_ATTR_PARSING_ERR = 20,
    OTHER_ERROR = 200,
    ATOMIC_NOT_SUPPORTED_OP = 201,
    ATOMIC_NON_LVALUE_EXPR = 202,
    INTERNAL_ERROR_PARAMS_NULL_OBJ = 203
};

namespace std {
template <>
struct is_error_code_enum<OkltTranspilerErrorCode> : true_type {};
}  // namespace std

std::error_code make_error_code(OkltTranspilerErrorCode);

namespace oklt {
Error makeError(OkltTranspilerErrorCode ec, std::string desc);
}  // namespace oklt
