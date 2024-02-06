#pragma once
#include <system_error>

// this should be in the global namespace to make make_error_code works for automatic convertions
enum class OkltNormalizerErrorCode {
  EMPTY_SOURCE_STRING = 1,
  NO_TOKENS_FROM_SOURCE = 10,
  OTHER_ERROR = 200,
};

namespace std {
template <>
struct is_error_code_enum<OkltNormalizerErrorCode> : true_type {};
}  // namespace std

std::error_code make_error_code(OkltNormalizerErrorCode);
