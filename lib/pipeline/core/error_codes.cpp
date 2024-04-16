#include "pipeline/core/error_codes.h"

namespace {
struct OkltPipelineErrorsCategory : std::error_category {
    [[nodiscard]] const char* name() const noexcept override;
    [[nodiscard]] std::string message(int ev) const override;
};

const char* OkltPipelineErrorsCategory::name() const noexcept {
    return "okl-transpiler-pipeline";
}

std::string OkltPipelineErrorsCategory::message(int ev) const {
    switch (static_cast<OkltPipelineErrorCode>(ev)) {
        case OkltPipelineErrorCode::EMPTY_SOURCE_STRING:
            return "input sourece string is empty";
        case OkltPipelineErrorCode::NO_TOKENS_FROM_SOURCE:
            return "no tokens fetched from input source";
        case OkltPipelineErrorCode::OKL_ATTR_PARSING_ERR:
            return "other error";
        case OkltPipelineErrorCode::OTHER_ERROR:
            return "other error";
        case OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP:
            return "not supported atomic expression";
        case OkltPipelineErrorCode::ATOMIC_NON_LVALUE_EXPR:
            return "atomic left expression must be lvalue";
        case OkltPipelineErrorCode::INTERNAL_ERROR_PARAMS_NULL_OBJ:
            return "internal error, parameter object can't be null pointer";
        case OkltPipelineErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL:
            return "internal error, kernel info object can't be null pointer";
        case OkltPipelineErrorCode::AT_LEAST_ONE_OUTER_REQUIRED:
            return "okl sema error, at least one outer loop is requied in kernel";
        case OkltPipelineErrorCode::AT_LEAST_ONE_INNER_REQUIRED:
            return "okl sema error, at least one inner loop is requied in kernel";
        case OkltPipelineErrorCode::MISSING_INNER_LOOP:
            return "okl sema error, missed inner loop in kernel";
    }
    return "unrecognized error";
}
// this object is used to distinguish camera category from other categries
const OkltPipelineErrorsCategory okltPipelineErrorsCategory{};

}  // namespace

std::error_code make_error_code(OkltPipelineErrorCode e) {
    return {static_cast<int>(e), okltPipelineErrorsCategory};
}

namespace oklt {
Error makeError(OkltPipelineErrorCode ec, std::string desc) {
    return {.ec = make_error_code(ec), .desc = std::move(desc)};
}
}  // namespace oklt
