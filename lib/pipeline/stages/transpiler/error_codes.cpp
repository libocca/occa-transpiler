#include "pipeline/stages/transpiler/error_codes.h"

namespace {
struct TranspilerErrorsCategory : std::error_category {
    [[nodiscard]] const char* name() const noexcept override;
    [[nodiscard]] std::string message(int ev) const override;
};

const char* TranspilerErrorsCategory::name() const noexcept {
    return "transpiler";
}

std::string TranspilerErrorsCategory::message(int ev) const {
    switch (static_cast<OkltTranspilerErrorCode>(ev)) {
        case OkltTranspilerErrorCode::EMPTY_SOURCE_STRING:
            return "input sourece string is empty";
        case OkltTranspilerErrorCode::NO_TOKENS_FROM_SOURCE:
            return "no tokens fetched from input source";
        case OkltTranspilerErrorCode::OKL_ATTR_PARSING_ERR:
            return "other error";
        case OkltTranspilerErrorCode::OTHER_ERROR:
            return "other error";
        case OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP:
            return "not supported atomic expression";
        case OkltTranspilerErrorCode::ATOMIC_NON_LVALUE_EXPR:
            return "atomic left expression must be lvalue";
    }
    return "unrecognized error";
}

// this object is used to distinguish camera category from other categries
const TranspilerErrorsCategory theTranspilerErrorsCategory{};

}  // namespace

std::error_code make_error_code(OkltTranspilerErrorCode e) {
    return {static_cast<int>(e), theTranspilerErrorsCategory};
}

namespace oklt {
Error makeError(OkltTranspilerErrorCode ec, std::string desc) {
    return {.ec = make_error_code(ec), .desc = std::move(desc)};
}
}  // namespace oklt
