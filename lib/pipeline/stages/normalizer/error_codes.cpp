#include "pipeline/stages/normalizer/error_codes.h"

namespace {
struct NormalizerErrorsCategory : std::error_category {
    [[nodiscard]] const char* name() const noexcept override;
    [[nodiscard]] std::string message(int ev) const override;
};

const char* NormalizerErrorsCategory::name() const noexcept {
    return "normalizer";
}

std::string NormalizerErrorsCategory::message(int ev) const {
    switch (static_cast<OkltNormalizerErrorCode>(ev)) {
        case OkltNormalizerErrorCode::EMPTY_SOURCE_STRING:
            return "input source string is empty";
        case OkltNormalizerErrorCode::NO_TOKENS_FROM_SOURCE:
            return "no tokens fetched from input source";
        case OkltNormalizerErrorCode::OTHER_ERROR:
            return "other error";
    }
    return "unrecognized error";
}

// this object is used to distinguish camera category from other categries
const NormalizerErrorsCategory theNormalizerErrorsCategory{};

}  // namespace

std::error_code make_error_code(OkltNormalizerErrorCode e) {
    return {static_cast<int>(e), theNormalizerErrorsCategory};
}

namespace oklt {
Error makeError(OkltNormalizerErrorCode ec, std::string desc) {
    return {.ec = make_error_code(ec), .desc = std::move(desc)};
}
}  // namespace oklt
