#include "attributes/utils/kernel_utils.h"
#include "oklt/core/kernel_metadata.h"
#include "pipeline/stages/transpiler/error_codes.h"
#include "tl/expected.hpp"

namespace oklt {

tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo) {
    auto& topOuterLoops = kernelInfo.children;
    if (topOuterLoops.empty()) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::AT_LEAST_ONE_OUTER_REQUIRED,
                                         "[@kernel] requires at least one [@outer] for-loop"});
    }

    size_t nMissingInner = 0;
    for (auto& loop : topOuterLoops) {
        if (!loop.is(LoopType::Outer, LoopType::Inner) &&
            !loop.getFirstAttributedChild(
                [](OklLoopInfo& info) { return info.has(LoopType::Inner); })) {
            ++nMissingInner;
        }
    }
    if (nMissingInner) {
        if (nMissingInner == topOuterLoops.size()) {
            return tl::make_unexpected(Error{OkltTranspilerErrorCode::AT_LEAST_ONE_INNER_REQUIRED,
                                             "[@kernel] requires at least one [@inner] for-loop"});
        }
        return tl::make_unexpected(
            Error{OkltTranspilerErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
    }
    return {};
    // return tl::make_unexpected(
    // Error{OkltTranspilerErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
}
}  // namespace oklt
