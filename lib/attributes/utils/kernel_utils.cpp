#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "oklt/core/kernel_metadata.h"
#include "pipeline/core/error_codes.h"

#include "tl/expected.hpp"

#include <clang/AST/ParentMapContext.h>

namespace oklt {
using namespace clang;

tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo) {
    auto& topOuterLoops = kernelInfo.topLevelOuterLoops;
    if (topOuterLoops.empty()) {
        return tl::make_unexpected(Error{OkltPipelineErrorCode::AT_LEAST_ONE_OUTER_REQUIRED,
                                         "[@kernel] requires at least one [@outer] for-loop"});
    }

    size_t nMissingInner = 0;
    for (auto& loop : topOuterLoops) {
        if (!loop->is(LoopType::Outer, LoopType::Inner) &&
            !loop->getFirstAttributedChild(
                [](OklLoopInfo& info) { return info.has(LoopType::Inner); })) {
            ++nMissingInner;
        }
    }
    if (nMissingInner) {
        if (nMissingInner == topOuterLoops.size()) {
            return tl::make_unexpected(Error{OkltPipelineErrorCode::AT_LEAST_ONE_INNER_REQUIRED,
                                             "[@kernel] requires at least one [@inner] for-loop"});
        }
        return tl::make_unexpected(
            Error{OkltPipelineErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
    }
    return {};
    // return tl::make_unexpected(
    // Error{OkltPipelineErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
}

const clang::AttributedStmt* getAttributedStmt(SessionStage& s, const clang::Stmt& stmt) {
    auto& ctx = s.getCompiler().getASTContext();
    const auto parents = ctx.getParentMapContext().getParents(stmt);
    if (parents.empty())
        return nullptr;

    return parents[0].get<clang::AttributedStmt>();
}

tl::expected<std::any, Error> handleChildAttr(SessionStage& s,
                                              const clang::Stmt& stmt,
                                              std::string_view name) {
    auto* attributedStmt = getAttributedStmt(s, stmt);
    if (!attributedStmt) {
        return {};
    }

    auto& am = s.getAttrManager();
    for (const auto* attr : attributedStmt->getAttrs()) {
        if (!attr) {
            continue;
        }

        if (attr->getNormalizedFullName() != name) {
            continue;
        }

        auto params = am.parseAttr(s, *attr);
        if (!params) {
            return tl::make_unexpected(std::move(params.error()));
        }

        auto node = DynTypedNode::create(stmt);
        return am.handleAttr(s, node, *attr, &params.value());
    }

    return {};
}

}  // namespace oklt
