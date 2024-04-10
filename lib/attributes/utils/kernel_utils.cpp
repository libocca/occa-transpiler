#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "oklt/core/kernel_metadata.h"
#include "pipeline/stages/transpiler/error_codes.h"

#include "tl/expected.hpp"

#include <clang/AST/ParentMapContext.h>

#include <deque>

namespace oklt {

namespace {
struct LoopTreeNode {
    OklLoopInfo* loop;
    uint32_t typeIdx = 0;  // type can be complex (@tile), so index indicates concrete type
};

bool verifyLoopStructureRecursiveBFS(std::deque<LoopTreeNode>& queue) {
    // Leaf - stop condition
    if (queue.empty()) {
        return true;
    }
    std::deque<LoopTreeNode> nextQueue;

    auto& firstNode = queue.front();

    auto levelType = firstNode.loop->type[firstNode.typeIdx];

    bool hasChildren = !firstNode.loop->children.empty();
    bool hasTiledChildren = (firstNode.loop->type.size() - 1) > firstNode.typeIdx;
    bool isLeafLevel = !hasChildren && !hasTiledChildren;

    for (auto node : queue) {
        // Verify same type
        auto currLevelType = node.loop->type[node.typeIdx];
        if (currLevelType != levelType) {
            return false;
        }

        // Veryfi same depth
        bool currHasChildren = !node.loop->children.empty();
        bool currHasTiledChildren = (node.loop->type.size() - 1) > node.typeIdx;
        if (isLeafLevel) {
            if (currHasChildren || currHasTiledChildren) {
                return false;
            }
        } else {
            // We have to visist complex types (@tile) multiple times
            if (currHasTiledChildren) {
                node.typeIdx++;
                nextQueue.push_back(node);
            } else if (currHasChildren) {
                for (auto& child : node.loop->children) {
                    nextQueue.push_back(LoopTreeNode{&child});
                }
            } else {
                // Not all children depth is the same = bad structure
                return false;
            }
        }
    }
    return verifyLoopStructureRecursiveBFS(nextQueue);
}

// All leafs must be at the same level and all levels must have the same type
bool verifyLoopStructure(const std::list<OklLoopInfo*>& topLevelOuterLoops) {
    std::deque<LoopTreeNode> q;
    for (auto* loop : topLevelOuterLoops) {
        q.push_back({loop});
    }
    return verifyLoopStructureRecursiveBFS(q);
}

}  // namespace

tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo) {
    auto& topOuterLoops = kernelInfo.topLevelOuterLoops;
    if (topOuterLoops.empty()) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::AT_LEAST_ONE_OUTER_REQUIRED,
                                         "[@kernel] requires at least one [@outer] for-loop"});
    }

    // Verify inner are present
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
            return tl::make_unexpected(Error{OkltTranspilerErrorCode::AT_LEAST_ONE_INNER_REQUIRED,
                                             "[@kernel] requires at least one [@inner] for-loop"});
        }
        return tl::make_unexpected(
            Error{OkltTranspilerErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
    }

    // Verify loops hierarchy
    if (!verifyLoopStructure(topOuterLoops)) {
        return tl::make_unexpected(
            Error{std::error_code(), "Incorrect [@outer]/[@inner] loops structure"});
    }
    return {};
}

const clang::AttributedStmt* getAttributedStmt(const clang::Stmt& stmt, SessionStage& s) {
    auto& ctx = s.getCompiler().getASTContext();
    const auto parents = ctx.getParentMapContext().getParents(stmt);
    if (parents.empty())
        return nullptr;

    return parents[0].get<clang::AttributedStmt>();
}

tl::expected<void, Error> handleChildAttr(const clang::Stmt& stmt,
                                          std::string_view name,
                                          SessionStage& s) {
    auto* attributedStmt = getAttributedStmt(stmt, s);
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

        auto params = am.parseAttr(*attr, s);
        if (!params) {
            return tl::make_unexpected(std::move(params.error()));
        }

        return am.handleAttr(*attr, stmt, &params.value(), s);
    }

    return {};
}

}  // namespace oklt
