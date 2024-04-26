#include "attributes/utils/kernel_utils.h"
#include <oklt/core/kernel_metadata.h>
#include "attributes/attribute_names.h"
#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_info.h"
#include "core/transpiler_session/session_stage.h"
#include "pipeline/core/error_codes.h"

#include <clang/AST/ParentMapContext.h>

#include <spdlog/fmt/fmt.h>

#include <deque>
#include <tl/expected.hpp>

namespace oklt {
using namespace clang;

namespace {
/**
 * @brief A structure representing a node in the loop tree.
 */
struct LoopTreeNode {
    OklLoopInfo* loop;     ///< A pointer to the loop information.
    uint32_t typeIdx = 0;  ///< The index indicating the concrete type of the loop.
};

/**
 * @brief Function to get the next attributed type index in the loop types.
 * @param loopTypes The types of the loops.
 * @param currIdx The current index.
 * @return The next attributed type index if it exists, -1 otherwise.
 */
int32_t getNextAttributedTypeIdx(const LoopTypes& loopTypes, int32_t currIdx) {
    while (++currIdx < loopTypes.size()) {
        if (loopTypes[currIdx] != LoopType::Regular) {
            return currIdx;
        }
    }
    return -1;
}

/**
 * @brief Function to verify the loop structure recursively using Breadth-First Search (BFS).
 * @param queue A queue of loop tree nodes to verify.
 * @return An expected void value if the loop structure is valid, an unexpected pair of a pointer to
 * the problematic statement and an error message otherwise.
 */
tl::expected<void, std::pair<const clang::ForStmt*, std::string>> verifyLoopStructureRecursiveBFS(
    std::deque<LoopTreeNode>& queue) {
    // Leaf - stop condition
    if (queue.empty()) {
        return {};
    }
    std::deque<LoopTreeNode> nextQueue;

    auto& firstNode = queue.front();

    auto levelType = firstNode.loop->type[firstNode.typeIdx];

    bool hasChildren = firstNode.loop->getFirstAttributedChild();
    bool hasTiledChildren = getNextAttributedTypeIdx(firstNode.loop->type, firstNode.typeIdx) != -1;
    bool isLeafLevel = !hasChildren && !hasTiledChildren;

    for (auto node : queue) {
        // Verify same type
        auto currType = node.loop->type[node.typeIdx];
        if (currType != levelType) {
            return tl::make_unexpected(
                std::make_pair(&node.loop->stmt,
                               fmt::format("Loop type mismatch: Expected [@{}], received [@{}]",
                                           toString(levelType),
                                           toString(currType))));
        }

        // Veryfy same depth
        bool currHasChildren = node.loop->getFirstAttributedChild();
        auto nextTiledTypeIdx = getNextAttributedTypeIdx(node.loop->type, node.typeIdx);
        bool currHasTiledChildren = nextTiledTypeIdx != -1;
        if (isLeafLevel) {
            // Didn't expect child case
            if (currHasChildren || currHasTiledChildren) {
                const clang::ForStmt* firstChildLoop;
                LoopType firstChildType;
                if (currHasChildren) {
                    firstChildLoop = &node.loop->children.front().stmt;
                    firstChildType = node.loop->children.front().type.front();
                }
                if (currHasTiledChildren) {
                    firstChildLoop = &node.loop->stmt;
                    firstChildType = node.loop->type[nextTiledTypeIdx];
                }
                return tl::make_unexpected(
                    std::make_pair(firstChildLoop,
                                   fmt::format("Mismatch of [@{}] loops: didn't expect this loop",
                                               toString(firstChildType))));
            }
        } else {
            // We have to visist complex types (@tile) multiple times
            if (currHasTiledChildren) {
                node.typeIdx = nextTiledTypeIdx;
                nextQueue.push_back(node);
            } else if (currHasChildren) {
                for (auto& child : node.loop->children) {
                    nextQueue.push_back(LoopTreeNode{&child});
                }
            } else {
                // Missing child loop case
                return tl::make_unexpected(
                    std::make_pair(&node.loop->stmt,
                                   fmt::format("Mistmatch of [@{}] loops: expected loop inside",
                                               toString(currType))));
            }
        }
    }
    return verifyLoopStructureRecursiveBFS(nextQueue);
}

/**
 * @brief Function to verify the loop structure. All leafs must be at the same level and all levels
 * must have the same type.
 * @param topLevelOuterLoops A list of pointers to the top level outer loops.
 * @return An expected void value if the loop structure is valid, an unexpected pair of a pointer to
 * the problematic statement and an error message otherwise.
 */
tl::expected<void, std::pair<const clang::ForStmt*, std::string>> verifyLoopStructure(
    const std::list<OklLoopInfo*>& topLevelOuterLoops) {
    std::deque<LoopTreeNode> q;
    for (auto* loop : topLevelOuterLoops) {
        q.push_back({loop});
    }
    return verifyLoopStructureRecursiveBFS(q);
}

const clang::Attr* getFirstLoopAttribute(const clang::ForStmt* forStmt, SessionStage& stage) {
    static std::set<std::string> loopAttributesNames{
        TILE_ATTR_NAME, INNER_ATTR_NAME, OUTER_ATTR_NAME};
    if (!forStmt) {
        return nullptr;
    }
    auto attributedLoop = getAttributedStmt(stage, *clang::dyn_cast<clang::Stmt>(forStmt));
    if (!attributedLoop) {
        return nullptr;
    }
    for (const auto* attr : attributedLoop->getAttrs()) {
        auto name = attr->getNormalizedFullName();
        if (loopAttributesNames.count(name)) {
            return attr;
        }
    }
    return nullptr;
}

}  // namespace

/**
 * @brief Function to verify the loops tree in a kernel.
 * @param stage The current session stage.
 * @param kernelInfo The parsed kernel information.
 * @return An expected void value if the loops are valid, an unexpected error otherwise.
 */
tl::expected<void, Error> verifyLoops(SessionStage& stage,
                                      OklSemaCtx::ParsedKernelInfo& kernelInfo) {
    auto& topOuterLoops = kernelInfo.topLevelOuterLoops;
    if (topOuterLoops.empty()) {
        // If there is outer somewhere, but not on the top level
        for (auto& loop : kernelInfo.topLevelLoops) {
            if (loop.getFirstAttributedChild(
                    [](OklLoopInfo& info) { return info.has(LoopType::Outer); })) {
                return tl::make_unexpected(
                    Error{{}, "Cannot have [@inner] loop outside of an [@outer] loop"});
            }
        }

        // If there is no outer at all
        return tl::make_unexpected(Error{OkltPipelineErrorCode::AT_LEAST_ONE_OUTER_REQUIRED,
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
            return tl::make_unexpected(Error{OkltPipelineErrorCode::AT_LEAST_ONE_INNER_REQUIRED,
                                             "[@kernel] requires at least one [@inner] for-loop"});
        }
        return tl::make_unexpected(
            Error{OkltPipelineErrorCode::MISSING_INNER_LOOP, "Missing an [@inner] loop"});
    }

    // Verify loops hierarchy
    auto structureOk = verifyLoopStructure(topOuterLoops);
    if (!structureOk) {
        auto* failedLoopAttribute = getFirstLoopAttribute(structureOk.error().first, stage);
        return tl::make_unexpected(
            Error{{},
                  structureOk.error().second,
                  failedLoopAttribute == nullptr ? std::any() : failedLoopAttribute->getRange()});
    }
    return {};
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
