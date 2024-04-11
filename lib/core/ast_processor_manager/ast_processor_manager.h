#pragma once

#include <oklt/core/ast_processor_types.h>
#include <oklt/util/string_utils.h>

#include "core/attribute_manager/result.h"
#include "util/type_traits.h"

#include <clang/AST/ASTTypeTraits.h>

#include <map>

namespace oklt {

class SessionStage;
class OklSemaCtx;

class AstProcessorManager {
   protected:
    AstProcessorManager() = default;
    ~AstProcessorManager() = default;

   public:
    using KeyType = std::tuple<AstProcessorType, std::string, clang::ASTNodeKind>;
    using HandleType =
        std::function<HandleResult(SessionStage&, const clang::DynTypedNode&, const clang::Attr*)>;

    struct NodeHandle {
        // run in direction from parent to child
        HandleType preAction;
        // run in direction from child to parent
        HandleType postAction;
    };

    static AstProcessorManager& instance();

    AstProcessorManager(const AstProcessorManager&) = delete;
    AstProcessorManager(AstProcessorManager&&) = delete;
    AstProcessorManager& operator=(const AstProcessorManager&) = delete;
    AstProcessorManager& operator=(AstProcessorManager&&) = delete;

    bool registerHandle(KeyType key, NodeHandle handle);

    HandleResult runPreActionNodeHandle(AstProcessorType procType,
                                        SessionStage& stage,
                                        const clang::DynTypedNode& node,
                                        const clang::Attr* attr);
    HandleResult runPostActionNodeHandle(AstProcessorType procType,
                                         SessionStage& stage,
                                         const clang::DynTypedNode& node,
                                         const clang::Attr* attr);

   private:
    std::map<KeyType, NodeHandle> _nodeHandlers;
};

namespace detail {
constexpr size_t HANDLE_NUM_OF_ARGS = 3;
template <typename Handler, typename HandleType>
HandleType makeSemaXXXHandle(Handler& handler) {
    using NodeType = typename std::remove_reference_t<typename func_param_type<Handler, 2>::type>;
    constexpr size_t nargs = func_num_arguments<Handler>::value;

    return HandleType{[&handler, nargs](SessionStage& stage,
                                        const clang::DynTypedNode& node,
                                        const clang::Attr* attr) -> HandleResult {
        static_assert(nargs == HANDLE_NUM_OF_ARGS, "Handler must have 3 arguments");

        const auto localNode = node.get<NodeType>();
        if (!localNode) {
            auto baseNodeTypeName = node.getNodeKind().asStringRef();
            auto handleNodeTypeName = clang::ASTNodeKind::getFromNodeKind<NodeType>().asStringRef();
            return tl::make_unexpected(Error{
                {},
                util::fmt(
                    "Failed to cast {} to {}", baseNodeTypeName.str(), handleNodeTypeName.str())
                    .value()});
        }

        if constexpr (!std::is_reference_v<typename func_param_type<Handler, 3>::type>) {
            return handler(stage, *localNode, attr);
        } else {
            if (!attr) {
                auto handleNodeTypeName =
                    clang::ASTNodeKind::getFromNodeKind<NodeType>().asStringRef();
                return tl::make_unexpected(
                    Error{{}, util::fmt("nullptr attr for {}", handleNodeTypeName.str()).value()});
            }
            return handler(stage, *localNode, *attr);
        }
    }};
};

}  // namespace detail

template <typename Handler>
auto makeSemaHandle(Handler& handler) {
    return detail::makeSemaXXXHandle<Handler, AstProcessorManager::HandleType>(handler);
}

}  // namespace oklt
