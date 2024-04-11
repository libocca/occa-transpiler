#pragma once

#include <oklt/core/target_backends.h>

#include "core/attribute_manager/implicit_handlers/node_handler.h"
#include "core/attribute_manager/result.h"

#include <any>
#include <map>
#include <tl/expected.hpp>
#include <tuple>

namespace oklt {
class ImplicitHandlerMap {
   public:
    using KeyType = std::tuple<TargetBackend, clang::ASTNodeKind>;
    using NodeHandlers = std::map<KeyType, NodeHandler>;

    ImplicitHandlerMap() = default;
    ~ImplicitHandlerMap() = default;

    bool registerHandler(KeyType key, NodeHandler handler);

    bool hasHandler(const KeyType& key) const;

    HandleResult operator()(SessionStage& stage, const clang::DynTypedNode& node);

   private:
    NodeHandlers _nodeHandlers;
};
}  // namespace oklt
