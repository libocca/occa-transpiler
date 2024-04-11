#pragma once

#include <oklt/core/target_backends.h>
#include "core/attribute_manager/attr_handler.h"
#include "core/attribute_manager/result.h"
#include "util/type_traits.h"

#include <map>
#include <tuple>

namespace oklt {
constexpr size_t N_ARGUMENTS_WITH_PARAMS = 4;
constexpr size_t N_ARGUMENTS_WITHOUT_PARAMS = 3;

class BackendAttributeMap {
   public:
    using KeyType = std::tuple<TargetBackend, std::string, clang::ASTNodeKind>;
    using NodeHandlers = std::map<KeyType, AttrHandler>;

    BackendAttributeMap() = default;
    ~BackendAttributeMap() = default;

    bool registerHandler(KeyType key, AttrHandler handler);

    HandleResult handleAttr(SessionStage& stage,
                            const clang::DynTypedNode& node,
                            const clang::Attr& attr,
                            const std::any* params);

    bool hasHandler(const KeyType& key);

   private:
    NodeHandlers _nodeHandlers;
};
}  // namespace oklt
