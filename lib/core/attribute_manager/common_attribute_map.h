#pragma once

#include "core/attribute_manager/attr_handler.h"
#include "core/attribute_manager/result.h"

#include <map>
namespace oklt {

class CommonAttributeMap {
   public:
    using KeyType = std::tuple<std::string, clang::ASTNodeKind>;
    using NodeHandlers = std::map<KeyType, AttrHandler>;

    CommonAttributeMap() = default;
    ~CommonAttributeMap() = default;

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
