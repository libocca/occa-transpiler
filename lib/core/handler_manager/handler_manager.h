#pragma once
#include "core/attribute_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>

#include <unordered_map>

namespace oklt {
struct SessionStage;

struct HandlerManager {
    using HandlerType = std::function<
        HandleResult(const clang::Attr* attr, const clang::DynTypedNode node, SessionStage& stage)>;
    using KeyType = size_t;

    static HandlerManager& instance();

    bool registerHandler(KeyType, HandlerType);
    bool hasHandler(KeyType) const;

    HandlerType* tryGetHandler(KeyType);
    const HandlerType* tryGetHandler(KeyType) const;

    HandleResult runHandler(KeyType, const clang::Attr*, const clang::DynTypedNode, SessionStage&);

   private:
    std::unordered_map<size_t, HandlerType> _handlerMap;
};
}  // namespace oklt
