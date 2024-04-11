#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>

#include <functional>
#include <tl/expected.hpp>

namespace oklt {

class SessionStage;

class NodeHandler {
   public:
    using HandleType = std::function<HandleResult(SessionStage&, const clang::DynTypedNode&)>;

    explicit NodeHandler(HandleType h);
    ~NodeHandler() = default;

    HandleResult operator()(SessionStage& stage, const clang::DynTypedNode&);

   private:
    HandleType _handler;
};
}  // namespace oklt
