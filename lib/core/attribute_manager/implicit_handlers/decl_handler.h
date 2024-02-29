#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/Attr.h>

#include <functional>
#include <tl/expected.hpp>

namespace oklt {

class SessionStage;

class DeclHandler {
   public:
    using HandleType = std::function<HandleResult(const clang::Decl&, SessionStage&)>;

    explicit DeclHandler(HandleType h);
    ~DeclHandler() = default;

    HandleResult operator()(const clang::Decl&, SessionStage& stage);

   private:
    HandleType _handler;
};
}  // namespace oklt
