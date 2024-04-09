#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/Attr.h>

#include <functional>
#include <tl/expected.hpp>

namespace oklt {

class SessionStage;

class DeclHandler {
   public:
    using HandleType = std::function<HandleResult(SessionStage&, const clang::Decl&)>;

    explicit DeclHandler(HandleType h);
    ~DeclHandler() = default;

    HandleResult operator()(SessionStage& stage, const clang::Decl&);

   private:
    HandleType _handler;
};
}  // namespace oklt
